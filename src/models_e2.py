"""
Cấu hình E2 nâng cấp (Gói 1):
Salience Context-Window (title + text) + Dual-Hypothesis NLI (Epistemic head + Aleatoric head)
trên GPU CUDA (RTX 4050) với MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (fp32):
- Đóng băng 6 tầng Transformer đáy để giảm phương sai khi train trên dữ liệu vừa/nhỏ
- Phạt trọng số Hard-Negative + Label Smoothing
- 5-Fold OOF Cross-Validation Ensemble (hoán đổi state_dict từng fold trên 1 backbone duy nhất để tiết kiệm VRAM)
- Tích hợp Supervised Evidence Scorer từ xác suất câu (P_epi, P_alea, P_max)
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from evidence_scorer import SupervisedEvidenceScorer
from models_e0 import BBDVNRuleModel
from text_utils import (
    clean_for_model,
    normalize_text,
    select_threshold_with_hn_penalty,
    split_sentences_vi,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_E0_CFG = ROOT_DIR / "configs" / "E0.json"

_SALIENCE_PATTERNS = [
    (
        re.compile(
            r"\b(?:bất định|không chắc chắn|chưa rõ|khó lường|khó đoán|rủi ro|lo ngại|quan ngại|bỏ ngỏ|"
            r"cân nhắc|dự thảo|đề xuất|kiến nghị|lấy ý kiến|câu hỏi đặt ra|bất ổn|thận trọng|biến động|"
            r"biến số|có thể|dự kiến|đảo chiều|quay đầu|lao dốc|rơi thẳng đứng|trượt thẳng đứng|bất ngờ|"
            r"ngược chiều|thất thường|rung lắc|trồi sụt|sập mạnh|kịch trần|bán tháo|bốc hơi|tăng vọt|"
            r"tăng đột biến|gia tăng bất thường|tăng trần|giảm sâu|đột phá thể chế)\b",
            re.I,
        ),
        3.0,
    ),
    (
        re.compile(
            r"\b(?:chính sách|chính phủ|ngân hàng nhà nước|nhnn|fed|thuế|thuế quan|lãi suất|tỷ giá|"
            r"lạm phát|nghị định|thông tư|quy định|hạn mức|tín dụng|gdp|đầu tư công|chống bán phá giá|phá sản)\b",
            re.I,
        ),
        2.0,
    ),
    (
        re.compile(
            r"\b(?:kinh tế|thị trường|doanh nghiệp|xuất khẩu|nhập khẩu|chứng khoán|trái phiếu|giá vàng|"
            r"bất động sản|thanh khoản|usd|vàng miếng|vàng nhẫn|sjc|vn-index|cổ phiếu|giá gas|giá bạc|tài sản số)\b",
            re.I,
        ),
        1.0,
    ),
    (re.compile(r"-?\d+(?:[.,]\d+)?%"), 0.8),
]


def select_salience_context(
    text: str, title: Optional[str] = None, top_k: int = 3
) -> str:
    """
    Chọn tối đa top_k câu giàu tín hiệu từ `text` đã làm sạch (bỏ chú thích ảnh/byline,
    phạt câu nhiễu < 30 ký tự), giữ nguyên thứ tự gốc và ghép tiêu đề ở đầu nếu có.
    """
    sents = split_sentences_vi(text)
    cleaned_sents = [clean_for_model(s) for s in sents]
    cleaned_sents = [s for s in cleaned_sents if s]

    if not cleaned_sents:
        body = clean_for_model(text)
    elif len(cleaned_sents) <= top_k:
        body = " ".join(cleaned_sents)
    else:
        scored = []
        for idx, s in enumerate(cleaned_sents):
            score = 0.0
            for pat, w in _SALIENCE_PATTERNS:
                if pat.search(s):
                    score += w
            if idx == 0:
                score += 0.25
            if len(s) < 30:
                score -= 2.5
            scored.append((score, idx, s))
        top_indices = sorted([idx for _, idx, _ in sorted(scored, key=lambda x: (-x[0], x[1]))[:top_k]])
        body = " ".join(cleaned_sents[i] for i in top_indices)

    t_clean = clean_for_model(title) if title and str(title).strip().lower() != "nan" else ""
    if t_clean and not body.lower().startswith(t_clean.lower()[: min(30, len(t_clean))]):
        sep = " " if t_clean.endswith((".", "!", "?", ":")) else ". "
        return f"{t_clean}{sep}{body}".strip()
    return body.strip()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_nli_indices(label2id: Dict[str, int]) -> Tuple[int, int]:
    """Đọc động chỉ số entailment và contradiction từ model.config.label2id."""
    norm_map = {str(k).lower(): int(v) for k, v in label2id.items()}
    ent_idx = None
    con_idx = None
    for k, v in norm_map.items():
        if "entail" in k:
            ent_idx = v
        elif "contrad" in k:
            con_idx = v
    if ent_idx is None or con_idx is None:
        raise ValueError(f"Không tìm thấy nhãn entailment/contradiction trong label2id={label2id}")
    return ent_idx, con_idx


class DualNLIDataset(Dataset):
    """Sinh 2 cặp (premise, H_epistemic) và (premise, H_aleatoric) kèm trọng số từng mẫu."""

    def __init__(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        labels: Sequence[int],
        weights: Sequence[float],
        tokenizer,
        max_length: int,
    ):
        self.premises = list(premises)
        self.hypotheses = list(hypotheses)
        self.labels = list(labels)
        self.weights = list(weights)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.premises)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = int(self.labels[idx])
        enc["sample_weight"] = float(self.weights[idx])
        return enc


def _collate_fn(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attn_masks, labels, weights = [], [], [], []
    for x in batch:
        pad_n = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_n)
        attn_masks.append(x["attention_mask"] + [0] * pad_n)
        labels.append(x["labels"])
        weights.append(x["sample_weight"])
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_weight": torch.tensor(weights, dtype=torch.float32),
    }
    if "token_type_ids" in batch[0]:
        tti = [x["token_type_ids"] + [0] * (max_len - len(x["token_type_ids"])) for x in batch]
        out["token_type_ids"] = torch.tensor(tti, dtype=torch.long)
    return out


class SalienceDefinitionNLI:
    def __init__(self, config_path: Path, device: str | None = None):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.cfg["model_name"]
        self.h_epi = self.cfg.get("hypothesis_epistemic", self.cfg.get("hypothesis_template", ""))
        self.h_alea = self.cfg.get("hypothesis_aleatoric", self.h_epi)
        self.freeze_bottom = int(self.cfg.get("freeze_bottom_layers", 6))
        self.top_k = int(self.cfg.get("salience_top_k_sentences", 3))
        self.max_length = int(self.cfg.get("max_length", 256))
        self.batch_size = int(self.cfg.get("batch_size", 4))
        self.grad_accum = int(self.cfg.get("gradient_accumulation_steps", 4))
        self.lr = float(self.cfg.get("learning_rate", 2.5e-5))
        self.weight_decay = float(self.cfg.get("weight_decay", 0.05))
        self.label_smoothing = float(self.cfg.get("label_smoothing", 0.05))
        self.hn_weight = float(self.cfg.get("hard_negative_weight", 1.6))
        self.epochs = int(self.cfg.get("epochs", 6))
        self.patience = int(self.cfg.get("patience", 3))
        # Cảnh báo kỹ thuật: luôn giữ fp32 cho mDeBERTa-v3 để tránh tràn số / NaN
        self.use_bf16 = False

        self.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)
        self.tokenizer = None
        self.model = None
        self.ent_idx = 0
        self.con_idx = 2

        self.best_threshold = 0.5
        self.best_fold_idx = 0
        self.fold_trainable_states: List[Dict[str, torch.Tensor]] = []
        self.fold_ev_scorers: List[SupervisedEvidenceScorer] = []

    def _apply_freeze(self, model) -> None:
        if self.freeze_bottom <= 0:
            return
        deberta = getattr(model, "deberta", None)
        if deberta is None:
            return
        if hasattr(deberta, "embeddings"):
            for p in deberta.embeddings.parameters():
                p.requires_grad = False
        enc = getattr(deberta, "encoder", None)
        if enc is not None and hasattr(enc, "layer"):
            for layer in enc.layer[: self.freeze_bottom]:
                for p in layer.parameters():
                    p.requires_grad = False

    def _init_backbone(self, model_path_or_name: str | None = None, apply_freeze: bool = True) -> None:
        src = model_path_or_name or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.float32)
        self.ent_idx, self.con_idx = resolve_nli_indices(self.model.config.label2id)
        if apply_freeze:
            self._apply_freeze(self.model)
        self.model.to(self.device)

    def _extract_trainable_state(self) -> Dict[str, torch.Tensor]:
        assert self.model is not None
        trainable_names = {n for n, p in self.model.named_parameters() if p.requires_grad}
        state: Dict[str, torch.Tensor] = {}
        for k, v in self.model.state_dict().items():
            if k in trainable_names or not (
                k.startswith("deberta.embeddings.")
                or any(k.startswith(f"deberta.encoder.layer.{idx}.") for idx in range(self.freeze_bottom))
            ):
                state[k] = v.detach().cpu().clone()
        return state

    def _swap_fold_state(self, fold_idx: int) -> None:
        assert self.model is not None and self.fold_trainable_states
        st = self.fold_trainable_states[fold_idx]
        missing, unexpected = self.model.load_state_dict(
            {k: v.to(self.device) for k, v in st.items()},
            strict=False,
        )
        # Chỉ các tầng đã đóng băng mới được phép không nằm trong state rút gọn
        assert not unexpected, f"Unexpected keys in fold state: {unexpected}"
        for m in missing:
            assert (
                m.startswith("deberta.embeddings.")
                or any(m.startswith(f"deberta.encoder.layer.{idx}.") for idx in range(self.freeze_bottom))
            ), f"Thiếu trọng số tầng trainable khi nạp fold {fold_idx}: {m}"

    @torch.inference_mode()
    def _predict_pair_probs(
        self, premises: Sequence[str], hypothesis: str, infer_batch_size: int = 16, max_len: Optional[int] = None
    ) -> np.ndarray:
        assert self.model is not None and self.tokenizer is not None
        self.model.eval()
        ml = int(max_len or self.max_length)
        all_probs: List[float] = []
        for i in range(0, len(premises), infer_batch_size):
            batch_p = list(premises[i : i + infer_batch_size])
            enc = self.tokenizer(
                batch_p,
                [hypothesis] * len(batch_p),
                truncation=True,
                max_length=ml,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits.float()
            pair_logits = torch.stack([logits[:, self.con_idx], logits[:, self.ent_idx]], dim=-1)
            probs = torch.softmax(pair_logits, dim=-1)[:, 1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())
        return np.asarray(all_probs, dtype=np.float32)

    def predict_dual_probs(
        self, premises: Sequence[str], infer_batch_size: int = 16, max_len: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trả về (p_epistemic, p_aleatoric) cho danh sách premise ở trạng thái weight hiện hành."""
        p_epi = self._predict_pair_probs(premises, self.h_epi, infer_batch_size=infer_batch_size, max_len=max_len)
        p_alea = self._predict_pair_probs(premises, self.h_alea, infer_batch_size=infer_batch_size, max_len=max_len)
        return p_epi, p_alea

    def eval_zero_shot_on_dev(
        self,
        dev_texts: Sequence[str],
        dev_titles: Optional[Sequence[Optional[str]]],
        dev_y: Sequence[int],
        dev_boundary_types: Sequence[str],
    ) -> Dict:
        self._init_backbone(apply_freeze=False)
        if dev_titles is None:
            dev_titles = [None] * len(dev_texts)
        dv_prem = [select_salience_context(t, title=ttl, top_k=self.top_k) for t, ttl in zip(dev_texts, dev_titles)]
        p_epi, p_alea = self.predict_dual_probs(dv_prem)
        p_unc = np.maximum(p_epi, p_alea)
        thr_grid = [float(t) for t in self.cfg.get("threshold_grid", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])]
        thr, macro, f1_pos, hn_fpr = select_threshold_with_hn_penalty(
            p_unc, dev_y, dev_boundary_types, thr_grid, hn_penalty=0.0, hn_target=0.10
        )
        return {
            "zero_shot_dev_threshold": thr,
            "zero_shot_dev_macro_f1": round(macro, 4),
            "zero_shot_dev_f1_uncertain": round(f1_pos, 4),
            "zero_shot_dev_hn_fpr": round(hn_fpr, 4),
        }

    def _fit_single_fold(
        self,
        tr_texts: Sequence[str],
        tr_titles: Sequence[Optional[str]],
        tr_y_bin: Sequence[int],
        tr_y_epi: Sequence[int],
        tr_y_alea: Sequence[int],
        tr_bt: Sequence[str],
        tr_gold_ev: Sequence[str],
        va_texts: Sequence[str],
        va_titles: Sequence[Optional[str]],
        va_y_bin: Sequence[int],
        va_bt: Sequence[str],
        seed: int,
        log_prefix: str = "E2",
    ) -> Dict[str, Any]:
        set_all_seeds(seed)
        self._init_backbone(apply_freeze=True)
        total_p = sum(p.numel() for p in self.model.parameters())
        train_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  [{log_prefix} seed={seed}] Trainable params: {train_p:,} / {total_p:,} ({100.0*train_p/total_p:.1f}%, fp32)")

        tr_prem = [select_salience_context(t, title=ttl, top_k=self.top_k) for t, ttl in zip(tr_texts, tr_titles)]
        va_prem = [select_salience_context(t, title=ttl, top_k=self.top_k) for t, ttl in zip(va_texts, va_titles)]

        # Xây tập cặp kép (H_epi + H_alea) kèm trọng số cân bằng lớp + phạt Hard-Negative
        ye_arr = np.asarray(tr_y_epi, dtype=int)
        ya_arr = np.asarray(tr_y_alea, dtype=int)
        n_tr = len(tr_prem)
        w1_epi = min(3.5, n_tr / (2.0 * max(1, int((ye_arr == 1).sum()))))
        w0_epi = n_tr / (2.0 * max(1, int((ye_arr == 0).sum())))
        w1_alea = min(3.5, n_tr / (2.0 * max(1, int((ya_arr == 1).sum()))))
        w0_alea = n_tr / (2.0 * max(1, int((ya_arr == 0).sum())))

        pair_p, pair_h, pair_y, pair_w = [], [], [], []
        for prem, ye, ya, bt in zip(tr_prem, ye_arr, ya_arr, tr_bt):
            is_hn = str(bt) == "negative_certain_hard_negative"
            # Cặp Epistemic
            pair_p.append(prem)
            pair_h.append(self.h_epi)
            pair_y.append(int(ye))
            we = w1_epi if ye == 1 else (w0_epi * (self.hn_weight if is_hn else 1.0))
            pair_w.append(float(we))
            # Cặp Aleatoric
            pair_p.append(prem)
            pair_h.append(self.h_alea)
            pair_y.append(int(ya))
            wa = w1_alea if ya == 1 else (w0_alea * (self.hn_weight if is_hn else 1.0))
            pair_w.append(float(wa))

        ds = DualNLIDataset(pair_p, pair_h, pair_y, pair_w, self.tokenizer, self.max_length)
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: _collate_fn(b, self.tokenizer.pad_token_id or 0),
        )

        loss_fn = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smoothing)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = max(1, (len(loader) // self.grad_accum + 1) * self.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * float(self.cfg.get("warmup_ratio", 0.15)))),
            num_training_steps=total_steps,
        )

        thr_grid = [float(t) for t in self.cfg.get("threshold_grid", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])]
        hn_pen = float(self.cfg.get("hn_penalty", 0.08))
        hn_tgt = float(self.cfg.get("hn_target", 0.10))

        best_key = -1e9
        best_trainable: Optional[Dict[str, torch.Tensor]] = None
        best_thr = 0.5
        best_macro = 0.0
        best_f1_pos = 0.0
        best_epoch = 0
        no_improve = 0
        best_va_epi = None
        best_va_alea = None

        va_y_arr = np.asarray(va_y_bin, dtype=float)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            for step, batch in enumerate(loader, start=1):
                labels = batch.pop("labels").to(self.device)
                sw = batch.pop("sample_weight").to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**inputs).logits
                pair_logits = torch.stack([logits[:, self.con_idx], logits[:, self.ent_idx]], dim=-1).float()
                per_loss = loss_fn(pair_logits, labels) * sw
                loss = per_loss.mean() / self.grad_accum
                if torch.isnan(loss):
                    raise RuntimeError("Phát hiện NaN loss trong huấn luyện Dual-Hypothesis mDeBERTa!")
                loss.backward()
                epoch_loss += float(loss.item()) * self.grad_accum

                if step % self.grad_accum == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            va_epi, va_alea = self.predict_dual_probs(va_prem)
            va_unc = np.clip(np.maximum(va_epi, va_alea), 1e-6, 1.0 - 1e-6)
            bce = float(-np.mean(va_y_arr * np.log(va_unc) + (1.0 - va_y_arr) * np.log(1.0 - va_unc)))
            thr, macro, f1_pos, hn_fpr = select_threshold_with_hn_penalty(
                va_unc, va_y_bin, va_bt, thr_grid, hn_penalty=hn_pen, hn_target=hn_tgt
            )
            key = macro - hn_pen * max(0.0, hn_fpr - hn_tgt) + 1e-4 * f1_pos - 1e-3 * bce
            print(
                f"  [{log_prefix} seed={seed}] ep {epoch}/{self.epochs} | loss={epoch_loss/len(loader):.4f} | "
                f"val_macro={macro:.4f} | hn_fpr={hn_fpr:.3f} | thr={thr:.3f}"
            )
            if key > best_key:
                best_key = key
                best_macro = macro
                best_f1_pos = f1_pos
                best_thr = thr
                best_epoch = epoch
                best_trainable = self._extract_trainable_state()
                best_va_epi = va_epi.copy()
                best_va_alea = va_alea.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        assert best_trainable is not None
        self.model.load_state_dict({k: v.to(self.device) for k, v in best_trainable.items()}, strict=False)

        # Chấm điểm từng câu để huấn luyện SupervisedEvidenceScorer riêng trong fold
        tr_ev_records = []
        tr_all_sents, tr_offsets = self._gather_sentences(tr_texts)
        tr_se, tr_sa = self.predict_dual_probs(tr_all_sents, infer_batch_size=32)
        for i, (s_start, s_end) in enumerate(tr_offsets):
            se_i = tr_se[s_start:s_end]
            sa_i = tr_sa[s_start:s_end]
            s_mat = np.column_stack([se_i, sa_i, np.maximum(se_i, sa_i)]).astype(np.float32)
            tr_ev_records.append(
                {
                    "text": tr_texts[i],
                    "title": tr_titles[i],
                    "gold_evidence": tr_gold_ev[i],
                    "label": int(tr_y_bin[i]),
                    "model_sent_scores": s_mat,
                }
            )
        ev_scorer = SupervisedEvidenceScorer(C=0.5, seed=seed).fit(tr_ev_records, rule_model=self.rule_model)

        va_all_sents, va_offsets = self._gather_sentences(va_texts)
        va_se, va_sa = self.predict_dual_probs(va_all_sents, infer_batch_size=32)
        va_evidences: List[str] = []
        va_sent_scores: List[np.ndarray] = []
        for i, (s_start, s_end) in enumerate(va_offsets):
            se_i = va_se[s_start:s_end]
            sa_i = va_sa[s_start:s_end]
            s_mat = np.column_stack([se_i, sa_i, np.maximum(se_i, sa_i)]).astype(np.float32)
            va_sent_scores.append(s_mat)
            va_evidences.append(
                ev_scorer.select_best_sentence(
                    raw_text=va_texts[i],
                    title=va_titles[i],
                    rule_model=self.rule_model,
                    model_sent_scores=s_mat,
                )
            )

        return {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_threshold": best_thr,
            "fold_macro_f1": best_macro,
            "fold_f1_uncertain": best_f1_pos,
            "trainable_state": best_trainable,
            "ev_scorer": ev_scorer,
            "va_p_epi": best_va_epi,
            "va_p_alea": best_va_alea,
            "va_evidences": va_evidences,
            "va_sent_scores": va_sent_scores,
        }

    @staticmethod
    def _gather_sentences(texts: Sequence[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
        all_s: List[str] = []
        offsets: List[Tuple[int, int]] = []
        for t in texts:
            sents = split_sentences_vi(t)
            if not sents:
                fb = normalize_text(t)
                sents = [fb] if fb else [""]
            start = len(all_s)
            all_s.extend(sents)
            offsets.append((start, len(all_s)))
        return all_s, offsets

    def fit_cv(
        self,
        texts: Sequence[str],
        titles: Sequence[Optional[str]],
        y_bin: Sequence[int],
        y_epi: Sequence[int],
        y_alea: Sequence[int],
        boundary_types: Sequence[str],
        gold_evidences: Sequence[str],
        fold_ids: Sequence[int],
        seeds: Optional[Sequence[int]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Huấn luyện 5-Fold OOF Ensemble (hoặc 3-Fold trong LOSO):
        - Lưu 1 state_dict trainable cho mỗi fold + kiểm tra parity khi nạp lại Fold 0.
        - Trả về (info, oof_preds, per_fold_records).
        """
        n = len(texts)
        texts_l = list(texts)
        titles_l = list(titles)
        y_bin_arr = np.asarray(y_bin, dtype=int)
        y_epi_arr = np.asarray(y_epi, dtype=int)
        y_alea_arr = np.asarray(y_alea, dtype=int)
        fold_arr = np.asarray(fold_ids, dtype=int)
        uniq_folds = sorted(np.unique(fold_arr).tolist())

        cfg_seeds = [int(s) for s in (seeds if seeds is not None else self.cfg.get("fold_seeds", [42, 123, 2026, 777, 999]))]
        fold_seeds = [cfg_seeds[i % len(cfg_seeds)] for i in range(len(uniq_folds))]

        self.fold_trainable_states = []
        self.fold_ev_scorers = []
        oof_epi = np.zeros(n, dtype=np.float32)
        oof_alea = np.zeros(n, dtype=np.float32)
        oof_evidences = [""] * n
        oof_sent_scores: List[np.ndarray] = [np.zeros((1, 3), dtype=np.float32) for _ in range(n)]
        fold_logs: List[Dict[str, Any]] = []

        best_fold_score = -1.0
        best_fold_idx = 0

        for pos_idx, (f, sd) in enumerate(zip(uniq_folds, fold_seeds)):
            tr_idx = np.where(fold_arr != f)[0]
            va_idx = np.where(fold_arr == f)[0]
            res = self._fit_single_fold(
                tr_texts=[texts_l[i] for i in tr_idx],
                tr_titles=[titles_l[i] for i in tr_idx],
                tr_y_bin=y_bin_arr[tr_idx],
                tr_y_epi=y_epi_arr[tr_idx],
                tr_y_alea=y_alea_arr[tr_idx],
                tr_bt=[boundary_types[i] for i in tr_idx],
                tr_gold_ev=[gold_evidences[i] for i in tr_idx],
                va_texts=[texts_l[i] for i in va_idx],
                va_titles=[titles_l[i] for i in va_idx],
                va_y_bin=y_bin_arr[va_idx],
                va_bt=[boundary_types[i] for i in va_idx],
                seed=sd,
                log_prefix=f"E2 fold={f}",
            )

            # Kiểm tra unit assertion trên Fold 0 để chắc chắn nạp trainable state tái lập nguyên vẹn logits
            if pos_idx == 0:
                va_sample_prem = [
                    select_salience_context(texts_l[i], title=titles_l[i], top_k=self.top_k)
                    for i in va_idx[: min(6, len(va_idx))]
                ]
                orig_e, orig_a = self.predict_dual_probs(va_sample_prem)
                self._init_backbone(apply_freeze=True)
                self.model.load_state_dict({k: v.to(self.device) for k, v in res["trainable_state"].items()}, strict=False)
                re_e, re_a = self.predict_dual_probs(va_sample_prem)
                diff = max(float(np.max(np.abs(orig_e - re_e))), float(np.max(np.abs(orig_a - re_a))))
                assert diff < 1e-4, f"Fold 0 trainable reload parity failed: max_abs_diff={diff}"
                print(f"  [E2 Parity Check] Fold 0 reload max_abs_diff = {diff:.2e} (< 1e-4 PASS)")

            self.fold_trainable_states.append(res["trainable_state"])
            self.fold_ev_scorers.append(res["ev_scorer"])

            for local_k, orig_i in enumerate(va_idx):
                oof_epi[orig_i] = res["va_p_epi"][local_k]
                oof_alea[orig_i] = res["va_p_alea"][local_k]
                oof_evidences[orig_i] = res["va_evidences"][local_k]
                oof_sent_scores[orig_i] = res["va_sent_scores"][local_k]

            f_key = res["fold_macro_f1"] + 1e-4 * res["fold_f1_uncertain"]
            if f_key > best_fold_score:
                best_fold_score = f_key
                best_fold_idx = pos_idx

            fold_logs.append(
                {
                    "fold": f,
                    "seed": sd,
                    "best_epoch": res["best_epoch"],
                    "best_threshold": res["best_threshold"],
                    "oof_fold_macro_f1": round(res["fold_macro_f1"], 4),
                    "oof_fold_f1_uncertain": round(res["fold_f1_uncertain"], 4),
                }
            )
            # Dọn dẹp VRAM giữa các fold
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Nạp lại 1 backbone duy nhất để suy luận tiếp mà không tái tải từ đĩa
        self._init_backbone(apply_freeze=True)
        self.best_fold_idx = best_fold_idx

        oof_unc = np.maximum(oof_epi, oof_alea)
        thr_grid = [float(t) for t in self.cfg.get("threshold_grid", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])]
        hn_pen = float(self.cfg.get("hn_penalty", 0.08))
        hn_tgt = float(self.cfg.get("hn_target", 0.10))
        best_thr, oof_macro, oof_f1_pos, oof_hn_fpr = select_threshold_with_hn_penalty(
            oof_unc, y_bin_arr, boundary_types, thr_grid, hn_penalty=hn_pen, hn_target=hn_tgt
        )
        self.best_threshold = float(best_thr)

        oof_preds: List[Dict[str, Any]] = []
        for i in range(n):
            p = float(oof_unc[i])
            pe = float(oof_epi[i])
            pa = float(oof_alea[i])
            is_u = p >= self.best_threshold
            if not is_u:
                ut = "none"
            elif pe >= 0.45 and pa >= 0.45:
                ut = "both"
            elif pa > pe:
                ut = "aleatoric"
            else:
                ut = "epistemic"
            oof_preds.append(
                {
                    "pred_label": "uncertain" if is_u else "not_uncertain",
                    "prob_uncertain": round(p, 6),
                    "prob_epistemic": round(pe, 6),
                    "prob_aleatoric": round(pa, 6),
                    "pred_uncertainty_type": ut,
                    "pred_evidence": oof_evidences[i],
                    "sent_scores": oof_sent_scores[i],
                }
            )

        info = {
            "best_threshold": self.best_threshold,
            "best_fold_idx": self.best_fold_idx,
            "oof_macro_f1": round(oof_macro, 4),
            "oof_f1_uncertain": round(oof_f1_pos, 4),
            "oof_hn_fpr": round(oof_hn_fpr, 4),
            "n_folds": len(uniq_folds),
        }
        return info, oof_preds, fold_logs

    def predict_batch(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        extract_evidence: bool = True,
        infer_batch_size: int = 16,
        use_ensemble: bool = True,
        return_sent_scores: bool = False,
        specific_fold_idx: Optional[int] = None,
    ) -> List[Dict]:
        assert self.model is not None and self.fold_trainable_states, "Mô hình E2 chưa được huấn luyện!"
        if titles is None:
            titles = [None] * len(texts)

        premises = [select_salience_context(t, title=ttl, top_k=self.top_k) for t, ttl in zip(texts, titles)]
        if specific_fold_idx is not None:
            fold_indices = [int(specific_fold_idx)]
        elif use_ensemble:
            fold_indices = list(range(len(self.fold_trainable_states)))
        else:
            fold_indices = [self.best_fold_idx]

        n = len(texts)
        acc_epi = np.zeros(n, dtype=np.float64)
        acc_alea = np.zeros(n, dtype=np.float64)

        all_sents: List[str] = []
        offsets: List[Tuple[int, int]] = []
        if extract_evidence or return_sent_scores:
            all_sents, offsets = self._gather_sentences(texts)

        acc_sent_epi = np.zeros(len(all_sents), dtype=np.float64) if all_sents else None
        acc_sent_alea = np.zeros(len(all_sents), dtype=np.float64) if all_sents else None

        for fi in fold_indices:
            self._swap_fold_state(fi)
            pe, pa = self.predict_dual_probs(premises, infer_batch_size=infer_batch_size)
            acc_epi += pe
            acc_alea += pa
            if (extract_evidence or return_sent_scores) and all_sents:
                se, sa = self.predict_dual_probs(all_sents, infer_batch_size=infer_batch_size * 2)
                acc_sent_epi += se
                acc_sent_alea += sa

        k = float(len(fold_indices))
        p_epi = acc_epi / k
        p_alea = acc_alea / k
        p_unc = np.maximum(p_epi, p_alea)

        best_evidences = [normalize_text(t) for t in texts]
        doc_sent_scores_list: List[Optional[np.ndarray]] = [None] * n
        if (extract_evidence or return_sent_scores) and all_sents:
            se_avg = (acc_sent_epi / k).astype(np.float32)
            sa_avg = (acc_sent_alea / k).astype(np.float32)
            best_evidences = []
            for idx, (s_start, s_end) in enumerate(offsets):
                se_i = se_avg[s_start:s_end]
                sa_i = sa_avg[s_start:s_end]
                s_mat = np.column_stack([se_i, sa_i, np.maximum(se_i, sa_i)]).astype(np.float32)
                doc_sent_scores_list[idx] = s_mat
                if extract_evidence:
                    fold_ev_p = np.zeros(s_end - s_start, dtype=np.float64)
                    for fi in fold_indices:
                        scorer = self.fold_ev_scorers[fi] if fi < len(self.fold_ev_scorers) else None
                        if scorer is not None:
                            _, ev_p = scorer.score_sentences(
                                raw_text=texts[idx],
                                title=titles[idx],
                                rule_model=self.rule_model,
                                model_sent_scores=s_mat,
                            )
                            fold_ev_p += ev_p
                        else:
                            fold_ev_p += s_mat[:, 2]
                    best_evidences.append(all_sents[s_start + int(np.argmax(fold_ev_p))])
                else:
                    best_evidences.append(all_sents[s_start])

        results: List[Dict] = []
        for i in range(n):
            p = float(p_unc[i])
            pe = float(p_epi[i])
            pa = float(p_alea[i])
            is_u = p >= self.best_threshold
            if not is_u:
                ut = "none"
            elif pe >= 0.45 and pa >= 0.45:
                ut = "both"
            elif pa > pe:
                ut = "aleatoric"
            else:
                ut = "epistemic"
            rec: Dict[str, Any] = {
                "pred_label": "uncertain" if is_u else "not_uncertain",
                "prob_uncertain": round(p, 6),
                "prob_epistemic": round(pe, 6),
                "prob_aleatoric": round(pa, 6),
                "pred_uncertainty_type": ut,
                "pred_evidence": best_evidences[i],
                "salience_premise": premises[i],
            }
            if return_sent_scores:
                rec["sent_scores"] = doc_sent_scores_list[i]
            results.append(rec)
        return results

    def predict_probs_single_fold_fast(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        infer_batch_size: int = 16,
        max_len: int = 224,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Suy luận nhanh toàn corpus (39.028 bài) trên GPU bằng best_fold_idx + length-sorting."""
        assert self.model is not None and self.fold_trainable_states
        if titles is None:
            titles = [None] * len(texts)
        self._swap_fold_state(self.best_fold_idx)
        premises = [select_salience_context(t, title=ttl, top_k=self.top_k) for t, ttl in zip(texts, titles)]
        order = np.argsort([len(p) for p in premises])
        sorted_prem = [premises[i] for i in order]
        s_epi, s_alea = self.predict_dual_probs(sorted_prem, infer_batch_size=infer_batch_size, max_len=max_len)
        p_epi = np.empty_like(s_epi)
        p_alea = np.empty_like(s_alea)
        p_epi[order] = s_epi
        p_alea[order] = s_alea
        p_unc = np.maximum(p_epi, p_alea)
        return p_unc, p_epi, p_alea

    def save(self, save_dir: Path | str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        for idx, st in enumerate(self.fold_trainable_states):
            torch.save(st, save_dir / f"fold_{idx}_trainable.pt")
        joblib.dump(self.fold_ev_scorers, save_dir / "ev_scorers.joblib")
        meta = {
            "cfg": self.cfg,
            "ent_idx": self.ent_idx,
            "con_idx": self.con_idx,
            "best_threshold": self.best_threshold,
            "best_fold_idx": self.best_fold_idx,
            "n_folds": len(self.fold_trainable_states),
        }
        (save_dir / "e2_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: Path | str, device: str | None = None) -> "SalienceDefinitionNLI":
        save_dir = Path(save_dir)
        meta = json.loads((save_dir / "e2_meta.json").read_text(encoding="utf-8"))
        inst = cls.__new__(cls)
        inst.cfg = meta["cfg"]
        inst.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        inst.model_name = inst.cfg["model_name"]
        inst.h_epi = inst.cfg.get("hypothesis_epistemic", inst.cfg.get("hypothesis_template", ""))
        inst.h_alea = inst.cfg.get("hypothesis_aleatoric", inst.h_epi)
        inst.freeze_bottom = int(inst.cfg.get("freeze_bottom_layers", 6))
        inst.top_k = int(inst.cfg.get("salience_top_k_sentences", 3))
        inst.max_length = int(inst.cfg.get("max_length", 256))
        inst.batch_size = int(inst.cfg.get("batch_size", 4))
        inst.grad_accum = int(inst.cfg.get("gradient_accumulation_steps", 4))
        inst.lr = float(inst.cfg.get("learning_rate", 2.5e-5))
        inst.weight_decay = float(inst.cfg.get("weight_decay", 0.05))
        inst.label_smoothing = float(inst.cfg.get("label_smoothing", 0.05))
        inst.hn_weight = float(inst.cfg.get("hard_negative_weight", 1.6))
        inst.epochs = int(inst.cfg.get("epochs", 6))
        inst.patience = int(inst.cfg.get("patience", 3))
        inst.use_bf16 = False
        inst.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)

        inst.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        inst.model = AutoModelForSequenceClassification.from_pretrained(inst.model_name, dtype=torch.float32)
        inst.ent_idx = int(meta["ent_idx"])
        inst.con_idx = int(meta["con_idx"])
        inst._apply_freeze(inst.model)
        inst.model.to(inst.device)

        n_folds = int(meta["n_folds"])
        inst.fold_trainable_states = [
            torch.load(save_dir / f"fold_{idx}_trainable.pt", map_location="cpu", weights_only=True)
            for idx in range(n_folds)
        ]
        ev_path = save_dir / "ev_scorers.joblib"
        inst.fold_ev_scorers = joblib.load(ev_path) if ev_path.exists() else []
        inst.best_threshold = float(meta["best_threshold"])
        inst.best_fold_idx = int(meta.get("best_fold_idx", 0))
        inst._swap_fold_state(inst.best_fold_idx)
        return inst

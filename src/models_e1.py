"""
Cấu hình E1 nâng cấp (Gói 1):
Sentence-Pooled Vietnamese Bi-Encoder ([title_emb, mean_sent_emb, max_sent_emb] = 2304-d)
chạy trên GPU CUDA + Dual-Head L2 Logistic Regression (5-Fold OOF CV phạt Hard-Negative)
+ Sentence Probe + Supervised Evidence Scorer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from evidence_scorer import SupervisedEvidenceScorer
from models_e0 import BBDVNRuleModel
from text_utils import (
    clean_for_model,
    normalize_text,
    segment_vi_words,
    select_threshold_with_hn_penalty,
    split_sentences_vi,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_E0_CFG = ROOT_DIR / "configs" / "E0.json"


def _l2_norm_vec(v: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-12:
        return v
    return (v / nrm).astype(np.float32)


def _fit_lr_head_dense(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> LogisticRegression:
    uniq = np.unique(y)
    clf = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    if len(uniq) < 2:
        clf.fit(X[:2], np.array([0, 1], dtype=int))
        return clf
    clf.fit(X, y)
    return clf


class VietnameseBiEncoderProbe:
    def __init__(self, config_path: Path, device: str | None = None):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.seed = int(self.cfg.get("seed", 42))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = self.cfg["model_name"]
        self.use_pyvi = bool(self.cfg.get("use_pyvi_segmentation", True))
        self.max_length = int(self.cfg.get("max_length", 256))
        self.batch_size = int(self.cfg.get("batch_size", 64))
        self.normalize_emb = bool(self.cfg.get("normalize_embeddings", True))

        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        self.encoder.max_seq_length = self.max_length
        self.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)

        self.fold_bundles: List[Dict[str, Any]] = []
        self.best_C = 1.0
        self.best_alpha = 0.7
        self.best_threshold = 0.5

    def _prep_text(self, t: Optional[str]) -> str:
        c = clean_for_model(t)
        if not c:
            c = normalize_text(t)
        if not c:
            return ""
        return segment_vi_words(c) if self.use_pyvi else c

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        prepped = [self._prep_text(t) for t in texts]
        embs = self.encoder.encode(
            prepped,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_emb,
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype=np.float32)

    def encode_multiview(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        show_progress: bool = False,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[str]], List[np.ndarray], List[np.ndarray]]:
        """
        Mã hóa 1 lượt trên GPU tất cả câu và tiêu đề:
        Trả về:
          - X_doc [N, 3*d] = [title_emb, mean_sent_emb, max_sent_emb]
          - sent_embs_list: list mảng [n_sents_i, d]
          - sents_list: list câu nguyên văn của từng đoạn
          - sent_title_cos_list, sent_doc_cos_list
        """
        if titles is None:
            titles = [None] * len(texts)

        sents_list: List[List[str]] = []
        all_prepped_sents: List[str] = []
        offsets: List[Tuple[int, int]] = []

        non_empty_titles: List[str] = []
        title_slot_map: List[int] = []  # -1 nếu không có tiêu đề

        for t, ttl in zip(texts, titles):
            sents = split_sentences_vi(t)
            if not sents:
                fb = normalize_text(t)
                sents = [fb] if fb else [""]
            sents_list.append(sents)
            start = len(all_prepped_sents)
            for s in sents:
                all_prepped_sents.append(self._prep_text(s))
            offsets.append((start, len(all_prepped_sents)))

            ttl_str = "" if ttl is None or str(ttl).strip().lower() == "nan" else str(ttl).strip()
            ttl_prepped = self._prep_text(ttl_str) if ttl_str else ""
            if ttl_prepped:
                title_slot_map.append(len(non_empty_titles))
                non_empty_titles.append(ttl_prepped)
            else:
                title_slot_map.append(-1)

        all_sent_embs = self.encoder.encode(
            all_prepped_sents,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        if non_empty_titles:
            title_embs_pool = self.encoder.encode(
                non_empty_titles,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
        else:
            title_embs_pool = np.zeros((0, all_sent_embs.shape[1]), dtype=np.float32)

        doc_vecs: List[np.ndarray] = []
        sent_embs_list: List[np.ndarray] = []
        sent_title_cos_list: List[np.ndarray] = []
        sent_doc_cos_list: List[np.ndarray] = []

        for idx, (start, end) in enumerate(offsets):
            s_emb = all_sent_embs[start:end]
            sent_embs_list.append(s_emb)
            mean_v = _l2_norm_vec(s_emb.mean(axis=0))
            max_v = _l2_norm_vec(s_emb.max(axis=0))
            t_slot = title_slot_map[idx]
            t_emb = _l2_norm_vec(title_embs_pool[t_slot]) if t_slot >= 0 else _l2_norm_vec(s_emb[0])
            doc_vecs.append(np.concatenate([t_emb, mean_v, max_v], axis=0))
            sent_title_cos_list.append((s_emb @ t_emb).astype(np.float32))
            sent_doc_cos_list.append((s_emb @ mean_v).astype(np.float32))

        X_doc = np.vstack(doc_vecs).astype(np.float32)
        return X_doc, sent_embs_list, sents_list, sent_title_cos_list, sent_doc_cos_list

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
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Huấn luyện 5-Fold OOF CV không rò rỉ trên GPU (mã hóa sentence+title 1 lượt duy nhất):
        - Dual-head (bin, epi, alea) trên X_doc [title, mean, max].
        - Sentence probe (sent_clf_bin, sent_clf_epi, sent_clf_alea) + SupervisedEvidenceScorer trong từng fold.
        """
        n = len(texts)
        texts_l = list(texts)
        titles_l = list(titles)
        y_bin_arr = np.asarray(y_bin, dtype=int)
        y_epi_arr = np.asarray(y_epi, dtype=int)
        y_alea_arr = np.asarray(y_alea, dtype=int)
        fold_arr = np.asarray(fold_ids, dtype=int)
        uniq_folds = sorted(np.unique(fold_arr).tolist())

        X_doc, sent_embs_list, sents_list, sent_title_cos_list, sent_doc_cos_list = self.encode_multiview(
            texts_l, titles_l, show_progress=False
        )

        # Ánh xạ gold sentence index để dùng làm nhãn huấn luyện bộ câu trong mỗi fold
        gold_idx_map: List[Optional[int]] = [
            SupervisedEvidenceScorer._find_gold_sent_idx(sents_list[i], gold_evidences[i])
            if int(y_bin_arr[i]) == 1
            else None
            for i in range(n)
        ]

        C_grid = [float(c) for c in self.cfg.get("C_grid", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])]
        alpha_grid = [float(a) for a in self.cfg.get("alpha_grid", [0.5, 0.7, 1.0])]
        thr_grid = [float(t) for t in self.cfg.get("threshold_grid", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])]
        hn_pen = float(self.cfg.get("hn_penalty", 0.08))
        hn_tgt = float(self.cfg.get("hn_target", 0.10))

        best_key = -1e9
        best_C = C_grid[0]
        best_alpha = alpha_grid[0]
        best_thr = 0.5
        best_oof_macro = 0.0
        best_oof_f1_pos = 0.0
        best_oof_hn_fpr = 0.0
        best_fold_clfs: List[Tuple[LogisticRegression, LogisticRegression, LogisticRegression]] = []
        best_p_bin = np.zeros(n, dtype=np.float32)
        best_p_epi = np.zeros(n, dtype=np.float32)
        best_p_alea = np.zeros(n, dtype=np.float32)

        fold_splits = [(f, np.where(fold_arr != f)[0], np.where(fold_arr == f)[0]) for f in uniq_folds]

        for c_val in C_grid:
            oof_bin = np.zeros(n, dtype=np.float32)
            oof_epi = np.zeros(n, dtype=np.float32)
            oof_alea = np.zeros(n, dtype=np.float32)
            cur_clfs = []
            for _, tr_idx, va_idx in fold_splits:
                clf_b = _fit_lr_head_dense(X_doc[tr_idx], y_bin_arr[tr_idx], c_val, self.seed)
                clf_e = _fit_lr_head_dense(X_doc[tr_idx], y_epi_arr[tr_idx], c_val, self.seed)
                clf_a = _fit_lr_head_dense(X_doc[tr_idx], y_alea_arr[tr_idx], c_val, self.seed)
                oof_bin[va_idx] = clf_b.predict_proba(X_doc[va_idx])[:, 1]
                oof_epi[va_idx] = clf_e.predict_proba(X_doc[va_idx])[:, 1]
                oof_alea[va_idx] = clf_a.predict_proba(X_doc[va_idx])[:, 1]
                cur_clfs.append((clf_b, clf_e, clf_a))

            max_dual = np.maximum(oof_epi, oof_alea)
            for alpha in alpha_grid:
                p_comb = alpha * oof_bin + (1.0 - alpha) * max_dual
                thr, macro, f1_pos, hn_fpr = select_threshold_with_hn_penalty(
                    p_comb, y_bin_arr, boundary_types, thr_grid, hn_penalty=hn_pen, hn_target=hn_tgt
                )
                key = macro - hn_pen * max(0.0, hn_fpr - hn_tgt) + 1e-4 * f1_pos
                if key > best_key:
                    best_key = key
                    best_C = float(c_val)
                    best_alpha = float(alpha)
                    best_thr = float(thr)
                    best_oof_macro = float(macro)
                    best_oof_f1_pos = float(f1_pos)
                    best_oof_hn_fpr = float(hn_fpr)
                    best_fold_clfs = cur_clfs
                    best_p_bin = oof_bin.copy()
                    best_p_epi = oof_epi.copy()
                    best_p_alea = oof_alea.copy()

        # Huấn luyện sentence probe & SupervisedEvidenceScorer hoàn toàn trong từng fold
        self.fold_bundles = []
        oof_evidences = [""] * n
        oof_sent_scores_list: List[np.ndarray] = [np.zeros((1, 3), dtype=np.float32) for _ in range(n)]

        for (f, tr_idx, va_idx), (clf_b, clf_e, clf_a) in zip(fold_splits, best_fold_clfs):
            s_X_parts, s_yb, s_ye, s_ya = [], [], [], []
            for idx in tr_idx:
                s_emb = sent_embs_list[idx]
                if int(y_bin_arr[idx]) == 0:
                    for j in range(len(s_emb)):
                        s_X_parts.append(s_emb[j])
                        s_yb.append(0)
                        s_ye.append(0)
                        s_ya.append(0)
                else:
                    g_idx = gold_idx_map[idx]
                    if g_idx is not None:
                        s_X_parts.append(s_emb[g_idx])
                        s_yb.append(1)
                        s_ye.append(int(y_epi_arr[idx]))
                        s_ya.append(int(y_alea_arr[idx]))

            s_X = np.vstack(s_X_parts).astype(np.float32)
            sent_clf_b = _fit_lr_head_dense(s_X, np.asarray(s_yb, dtype=int), 1.0, self.seed)
            sent_clf_e = _fit_lr_head_dense(s_X, np.asarray(s_ye, dtype=int), 1.0, self.seed)
            sent_clf_a = _fit_lr_head_dense(s_X, np.asarray(s_ya, dtype=int), 1.0, self.seed)

            ev_records = []
            for idx in tr_idx:
                se = sent_embs_list[idx]
                s_sc = np.column_stack(
                    [
                        sent_clf_b.predict_proba(se)[:, 1],
                        sent_clf_e.predict_proba(se)[:, 1],
                        sent_clf_a.predict_proba(se)[:, 1],
                    ]
                ).astype(np.float32)
                ev_records.append(
                    {
                        "text": texts_l[idx],
                        "title": titles_l[idx],
                        "gold_evidence": gold_evidences[idx],
                        "label": int(y_bin_arr[idx]),
                        "model_sent_scores": s_sc,
                        "sent_title_cos": sent_title_cos_list[idx],
                        "sent_doc_cos": sent_doc_cos_list[idx],
                    }
                )
            ev_scorer = SupervisedEvidenceScorer(C=0.5, seed=self.seed).fit(ev_records, rule_model=self.rule_model)

            bundle = {
                "fold": f,
                "clf_bin": clf_b,
                "clf_epi": clf_e,
                "clf_alea": clf_a,
                "sent_clf_bin": sent_clf_b,
                "sent_clf_epi": sent_clf_e,
                "sent_clf_alea": sent_clf_a,
                "ev_scorer": ev_scorer,
            }
            self.fold_bundles.append(bundle)

            for idx in va_idx:
                se = sent_embs_list[idx]
                s_sc = np.column_stack(
                    [
                        sent_clf_b.predict_proba(se)[:, 1],
                        sent_clf_e.predict_proba(se)[:, 1],
                        sent_clf_a.predict_proba(se)[:, 1],
                    ]
                ).astype(np.float32)
                oof_sent_scores_list[idx] = s_sc
                oof_evidences[idx] = ev_scorer.select_best_sentence(
                    raw_text=texts_l[idx],
                    title=titles_l[idx],
                    rule_model=self.rule_model,
                    model_sent_scores=s_sc,
                    sent_title_cos=sent_title_cos_list[idx],
                    sent_doc_cos=sent_doc_cos_list[idx],
                )

        self.best_C = best_C
        self.best_alpha = best_alpha
        self.best_threshold = best_thr

        oof_comb = best_alpha * best_p_bin + (1.0 - best_alpha) * np.maximum(best_p_epi, best_p_alea)
        oof_preds: List[Dict[str, Any]] = []
        for i in range(n):
            p = float(oof_comb[i])
            pe = float(best_p_epi[i])
            pa = float(best_p_alea[i])
            is_u = p >= best_thr
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
                    "prob_bin": round(float(best_p_bin[i]), 6),
                    "prob_epistemic": round(pe, 6),
                    "prob_aleatoric": round(pa, 6),
                    "pred_uncertainty_type": ut,
                    "pred_evidence": oof_evidences[i],
                    "sent_scores": oof_sent_scores_list[i],
                }
            )

        info = {
            "best_C": best_C,
            "best_alpha": best_alpha,
            "best_threshold": best_thr,
            "oof_macro_f1": round(best_oof_macro, 4),
            "oof_f1_uncertain": round(best_oof_f1_pos, 4),
            "oof_hn_fpr": round(best_oof_hn_fpr, 4),
            "n_folds": len(uniq_folds),
        }
        return info, oof_preds

    def predict_probs_fast(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Suy luận nhanh hàng loạt trên GPU (39.028 bài) không trích từng câu chứng cứ."""
        assert self.fold_bundles, "Mô hình E1 chưa được huấn luyện!"
        X_doc, _, _, _, _ = self.encode_multiview(texts, titles=titles, show_progress=show_progress)
        n = len(texts)
        k = float(len(self.fold_bundles))
        acc_bin = np.zeros(n, dtype=np.float64)
        acc_epi = np.zeros(n, dtype=np.float64)
        acc_alea = np.zeros(n, dtype=np.float64)
        for b in self.fold_bundles:
            acc_bin += b["clf_bin"].predict_proba(X_doc)[:, 1]
            acc_epi += b["clf_epi"].predict_proba(X_doc)[:, 1]
            acc_alea += b["clf_alea"].predict_proba(X_doc)[:, 1]
        p_bin = (acc_bin / k).astype(np.float32)
        p_epi = (acc_epi / k).astype(np.float32)
        p_alea = (acc_alea / k).astype(np.float32)
        p_comb = (self.best_alpha * p_bin + (1.0 - self.best_alpha) * np.maximum(p_epi, p_alea)).astype(np.float32)
        return p_comb, p_bin, p_epi, p_alea

    def predict_batch(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        extract_evidence: bool = True,
        show_progress: bool = False,
        return_sent_scores: bool = False,
    ) -> List[Dict]:
        assert self.fold_bundles, "Mô hình E1 chưa được huấn luyện!"
        if titles is None:
            titles = [None] * len(texts)

        X_doc, sent_embs_list, sents_list, sent_title_cos_list, sent_doc_cos_list = self.encode_multiview(
            texts, titles=titles, show_progress=show_progress
        )
        n = len(texts)
        k = float(len(self.fold_bundles))
        acc_bin = np.zeros(n, dtype=np.float64)
        acc_epi = np.zeros(n, dtype=np.float64)
        acc_alea = np.zeros(n, dtype=np.float64)
        for b in self.fold_bundles:
            acc_bin += b["clf_bin"].predict_proba(X_doc)[:, 1]
            acc_epi += b["clf_epi"].predict_proba(X_doc)[:, 1]
            acc_alea += b["clf_alea"].predict_proba(X_doc)[:, 1]

        p_bin = acc_bin / k
        p_epi = acc_epi / k
        p_alea = acc_alea / k
        p_comb = self.best_alpha * p_bin + (1.0 - self.best_alpha) * np.maximum(p_epi, p_alea)

        results: List[Dict] = []
        for idx, (orig_t, ttl, p, pb, pe, pa) in enumerate(zip(texts, titles, p_comb, p_bin, p_epi, p_alea)):
            sents = sents_list[idx]
            avg_s_sc = None
            if extract_evidence or return_sent_scores:
                se = sent_embs_list[idx]
                fold_ev_scores = np.zeros(len(sents), dtype=np.float64)
                sum_s_sc = np.zeros((len(sents), 3), dtype=np.float64)
                for b in self.fold_bundles:
                    s_sc = np.column_stack(
                        [
                            b["sent_clf_bin"].predict_proba(se)[:, 1],
                            b["sent_clf_epi"].predict_proba(se)[:, 1],
                            b["sent_clf_alea"].predict_proba(se)[:, 1],
                        ]
                    )
                    sum_s_sc += s_sc
                    _, ev_p = b["ev_scorer"].score_sentences(
                        raw_text=orig_t,
                        title=ttl,
                        rule_model=self.rule_model,
                        model_sent_scores=s_sc,
                        sent_title_cos=sent_title_cos_list[idx],
                        sent_doc_cos=sent_doc_cos_list[idx],
                    )
                    fold_ev_scores += ev_p
                avg_s_sc = (sum_s_sc / k).astype(np.float32)
                best_sent = sents[int(np.argmax(fold_ev_scores))]
            else:
                best_sent = sents[0]

            is_u = float(p) >= self.best_threshold
            if not is_u:
                ut = "none"
            elif float(pe) >= 0.45 and float(pa) >= 0.45:
                ut = "both"
            elif float(pa) > float(pe):
                ut = "aleatoric"
            else:
                ut = "epistemic"

            rec = {
                "pred_label": "uncertain" if is_u else "not_uncertain",
                "prob_uncertain": round(float(p), 6),
                "prob_bin": round(float(pb), 6),
                "prob_epistemic": round(float(pe), 6),
                "prob_aleatoric": round(float(pa), 6),
                "pred_uncertainty_type": ut,
                "pred_evidence": best_sent,
            }
            if return_sent_scores:
                rec["sent_scores"] = avg_s_sc
            results.append(rec)
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "cfg": self.cfg,
                "fold_bundles": self.fold_bundles,
                "best_C": self.best_C,
                "best_alpha": self.best_alpha,
                "best_threshold": self.best_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> "VietnameseBiEncoderProbe":
        obj = joblib.load(path)
        inst = cls.__new__(cls)
        inst.cfg = obj["cfg"]
        inst.seed = int(inst.cfg.get("seed", 42))
        inst.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        inst.model_name = inst.cfg["model_name"]
        inst.use_pyvi = bool(inst.cfg.get("use_pyvi_segmentation", True))
        inst.max_length = int(inst.cfg.get("max_length", 256))
        inst.batch_size = int(inst.cfg.get("batch_size", 64))
        inst.normalize_emb = bool(inst.cfg.get("normalize_embeddings", True))
        inst.encoder = SentenceTransformer(inst.model_name, device=inst.device)
        inst.encoder.max_seq_length = inst.max_length
        inst.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)
        inst.fold_bundles = obj["fold_bundles"]
        inst.best_C = float(obj["best_C"])
        inst.best_alpha = float(obj.get("best_alpha", 1.0))
        inst.best_threshold = float(obj["best_threshold"])
        return inst

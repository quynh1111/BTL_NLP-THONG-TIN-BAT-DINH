"""
Bộ chấm điểm trích xuất câu chứng cứ có giám sát (Supervised Evidence Scorer):
Kết hợp đặc trưng vị trí câu, chồng lấp tiêu đề, từ điển BBD-VN v2.1,
độ tương đồng cosine câu-tiêu đề/câu-đoạn (nếu có) và điểm xác suất cấp câu của mô hình.
Luôn trả về exact substring từ `split_sentences_vi(raw_text)`.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from text_utils import clean_for_model, normalize_text, split_sentences_vi

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_E0_CFG = ROOT_DIR / "configs" / "E0.json"
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*%?")
_TOK_RE = re.compile(r"\w+")


def _toks(s: str) -> List[str]:
    return _TOK_RE.findall((s or "").lower())


def _token_f1(a: str, b: str) -> float:
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return 0.0
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p = common / len(ta)
    r = common / len(tb)
    return 2.0 * p * r / (p + r)


def _get_default_rule_model():
    from models_e0 import BBDVNRuleModel

    return BBDVNRuleModel(DEFAULT_E0_CFG)


def extract_sentence_features(
    raw_text: str,
    title: Optional[str] = None,
    rule_model: Any = None,
    model_sent_scores: Optional[np.ndarray] = None,
    sent_title_cos: Optional[np.ndarray] = None,
    sent_doc_cos: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Trích ma trận đặc trưng [n_sents, n_features] cho tất cả câu nguyên văn trong `raw_text`.
    """
    sents = split_sentences_vi(raw_text)
    if not sents:
        fallback = normalize_text(raw_text)
        sents = [fallback] if fallback else [""]

    rm = rule_model if rule_model is not None else _get_default_rule_model()
    n_s = len(sents)
    doc_len = max(1, len(raw_text or ""))
    t_clean = clean_for_model(title) if title and str(title).strip().lower() != "nan" else ""
    t_set = set(_toks(t_clean))

    # Tính trước rule features trên từng câu để lấy giá trị tương đối trong cùng đoạn
    rule_list = [rm.extract_rule_features(s, title=None) for s in sents]
    sig_totals = [
        float(rf["u"] + rf["prop"] + rf["a_strong"] + rf["a_mod"])
        for rf in rule_list
    ]
    max_sig_total = max(sig_totals) if sig_totals else 0.0

    extra_score_mat = None
    if model_sent_scores is not None:
        arr = np.asarray(model_sent_scores, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if len(arr) == n_s and arr.shape[1] > 0:
            col_max = arr.max(axis=0, keepdims=True)
            col_mean = arr.mean(axis=0, keepdims=True)
            extra_score_mat = np.hstack([arr, arr - col_max, arr - col_mean])

    title_cos_mat = None
    if sent_title_cos is not None:
        tc = np.asarray(sent_title_cos, dtype=float).reshape(-1)
        if len(tc) == n_s:
            title_cos_mat = np.column_stack([tc, tc - float(np.max(tc))])

    doc_cos_mat = None
    if sent_doc_cos is not None:
        dc = np.asarray(sent_doc_cos, dtype=float).reshape(-1)
        if len(dc) == n_s:
            doc_cos_mat = np.column_stack([dc, dc - float(np.max(dc))])

    rows = []
    for i, (s, rf, tot_sig) in enumerate(zip(sents, rule_list, sig_totals)):
        s_clean = clean_for_model(s)
        s_set = set(_toks(s_clean))
        if t_set and s_set:
            jacc = float(len(t_set & s_set)) / float(len(t_set | s_set))
        else:
            jacc = 0.0

        base_vec = [
            float(i == 0),
            float(i == 1),
            float(i / max(1, n_s - 1)),
            float(len(s) / doc_len),
            float(len(s.strip()) < 35),
            jacc,
            float(rf["e"]),
            float(rf["p"]),
            float(rf["u"]),
            float(rf["prop"]),
            float(rf["a_strong"]),
            float(rf["a_mod"]),
            float(rf["asset"]),
            float(rf["neg"]),
            float(rf["corp_guard"]),
            float(tot_sig - max_sig_total),
            float("?" in s),
            float(bool(_NUM_RE.search(s))),
        ]
        if extra_score_mat is not None:
            base_vec.extend(extra_score_mat[i].tolist())
        if title_cos_mat is not None:
            base_vec.extend(title_cos_mat[i].tolist())
        if doc_cos_mat is not None:
            base_vec.extend(doc_cos_mat[i].tolist())
        rows.append(base_vec)

    return sents, np.asarray(rows, dtype=np.float32)


class SupervisedEvidenceScorer:
    """
    Học có giám sát từ nhãn `evidence_sentence` ở tập huấn luyện để chọn câu chứng cứ
    nguyên văn tốt nhất trong đoạn (tránh lệch về câu mở đầu hay câu byline phóng viên).
    """

    def __init__(self, C: float = 0.5, seed: int = 42):
        self.C = float(C)
        self.seed = int(seed)
        self.pipeline: Optional[Pipeline] = None
        self.n_features_: int = 0

    @staticmethod
    def _find_gold_sent_idx(sents: List[str], gold_evidence: str) -> Optional[int]:
        g_norm = str(gold_evidence or "").strip()
        if not g_norm:
            return None
        for idx, s in enumerate(sents):
            if s.strip() == g_norm:
                return idx
        f1s = [_token_f1(s, g_norm) for s in sents]
        best_idx = int(np.argmax(f1s))
        if f1s[best_idx] >= 0.7:
            return best_idx
        return None

    def fit(
        self,
        records: Sequence[Dict[str, Any]],
        rule_model: Any = None,
    ) -> "SupervisedEvidenceScorer":
        """
        Mỗi phần tử trong `records` là dict có:
          - text (str), title (str|None), gold_evidence (str), label (int)
          - tùy chọn: model_sent_scores, sent_title_cos, sent_doc_cos
        Ưu tiên đoạn >= 2 câu thuộc lớp uncertain (`label == 1`); nếu < 15 đoạn thì mở rộng toàn tập đa câu.
        """
        rm = rule_model if rule_model is not None else _get_default_rule_model()

        def _collect(only_pos: bool) -> Tuple[List[np.ndarray], List[int]]:
            X_parts: List[np.ndarray] = []
            y_parts: List[int] = []
            for rec in records:
                lbl = int(rec.get("label", 0))
                if only_pos and lbl != 1:
                    continue
                raw_text = str(rec["text"])
                sents = split_sentences_vi(raw_text)
                if len(sents) < 2:
                    continue
                gold_idx = self._find_gold_sent_idx(sents, rec.get("gold_evidence", ""))
                if gold_idx is None:
                    continue
                _, feat_mat = extract_sentence_features(
                    raw_text=raw_text,
                    title=rec.get("title"),
                    rule_model=rm,
                    model_sent_scores=rec.get("model_sent_scores"),
                    sent_title_cos=rec.get("sent_title_cos"),
                    sent_doc_cos=rec.get("sent_doc_cos"),
                )
                X_parts.append(feat_mat)
                for j in range(len(sents)):
                    y_parts.append(1 if j == gold_idx else 0)
            return X_parts, y_parts

        X_parts, y_parts = _collect(only_pos=True)
        if len(X_parts) < 15:
            X_parts, y_parts = _collect(only_pos=False)

        if not X_parts or len(set(y_parts)) < 2:
            self.pipeline = None
            return self

        X = np.vstack(X_parts)
        y = np.asarray(y_parts, dtype=int)
        self.n_features_ = int(X.shape[1])
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def score_sentences(
        self,
        raw_text: str,
        title: Optional[str] = None,
        rule_model: Any = None,
        model_sent_scores: Optional[np.ndarray] = None,
        sent_title_cos: Optional[np.ndarray] = None,
        sent_doc_cos: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], np.ndarray]:
        sents, feat_mat = extract_sentence_features(
            raw_text=raw_text,
            title=title,
            rule_model=rule_model,
            model_sent_scores=model_sent_scores,
            sent_title_cos=sent_title_cos,
            sent_doc_cos=sent_doc_cos,
        )
        if len(sents) == 1:
            return sents, np.array([1.0], dtype=np.float32)

        if self.pipeline is not None and feat_mat.shape[1] == self.n_features_:
            probs = self.pipeline.predict_proba(feat_mat)[:, 1].astype(np.float32)
            return sents, probs

        # Fallback heuristic nếu không đủ mẫu huấn luyện
        if model_sent_scores is not None:
            arr = np.asarray(model_sent_scores, dtype=float)
            if arr.ndim == 2 and len(arr) == len(sents):
                return sents, arr.max(axis=1).astype(np.float32)
            if arr.ndim == 1 and len(arr) == len(sents):
                return sents, arr.astype(np.float32)
        rm = rule_model if rule_model is not None else _get_default_rule_model()
        h_scores = []
        for s in sents:
            rf = rm.extract_rule_features(s, title=None)
            sc = (
                3.0 * (rf["u"] + rf["a_strong"])
                + 2.2 * rf["prop"]
                + 1.8 * rf["a_mod"]
                + 1.2 * rf["p"]
                + 0.8 * rf["e"]
                - 3.0 * rf["neg"]
                - 2.0 * rf["corp_guard"]
                - (1.5 if len(s.strip()) < 35 else 0.0)
            )
            h_scores.append(sc)
        return sents, np.asarray(h_scores, dtype=np.float32)

    def select_best_sentence(
        self,
        raw_text: str,
        title: Optional[str] = None,
        rule_model: Any = None,
        model_sent_scores: Optional[np.ndarray] = None,
        sent_title_cos: Optional[np.ndarray] = None,
        sent_doc_cos: Optional[np.ndarray] = None,
    ) -> str:
        sents, scores = self.score_sentences(
            raw_text=raw_text,
            title=title,
            rule_model=rule_model,
            model_sent_scores=model_sent_scores,
            sent_title_cos=sent_title_cos,
            sent_doc_cos=sent_doc_cos,
        )
        return sents[int(np.argmax(scores))]

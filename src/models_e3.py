"""
Cấu hình E3 (Gói 1): Calibrated Hybrid Stacking Ensemble kết hợp có kiểm soát (không rò rỉ)
cả 4 nhánh E0 (luật v2.1) + E0_tfidf (multi-field) + E1 (sentence-pooled Bi-Encoder) + E2 (Dual-Hypothesis NLI)
cùng bộ chọn câu chứng cứ Fused Supervised Evidence Scorer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evidence_scorer import SupervisedEvidenceScorer
from models_e0 import BBDVNRuleModel
from text_utils import normalize_text, select_threshold_with_hn_penalty, split_sentences_vi

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_E0_CFG = ROOT_DIR / "configs" / "E0.json"


class HybridStackingEnsemble:
    def __init__(self, config_path: Path):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.seed = int(self.cfg.get("seed", 42))
        self.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)
        self.fold_bundles: List[Dict[str, Any]] = []
        self.best_C = 0.25
        self.best_threshold = 0.5

    @staticmethod
    def build_meta_features(
        r0_list: Sequence[Dict[str, Any]],
        r0t_list: Sequence[Dict[str, Any]],
        r1_list: Sequence[Dict[str, Any]],
        r2_list: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        rows = []
        for r0, r0t, r1, r2 in zip(r0_list, r0t_list, r1_list, r2_list):
            rows.append(
                [
                    float(r0["prob_uncertain"]),
                    1.0 if r0["pred_label"] == "uncertain" else 0.0,
                    1.0 if float(r0.get("corp_guard_hits", 0)) > 0 else 0.0,
                    float(r0t["prob_bin"]),
                    float(r0t["prob_epistemic"]),
                    float(r0t["prob_aleatoric"]),
                    float(r1["prob_bin"]),
                    float(r1["prob_epistemic"]),
                    float(r1["prob_aleatoric"]),
                    float(r2["prob_epistemic"]),
                    float(r2["prob_aleatoric"]),
                    float(r2["prob_uncertain"]),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def build_meta_features_arrays(
        e0_prob: np.ndarray,
        e0_pred: np.ndarray,
        e0_guard: np.ndarray,
        e0t_p_bin: np.ndarray,
        e0t_p_epi: np.ndarray,
        e0t_p_alea: np.ndarray,
        e1_p_bin: np.ndarray,
        e1_p_epi: np.ndarray,
        e1_p_alea: np.ndarray,
        e2_p_epi: np.ndarray,
        e2_p_alea: np.ndarray,
        e2_p_max: np.ndarray,
    ) -> np.ndarray:
        return np.column_stack(
            [
                e0_prob,
                e0_pred,
                e0_guard,
                e0t_p_bin,
                e0t_p_epi,
                e0t_p_alea,
                e1_p_bin,
                e1_p_epi,
                e1_p_alea,
                e2_p_epi,
                e2_p_alea,
                e2_p_max,
            ]
        ).astype(np.float32)

    @staticmethod
    def _combine_sent_scores(
        r0t_list: Sequence[Dict[str, Any]],
        r1_list: Sequence[Dict[str, Any]],
        r2_list: Sequence[Dict[str, Any]],
    ) -> List[Optional[np.ndarray]]:
        out: List[Optional[np.ndarray]] = []
        for r0t, r1, r2 in zip(r0t_list, r1_list, r2_list):
            parts = []
            for d in (r0t, r1, r2):
                sc = d.get("sent_scores")
                if sc is not None:
                    parts.append(np.asarray(sc, dtype=np.float32))
            if len(parts) == 3 and parts[0].shape[0] == parts[1].shape[0] == parts[2].shape[0]:
                out.append(np.hstack(parts))
            else:
                out.append(None)
        return out

    def fit_cv(
        self,
        r0_list: Sequence[Dict[str, Any]],
        r0t_list: Sequence[Dict[str, Any]],
        r1_list: Sequence[Dict[str, Any]],
        r2_list: Sequence[Dict[str, Any]],
        y_bin: Sequence[int],
        boundary_types: Sequence[str],
        fold_ids: Sequence[int],
        raw_texts: Sequence[str],
        titles: Sequence[Optional[str]],
        gold_evidences: Sequence[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Huấn luyện Meta-Learner lồng nhau trên đúng `fold_ids` dùng chung:
        Mỗi fold `f` chỉ học từ cột dự đoán OOF của `fold != f`, sau đó chấm OOF trên `fold == f`.
        """
        meta_X = self.build_meta_features(r0_list, r0t_list, r1_list, r2_list)
        fused_sent_scores = self._combine_sent_scores(r0t_list, r1_list, r2_list)
        y_arr = np.asarray(y_bin, dtype=int)
        fold_arr = np.asarray(fold_ids, dtype=int)
        uniq_folds = sorted(np.unique(fold_arr).tolist())
        n = len(y_arr)

        C_grid = [float(c) for c in self.cfg.get("C_grid", [0.05, 0.1, 0.25, 0.5, 1.0, 2.0])]
        thr_grid = [float(t) for t in self.cfg.get("threshold_grid", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])]
        hn_pen = float(self.cfg.get("hn_penalty", 0.08))
        hn_tgt = float(self.cfg.get("hn_target", 0.10))

        fold_splits = [(f, np.where(fold_arr != f)[0], np.where(fold_arr == f)[0]) for f in uniq_folds]

        best_key = -1e9
        best_C = C_grid[0]
        best_thr = 0.5
        best_oof_macro = 0.0
        best_oof_f1_pos = 0.0
        best_oof_hn_fpr = 0.0
        best_fold_models: List[Tuple[StandardScaler, LogisticRegression]] = []
        best_oof_probs = np.zeros(n, dtype=np.float32)

        for c_val in C_grid:
            oof_p = np.zeros(n, dtype=np.float32)
            cur_models = []
            for _, tr_idx, va_idx in fold_splits:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(meta_X[tr_idx])
                X_va = scaler.transform(meta_X[va_idx])
                clf = LogisticRegression(
                    C=c_val,
                    class_weight=self.cfg.get("class_weight", "balanced"),
                    max_iter=2000,
                    random_state=self.seed,
                )
                clf.fit(X_tr, y_arr[tr_idx])
                oof_p[va_idx] = clf.predict_proba(X_va)[:, 1]
                cur_models.append((scaler, clf))

            thr, macro, f1_pos, hn_fpr = select_threshold_with_hn_penalty(
                oof_p, y_arr, boundary_types, thr_grid, hn_penalty=hn_pen, hn_target=hn_tgt
            )
            key = macro - hn_pen * max(0.0, hn_fpr - hn_tgt) + 1e-4 * f1_pos
            if key > best_key:
                best_key = key
                best_C = float(c_val)
                best_thr = float(thr)
                best_oof_macro = float(macro)
                best_oof_f1_pos = float(f1_pos)
                best_oof_hn_fpr = float(hn_fpr)
                best_fold_models = cur_models
                best_oof_probs = oof_p.copy()

        self.fold_bundles = []
        oof_evidences = [""] * n
        for (f, tr_idx, va_idx), (scaler, clf) in zip(fold_splits, best_fold_models):
            ev_records = [
                {
                    "text": raw_texts[i],
                    "title": titles[i],
                    "gold_evidence": gold_evidences[i],
                    "label": int(y_arr[i]),
                    "model_sent_scores": fused_sent_scores[i],
                }
                for i in tr_idx
            ]
            ev_scorer = SupervisedEvidenceScorer(C=0.5, seed=self.seed).fit(ev_records, rule_model=self.rule_model)
            self.fold_bundles.append({"fold": f, "scaler": scaler, "clf": clf, "ev_scorer": ev_scorer})
            for i in va_idx:
                oof_evidences[i] = ev_scorer.select_best_sentence(
                    raw_text=raw_texts[i],
                    title=titles[i],
                    rule_model=self.rule_model,
                    model_sent_scores=fused_sent_scores[i],
                )

        self.best_C = best_C
        self.best_threshold = best_thr

        oof_preds: List[Dict[str, Any]] = []
        for i in range(n):
            p = float(best_oof_probs[i])
            pe = (float(r0t_list[i]["prob_epistemic"]) + float(r1_list[i]["prob_epistemic"]) + float(r2_list[i]["prob_epistemic"])) / 3.0
            pa = (float(r0t_list[i]["prob_aleatoric"]) + float(r1_list[i]["prob_aleatoric"]) + float(r2_list[i]["prob_aleatoric"])) / 3.0
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
                    "prob_epistemic": round(pe, 6),
                    "prob_aleatoric": round(pa, 6),
                    "pred_uncertainty_type": ut,
                    "pred_evidence": oof_evidences[i],
                }
            )

        info = {
            "best_C": best_C,
            "best_threshold": best_thr,
            "oof_macro_f1": round(best_oof_macro, 4),
            "oof_f1_uncertain": round(best_oof_f1_pos, 4),
            "oof_hn_fpr": round(best_oof_hn_fpr, 4),
            "n_folds": len(uniq_folds),
        }
        return info, oof_preds

    def predict_probs_from_arrays(
        self,
        e0_prob: np.ndarray,
        e0_pred: np.ndarray,
        e0_guard: np.ndarray,
        e0t_p_bin: np.ndarray,
        e0t_p_epi: np.ndarray,
        e0t_p_alea: np.ndarray,
        e1_p_bin: np.ndarray,
        e1_p_epi: np.ndarray,
        e1_p_alea: np.ndarray,
        e2_p_epi: np.ndarray,
        e2_p_alea: np.ndarray,
        e2_p_max: np.ndarray,
    ) -> np.ndarray:
        assert self.fold_bundles, "Mô hình E3 chưa được huấn luyện!"
        meta_X = self.build_meta_features_arrays(
            e0_prob, e0_pred, e0_guard,
            e0t_p_bin, e0t_p_epi, e0t_p_alea,
            e1_p_bin, e1_p_epi, e1_p_alea,
            e2_p_epi, e2_p_alea, e2_p_max,
        )
        acc = np.zeros(len(meta_X), dtype=np.float64)
        for b in self.fold_bundles:
            acc += b["clf"].predict_proba(b["scaler"].transform(meta_X))[:, 1]
        return (acc / float(len(self.fold_bundles))).astype(np.float32)

    def predict_from_base_preds(
        self,
        r0_list: Sequence[Dict[str, Any]],
        r0t_list: Sequence[Dict[str, Any]],
        r1_list: Sequence[Dict[str, Any]],
        r2_list: Sequence[Dict[str, Any]],
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        extract_evidence: bool = True,
    ) -> List[Dict[str, Any]]:
        assert self.fold_bundles, "Mô hình E3 chưa được huấn luyện!"
        if titles is None:
            titles = [None] * len(texts)
        meta_X = self.build_meta_features(r0_list, r0t_list, r1_list, r2_list)
        acc = np.zeros(len(meta_X), dtype=np.float64)
        for b in self.fold_bundles:
            acc += b["clf"].predict_proba(b["scaler"].transform(meta_X))[:, 1]
        probs = acc / float(len(self.fold_bundles))

        fused_sent_scores = self._combine_sent_scores(r0t_list, r1_list, r2_list)
        results: List[Dict[str, Any]] = []
        for idx, (orig_t, ttl, p, r0t, r1, r2, s_sc) in enumerate(
            zip(texts, titles, probs, r0t_list, r1_list, r2_list, fused_sent_scores)
        ):
            sents = split_sentences_vi(orig_t)
            if not sents:
                sents = [normalize_text(orig_t)]
            if extract_evidence:
                fold_ev_p = np.zeros(len(sents), dtype=np.float64)
                for b in self.fold_bundles:
                    _, ev_p = b["ev_scorer"].score_sentences(
                        raw_text=orig_t,
                        title=ttl,
                        rule_model=self.rule_model,
                        model_sent_scores=s_sc,
                    )
                    fold_ev_p += ev_p
                best_sent = sents[int(np.argmax(fold_ev_p))]
            else:
                best_sent = sents[0]

            pe = (float(r0t["prob_epistemic"]) + float(r1["prob_epistemic"]) + float(r2["prob_epistemic"])) / 3.0
            pa = (float(r0t["prob_aleatoric"]) + float(r1["prob_aleatoric"]) + float(r2["prob_aleatoric"])) / 3.0
            is_u = float(p) >= self.best_threshold
            if not is_u:
                ut = "none"
            elif pe >= 0.45 and pa >= 0.45:
                ut = "both"
            elif pa > pe:
                ut = "aleatoric"
            else:
                ut = "epistemic"
            results.append(
                {
                    "pred_label": "uncertain" if is_u else "not_uncertain",
                    "prob_uncertain": round(float(p), 6),
                    "prob_epistemic": round(pe, 6),
                    "prob_aleatoric": round(pa, 6),
                    "pred_uncertainty_type": ut,
                    "pred_evidence": best_sent,
                }
            )
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "cfg": self.cfg,
                "fold_bundles": self.fold_bundles,
                "best_C": self.best_C,
                "best_threshold": self.best_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "HybridStackingEnsemble":
        obj = joblib.load(path)
        inst = cls.__new__(cls)
        inst.cfg = obj["cfg"]
        inst.seed = int(inst.cfg.get("seed", 42))
        inst.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)
        inst.fold_bundles = obj["fold_bundles"]
        inst.best_C = float(obj["best_C"])
        inst.best_threshold = float(obj["best_threshold"])
        return inst

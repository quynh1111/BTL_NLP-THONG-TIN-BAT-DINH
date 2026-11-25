"""
Các mô hình Baseline nâng cấp (Gói 1) cho Đề tài 4:
- E0: Luật / Từ điển BBD-VN v2.1 (Epistemic E+P+U/Proposal + Guarded Aleatoric Volatility + Negation/Corporate Guardrail)
- E0_tfidf: Multi-Field TF-IDF (Title + Text word/char) + Đặc trưng cấu trúc BBD-VN + Dual-Head (Bin/Epi/Alea) 5-Fold CV + Supervised Evidence Scorer
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from evidence_scorer import SupervisedEvidenceScorer
from text_utils import (
    clean_for_model,
    compose_title_text,
    normalize_text,
    select_threshold_with_hn_penalty,
    split_sentences_vi,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_E0_CFG = ROOT_DIR / "configs" / "E0.json"
_TOK_RE = re.compile(r"\w+")


def _compile_kw_patterns(keywords: Sequence[str]) -> List[re.Pattern]:
    pats = []
    for kw in keywords:
        kw_esc = re.escape(kw.strip().lower())
        if not kw_esc:
            continue
        # Ranh giới từ Unicode-safe để tránh khớp nhầm từ ngắn như 'fed' hay 'gdp' bên trong từ khác
        pats.append(re.compile(rf"(?<!\w){kw_esc}(?!\w)", re.IGNORECASE))
    return pats


class BBDVNRuleModel:
    """Baseline E0 v2.1 dựa trên luật từ điển 2 nhánh (Bất định tri thức + Bất định ngẫu nhiên có kiểm soát PR/lãi suất)."""

    def __init__(self, config_path: Path = DEFAULT_E0_CFG):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.e_pats = _compile_kw_patterns(self.cfg["economy_keywords"])
        self.p_pats = _compile_kw_patterns(self.cfg["policy_keywords"])
        self.u_pats = _compile_kw_patterns(self.cfg["uncertainty_keywords"])
        self.prop_pats = _compile_kw_patterns(self.cfg.get("proposal_draft_keywords", []))
        self.a_strong_pats = _compile_kw_patterns(
            self.cfg.get("aleatoric_strong_keywords", self.cfg.get("aleatoric_volatility_keywords", []))
        )
        self.a_mod_pats = _compile_kw_patterns(self.cfg.get("aleatoric_moderate_keywords", []))
        self.asset_pats = _compile_kw_patterns(self.cfg.get("market_asset_keywords", []))
        self.corp_guard_pats = _compile_kw_patterns(self.cfg.get("corporate_result_guard_phrases", []))
        self.neg_pats = _compile_kw_patterns(self.cfg["negation_certainty_phrases"])

    def extract_rule_features(self, text: str, title: Optional[str] = None) -> Dict[str, float]:
        combined = compose_title_text(title, text).lower()
        t_clean = clean_for_model(title) if title and str(title).strip().lower() != "nan" else ""
        x_clean = clean_for_model(text)

        return {
            "e": float(sum(1 for p in self.e_pats if p.search(combined))),
            "p": float(sum(1 for p in self.p_pats if p.search(combined))),
            "u": float(sum(1 for p in self.u_pats if p.search(combined))),
            "prop": float(sum(1 for p in self.prop_pats if p.search(combined))),
            "a_strong": float(sum(1 for p in self.a_strong_pats if p.search(combined))),
            "a_mod": float(sum(1 for p in self.a_mod_pats if p.search(combined))),
            "asset": float(sum(1 for p in self.asset_pats if p.search(combined))),
            "neg": float(sum(1 for p in self.neg_pats if p.search(combined))),
            "corp_guard": float(sum(1 for p in self.corp_guard_pats if p.search(combined))),
            "has_title_qmark": float("?" in t_clean),
            "has_text_qmark": float("?" in x_clean),
        }

    def extract_struct_vector(self, text: str, title: Optional[str] = None) -> List[float]:
        rf = self.extract_rule_features(text, title=title)
        return [
            rf["e"],
            rf["p"],
            rf["u"],
            rf["prop"],
            rf["a_strong"],
            rf["a_mod"],
            rf["asset"],
            rf["neg"],
            rf["corp_guard"],
            rf["has_title_qmark"],
            rf["has_text_qmark"],
        ]

    def predict_one(self, text: str, title: Optional[str] = None) -> Dict:
        rf = self.extract_rule_features(text, title=title)
        e_hits = int(rf["e"])
        p_hits = int(rf["p"])
        u_hits = int(rf["u"])
        prop_hits = int(rf["prop"])
        a_strong_hits = int(rf["a_strong"])
        a_mod_hits = int(rf["a_mod"])
        asset_hits = int(rf["asset"])
        neg_hits = int(rf["neg"])
        corp_guard_hits = int(rf["corp_guard"])
        has_title_q = rf["has_title_qmark"] > 0

        min_e = int(self.cfg.get("min_e_hits", 1))
        min_u = int(self.cfg.get("min_u_hits", 1))
        req_p = bool(self.cfg.get("require_p_or_strong_u", True))

        epistemic_hit = (
            (e_hits >= min_e and u_hits >= min_u and (not req_p or p_hits >= 1 or u_hits >= 2))
            or (e_hits >= min_e and p_hits >= 1 and prop_hits >= 1 and corp_guard_hits == 0)
            or (e_hits >= min_e and (p_hits >= 1 or prop_hits >= 1) and has_title_q and corp_guard_hits == 0)
        )
        aleatoric_hit = (
            (e_hits >= min_e and a_strong_hits >= 1 and corp_guard_hits == 0)
            or (asset_hits >= 1 and a_mod_hits >= 1 and corp_guard_hits == 0)
        )

        is_uncertain = False
        if neg_hits > 0 and (u_hits + prop_hits + a_strong_hits) <= 1:
            is_uncertain = False
        elif epistemic_hit or aleatoric_hit:
            is_uncertain = True

        if not is_uncertain:
            pred_utype = "none"
        elif epistemic_hit and aleatoric_hit:
            pred_utype = "both"
        elif epistemic_hit:
            pred_utype = "epistemic"
        else:
            pred_utype = "aleatoric"

        # Điểm tin cậy heuristic chuẩn hóa về [0.05, 0.95]
        if e_hits == 0 and asset_hits == 0:
            prob = 0.05
        else:
            a_mod_weight = 1.0 if (asset_hits > 0 and corp_guard_hits == 0) else 0.35
            raw_score = (
                (u_hits + a_strong_hits) * 1.4
                + prop_hits * 0.9
                + a_mod_hits * a_mod_weight
                + p_hits * 0.5
                - neg_hits * 1.5
                - corp_guard_hits * 1.2
            )
            prob = float(1.0 / (1.0 + np.exp(-0.8 * (raw_score - 0.8))))
            prob = float(np.clip(prob, 0.05, 0.95))

        sents = split_sentences_vi(text)
        if not sents:
            sents = [normalize_text(text)]
        t_clean = clean_for_model(title) if title and str(title).strip().lower() != "nan" else ""
        t_set = set(_TOK_RE.findall(t_clean.lower()))

        best_sent = sents[0]
        best_score = -1e9
        for s in sents:
            srf = self.extract_rule_features(s, title=None)
            s_set = set(_TOK_RE.findall(clean_for_model(s).lower()))
            t_jacc = float(len(t_set & s_set)) / float(len(t_set | s_set)) if (t_set and s_set) else 0.0
            score = (
                3.0 * (srf["u"] + srf["a_strong"])
                + 2.2 * srf["prop"]
                + 1.8 * srf["a_mod"]
                + 1.2 * srf["p"]
                + 0.8 * srf["e"]
                + 1.5 * t_jacc
                - 3.0 * srf["neg"]
                - 2.0 * srf["corp_guard"]
                - (1.5 if len(s.strip()) < 35 else 0.0)
            )
            if score > best_score:
                best_score = score
                best_sent = s

        return {
            "pred_label": "uncertain" if is_uncertain else "not_uncertain",
            "prob_uncertain": round(prob, 6),
            "pred_uncertainty_type": pred_utype,
            "pred_evidence": best_sent,
            "e_hits": e_hits,
            "p_hits": p_hits,
            "u_hits": u_hits,
            "prop_hits": prop_hits,
            "a_strong_hits": a_strong_hits,
            "a_mod_hits": a_mod_hits,
            "asset_hits": asset_hits,
            "neg_hits": neg_hits,
            "corp_guard_hits": corp_guard_hits,
        }

    def predict_batch(
        self, texts: Sequence[str], titles: Optional[Sequence[Optional[str]]] = None
    ) -> List[Dict]:
        if titles is None:
            titles = [None] * len(texts)
        return [self.predict_one(t, title=ttl) for t, ttl in zip(texts, titles)]


def _safe_fit_dual_title_vec(
    titles_clean: Sequence[str], cfg: Dict[str, Any]
) -> Tuple[FeatureUnion, Any]:
    w_rng = tuple(cfg.get("word_ngram_range", [1, 2]))
    c_rng = tuple(cfg.get("char_ngram_range", [3, 5]))
    for m_df in (int(cfg.get("min_df", 2)), 1):
        vec = FeatureUnion(
            [
                (
                    "t_word",
                    TfidfVectorizer(
                        ngram_range=w_rng,
                        min_df=m_df,
                        max_features=int(cfg.get("title_max_features_word", 4000)),
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "t_char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=c_rng,
                        min_df=m_df,
                        max_features=int(cfg.get("title_max_features_char", 6000)),
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
            ]
        )
        try:
            X = vec.fit_transform(titles_clean)
            if X.shape[1] > 0:
                return vec, X
        except ValueError:
            continue
    # Dự phòng tuyệt đối nếu mọi tiêu đề đều rỗng
    vec = FeatureUnion(
        [
            ("t_word", TfidfVectorizer(min_df=1, lowercase=True, token_pattern=r"(?u)\b\w+\b")),
        ]
    )
    non_empty = [t if t.strip() else "none" for t in titles_clean]
    return vec, vec.fit_transform(non_empty)


def _fit_lr_head(X: csr_matrix, y: np.ndarray, C: float, seed: int) -> LogisticRegression:
    uniq = np.unique(y)
    clf = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    if len(uniq) < 2:
        # Fallback hiếm nếu 1 fold mất hẳn 1 lớp phụ
        dummy_X = X[:2]
        dummy_y = np.array([0, 1], dtype=int)
        clf.fit(dummy_X, dummy_y)
        return clf
    clf.fit(X, y)
    return clf


class TfidfBaselineModel:
    """
    Baseline thống kê nâng cấp E0_tfidf:
    Multi-field TF-IDF (title + text word/char) + đặc trưng cấu trúc BBD-VN v2.1
    + 2 đầu chuyên biệt (bin, epi, alea) huấn luyện 5-Fold OOF Ensemble + SupervisedEvidenceScorer.
    """

    def __init__(self, config_path: Path):
        self.cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.seed = int(self.cfg.get("seed", 42))
        self.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)
        self.fold_bundles: List[Dict[str, Any]] = []
        self.best_C = 1.0
        self.best_alpha = 0.7
        self.best_threshold = 0.5

    def _build_text_vectorizer(self) -> FeatureUnion:
        w_rng = tuple(self.cfg.get("word_ngram_range", [1, 2]))
        c_rng = tuple(self.cfg.get("char_ngram_range", [3, 5]))
        min_df = int(self.cfg.get("min_df", 2))
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=w_rng,
                        min_df=min_df,
                        max_features=int(self.cfg.get("max_features_word", 12000)),
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=c_rng,
                        min_df=min_df,
                        max_features=int(self.cfg.get("max_features_char", 18000)),
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
            ]
        )

    def _fit_transform_features(
        self, tr_texts: Sequence[str], tr_titles: Sequence[Optional[str]]
    ) -> Tuple[FeatureUnion, FeatureUnion, StandardScaler, csr_matrix]:
        x_clean = [clean_for_model(t) for t in tr_texts]
        t_clean = [clean_for_model(t) if t and str(t).strip().lower() != "nan" else "" for t in tr_titles]

        text_vec = self._build_text_vectorizer()
        X_text = text_vec.fit_transform(x_clean)

        if self.cfg.get("use_title", True):
            title_vec, X_title = _safe_fit_dual_title_vec(t_clean, self.cfg)
        else:
            title_vec, X_title = None, None

        if self.cfg.get("use_structural_features", True):
            raw_struct = np.asarray(
                [self.rule_model.extract_struct_vector(t, title=ttl) for t, ttl in zip(tr_texts, tr_titles)],
                dtype=np.float32,
            )
            struct_scaler = StandardScaler()
            scale = float(self.cfg.get("structural_scale", 0.25))
            X_struct = csr_matrix(struct_scaler.fit_transform(raw_struct) * scale)
        else:
            struct_scaler, X_struct = None, None

        blocks = [b for b in (X_text, X_title, X_struct) if b is not None]
        X_all = hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
        return text_vec, title_vec, struct_scaler, X_all

    def _transform_features(
        self,
        bundle: Dict[str, Any],
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
    ) -> csr_matrix:
        if titles is None:
            titles = [None] * len(texts)
        x_clean = [clean_for_model(t) for t in texts]
        t_clean = [clean_for_model(t) if t and str(t).strip().lower() != "nan" else "" for t in titles]

        blocks = [bundle["text_vec"].transform(x_clean)]
        if bundle.get("title_vec") is not None:
            blocks.append(bundle["title_vec"].transform(t_clean))
        if bundle.get("struct_scaler") is not None:
            raw_struct = np.asarray(
                [self.rule_model.extract_struct_vector(t, title=ttl) for t, ttl in zip(texts, titles)],
                dtype=np.float32,
            )
            scale = float(self.cfg.get("structural_scale", 0.25))
            blocks.append(csr_matrix(bundle["struct_scaler"].transform(raw_struct) * scale))
        return hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]

    def _sent_model_scores(
        self, bundle: Dict[str, Any], raw_text: str
    ) -> Tuple[List[str], np.ndarray]:
        sents = split_sentences_vi(raw_text)
        if not sents:
            sents = [normalize_text(raw_text)]
        X_s = self._transform_features(bundle, sents, titles=[None] * len(sents))
        p_bin = bundle["clf_bin"].predict_proba(X_s)[:, 1]
        p_epi = bundle["clf_epi"].predict_proba(X_s)[:, 1]
        p_alea = bundle["clf_alea"].predict_proba(X_s)[:, 1]
        return sents, np.column_stack([p_bin, p_epi, p_alea]).astype(np.float32)

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
        Huấn luyện 5-Fold CV không rò rỉ trên `train + dev`:
        - Tìm (best_C, best_alpha, best_threshold) hoàn toàn trên OOF probabilities.
        - Huấn luyện SupervisedEvidenceScorer riêng trong từng fold và trả về OOF predictions.
        """
        n = len(texts)
        texts_l = list(texts)
        titles_l = list(titles)
        y_bin_arr = np.asarray(y_bin, dtype=int)
        y_epi_arr = np.asarray(y_epi, dtype=int)
        y_alea_arr = np.asarray(y_alea, dtype=int)
        fold_arr = np.asarray(fold_ids, dtype=int)
        uniq_folds = sorted(np.unique(fold_arr).tolist())

        C_grid = [float(c) for c in self.cfg.get("C_grid", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0])]
        alpha_grid = [float(a) for a in self.cfg.get("alpha_grid", [0.5, 0.7, 1.0])]
        thr_grid = [float(t) for t in self.cfg.get("threshold_grid", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])]
        hn_pen = float(self.cfg.get("hn_penalty", 0.08))
        hn_tgt = float(self.cfg.get("hn_target", 0.10))

        # Fit đặc trưng theo từng fold 1 lần rồi duyệt lưới C
        fold_feats: List[Dict[str, Any]] = []
        for f in uniq_folds:
            tr_idx = np.where(fold_arr != f)[0]
            va_idx = np.where(fold_arr == f)[0]
            tv, ttl_v, sc, X_tr = self._fit_transform_features(
                [texts_l[i] for i in tr_idx], [titles_l[i] for i in tr_idx]
            )
            tmp_bundle = {"text_vec": tv, "title_vec": ttl_v, "struct_scaler": sc}
            X_va = self._transform_features(
                tmp_bundle, [texts_l[i] for i in va_idx], [titles_l[i] for i in va_idx]
            )
            fold_feats.append(
                {
                    "fold": f,
                    "tr_idx": tr_idx,
                    "va_idx": va_idx,
                    "text_vec": tv,
                    "title_vec": ttl_v,
                    "struct_scaler": sc,
                    "X_tr": X_tr,
                    "X_va": X_va,
                }
            )

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

        for c_val in C_grid:
            oof_bin = np.zeros(n, dtype=np.float32)
            oof_epi = np.zeros(n, dtype=np.float32)
            oof_alea = np.zeros(n, dtype=np.float32)
            cur_clfs = []
            for ff in fold_feats:
                tr_idx = ff["tr_idx"]
                va_idx = ff["va_idx"]
                clf_b = _fit_lr_head(ff["X_tr"], y_bin_arr[tr_idx], c_val, self.seed)
                clf_e = _fit_lr_head(ff["X_tr"], y_epi_arr[tr_idx], c_val, self.seed)
                clf_a = _fit_lr_head(ff["X_tr"], y_alea_arr[tr_idx], c_val, self.seed)
                oof_bin[va_idx] = clf_b.predict_proba(ff["X_va"])[:, 1]
                oof_epi[va_idx] = clf_e.predict_proba(ff["X_va"])[:, 1]
                oof_alea[va_idx] = clf_a.predict_proba(ff["X_va"])[:, 1]
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

        # Lắp fold bundles & huấn luyện SupervisedEvidenceScorer hoàn toàn trong từng fold
        self.fold_bundles = []
        oof_evidences = [""] * n
        oof_sent_scores_list: List[np.ndarray] = [np.zeros((1, 3), dtype=np.float32) for _ in range(n)]

        for ff, (clf_b, clf_e, clf_a) in zip(fold_feats, best_fold_clfs):
            bundle: Dict[str, Any] = {
                "fold": ff["fold"],
                "text_vec": ff["text_vec"],
                "title_vec": ff["title_vec"],
                "struct_scaler": ff["struct_scaler"],
                "clf_bin": clf_b,
                "clf_epi": clf_e,
                "clf_alea": clf_a,
            }
            ev_records = []
            for idx in ff["tr_idx"]:
                _, s_scores = self._sent_model_scores(bundle, texts_l[idx])
                ev_records.append(
                    {
                        "text": texts_l[idx],
                        "title": titles_l[idx],
                        "gold_evidence": gold_evidences[idx],
                        "label": int(y_bin_arr[idx]),
                        "model_sent_scores": s_scores,
                    }
                )
            ev_scorer = SupervisedEvidenceScorer(C=0.5, seed=self.seed).fit(ev_records, rule_model=self.rule_model)
            bundle["ev_scorer"] = ev_scorer
            self.fold_bundles.append(bundle)

            for idx in ff["va_idx"]:
                _, va_s_scores = self._sent_model_scores(bundle, texts_l[idx])
                oof_sent_scores_list[idx] = va_s_scores
                oof_evidences[idx] = ev_scorer.select_best_sentence(
                    raw_text=texts_l[idx],
                    title=titles_l[idx],
                    rule_model=self.rule_model,
                    model_sent_scores=va_s_scores,
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

    def predict_batch(
        self,
        texts: Sequence[str],
        titles: Optional[Sequence[Optional[str]]] = None,
        extract_evidence: bool = True,
        return_sent_scores: bool = False,
    ) -> List[Dict]:
        assert self.fold_bundles, "Mô hình E0_tfidf chưa được huấn luyện!"
        if titles is None:
            titles = [None] * len(texts)

        n = len(texts)
        k = float(len(self.fold_bundles))
        acc_bin = np.zeros(n, dtype=np.float64)
        acc_epi = np.zeros(n, dtype=np.float64)
        acc_alea = np.zeros(n, dtype=np.float64)

        for bundle in self.fold_bundles:
            X = self._transform_features(bundle, texts, titles)
            acc_bin += bundle["clf_bin"].predict_proba(X)[:, 1]
            acc_epi += bundle["clf_epi"].predict_proba(X)[:, 1]
            acc_alea += bundle["clf_alea"].predict_proba(X)[:, 1]

        p_bin = acc_bin / k
        p_epi = acc_epi / k
        p_alea = acc_alea / k
        p_comb = self.best_alpha * p_bin + (1.0 - self.best_alpha) * np.maximum(p_epi, p_alea)

        results: List[Dict] = []
        for idx, (orig_t, ttl, p, pb, pe, pa) in enumerate(zip(texts, titles, p_comb, p_bin, p_epi, p_alea)):
            sents = split_sentences_vi(orig_t)
            if not sents:
                sents = [normalize_text(orig_t)]

            avg_sent_scores = None
            if extract_evidence or return_sent_scores:
                fold_ev_scores = np.zeros(len(sents), dtype=np.float64)
                sum_s_scores = np.zeros((len(sents), 3), dtype=np.float64)
                for bundle in self.fold_bundles:
                    _, s_sc = self._sent_model_scores(bundle, orig_t)
                    sum_s_scores += s_sc
                    _, ev_prob = bundle["ev_scorer"].score_sentences(
                        raw_text=orig_t,
                        title=ttl,
                        rule_model=self.rule_model,
                        model_sent_scores=s_sc,
                    )
                    fold_ev_scores += ev_prob
                avg_sent_scores = (sum_s_scores / k).astype(np.float32)
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
                rec["sent_scores"] = avg_sent_scores
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
    def load(cls, path: Path) -> "TfidfBaselineModel":
        obj = joblib.load(path)
        inst = cls.__new__(cls)
        inst.cfg = obj["cfg"]
        inst.seed = int(inst.cfg.get("seed", 42))
        inst.rule_model = BBDVNRuleModel(DEFAULT_E0_CFG)
        inst.fold_bundles = obj["fold_bundles"]
        inst.best_C = float(obj["best_C"])
        inst.best_alpha = float(obj.get("best_alpha", 1.0))
        inst.best_threshold = float(obj["best_threshold"])
        return inst

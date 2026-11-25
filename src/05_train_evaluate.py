"""
Bước 5 (Gói 1): Huấn luyện 5-Fold OOF Cross-Validation không rò rỉ trên (train + dev = 184 mẫu)
cho tất cả 5 cấu hình:
  - E0: BBD-VN Rule/Lexicon v2.1 (Epistemic + Guarded Aleatoric + Proposal)
  - E0_tfidf: Multi-Field TF-IDF + BBD-VN Structural + Dual-Head 5-Fold CV
  - E1: Sentence-Pooled Vietnamese Bi-Encoder (GPU) + Dual-Head 5-Fold CV + Supervised Evidence Scorer
  - E2: Dual-Hypothesis mDeBERTa-v3 NLI (fp32, Bottom-6 Frozen, HN-Loss, 5-Fold Ensemble)
  - E3: Calibrated Hybrid Stacking Ensemble (OOF Meta-Learner + Fused Evidence Scorer)
Toàn bộ siêu tham số và ngưỡng phạt HN-FPR chốt hoàn toàn từ OOF CV trước khi chấm 1 lần trên tập test khóa (n = 60).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from evaluate_predictions import compute_metrics_for_df, evaluate_all
from models_e0 import BBDVNRuleModel, TfidfBaselineModel
from models_e1 import VietnameseBiEncoderProbe
from models_e2 import SalienceDefinitionNLI
from models_e3 import HybridStackingEnsemble
from text_utils import make_stratified_folds

ROOT_DIR = Path(__file__).resolve().parents[1]
ANN_DIR = ROOT_DIR / "data" / "annotations"
CFG_DIR = ROOT_DIR / "configs"
EVAL_DIR = ROOT_DIR / "evaluation"
CKPT_DIR = ROOT_DIR / "checkpoints"


def _save_preds(df_split: pd.DataFrame, preds: List[Dict], out_path: Path) -> None:
    rows = []
    for (_, row), pr in zip(df_split.iterrows(), preds):
        rows.append(
            {
                "item_id": row["item_id"],
                "document_id": row["document_id"],
                "source_name": row["source_name"],
                "published_at": row["published_at"],
                "gold_label": row["task_label"],
                "pred_label": pr["pred_label"],
                "prob_uncertain": pr["prob_uncertain"],
                "pred_uncertainty_type": pr.get("pred_uncertainty_type", "none"),
                "gold_evidence": row["evidence_sentence"],
                "pred_evidence": pr["pred_evidence"],
                "uncertainty_type": row["uncertainty_type"] if "uncertainty_type" in row else "none",
                "boundary_type": row["boundary_type"],
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def _macro_f1(df_split: pd.DataFrame, preds: List[Dict]) -> float:
    y_true = (df_split["task_label"] == "uncertain").astype(int).to_numpy()
    y_pred = np.array([1 if p["pred_label"] == "uncertain" else 0 for p in preds])
    return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)


def _compute_oof_row(run_id: str, df_tr_dv: pd.DataFrame, oof_preds: List[Dict]) -> Dict:
    tmp = pd.DataFrame(
        {
            "item_id": df_tr_dv["item_id"],
            "document_id": df_tr_dv["document_id"],
            "source_name": df_tr_dv["source_name"],
            "published_at": df_tr_dv["published_at"],
            "gold_label": df_tr_dv["task_label"],
            "pred_label": [p["pred_label"] for p in oof_preds],
            "prob_uncertain": [p["prob_uncertain"] for p in oof_preds],
            "gold_evidence": df_tr_dv["evidence_sentence"],
            "pred_evidence": [p["pred_evidence"] for p in oof_preds],
            "boundary_type": df_tr_dv["boundary_type"],
        }
    )
    m = compute_metrics_for_df(tmp)
    m["split"] = "oof_cv"
    m["run_id"] = run_id
    return m


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train/Eval Gói 1] Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    df = pd.read_csv(ANN_DIR / "gold_labels_split.csv", encoding="utf-8-sig")
    tr = df[df["split"] == "train"].copy().reset_index(drop=True)
    dv = df[df["split"] == "dev"].copy().reset_index(drop=True)
    te = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Hợp nhất train+dev làm bể 5-Fold OOF CV; tập test (n=60) giữ nguyên khóa tuyệt đối
    tr_dv = pd.concat([tr, dv], ignore_index=True)
    is_dev = (tr_dv["split"] == "dev").to_numpy()
    fold_ids = make_stratified_folds(tr_dv, n_splits=5, seed=42)

    texts_trdv = tr_dv["text"].astype(str).tolist()
    titles_trdv = tr_dv["title"].tolist()
    y_bin_trdv = (tr_dv["task_label"] == "uncertain").astype(int).tolist()
    ut_trdv = tr_dv["uncertainty_type"].astype(str)
    y_epi_trdv = ut_trdv.isin(["epistemic", "both"]).astype(int).tolist()
    y_alea_trdv = ut_trdv.isin(["aleatoric", "both"]).astype(int).tolist()
    bt_trdv = tr_dv["boundary_type"].astype(str).tolist()
    gold_ev_trdv = tr_dv["evidence_sentence"].astype(str).tolist()

    te_texts = te["text"].astype(str).tolist()
    te_titles = te["title"].tolist()

    dv_texts = dv["text"].astype(str).tolist()
    dv_titles = dv["title"].tolist()
    dv_y = (dv["task_label"] == "uncertain").astype(int).tolist()
    dv_bt = dv["boundary_type"].astype(str).tolist()

    dataset_ver = "VON_econ_bd8d2a86_gold_v2.0_human_signed"
    split_ver = f"5fold_oof_cv_trdv{len(tr_dv)}_locked_te{len(te)}"
    experiments = []
    oof_rows = []

    # 1. E0: BBD-VN Rule/Lexicon v2.1
    t0 = time.time()
    e0 = BBDVNRuleModel(CFG_DIR / "E0.json")
    e0_trdv_preds = e0.predict_batch(texts_trdv, titles=titles_trdv)
    e0_dev_preds = [p for p, flag in zip(e0_trdv_preds, is_dev) if flag]
    e0_test_preds = e0.predict_batch(te_texts, titles=te_titles)
    _save_preds(dv, e0_dev_preds, EVAL_DIR / "dev_predictions_E0.csv")
    _save_preds(te, e0_test_preds, EVAL_DIR / "test_predictions_E0.csv")
    oof_rows.append(_compute_oof_row("E0", tr_dv, e0_trdv_preds))
    e0_dev_f1, e0_test_f1 = _macro_f1(dv, e0_dev_preds), _macro_f1(te, e0_test_preds)
    experiments.append(
        {
            "run_id": "E0",
            "method": "BBD-VN Rule/Lexicon v2.1 (Epistemic + Guarded Aleatoric + Proposal)",
            "dataset_version": dataset_ver,
            "split_version": split_ver,
            "config_path": "configs/E0.json",
            "seed_or_prompt_version": "bbd_vn_lexicon_v2.1 (seed=42)",
            "dev_metric": e0_dev_f1,
            "test_metric": e0_test_f1,
            "prediction_file": "evaluation/test_predictions_E0.csv",
            "notes": f"OOF-184 Macro-F1={oof_rows[-1]['macro_f1']}; Rule v2.1 ({time.time()-t0:.2f}s)",
        }
    )
    print(f"[E0 v2.1] oof184={oof_rows[-1]['macro_f1']} | dev={e0_dev_f1} | test={e0_test_f1}")

    # 2. E0_tfidf: Multi-Field TF-IDF + Structural + Dual-Head 5-Fold OOF
    t0 = time.time()
    e0t = TfidfBaselineModel(CFG_DIR / "E0_tfidf.json")
    e0t_info, e0t_oof_preds = e0t.fit_cv(
        texts_trdv, titles_trdv, y_bin_trdv, y_epi_trdv, y_alea_trdv, bt_trdv, gold_ev_trdv, fold_ids
    )
    e0t.save(CKPT_DIR / "e0_tfidf.joblib")
    e0t_dev_preds = [p for p, flag in zip(e0t_oof_preds, is_dev) if flag]
    e0t_test_preds = e0t.predict_batch(te_texts, titles=te_titles, return_sent_scores=True)
    _save_preds(dv, e0t_dev_preds, EVAL_DIR / "dev_predictions_E0_tfidf.csv")
    _save_preds(te, e0t_test_preds, EVAL_DIR / "test_predictions_E0_tfidf.csv")
    oof_rows.append(_compute_oof_row("E0_tfidf", tr_dv, e0t_oof_preds))
    e0t_dev_f1, e0t_test_f1 = _macro_f1(dv, e0t_dev_preds), _macro_f1(te, e0t_test_preds)
    experiments.append(
        {
            "run_id": "E0_tfidf",
            "method": "Multi-Field TF-IDF (Title+Text) + Structural + Dual-Head 5-Fold CV",
            "dataset_version": dataset_ver,
            "split_version": split_ver,
            "config_path": "configs/E0_tfidf.json",
            "seed_or_prompt_version": "5fold_oof_cv (seed=42)",
            "dev_metric": e0t_dev_f1,
            "test_metric": e0t_test_f1,
            "prediction_file": "evaluation/test_predictions_E0_tfidf.csv",
            "notes": (
                f"OOF-184={e0t_info['oof_macro_f1']} (HN-FPR={e0t_info['oof_hn_fpr']}); "
                f"best_C={e0t_info['best_C']}, alpha={e0t_info['best_alpha']}, thr={e0t_info['best_threshold']} ({time.time()-t0:.2f}s)"
            ),
        }
    )
    print(f"[E0_tfidf] oof184={e0t_info['oof_macro_f1']} | dev={e0t_dev_f1} | test={e0t_test_f1} ({e0t_info})")

    # 3. E1: Sentence-Pooled Vietnamese Bi-Encoder + Dual-Head + Supervised Evidence Scorer (GPU)
    t0 = time.time()
    e1 = VietnameseBiEncoderProbe(CFG_DIR / "E1.json", device=device)
    e1_info, e1_oof_preds = e1.fit_cv(
        texts_trdv, titles_trdv, y_bin_trdv, y_epi_trdv, y_alea_trdv, bt_trdv, gold_ev_trdv, fold_ids
    )
    e1.save(CKPT_DIR / "e1_model.joblib")
    e1_dev_preds = [p for p, flag in zip(e1_oof_preds, is_dev) if flag]
    e1_test_preds = e1.predict_batch(te_texts, titles=te_titles, return_sent_scores=True)
    _save_preds(dv, e1_dev_preds, EVAL_DIR / "dev_predictions_E1.csv")
    _save_preds(te, e1_test_preds, EVAL_DIR / "test_predictions_E1.csv")
    oof_rows.append(_compute_oof_row("E1", tr_dv, e1_oof_preds))
    e1_dev_f1, e1_test_f1 = _macro_f1(dv, e1_dev_preds), _macro_f1(te, e1_test_preds)
    experiments.append(
        {
            "run_id": "E1",
            "method": "Sentence-Pooled vietnamese-bi-encoder ([title,mean,max]) + Dual-Head 5-Fold CV + Evidence Scorer",
            "dataset_version": dataset_ver,
            "split_version": split_ver,
            "config_path": "configs/E1.json",
            "seed_or_prompt_version": "5fold_oof_cv (seed=42)",
            "dev_metric": e1_dev_f1,
            "test_metric": e1_test_f1,
            "prediction_file": "evaluation/test_predictions_E1.csv",
            "notes": (
                f"OOF-184={e1_info['oof_macro_f1']} (HN-FPR={e1_info['oof_hn_fpr']}); "
                f"best_C={e1_info['best_C']}, alpha={e1_info['best_alpha']}, thr={e1_info['best_threshold']}, device={device} ({time.time()-t0:.2f}s)"
            ),
        }
    )
    print(f"[E1] oof184={e1_info['oof_macro_f1']} | dev={e1_dev_f1} | test={e1_test_f1} ({e1_info})")

    del e1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. E2: Dual-Hypothesis NLI + Bottom-6 Frozen + Hard-Negative Loss + 5-Fold Ensemble (GPU fp32)
    t0 = time.time()
    e2_zs = SalienceDefinitionNLI(CFG_DIR / "E2.json", device=device)
    zs_res = e2_zs.eval_zero_shot_on_dev(dv_texts, dv_titles, dv_y, dv_bt)
    print(f"[E2 Zero-Shot Dual-Hypothesis on Dev] {zs_res}")
    del e2_zs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    e2 = SalienceDefinitionNLI(CFG_DIR / "E2.json", device=device)
    e2_info, e2_oof_preds, fold_logs = e2.fit_cv(
        texts_trdv, titles_trdv, y_bin_trdv, y_epi_trdv, y_alea_trdv, bt_trdv, gold_ev_trdv, fold_ids
    )
    e2.save(CKPT_DIR / "e2_best")

    e2_dev_preds = [p for p, flag in zip(e2_oof_preds, is_dev) if flag]
    e2_test_preds = e2.predict_batch(te_texts, titles=te_titles, use_ensemble=True, return_sent_scores=True)
    _save_preds(dv, e2_dev_preds, EVAL_DIR / "dev_predictions_E2.csv")
    _save_preds(te, e2_test_preds, EVAL_DIR / "test_predictions_E2.csv")
    oof_rows.append(_compute_oof_row("E2", tr_dv, e2_oof_preds))
    e2_dev_f1, e2_test_f1 = _macro_f1(dv, e2_dev_preds), _macro_f1(te, e2_test_preds)

    # Ghi nhận chẩn đoán từng fold đơn lẻ trên dev/test để theo dõi phương sai (không dùng để chọn tham số)
    seed_rows = []
    te_y_arr = (te["task_label"] == "uncertain").astype(int).to_numpy()
    for fi, flog in enumerate(fold_logs):
        fi_dev = e2.predict_batch(dv_texts, titles=dv_titles, extract_evidence=False, specific_fold_idx=fi)
        fi_test = e2.predict_batch(te_texts, titles=te_titles, extract_evidence=False, specific_fold_idx=fi)
        fi_te_pred = np.array([1 if p["pred_label"] == "uncertain" else 0 for p in fi_test])
        seed_rows.append(
            {
                "fold": flog["fold"],
                "seed": flog["seed"],
                "best_epoch": flog["best_epoch"],
                "best_threshold": flog["best_threshold"],
                "oof_fold_macro_f1": flog["oof_fold_macro_f1"],
                "oof_fold_f1_uncertain": flog["oof_fold_f1_uncertain"],
                "dev_macro_f1": _macro_f1(dv, fi_dev),
                "dev_f1_uncertain": round(
                    float(f1_score(dv_y, [1 if p["pred_label"] == "uncertain" else 0 for p in fi_dev], pos_label=1, zero_division=0)),
                    4,
                ),
                "test_macro_f1": _macro_f1(te, fi_test),
                "test_f1_uncertain": round(float(f1_score(te_y_arr, fi_te_pred, pos_label=1, zero_division=0)), 4),
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(EVAL_DIR / "e2_seeds_summary.csv", index=False, encoding="utf-8-sig")

    mean_te = seed_df["test_macro_f1"].mean()
    std_te = seed_df["test_macro_f1"].std(ddof=1) if len(seed_df) > 1 else 0.0
    experiments.append(
        {
            "run_id": "E2",
            "method": "Dual-Hypothesis NLI (Bottom-6 Frozen + HN-Loss + 5-Fold Ensemble, mDeBERTa-v3 fp32)",
            "dataset_version": dataset_ver,
            "split_version": split_ver,
            "config_path": "configs/E2.json",
            "seed_or_prompt_version": "5fold_oof_ensemble [42,123,2026,777,999] | dual_hyp_v2",
            "dev_metric": e2_dev_f1,
            "test_metric": e2_test_f1,
            "prediction_file": "evaluation/test_predictions_E2.csv",
            "notes": (
                f"OOF-184={e2_info['oof_macro_f1']} (HN-FPR={e2_info['oof_hn_fpr']}); "
                f"thr={e2_info['best_threshold']}; single-fold test mean={mean_te:.4f}+/-{std_te:.4f}; "
                f"zero-shot dev={zs_res['zero_shot_dev_macro_f1']:.4f}; device={device} ({time.time()-t0:.1f}s)"
            ),
        }
    )
    print(f"[E2] oof184={e2_info['oof_macro_f1']} | dev={e2_dev_f1} | test(5-fold ens)={e2_test_f1} ({e2_info})")

    del e2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. E3: Calibrated Hybrid Stacking Ensemble (OOF không rò rỉ trên tr_dv 184 mẫu)
    t0 = time.time()
    e3 = HybridStackingEnsemble(CFG_DIR / "E3.json")
    e3_info, e3_oof_preds = e3.fit_cv(
        r0_list=e0_trdv_preds,
        r0t_list=e0t_oof_preds,
        r1_list=e1_oof_preds,
        r2_list=e2_oof_preds,
        y_bin=y_bin_trdv,
        boundary_types=bt_trdv,
        fold_ids=fold_ids,
        raw_texts=texts_trdv,
        titles=titles_trdv,
        gold_evidences=gold_ev_trdv,
    )
    e3.save(CKPT_DIR / "e3_stacking.joblib")
    e3_dev_preds = [p for p, flag in zip(e3_oof_preds, is_dev) if flag]
    e3_test_preds = e3.predict_from_base_preds(
        r0_list=e0_test_preds,
        r0t_list=e0t_test_preds,
        r1_list=e1_test_preds,
        r2_list=e2_test_preds,
        texts=te_texts,
        titles=te_titles,
        extract_evidence=True,
    )
    _save_preds(dv, e3_dev_preds, EVAL_DIR / "dev_predictions_E3.csv")
    _save_preds(te, e3_test_preds, EVAL_DIR / "test_predictions_E3.csv")
    oof_rows.append(_compute_oof_row("E3", tr_dv, e3_oof_preds))
    e3_dev_f1, e3_test_f1 = _macro_f1(dv, e3_dev_preds), _macro_f1(te, e3_test_preds)
    experiments.append(
        {
            "run_id": "E3",
            "method": "Calibrated Hybrid Stacking Ensemble (E0+E0_tfidf+E1+E2) + Fused Evidence Scorer",
            "dataset_version": dataset_ver,
            "split_version": split_ver,
            "config_path": "configs/E3.json",
            "seed_or_prompt_version": "nested_5fold_oof_stacking (seed=42)",
            "dev_metric": e3_dev_f1,
            "test_metric": e3_test_f1,
            "prediction_file": "evaluation/test_predictions_E3.csv",
            "notes": (
                f"OOF-184={e3_info['oof_macro_f1']} (HN-FPR={e3_info['oof_hn_fpr']}); "
                f"best_C={e3_info['best_C']}, thr={e3_info['best_threshold']} ({time.time()-t0:.2f}s)"
            ),
        }
    )
    print(f"[E3] oof184={e3_info['oof_macro_f1']} | dev={e3_dev_f1} | test={e3_test_f1} ({e3_info})")

    oof_df = pd.DataFrame(oof_rows)
    oof_cols = [
        "split", "run_id", "n_samples", "macro_f1", "f1_uncertain",
        "precision_uncertain", "recall_uncertain", "hard_neg_count", "hard_neg_fpr",
        "evidence_exact_match_uncertain", "evidence_token_f1_uncertain",
    ]
    oof_df[oof_cols].to_csv(EVAL_DIR / "oof_cv_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(experiments).to_csv(EVAL_DIR / "experiments.csv", index=False, encoding="utf-8-sig")

    print("\n=== FULL 184-ITEM OOF CV SUMMARY ===")
    print(oof_df[oof_cols].to_string(index=False))
    print("\n=== E2 PER-FOLD DIAGNOSTIC SUMMARY ===")
    print(seed_df.to_string(index=False))
    print("\n=== RUNNING INDEPENDENT EVALUATOR ===")
    evaluate_all(verify=True)


if __name__ == "__main__":
    main()

"""
Script đánh giá độc lập (Evaluation Verifier):
Đọc trực tiếp các file evaluation/test_predictions_*.csv và evaluation/dev_predictions_*.csv
để tính toán và kiểm chứng toàn bộ bảng kết quả trên cùng giao thức.
Chạy độc lập: python src/evaluate_predictions.py --verify
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT_DIR / "evaluation"

RUN_ORDER = ["E0", "E0_tfidf", "E1", "E2", "E3"]


def _norm_toks(s: str) -> List[str]:
    return re.findall(r"\w+", str(s or "").lower())


def evidence_token_f1(pred: str, gold: str) -> float:
    pt = _norm_toks(pred)
    gt = _norm_toks(gold)
    if not pt or not gt:
        return 0.0
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return 0.0
    p = common / len(pt)
    r = common / len(gt)
    return 2 * p * r / (p + r)


def compute_metrics_for_df(df: pd.DataFrame) -> Dict:
    y_true = (df["gold_label"] == "uncertain").astype(int).to_numpy()
    y_pred = (df["pred_label"] == "uncertain").astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_unc = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_not = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    p_unc = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r_unc = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    p_not = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    r_not = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # Hard-negative False Positive Rate (tin xấu đã xảy ra bị nhận nhầm thành bất định)
    hn_df = df[df["boundary_type"] == "negative_certain_hard_negative"]
    hn_fpr = float((hn_df["pred_label"] == "uncertain").mean()) if len(hn_df) > 0 else np.nan

    # Chất lượng câu chứng cứ trên tập mẫu uncertain thực tế (và toàn tập)
    unc_df = df[df["gold_label"] == "uncertain"]
    if len(unc_df) > 0 and "pred_evidence" in unc_df.columns and "gold_evidence" in unc_df.columns:
        ev_exact_unc = float(
            np.mean(
                [
                    str(p).strip() == str(g).strip()
                    for p, g in zip(unc_df["pred_evidence"], unc_df["gold_evidence"])
                ]
            )
        )
        ev_tok_f1_unc = float(
            np.mean(
                [
                    evidence_token_f1(p, g)
                    for p, g in zip(unc_df["pred_evidence"], unc_df["gold_evidence"])
                ]
            )
        )
    else:
        ev_exact_unc, ev_tok_f1_unc = np.nan, np.nan

    ev_exact_all = float(
        np.mean(
            [
                str(p).strip() == str(g).strip()
                for p, g in zip(df["pred_evidence"], df["gold_evidence"])
            ]
        )
    )
    ev_tok_f1_all = float(
        np.mean(
            [
                evidence_token_f1(p, g)
                for p, g in zip(df["pred_evidence"], df["gold_evidence"])
            ]
        )
    )

    return {
        "n_samples": int(len(df)),
        "macro_f1": round(float(macro_f1), 4),
        "f1_uncertain": round(float(f1_unc), 4),
        "precision_uncertain": round(float(p_unc), 4),
        "recall_uncertain": round(float(r_unc), 4),
        "f1_not_uncertain": round(float(f1_not), 4),
        "precision_not_uncertain": round(float(p_not), 4),
        "recall_not_uncertain": round(float(r_not), 4),
        "accuracy": round(float(acc), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "hard_neg_count": int(len(hn_df)),
        "hard_neg_fpr": round(float(hn_fpr), 4) if not np.isnan(hn_fpr) else np.nan,
        "evidence_exact_match_uncertain": round(ev_exact_unc, 4) if not np.isnan(ev_exact_unc) else np.nan,
        "evidence_token_f1_uncertain": round(ev_tok_f1_unc, 4) if not np.isnan(ev_tok_f1_unc) else np.nan,
        "evidence_exact_match_all": round(ev_exact_all, 4),
        "evidence_token_f1_all": round(ev_tok_f1_all, 4),
    }


def evaluate_all(verify: bool = False) -> None:
    summary_rows = []
    slice_rows = []
    boundary_rows = []
    utype_rows = []

    for split in ["dev", "test"]:
        for run_id in RUN_ORDER:
            pred_file = EVAL_DIR / f"{split}_predictions_{run_id}.csv"
            if not pred_file.exists():
                continue
            df = pd.read_csv(pred_file, encoding="utf-8-sig")
            m = compute_metrics_for_df(df)
            m["split"] = split
            m["run_id"] = run_id
            m["prediction_file"] = str(pred_file.relative_to(ROOT_DIR)).replace("\\", "/")
            summary_rows.append(m)

            if split == "test":
                # Lát cắt theo nguồn báo
                for src, g in df.groupby("source_name"):
                    sm = compute_metrics_for_df(g)
                    sm["slice_type"] = "source"
                    sm["slice_value"] = src
                    sm["run_id"] = run_id
                    slice_rows.append(sm)
                # Lát cắt theo năm công bố
                df["year"] = df["published_at"].astype(str).str.slice(0, 4)
                for yr, g in df.groupby("year"):
                    ym = compute_metrics_for_df(g)
                    ym["slice_type"] = "year"
                    ym["slice_value"] = yr
                    ym["run_id"] = run_id
                    slice_rows.append(ym)
                # Lát cắt theo nhóm biên giới (boundary_type)
                for bt, g in df.groupby("boundary_type"):
                    acc_bt = float((g["gold_label"] == g["pred_label"]).mean())
                    fp_rate = float((g["pred_label"] == "uncertain").mean())
                    boundary_rows.append(
                        {
                            "run_id": run_id,
                            "boundary_type": bt,
                            "n_samples": int(len(g)),
                            "gold_uncertain_rate": round(float((g["gold_label"] == "uncertain").mean()), 4),
                            "accuracy_on_group": round(acc_bt, 4),
                            "predicted_uncertain_rate": round(fp_rate, 4),
                        }
                    )
                # Lát cắt theo 2 dạng bất định (uncertainty_type: epistemic vs aleatoric vs both vs none)
                if "uncertainty_type" in df.columns:
                    for ut, g in df.groupby("uncertainty_type"):
                        acc_ut = float((g["gold_label"] == g["pred_label"]).mean())
                        pred_unc_rate = float((g["pred_label"] == "uncertain").mean())
                        utype_rows.append(
                            {
                                "run_id": run_id,
                                "uncertainty_type": ut,
                                "n_samples": int(len(g)),
                                "gold_uncertain_rate": round(float((g["gold_label"] == "uncertain").mean()), 4),
                                "accuracy_on_group": round(acc_ut, 4),
                                "predicted_uncertain_rate": round(pred_unc_rate, 4),
                            }
                        )

    summary_df = pd.DataFrame(summary_rows)
    cols_order = [
        "split", "run_id", "n_samples", "macro_f1", "f1_uncertain",
        "precision_uncertain", "recall_uncertain", "f1_not_uncertain",
        "precision_not_uncertain", "recall_not_uncertain", "accuracy",
        "tp", "fp", "fn", "tn", "hard_neg_count", "hard_neg_fpr",
        "evidence_exact_match_uncertain", "evidence_token_f1_uncertain",
        "evidence_exact_match_all", "evidence_token_f1_all", "prediction_file",
    ]
    summary_df = summary_df[[c for c in cols_order if c in summary_df.columns]]
    summary_df.to_csv(EVAL_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    if slice_rows:
        slice_df = pd.DataFrame(slice_rows)
        slice_df.to_csv(EVAL_DIR / "metrics_by_source_year.csv", index=False, encoding="utf-8-sig")
    if boundary_rows:
        pd.DataFrame(boundary_rows).to_csv(EVAL_DIR / "metrics_by_boundary.csv", index=False, encoding="utf-8-sig")
    if utype_rows:
        pd.DataFrame(utype_rows).to_csv(EVAL_DIR / "metrics_by_uncertainty_type.csv", index=False, encoding="utf-8-sig")

    if verify:
        exp_path = EVAL_DIR / "experiments.csv"
        if exp_path.exists():
            exp_df = pd.read_csv(exp_path, encoding="utf-8-sig")
            test_map = summary_df[summary_df["split"] == "test"].set_index("run_id")["macro_f1"].to_dict()
            dev_map = summary_df[summary_df["split"] == "dev"].set_index("run_id")["macro_f1"].to_dict()
            for _, row in exp_df.iterrows():
                rid = str(row["run_id"])
                if rid in test_map:
                    assert abs(float(row["test_metric"]) - float(test_map[rid])) < 1e-3, (
                        f"Sai lệch test_metric cho {rid}: {row['test_metric']} vs {test_map[rid]}"
                    )
                if rid in dev_map:
                    assert abs(float(row["dev_metric"]) - float(dev_map[rid])) < 1e-3, (
                        f"Sai lệch dev_metric cho {rid}: {row['dev_metric']} vs {dev_map[rid]}"
                    )
            print("[VERIFY PASS] Toàn bộ số liệu trong experiments.csv khớp 100% với file prediction!")

    print("=== METRICS SUMMARY ===")
    print(summary_df[["split", "run_id", "macro_f1", "f1_uncertain", "precision_uncertain", "recall_uncertain", "hard_neg_fpr", "evidence_token_f1_uncertain"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Kiểm tra đối khớp với experiments.csv")
    args = parser.parse_args()
    evaluate_all(verify=args.verify)


if __name__ == "__main__":
    main()

"""
Bước 6 (Phần mở rộng Mục 4.4.4 – Gói 1):
1. Cross-Source Robustness: Leave-One-Source-Out (LOSO) 3-Fold trên 3 báo VietOnlineNews
   + Zero-Transfer trên 34 đoạn từ 4 báo hoàn toàn mới của UVN-1 (cho cả 5 cấu hình E0..E3).
2. Quét toàn bộ corpus kinh tế (39.028 bài) trong 36 tháng (01/2023–12/2025) trên GPU:
   tính chuỗi thô vs chuẩn hóa tỷ lệ và chỉ số Chuẩn hóa BBD-style cho E0..E3.
3. Đối chiếu mô tả các đỉnh bất định với 4 biến cố kinh tế-chính sách có nguồn công khai.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from evaluate_predictions import compute_metrics_for_df
from models_e0 import BBDVNRuleModel, TfidfBaselineModel
from models_e1 import VietnameseBiEncoderProbe
from models_e2 import SalienceDefinitionNLI
from models_e3 import HybridStackingEnsemble
from text_utils import make_stratified_folds, split_sentences_vi

ROOT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT_DIR / "data" / "processed"
ANN_DIR = ROOT_DIR / "data" / "annotations"
DATA_DIR = ROOT_DIR / "data"
CFG_DIR = ROOT_DIR / "configs"
EVAL_DIR = ROOT_DIR / "evaluation"
CKPT_DIR = ROOT_DIR / "checkpoints"


def _make_rob_row(eval_type: str, target_source: str, run_id: str, df_sub: pd.DataFrame, preds: List[Dict]) -> Dict:
    tmp = pd.DataFrame(
        {
            "item_id": df_sub["item_id"],
            "document_id": df_sub["document_id"],
            "source_name": df_sub["source_name"],
            "published_at": df_sub["published_at"],
            "gold_label": df_sub["task_label"],
            "pred_label": [p["pred_label"] for p in preds],
            "prob_uncertain": [p["prob_uncertain"] for p in preds],
            "gold_evidence": df_sub["evidence_sentence"],
            "pred_evidence": [p["pred_evidence"] for p in preds],
            "boundary_type": df_sub["boundary_type"],
        }
    )
    m = compute_metrics_for_df(tmp)
    m["eval_type"] = eval_type
    m["target_source"] = target_source
    m["run_id"] = run_id
    return m


def run_loso(von_df: pd.DataFrame, device: str) -> List[Dict]:
    rows = []
    sources = sorted(von_df["source_name"].unique().tolist())
    e0 = BBDVNRuleModel(CFG_DIR / "E0.json")

    for held_out in sources:
        train_pool = von_df[von_df["source_name"] != held_out].copy().reset_index(drop=True)
        test_sub = von_df[von_df["source_name"] == held_out].copy().reset_index(drop=True)

        fold_ids = make_stratified_folds(train_pool, n_splits=3, seed=42)
        pool_texts = train_pool["text"].astype(str).tolist()
        pool_titles = train_pool["title"].tolist()
        pool_y_bin = (train_pool["task_label"] == "uncertain").astype(int).tolist()
        pool_ut = train_pool["uncertainty_type"].astype(str)
        pool_y_epi = pool_ut.isin(["epistemic", "both"]).astype(int).tolist()
        pool_y_alea = pool_ut.isin(["aleatoric", "both"]).astype(int).tolist()
        pool_bt = train_pool["boundary_type"].astype(str).tolist()
        pool_gold_ev = train_pool["evidence_sentence"].astype(str).tolist()

        sub_texts = test_sub["text"].astype(str).tolist()
        sub_titles = test_sub["title"].tolist()

        # E0
        p_e0_pool = e0.predict_batch(pool_texts, titles=pool_titles)
        p_e0_sub = e0.predict_batch(sub_texts, titles=sub_titles)
        rows.append(_make_rob_row("LOSO", held_out, "E0", test_sub, p_e0_sub))

        # E0_tfidf (3-fold CV trên train_pool)
        e0t = TfidfBaselineModel(CFG_DIR / "E0_tfidf.json")
        _, e0t_oof = e0t.fit_cv(
            pool_texts, pool_titles, pool_y_bin, pool_y_epi, pool_y_alea, pool_bt, pool_gold_ev, fold_ids
        )
        p_e0t_sub = e0t.predict_batch(sub_texts, titles=sub_titles, extract_evidence=True, return_sent_scores=True)
        rows.append(_make_rob_row("LOSO", held_out, "E0_tfidf", test_sub, p_e0t_sub))

        # E1 (3-fold CV trên train_pool, GPU)
        e1 = VietnameseBiEncoderProbe(CFG_DIR / "E1.json", device=device)
        _, e1_oof = e1.fit_cv(
            pool_texts, pool_titles, pool_y_bin, pool_y_epi, pool_y_alea, pool_bt, pool_gold_ev, fold_ids
        )
        p_e1_sub = e1.predict_batch(sub_texts, titles=sub_titles, extract_evidence=True, return_sent_scores=True)
        rows.append(_make_rob_row("LOSO", held_out, "E1", test_sub, p_e1_sub))
        del e1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # E2 (3-fold Dual-Hypothesis bottom-6 frozen trên GPU)
        e2 = SalienceDefinitionNLI(CFG_DIR / "E2.json", device=device)
        _, e2_oof, _ = e2.fit_cv(
            pool_texts,
            pool_titles,
            pool_y_bin,
            pool_y_epi,
            pool_y_alea,
            pool_bt,
            pool_gold_ev,
            fold_ids,
            seeds=[42, 123, 2026],
        )
        p_e2_sub = e2.predict_batch(
            sub_texts, titles=sub_titles, extract_evidence=True, use_ensemble=True, return_sent_scores=True
        )
        rows.append(_make_rob_row("LOSO", held_out, "E2", test_sub, p_e2_sub))
        del e2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # E3 (OOF stacking lồng trên train_pool)
        e3 = HybridStackingEnsemble(CFG_DIR / "E3.json")
        e3.fit_cv(
            r0_list=p_e0_pool,
            r0t_list=e0t_oof,
            r1_list=e1_oof,
            r2_list=e2_oof,
            y_bin=pool_y_bin,
            boundary_types=pool_bt,
            fold_ids=fold_ids,
            raw_texts=pool_texts,
            titles=pool_titles,
            gold_evidences=pool_gold_ev,
        )
        p_e3_sub = e3.predict_from_base_preds(
            r0_list=p_e0_sub,
            r0t_list=p_e0t_sub,
            r1_list=p_e1_sub,
            r2_list=p_e2_sub,
            texts=sub_texts,
            titles=sub_titles,
            extract_evidence=True,
        )
        rows.append(_make_rob_row("LOSO", held_out, "E3", test_sub, p_e3_sub))

        print(f"[LOSO held_out={held_out}] hoàn thành E0..E3 (n={len(test_sub)})")
    return rows


def run_external_uvn1(uvn_df: pd.DataFrame, device: str) -> List[Dict]:
    rows = []
    texts = uvn_df["text"].astype(str).tolist()
    titles = uvn_df["title"].tolist()

    e0 = BBDVNRuleModel(CFG_DIR / "E0.json")
    e0t = TfidfBaselineModel.load(CKPT_DIR / "e0_tfidf.joblib")
    e1 = VietnameseBiEncoderProbe.load(CKPT_DIR / "e1_model.joblib", device=device)
    e2 = SalienceDefinitionNLI.load(CKPT_DIR / "e2_best", device=device)
    e3 = HybridStackingEnsemble.load(CKPT_DIR / "e3_stacking.joblib")

    p0 = e0.predict_batch(texts, titles=titles)
    p0t = e0t.predict_batch(texts, titles=titles, extract_evidence=True, return_sent_scores=True)
    p1 = e1.predict_batch(texts, titles=titles, extract_evidence=True, return_sent_scores=True)
    p2 = e2.predict_batch(texts, titles=titles, extract_evidence=True, use_ensemble=True, return_sent_scores=True)
    p3 = e3.predict_from_base_preds(p0, p0t, p1, p2, texts=texts, titles=titles, extract_evidence=True)

    for run_id, preds in [("E0", p0), ("E0_tfidf", p0t), ("E1", p1), ("E2", p2), ("E3", p3)]:
        out_df = pd.DataFrame(
            {
                "item_id": uvn_df["item_id"],
                "document_id": uvn_df["document_id"],
                "source_name": uvn_df["source_name"],
                "published_at": uvn_df["published_at"],
                "gold_label": uvn_df["task_label"],
                "pred_label": [p["pred_label"] for p in preds],
                "prob_uncertain": [p["prob_uncertain"] for p in preds],
                "pred_uncertainty_type": [p.get("pred_uncertainty_type", "none") for p in preds],
                "gold_evidence": uvn_df["evidence_sentence"],
                "pred_evidence": [p["pred_evidence"] for p in preds],
                "uncertainty_type": uvn_df["uncertainty_type"] if "uncertainty_type" in uvn_df.columns else "none",
                "boundary_type": uvn_df["boundary_type"],
            }
        )
        out_df.to_csv(EVAL_DIR / f"external_uvn1_predictions_{run_id}.csv", index=False, encoding="utf-8-sig")
        rows.append(_make_rob_row("External_UVN1_All", "UVN-1 (4 unseen outlets)", run_id, uvn_df, preds))

    del e1, e2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[External UVN-1] Đã đánh giá xong 5 cấu hình (E0..E3) trên {len(uvn_df)} mẫu ngoài nguồn.")
    return rows


def run_corpus_timeline(device: str) -> None:
    """
    Chạy suy luận trên toàn bộ 39.028 bài kinh tế sạch (2023-01 .. 2025-12) trên GPU
    cho tất cả 5 mô hình E0, E0_tfidf, E1, E2 (best_fold_idx length-sorted) và E3.
    Tự động chọn mô hình chính dựa trên OOF CV (n=184), tuyệt đối không đụng tập test.
    """
    articles = pd.read_parquet(PROC_DIR / "clean_econ_articles.parquet")
    articles = articles[articles["year_month"].between("2023-01", "2025-12")].copy().reset_index(drop=True)
    n_arts = len(articles)
    print(f"[Timeline] Chuẩn bị suy luận toàn corpus: {n_arts:,} bài kinh tế sạch...")

    oof_path = EVAL_DIR / "oof_cv_summary.csv"
    if oof_path.exists():
        sel_df = pd.read_csv(oof_path, encoding="utf-8-sig").sort_values(["macro_f1", "f1_uncertain"], ascending=False)
        best_run_id = str(sel_df.iloc[0]["run_id"])
    else:
        ms = pd.read_csv(EVAL_DIR / "metrics_summary.csv", encoding="utf-8-sig")
        dev_ms = ms[ms["split"] == "dev"].sort_values(["macro_f1", "f1_uncertain"], ascending=False)
        best_run_id = str(dev_ms.iloc[0]["run_id"])
    print(f"[Timeline] Mô hình chính chọn tự động từ OOF CV (không chạm test): {best_run_id}")

    texts = articles["paragraph"].astype(str).tolist()
    titles = articles["title"].tolist()

    # 1. E0 trên toàn corpus
    t0 = time.time()
    e0 = BBDVNRuleModel(CFG_DIR / "E0.json")
    e0_res = [e0.predict_one(t, title=ttl) for t, ttl in zip(texts, titles)]
    e0_prob = np.array([r["prob_uncertain"] for r in e0_res], dtype=np.float32)
    e0_pred = np.array([1.0 if r["pred_label"] == "uncertain" else 0.0 for r in e0_res], dtype=np.float32)
    e0_guard = np.array([1.0 if float(r.get("corp_guard_hits", 0)) > 0 else 0.0 for r in e0_res], dtype=np.float32)
    articles["prob_E0"] = e0_prob
    articles["pred_E0"] = e0_pred.astype(int)
    print(f"  -> E0 xong {n_arts:,} bài trong {time.time()-t0:.1f}s")

    # 2. E0_tfidf trên toàn corpus
    t0 = time.time()
    e0t = TfidfBaselineModel.load(CKPT_DIR / "e0_tfidf.joblib")
    e0t_res = e0t.predict_batch(texts, titles=titles, extract_evidence=False)
    e0t_prob = np.array([r["prob_uncertain"] for r in e0t_res], dtype=np.float32)
    e0t_p_bin = np.array([r["prob_bin"] for r in e0t_res], dtype=np.float32)
    e0t_p_epi = np.array([r["prob_epistemic"] for r in e0t_res], dtype=np.float32)
    e0t_p_alea = np.array([r["prob_aleatoric"] for r in e0t_res], dtype=np.float32)
    articles["prob_E0_tfidf"] = e0t_prob
    articles["pred_E0_tfidf"] = np.array([1 if r["pred_label"] == "uncertain" else 0 for r in e0t_res], dtype=int)
    print(f"  -> E0_tfidf xong {n_arts:,} bài trong {time.time()-t0:.1f}s")

    # 3. E1 trên toàn corpus (GPU)
    t0 = time.time()
    e1 = VietnameseBiEncoderProbe.load(CKPT_DIR / "e1_model.joblib", device=device)
    e1.batch_size = 128 if device == "cuda" else 32
    e1_prob, e1_p_bin, e1_p_epi, e1_p_alea = e1.predict_probs_fast(texts, titles=titles, show_progress=True)
    articles["prob_E1"] = e1_prob
    articles["pred_E1"] = (e1_prob >= e1.best_threshold).astype(int)
    print(f"  -> E1 GPU xong {n_arts:,} bài trong {time.time()-t0:.1f}s")
    del e1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. E2 trên toàn corpus (GPU fp32, best_fold_idx length-sorted)
    t0 = time.time()
    e2 = SalienceDefinitionNLI.load(CKPT_DIR / "e2_best", device=device)
    e2_prob, e2_p_epi, e2_p_alea = e2.predict_probs_single_fold_fast(
        texts, titles=titles, infer_batch_size=24 if device == "cuda" else 16, max_len=224
    )
    articles["prob_E2"] = e2_prob
    articles["pred_E2"] = (e2_prob >= e2.best_threshold).astype(int)
    print(f"  -> E2 GPU xong {n_arts:,} bài trong {time.time()-t0:.1f}s")

    # 5. E3 trên toàn corpus (<0.1s)
    t0 = time.time()
    e3 = HybridStackingEnsemble.load(CKPT_DIR / "e3_stacking.joblib")
    e3_prob = e3.predict_probs_from_arrays(
        e0_prob, e0_pred, e0_guard,
        e0t_p_bin, e0t_p_epi, e0t_p_alea,
        e1_p_bin, e1_p_epi, e1_p_alea,
        e2_p_epi, e2_p_alea, e2_prob,
    )
    articles["prob_E3"] = e3_prob
    articles["pred_E3"] = (e3_prob >= e3.best_threshold).astype(int)
    print(f"  -> E3 xong {n_arts:,} bài trong {time.time()-t0:.2f}s")

    best_col = f"pred_{best_run_id}" if f"pred_{best_run_id}" in articles.columns else "pred_E3"
    best_prob_col = f"prob_{best_run_id}" if f"prob_{best_run_id}" in articles.columns else "prob_E3"
    articles["pred_best"] = articles[best_col]
    articles["prob_best"] = articles[best_prob_col]

    # Tổng hợp theo (source_name, year_month)
    monthly = (
        articles.groupby(["source_name", "year_month"])
        .agg(
            total_clean_econ=("document_id", "count"),
            raw_uncertain_best=("pred_best", "sum"),
            raw_uncertain_E0=("pred_E0", "sum"),
            raw_uncertain_E0_tfidf=("pred_E0_tfidf", "sum"),
            raw_uncertain_E1=("pred_E1", "sum"),
            raw_uncertain_E2=("pred_E2", "sum"),
            raw_uncertain_E3=("pred_E3", "sum"),
        )
        .reset_index()
    )
    for tag, col_raw in [
        ("best", "raw_uncertain_best"),
        ("E0", "raw_uncertain_E0"),
        ("E0_tfidf", "raw_uncertain_E0_tfidf"),
        ("E1", "raw_uncertain_E1"),
        ("E2", "raw_uncertain_E2"),
        ("E3", "raw_uncertain_E3"),
    ]:
        sh = f"share_{tag}"
        monthly[sh] = monthly[col_raw] / monthly["total_clean_econ"]
        stds = monthly.groupby("source_name")[sh].transform("std").replace(0, 1.0)
        monthly[f"std_norm_{tag}"] = monthly[sh] / stds

    monthly.to_csv(EVAL_DIR / "extension_monthly_by_source.csv", index=False, encoding="utf-8-sig")

    # Tổng hợp toàn bộ 36 tháng (cấp tháng)
    agg = (
        monthly.groupby("year_month")
        .agg(
            total_clean_econ=("total_clean_econ", "sum"),
            raw_uncertain_best=("raw_uncertain_best", "sum"),
            raw_uncertain_E0=("raw_uncertain_E0", "sum"),
            raw_uncertain_E0_tfidf=("raw_uncertain_E0_tfidf", "sum"),
            raw_uncertain_E1=("raw_uncertain_E1", "sum"),
            raw_uncertain_E2=("raw_uncertain_E2", "sum"),
            raw_uncertain_E3=("raw_uncertain_E3", "sum"),
            mean_z_best=("std_norm_best", "mean"),
            mean_z_E0=("std_norm_E0", "mean"),
            mean_z_E0_tfidf=("std_norm_E0_tfidf", "mean"),
            mean_z_E1=("std_norm_E1", "mean"),
            mean_z_E2=("std_norm_E2", "mean"),
            mean_z_E3=("std_norm_E3", "mean"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    agg["naive_share_best"] = agg["raw_uncertain_best"] / agg["total_clean_econ"]
    for tag in ["best", "E0", "E0_tfidf", "E1", "E2", "E3"]:
        m_val = agg[f"mean_z_{tag}"].mean()
        agg[f"bbd_index_{tag}"] = 100.0 * agg[f"mean_z_{tag}"] / (m_val if m_val > 0 else 1.0)
    agg["raw_count_scaled_100"] = 100.0 * agg["raw_uncertain_best"] / agg["raw_uncertain_best"].mean()
    agg["best_model_used"] = best_run_id
    agg.to_csv(EVAL_DIR / "extension_monthly_index.csv", index=False, encoding="utf-8-sig")

    # Trích ví dụ câu chứng cứ tiêu biểu tại các tháng đỉnh / biến cố (dùng bộ chọn chứng cứ của best_run_id)
    events_df = pd.read_csv(DATA_DIR / "external_events_reference.csv", encoding="utf-8-sig")
    peak_months = set(events_df["year_month"].tolist())
    top_months = agg.sort_values("bbd_index_best", ascending=False)["year_month"].head(6).tolist()
    target_months = sorted(peak_months | set(top_months))

    sub_unc = articles[(articles["pred_best"] == 1) & (articles["year_month"].isin(target_months))].copy()
    sub_unc = sub_unc.sort_values("prob_best", ascending=False)
    picked_rows = []
    for (src, ym), grp in sub_unc.groupby(["source_name", "year_month"]):
        for _, r in grp.head(2).iterrows():
            picked_rows.append(r)

    sample_rows = []
    if picked_rows:
        pick_df = pd.DataFrame(picked_rows).reset_index(drop=True)
        pk_texts = pick_df["paragraph"].astype(str).tolist()
        pk_titles = pick_df["title"].tolist()
        if best_run_id == "E0_tfidf":
            pk_preds = e0t.predict_batch(pk_texts, titles=pk_titles, extract_evidence=True)
        elif best_run_id == "E2":
            pk_preds = e2.predict_batch(pk_texts, titles=pk_titles, extract_evidence=True, use_ensemble=True)
        elif best_run_id == "E3":
            pk0 = e0.predict_batch(pk_texts, titles=pk_titles)
            pk0t = e0t.predict_batch(pk_texts, titles=pk_titles, extract_evidence=True, return_sent_scores=True)
            e1_tmp = VietnameseBiEncoderProbe.load(CKPT_DIR / "e1_model.joblib", device=device)
            pk1 = e1_tmp.predict_batch(pk_texts, titles=pk_titles, extract_evidence=True, return_sent_scores=True)
            del e1_tmp
            pk2 = e2.predict_batch(pk_texts, titles=pk_titles, extract_evidence=True, use_ensemble=True, return_sent_scores=True)
            pk_preds = e3.predict_from_base_preds(pk0, pk0t, pk1, pk2, texts=pk_texts, titles=pk_titles, extract_evidence=True)
        elif best_run_id == "E1":
            e1_tmp = VietnameseBiEncoderProbe.load(CKPT_DIR / "e1_model.joblib", device=device)
            pk_preds = e1_tmp.predict_batch(pk_texts, titles=pk_titles, extract_evidence=True)
            del e1_tmp
        else:
            pk_preds = e0.predict_batch(pk_texts, titles=pk_titles)

        for (_, r), pr in zip(pick_df.iterrows(), pk_preds):
            sample_rows.append(
                {
                    "document_id": r["document_id"],
                    "source_name": r["source_name"],
                    "published_at": r["published_at"],
                    "year_month": r["year_month"],
                    "title": r["title"],
                    "pred_evidence": pr["pred_evidence"],
                    "prob_uncertain": round(float(r["prob_best"]), 4),
                    "model_used": best_run_id,
                }
            )
    pd.DataFrame(sample_rows).to_csv(EVAL_DIR / "extension_top_evidence_samples.csv", index=False, encoding="utf-8-sig")

    # Đối chiếu biến cố kinh tế
    ev_rows = []
    for _, er in events_df.iterrows():
        win = agg[(agg["year_month"] >= er["start_month"]) & (agg["year_month"] <= er["end_month"])]
        non_win = agg[~((agg["year_month"] >= er["start_month"]) & (agg["year_month"] <= er["end_month"]))]
        ev_rows.append(
            {
                "event_id": er["event_id"],
                "event_label": er["event_label"],
                "window": f"{er['start_month']}..{er['end_month']}",
                "mean_bbd_index_in_window": round(float(win["bbd_index_best"].mean()), 2),
                "peak_month_in_window": str(win.loc[win["bbd_index_best"].idxmax(), "year_month"]),
                "peak_bbd_index_in_window": round(float(win["bbd_index_best"].max()), 2),
                "mean_raw_count_in_window": round(float(win["raw_uncertain_best"].mean()), 1),
                "overall_non_window_median_index": round(float(non_win["bbd_index_best"].median()), 2),
                "reference_source": er["reference_source"],
                "caveat": "Mô tả đồng biến động tần suất văn bản trên 3 báo; KHÔNG phải chỉ số EPU chính thức và KHÔNG suy luận nhân quả.",
            }
        )
    pd.DataFrame(ev_rows).to_csv(EVAL_DIR / "extension_event_comparison.csv", index=False, encoding="utf-8-sig")

    corr_raw_norm = float(agg["raw_count_scaled_100"].corr(agg["bbd_index_best"]))
    corr_e0_e2 = float(agg["bbd_index_E0"].corr(agg["bbd_index_E2"]))
    corr_e0t_e2 = float(agg["bbd_index_E0_tfidf"].corr(agg["bbd_index_E2"]))
    corr_e1_e2 = float(agg["bbd_index_E1"].corr(agg["bbd_index_E2"]))
    corr_e3_e2 = float(agg["bbd_index_E3"].corr(agg["bbd_index_E2"]))
    summary_stats = {
        "total_clean_articles_scanned": int(n_arts),
        "best_model_used": best_run_id,
        "corr_raw_vs_bbd_normalized": round(corr_raw_norm, 4),
        "corr_bbd_E0_vs_E2": round(corr_e0_e2, 4),
        "corr_bbd_E0_tfidf_vs_E2": round(corr_e0t_e2, 4),
        "corr_bbd_E1_vs_E2": round(corr_e1_e2, 4),
        "corr_bbd_E3_vs_E2": round(corr_e3_e2, 4),
    }
    (EVAL_DIR / "extension_timeline_stats.json").write_text(
        json.dumps(summary_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Timeline Summary] {summary_stats}")

    del e2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(ANN_DIR / "gold_labels_split.csv", encoding="utf-8-sig")
    von_df = df[df["dataset_source"] == "VietOnlineNews"].copy().reset_index(drop=True)
    uvn_df = df[df["dataset_source"] == "UVN-1"].copy().reset_index(drop=True)

    rob_rows = []
    rob_rows.extend(run_loso(von_df, device=device))
    rob_rows.extend(run_external_uvn1(uvn_df, device=device))
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(EVAL_DIR / "extension_robustness.csv", index=False, encoding="utf-8-sig")

    print("\n=== CROSS-SOURCE ROBUSTNESS SUMMARY ===")
    print(rob_df[["eval_type", "target_source", "run_id", "n_samples", "macro_f1", "f1_uncertain", "hard_neg_fpr"]].to_string(index=False))

    run_corpus_timeline(device=device)


if __name__ == "__main__":
    main()

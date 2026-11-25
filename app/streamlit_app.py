r"""
Studio Tương tác Đề tài 4 (CSC4007 – Gói 1): Phát hiện & Diễn giải Thông tin Bất định Kinh tế
trong Tin tức Tiếng Việt (Hội đồng E0–E3, Bản đồ Epistemic×Aleatoric 2D, Heatmap Chứng cứ,
Phòng thí nghiệm Phản chứng What-If, Quét Bài báo Dài / Tệp CSV-TXT & Kiểm định Gold).
Chạy bằng môi trường ảo BTL_NPL_TQ (hỗ trợ GPU CUDA):
  .\BTL_NPL_TQ\Scripts\python.exe -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import io
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models_e0 import BBDVNRuleModel, TfidfBaselineModel  # noqa: E402
from models_e1 import VietnameseBiEncoderProbe  # noqa: E402
from models_e2 import SalienceDefinitionNLI  # noqa: E402
from models_e3 import HybridStackingEnsemble  # noqa: E402
from text_utils import clean_for_model, compose_title_text, normalize_text, split_sentences_vi  # noqa: E402

CFG_DIR = ROOT_DIR / "configs"
CKPT_DIR = ROOT_DIR / "checkpoints"
EVAL_DIR = ROOT_DIR / "evaluation"
DATA_DIR = ROOT_DIR / "data"
ANN_DIR = DATA_DIR / "annotations"

SAMPLE_TITLES = {
    "1. Bất định tri thức — Chính sách thuế quan (Ví dụ điển hình)": "Doanh nghiệp xuất khẩu dệt may, gỗ chờ lộ trình áp thuế đối ứng từ Mỹ",
    "2. Bất định ngẫu nhiên — Giá vàng SJC biến động mạnh trong ngày (Ví dụ của GV)": "Giá vàng hôm nay biến động mạnh, sáng vượt 68 triệu nhưng chiều giảm nhanh mất mốc này",
    "3. Bất định kết hợp (Tri thức + Ngẫu nhiên) — Tỷ giá & Fed": "Áp lực tỷ giá USD/VND tăng trước kỳ họp lãi suất của Fed",
    "4. Tin xấu ĐÃ CHẮC CHẮN — Hard Negative (Không phải bất định)": "Xây dựng An Phát lỗ ròng 142 tỷ đồng quý III và bị UBCKNN xử phạt",
    "5. Chính sách đã ban hành dứt khoát (Not Uncertain)": "Chính thức tăng lương tối thiểu vùng bình quân 6% từ ngày 1/7",
    "6. Tin PR / Quảng bá dự án Bất động sản (Not Uncertain)": "Ra mắt phân khu căn hộ cao cấp ven sông kèm ưu đãi thanh toán sớm 5%",
}

SAMPLE_INPUTS = {
    "1. Bất định tri thức — Chính sách thuế quan (Ví dụ điển hình)": (
        "Nhiều doanh nghiệp dệt may và gỗ xuất khẩu sang Mỹ cho biết đang tạm hoãn ký đơn hàng dài hạn "
        "vì lộ trình áp thuế đối ứng vẫn chưa ngã ngũ sau các vòng đàm phán. Nếu mức thuế cuối cùng "
        "thay đổi đột ngột vào quý tới, biên lợi nhuận của toàn ngành có thể chịu biến động khó lường."
    ),
    "2. Bất định ngẫu nhiên — Giá vàng SJC biến động mạnh trong ngày (Ví dụ của GV)": (
        "Giá vàng miếng SJC hôm nay biến động mạnh. Sau khi tăng cao vào buổi sáng, vượt mốc 68 triệu đồng/lượng, "
        "giá vàng miếng SJC đã giảm nhanh vào buổi chiều, mất mốc này."
    ),
    "3. Bất định kết hợp (Tri thức + Ngẫu nhiên) — Tỷ giá & Fed": (
        "Áp lực tỷ giá USD/VND khiến dư địa nới lỏng tiền tệ trong nước trở nên thu hẹp, trong khi thời điểm "
        "Fed cắt giảm lãi suất tiếp tục phụ thuộc sát vào số liệu lạm phát sắp công bố. Giới phân tích "
        "khuyến nghị doanh nghiệp vay ngoại tệ cần chuẩn bị kịch bản phòng ngừa biến động mạnh."
    ),
    "4. Tin xấu ĐÃ CHẮC CHẮN — Hard Negative (Không phải bất định)": (
        "Quý III/2024, Công ty Cổ phần Xây dựng An Phát ghi nhận doanh thu giảm 38% so với cùng kỳ và lỗ ròng "
        "142 tỷ đồng do chi phí lãi vay tăng cao. Trước đó một tuần, doanh nghiệp này cũng đã bị Ủy ban Chứng khoán "
        "Nhà nước xử phạt hành chính 92,5 triệu đồng vì chậm công bố báo cáo tài chính kiểm toán."
    ),
    "5. Chính sách đã ban hành dứt khoát (Not Uncertain)": (
        "Từ ngày 1/7, Nghị định điều chỉnh mức lương tối thiểu vùng tăng bình quân 6% chính thức có hiệu lực "
        "trên toàn quốc. Các doanh nghiệp đã hoàn tất việc rà soát thang bảng lương mới để áp dụng ngay trong kỳ trả lương tháng 7."
    ),
    "6. Tin PR / Quảng bá dự án Bất động sản (Not Uncertain)": (
        "Tổ hợp căn hộ cao cấp ven sông chính thức giới thiệu phân khu mới với hệ thống tiện ích khép kín, "
        "công viên cây xanh rộng hơn 3 ha và chính sách chiết khấu 5% cho khách hàng thanh toán sớm trong tháng."
    ),
}

UTYPE_META = {
    "epistemic": {"vi": "Bất định tri thức (Epistemic)", "short": "Epistemic (Tri thức)", "color": "#ef4444", "bg": "rgba(239,68,68,0.16)"},
    "aleatoric": {"vi": "Bất định ngẫu nhiên (Aleatoric)", "short": "Aleatoric (Ngẫu nhiên)", "color": "#f59e0b", "bg": "rgba(245,158,11,0.16)"},
    "both": {"vi": "Bất định kết hợp (Tri thức + Ngẫu nhiên)", "short": "Both (Kết hợp)", "color": "#a855f7", "bg": "rgba(168,85,247,0.18)"},
    "none": {"vi": "Không có bất định (Not Uncertain)", "short": "None (Chắc chắn)", "color": "#10b981", "bg": "rgba(16,185,129,0.16)"},
}

MODEL_NAMES_VI = {
    "E3": "E3: Hybrid Stacking Ensemble (Chính)",
    "E2": "E2: Dual-Hypothesis NLI (mDeBERTa-v3 5-Fold)",
    "E1": "E1: Sentence-Pooled Bi-Encoder (5-Fold)",
    "E0_tfidf": "E0_tfidf: Multi-Field TF-IDF + Structural",
    "E0": "E0: Luật từ điển BBD-VN v2.1",
}

EVENT_BANDS = [
    {"event": "NHNN 4 đợt hạ lãi suất (2023)", "start": "2023-03", "end": "2023-06", "color": "#38bdf8"},
    {"event": "Sốc Tỷ giá & Vàng SJC (2024)", "start": "2024-04", "end": "2024-06", "color": "#fb923c"},
    {"event": "Bão số 3 Yagi (2024)", "start": "2024-09", "end": "2024-10", "color": "#34d399"},
    {"event": "Thuế quan đối ứng Mỹ (2025)", "start": "2025-04", "end": "2025-08", "color": "#c084fc"},
]


def _inject_studio_css() -> None:
    st.markdown(
        """
        <style>
        .studio-card {
            background: rgba(30, 41, 59, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 10px;
            padding: 16px 18px;
            color: #f8fafc;
            margin-bottom: 12px;
        }
        .studio-verdict {
            border-left: 5px solid var(--accent, #38bdf8);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.85));
            border-radius: 10px;
            padding: 16px 20px;
            color: #f8fafc;
            border-top: 1px solid rgba(148, 163, 184, 0.22);
            border-right: 1px solid rgba(148, 163, 184, 0.22);
            border-bottom: 1px solid rgba(148, 163, 184, 0.22);
            margin-bottom: 14px;
        }
        .badge-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            margin: 2px 5px 2px 0;
        }
        .sent-heat-box {
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 10px;
            padding: 16px 18px;
            line-height: 1.85;
            font-size: 15px;
            color: #e2e8f0;
        }
        .sent-top {
            background: rgba(250, 204, 21, 0.22);
            border-bottom: 2px solid #facc15;
            color: #fef9c3;
            padding: 2px 6px;
            border-radius: 5px;
            font-weight: 600;
        }
        .sent-mid {
            background: rgba(56, 189, 248, 0.14);
            border-bottom: 1px dashed #38bdf8;
            color: #e0f2fe;
            padding: 1px 4px;
            border-radius: 4px;
        }
        .sent-low {
            color: #cbd5e1;
        }
        .whatif-box {
            background: rgba(15, 23, 42, 0.78);
            border: 1px dashed rgba(96, 165, 250, 0.55);
            border-radius: 10px;
            padding: 14px 16px;
            color: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Đang nạp mô hình NLP vào bộ nhớ...")
def load_model(model_key: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_key == "E0":
        return BBDVNRuleModel(CFG_DIR / "E0.json"), "cpu"
    if model_key == "E0_tfidf":
        return TfidfBaselineModel.load(CKPT_DIR / "e0_tfidf.joblib"), "cpu"
    if model_key == "E1":
        return VietnameseBiEncoderProbe.load(CKPT_DIR / "e1_model.joblib", device=device), device
    if model_key == "E2":
        return SalienceDefinitionNLI.load(CKPT_DIR / "e2_best", device=device), device
    if model_key == "E3":
        return HybridStackingEnsemble.load(CKPT_DIR / "e3_stacking.joblib"), device
    raise ValueError(model_key)


def _extract_e0_sent_scores(e0: BBDVNRuleModel, raw_text: str, title: Optional[str]) -> Tuple[List[str], np.ndarray]:
    sents = split_sentences_vi(raw_text)
    if not sents:
        fb = normalize_text(raw_text)
        sents = [fb] if fb else [""]
    t_clean = clean_for_model(title) if title and str(title).strip().lower() != "nan" else ""
    t_set = set(re.findall(r"\w+", t_clean.lower()))
    raw_scores = []
    for s in sents:
        srf = e0.extract_rule_features(s, title=None)
        s_set = set(re.findall(r"\w+", clean_for_model(s).lower()))
        t_jacc = float(len(t_set & s_set)) / float(len(t_set | s_set)) if (t_set and s_set) else 0.0
        sc = (
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
        raw_scores.append(sc)
    arr = np.asarray(raw_scores, dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-0.5 * arr))
    return sents, probs.astype(np.float32)


def _score_sentences_for_model(
    model_key: str,
    raw_text: str,
    title: Optional[str],
    base_res: Dict[str, Any],
    r0t: Optional[Dict[str, Any]] = None,
    r1: Optional[Dict[str, Any]] = None,
    r2: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], np.ndarray]:
    sents = split_sentences_vi(raw_text)
    if not sents:
        fb = normalize_text(raw_text)
        sents = [fb] if fb else [""]

    if model_key == "E0":
        e0, _ = load_model("E0")
        return _extract_e0_sent_scores(e0, raw_text, title)

    if model_key == "E0_tfidf":
        e0t, _ = load_model("E0_tfidf")
        ev_acc = np.zeros(len(sents), dtype=np.float64)
        for b in e0t.fold_bundles:
            _, s_sc = e0t._sent_model_scores(b, raw_text)
            _, ep = b["ev_scorer"].score_sentences(
                raw_text=raw_text, title=title, rule_model=e0t.rule_model, model_sent_scores=s_sc
            )
            ev_acc += ep
        return sents, (ev_acc / float(len(e0t.fold_bundles))).astype(np.float32)

    if model_key == "E1":
        e1, _ = load_model("E1")
        _, sent_embs_list, _, s_t_cos, s_d_cos = e1.encode_multiview([raw_text], titles=[title])
        se = sent_embs_list[0]
        ev_acc = np.zeros(len(sents), dtype=np.float64)
        for b in e1.fold_bundles:
            s_sc = np.column_stack(
                [
                    b["sent_clf_bin"].predict_proba(se)[:, 1],
                    b["sent_clf_epi"].predict_proba(se)[:, 1],
                    b["sent_clf_alea"].predict_proba(se)[:, 1],
                ]
            )
            _, ep = b["ev_scorer"].score_sentences(
                raw_text=raw_text,
                title=title,
                rule_model=e1.rule_model,
                model_sent_scores=s_sc,
                sent_title_cos=s_t_cos[0],
                sent_doc_cos=s_d_cos[0],
            )
            ev_acc += ep
        return sents, (ev_acc / float(len(e1.fold_bundles))).astype(np.float32)

    if model_key == "E2":
        e2, _ = load_model("E2")
        s_mat = base_res.get("sent_scores")
        ev_acc = np.zeros(len(sents), dtype=np.float64)
        if e2.fold_ev_scorers and s_mat is not None:
            for sc in e2.fold_ev_scorers:
                _, ep = sc.score_sentences(
                    raw_text=raw_text, title=title, rule_model=e2.rule_model, model_sent_scores=s_mat
                )
                ev_acc += ep
            return sents, (ev_acc / float(len(e2.fold_ev_scorers))).astype(np.float32)
        return _extract_e0_sent_scores(e2.rule_model, raw_text, title)

    if model_key == "E3":
        e3, _ = load_model("E3")
        fused_sc = HybridStackingEnsemble._combine_sent_scores([r0t], [r1], [r2])[0]
        ev_acc = np.zeros(len(sents), dtype=np.float64)
        for b in e3.fold_bundles:
            _, ep = b["ev_scorer"].score_sentences(
                raw_text=raw_text, title=title, rule_model=e3.rule_model, model_sent_scores=fused_sc
            )
            ev_acc += ep
        return sents, (ev_acc / float(len(e3.fold_bundles))).astype(np.float32)

    return sents, np.ones(len(sents), dtype=np.float32)


def extract_matched_signals(text: str, title: Optional[str]) -> Dict[str, List[str]]:
    e0, _ = load_model("E0")
    combined = compose_title_text(title, text).lower()

    def _hits(pats):
        seen = []
        for p in pats:
            for m in p.findall(combined):
                tok = m.strip().lower()
                if tok and tok not in seen:
                    seen.append(tok)
        return seen

    return {
        "Tri thức (U)": _hits(e0.u_pats),
        "Đề xuất / Dự thảo": _hits(e0.prop_pats),
        "Biến động mạnh (Aleatoric)": _hits(e0.a_strong_pats) + [k for k in _hits(e0.a_mod_pats)],
        "Chính sách (P)": _hits(e0.p_pats),
        "Tài sản thị trường": _hits(e0.asset_pats),
        "Phủ định / Guardrail PR": _hits(e0.neg_pats) + _hits(e0.corp_guard_pats),
    }


def _infer_e0_dual_coords(res: Dict[str, Any]) -> Tuple[float, float]:
    p = float(res["prob_uncertain"])
    ut = res.get("pred_uncertainty_type", "none")
    if ut == "both":
        return p, p
    if ut == "epistemic":
        return p, 0.08
    if ut == "aleatoric":
        return 0.08, p
    return 0.08, 0.08


def run_council_analysis(text: str, title: Optional[str]) -> Dict[str, Any]:
    """Chạy đồng thời cả 5 mô hình E0..E3 và chấm điểm chứng cứ cấp câu."""
    t0 = time.time()
    ttl = title.strip() if title and title.strip() else None
    e0, _ = load_model("E0")
    e0t, _ = load_model("E0_tfidf")
    e1, _ = load_model("E1")
    e2, dev = load_model("E2")
    e3, _ = load_model("E3")

    r0 = e0.predict_batch([text], titles=[ttl])[0]
    r0t = e0t.predict_batch([text], titles=[ttl], extract_evidence=True, return_sent_scores=True)[0]
    r1 = e1.predict_batch([text], titles=[ttl], extract_evidence=True, return_sent_scores=True)[0]
    r2 = e2.predict_batch([text], titles=[ttl], extract_evidence=True, use_ensemble=True, return_sent_scores=True)[0]
    r3 = e3.predict_from_base_preds([r0], [r0t], [r1], [r2], texts=[text], titles=[ttl], extract_evidence=True)[0]

    sents, sent_probs = _score_sentences_for_model("E3", text, ttl, r3, r0t=r0t, r1=r1, r2=r2)
    signals = extract_matched_signals(text, ttl)

    thresholds = {
        "E0": 0.50,
        "E0_tfidf": float(e0t.best_threshold),
        "E1": float(e1.best_threshold),
        "E2": float(e2.best_threshold),
        "E3": float(e3.best_threshold),
    }
    raw_preds = {"E0": r0, "E0_tfidf": r0t, "E1": r1, "E2": r2, "E3": r3}
    council_rows = []
    for k in ["E3", "E2", "E1", "E0_tfidf", "E0"]:
        pk = raw_preds[k]
        if k == "E0":
            pe, pa = _infer_e0_dual_coords(pk)
        else:
            pe = float(pk.get("prob_epistemic", pk["prob_uncertain"]))
            pa = float(pk.get("prob_aleatoric", pk["prob_uncertain"]))
        council_rows.append(
            {
                "run_id": k,
                "model_name": MODEL_NAMES_VI[k],
                "pred_label": pk["pred_label"],
                "prob_uncertain": float(pk["prob_uncertain"]),
                "prob_epistemic": pe,
                "prob_aleatoric": pa,
                "threshold": thresholds[k],
                "pred_uncertainty_type": pk.get("pred_uncertainty_type", "none"),
                "pred_evidence": pk["pred_evidence"],
            }
        )

    return {
        "mode": "council",
        "primary_key": "E3",
        "primary_res": r3,
        "council_rows": council_rows,
        "sents": sents,
        "sent_probs": sent_probs,
        "signals": signals,
        "device": dev.upper(),
        "elapsed_s": round(time.time() - t0, 3),
        "text": text,
        "title": ttl or "",
    }


def run_single_analysis(model_key: str, text: str, title: Optional[str]) -> Dict[str, Any]:
    if model_key == "E3":
        return run_council_analysis(text, title)
    t0 = time.time()
    ttl = title.strip() if title and title.strip() else None
    mdl, dev = load_model(model_key)
    if model_key == "E0":
        res = mdl.predict_batch([text], titles=[ttl])[0]
        thr = 0.50
        pe, pa = _infer_e0_dual_coords(res)
    elif model_key == "E2":
        res = mdl.predict_batch([text], titles=[ttl], extract_evidence=True, use_ensemble=True, return_sent_scores=True)[0]
        thr = float(mdl.best_threshold)
        pe, pa = float(res["prob_epistemic"]), float(res["prob_aleatoric"])
    else:
        res = mdl.predict_batch([text], titles=[ttl], extract_evidence=True, return_sent_scores=True)[0]
        thr = float(mdl.best_threshold)
        pe, pa = float(res["prob_epistemic"]), float(res["prob_aleatoric"])

    res["prob_epistemic"] = pe
    res["prob_aleatoric"] = pa
    sents, sent_probs = _score_sentences_for_model(model_key, text, ttl, res)
    signals = extract_matched_signals(text, ttl)

    return {
        "mode": "single",
        "primary_key": model_key,
        "primary_res": res,
        "council_rows": [
            {
                "run_id": model_key,
                "model_name": MODEL_NAMES_VI[model_key],
                "pred_label": res["pred_label"],
                "prob_uncertain": float(res["prob_uncertain"]),
                "prob_epistemic": pe,
                "prob_aleatoric": pa,
                "threshold": thr,
                "pred_uncertainty_type": res.get("pred_uncertainty_type", "none"),
                "pred_evidence": res["pred_evidence"],
            }
        ],
        "sents": sents,
        "sent_probs": sent_probs,
        "signals": signals,
        "device": dev.upper(),
        "elapsed_s": round(time.time() - t0, 3),
        "text": text,
        "title": ttl or "",
    }


@st.cache_data(show_spinner="Đang dựng bản đồ tham chiếu 2D từ 60 mẫu Test...")
def get_test_quadrant_reference() -> pd.DataFrame:
    """Tính tọa độ (P_epistemic, P_aleatoric) của 60 bài báo tập Test khóa làm nền tham chiếu."""
    g = pd.read_csv(ANN_DIR / "gold_labels_split.csv", encoding="utf-8-sig")
    te = g[g["split"] == "test"].copy().reset_index(drop=True)
    texts = te["text"].astype(str).tolist()
    titles = te["title"].tolist()

    e0t, _ = load_model("E0_tfidf")
    e1, _ = load_model("E1")
    e2, _ = load_model("E2")

    p0t = e0t.predict_batch(texts, titles=titles, extract_evidence=False)
    p1 = e1.predict_batch(texts, titles=titles, extract_evidence=False)
    p2 = e2.predict_batch(texts, titles=titles, extract_evidence=False, use_ensemble=False)

    rows = []
    for (_, row), a, b, c in zip(te.iterrows(), p0t, p1, p2):
        pe = (float(a["prob_epistemic"]) + float(b["prob_epistemic"]) + float(c["prob_epistemic"])) / 3.0
        pa = (float(a["prob_aleatoric"]) + float(b["prob_aleatoric"]) + float(c["prob_aleatoric"])) / 3.0
        ut = str(row.get("uncertainty_type", "none"))
        rows.append(
            {
                "item_id": row["item_id"],
                "source_name": row["source_name"],
                "title": str(row.get("title", ""))[:85],
                "uncertainty_type": ut,
                "loai_nhan": UTYPE_META.get(ut, UTYPE_META["none"])["short"],
                "prob_epistemic": round(pe, 4),
                "prob_aleatoric": round(pa, 4),
            }
        )
    return pd.DataFrame(rows)


def _render_sentence_heatmap(sents: List[str], sent_probs: np.ndarray) -> str:
    if not sents:
        return ""
    top_idx = int(np.argmax(sent_probs))
    p_top = float(sent_probs[top_idx])
    chunks = []
    for idx, (s, p) in enumerate(zip(sents, sent_probs)):
        esc = html.escape(s)
        if idx == top_idx:
            chunks.append(
                f"<span class='sent-top' title='Câu #{idx+1} (Chứng cứ chính): điểm={p:.3f}'>"
                f"[Câu {idx+1} ★ {p:.2f}] {esc}</span>"
            )
        elif p >= 0.45 or (p_top > 0 and p >= 0.72 * p_top):
            chunks.append(
                f"<span class='sent-mid' title='Câu #{idx+1}: điểm={p:.3f}'>"
                f"[Câu {idx+1} • {p:.2f}] {esc}</span>"
            )
        else:
            chunks.append(
                f"<span class='sent-low' title='Câu #{idx+1}: điểm={p:.3f}'>{esc}</span>"
            )
    return f"<div class='sent-heat-box'>{' '.join(chunks)}</div>"


def _render_quadrant_chart(ref_df: pd.DataFrame, cur_pe: float, cur_pa: float, cur_title: str) -> alt.Chart:
    domain = ["Epistemic (Tri thức)", "Aleatoric (Ngẫu nhiên)", "Both (Kết hợp)", "None (Chắc chắn)"]
    range_c = ["#ef4444", "#f59e0b", "#a855f7", "#10b981"]

    bg_points = (
        alt.Chart(ref_df)
        .mark_circle(size=75, opacity=0.45)
        .encode(
            x=alt.X("prob_epistemic:Q", title="Xác suất Bất định Tri thức — P(Epistemic)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("prob_aleatoric:Q", title="Xác suất Bất định Ngẫu nhiên — P(Aleatoric)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("loai_nhan:N", title="Nhóm Gold (60 mẫu Test)", scale=alt.Scale(domain=domain, range=range_c)),
            tooltip=[
                alt.Tooltip("item_id:N", title="Mã"),
                alt.Tooltip("source_name:N", title="Nguồn"),
                alt.Tooltip("loai_nhan:N", title="Dạng Gold"),
                alt.Tooltip("title:N", title="Tiêu đề"),
                alt.Tooltip("prob_epistemic:Q", format=".3f", title="P(Epistemic)"),
                alt.Tooltip("prob_aleatoric:Q", format=".3f", title="P(Aleatoric)"),
            ],
        )
    )

    cur_df = pd.DataFrame(
        [
            {
                "prob_epistemic": float(cur_pe),
                "prob_aleatoric": float(cur_pa),
                "label": "BÀI BÁO HIỆN TẠI",
                "title": cur_title or "(Đoạn tin người dùng nhập)",
            }
        ]
    )
    cur_point = (
        alt.Chart(cur_df)
        .mark_point(shape="diamond", size=340, filled=True, color="#facc15", stroke="#ffffff", strokeWidth=2.2)
        .encode(
            x="prob_epistemic:Q",
            y="prob_aleatoric:Q",
            tooltip=[
                alt.Tooltip("label:N", title="Vị trí"),
                alt.Tooltip("title:N", title="Tiêu đề"),
                alt.Tooltip("prob_epistemic:Q", format=".3f", title="P(Epistemic)"),
                alt.Tooltip("prob_aleatoric:Q", format=".3f", title="P(Aleatoric)"),
            ],
        )
    )
    cur_text = (
        alt.Chart(cur_df)
        .mark_text(align="left", dx=12, dy=-10, fontSize=12, fontWeight="bold", color="#fde047")
        .encode(x="prob_epistemic:Q", y="prob_aleatoric:Q", text="label:N")
    )

    vline = alt.Chart(pd.DataFrame({"x": [0.45]})).mark_rule(strokeDash=[5, 5], color="#94a3b8", opacity=0.7).encode(x="x:Q")
    hline = alt.Chart(pd.DataFrame({"y": [0.45]})).mark_rule(strokeDash=[5, 5], color="#94a3b8", opacity=0.7).encode(y="y:Q")

    quad_labels = pd.DataFrame(
        [
            {"x": 0.22, "y": 0.94, "txt": "GÓC NGẪU NHIÊN (Aleatoric)"},
            {"x": 0.75, "y": 0.94, "txt": "BẤT ĐỊNH KÉP (Both)"},
            {"x": 0.22, "y": 0.05, "txt": "CHẮC CHẮN (Not Uncertain)"},
            {"x": 0.75, "y": 0.05, "txt": "GÓC TRI THỨC (Epistemic)"},
        ]
    )
    q_txt = (
        alt.Chart(quad_labels)
        .mark_text(fontSize=10.5, fontWeight="bold", color="#cbd5e1", opacity=0.75)
        .encode(x="x:Q", y="y:Q", text="txt:N")
    )

    chart = (bg_points + vline + hline + q_txt + cur_point + cur_text).properties(height=360).configure_view(strokeOpacity=0)
    return chart


def _render_council_chart(council_df: pd.DataFrame) -> alt.Chart:
    df_c = council_df.copy()
    df_c["color_hex"] = np.where(df_c["pred_label"] == "uncertain", "#f97316", "#10b981")
    df_c["trang_thai"] = np.where(df_c["pred_label"] == "uncertain", "UNCERTAIN", "NOT_UNCERTAIN")

    bars = (
        alt.Chart(df_c)
        .mark_bar(cornerRadiusEnd=5, height=22)
        .encode(
            x=alt.X("prob_uncertain:Q", title="Xác suất P(uncertain)", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("run_id:N", sort=["E3", "E2", "E1", "E0_tfidf", "E0"], title="Cấu hình"),
            color=alt.Color(
                "trang_thai:N",
                title="Phán quyết",
                scale=alt.Scale(domain=["UNCERTAIN", "NOT_UNCERTAIN"], range=["#f97316", "#10b981"]),
            ),
            tooltip=[
                alt.Tooltip("run_id:N", title="Mô hình"),
                alt.Tooltip("trang_thai:N", title="Kết luận"),
                alt.Tooltip("prob_uncertain:Q", format=".3f", title="P(uncertain)"),
                alt.Tooltip("threshold:Q", format=".3f", title="Ngưỡng OOF"),
                alt.Tooltip("pred_uncertainty_type:N", title="Dạng dự đoán"),
            ],
        )
    )
    thr_ticks = (
        alt.Chart(df_c)
        .mark_tick(color="#fde047", thickness=3, size=24)
        .encode(
            x="threshold:Q",
            y=alt.Y("run_id:N", sort=["E3", "E2", "E1", "E0_tfidf", "E0"]),
            tooltip=[alt.Tooltip("threshold:Q", format=".3f", title="Ngưỡng quyết định")],
        )
    )
    labels = (
        alt.Chart(df_c)
        .mark_text(align="left", dx=6, fontSize=11.5, color="#f8fafc")
        .encode(
            x="prob_uncertain:Q",
            y=alt.Y("run_id:N", sort=["E3", "E2", "E1", "E0_tfidf", "E0"]),
            text=alt.Text("prob_uncertain:Q", format=".3f"),
        )
    )
    return (bars + thr_ticks + labels).properties(height=230).configure_view(strokeOpacity=0)


# --- Callbacks quản lý Session State sạch sẽ (không bao giờ ghi đè văn bản người dùng tự gõ) ---
def _cb_clear_inputs() -> None:
    st.session_state["input_title"] = ""
    st.session_state["input_text"] = ""
    st.session_state["last_analysis"] = None
    st.session_state["whatif_result"] = None


def main() -> None:
    st.set_page_config(
        page_title="CSC4007 - ĐT04: Studio Bất định Kinh tế",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_studio_css()

    # Khởi tạo an toàn state 1 lần duy nhất
    st.session_state.setdefault("input_title", "")
    st.session_state.setdefault("input_text", "")
    st.session_state.setdefault("last_analysis", None)
    st.session_state.setdefault("whatif_result", None)
    st.session_state.setdefault("long_scan_df", None)

    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # === SIDEBAR THÔNG TIN HỆ THỐNG ===
    with st.sidebar:
        st.markdown("### Đề tài 4 — CSC4007")
        st.caption(f"Thiết bị suy luận: **`{gpu_info}`**")
        exec_mode = st.radio(
            "Chế độ phân tích Tab 1:",
            ["Hội đồng 5 mô hình E0–E3 (Đầy đủ)", "Chế độ nhanh 1 mô hình"],
            index=0,
        )
        fast_model_key = "E3"
        if exec_mode == "Chế độ nhanh 1 mô hình":
            fast_model_key = st.selectbox(
                "Chọn mô hình đơn:",
                ["E0_tfidf", "E2", "E1", "E0", "E3"],
                format_func=lambda k: MODEL_NAMES_VI[k],
            )
        show_ref_cloud = st.checkbox("Hiện 60 bài Test tham chiếu trên Bản đồ 2D", value=True)

        st.divider()
        st.markdown("**Thông số đã xác minh (`Gói 1`):**")
        st.metric("Test Macro-F1 (`E3`)", "0.9333", "+0.084 vs trước Gói 1")
        st.metric("Hard-Negative FPR (`E3` / `E0_tfidf`)", "0.00%", "Giảm từ 18.2%–54.5%")
        st.metric("Evidence Token-F1 (`E3`)", "0.8230", "100% Exact Substring")
        st.caption("Bộ Gold ký xác nhận: **279/279** mẫu (`Kappa = 0.9282`) • Chuẩn `Guideline v2.0` (Tri thức + Ngẫu nhiên).")

    st.title("Studio Phát hiện Thông tin Bất định Kinh tế Tiếng Việt (ĐT04)")
    st.caption(
        "Dán bất kỳ bài báo/đoạn tin kinh tế mới nào để soi **Hội đồng 5 mô hình E0–E3**, "
        "xem vị trí trên **Bản đồ 2D Epistemic × Aleatoric**, **Heatmap Chứng cứ cấp câu** và **Thử nghiệm Phản chứng What-If**."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Studio Phân tích Tin Mới (Live AI Council)",
            "2. Quét Bài Báo Dài & Tệp Hàng Loạt (.txt / .csv)",
            "3. Chuỗi Tần suất 36 Tháng (39.028 bài báo)",
            "4. Thực nghiệm E0–E3 & Hậu kiểm Ký xác nhận Gold",
        ]
    )

    # =====================================================================
    # TAB 1: STUDIO PHÂN TÍCH TIN MỚI
    # =====================================================================
    with tab1:
        col_in_t, col_in_a = st.columns([3, 1.15])
        with col_in_t:
            st.text_input(
                "Tiêu đề bài báo kinh tế (tùy chọn — giúp tăng độ chính xác):",
                key="input_title",
                placeholder="Dán tiêu đề bài báo mới vào đây (hoặc bỏ trống nếu chỉ có đoạn văn)...",
            )
            st.text_area(
                "Nội dung đoạn tin kinh tế tiếng Việt (hỗ trợ dán trực tiếp bài mới hoàn toàn):",
                key="input_text",
                height=175,
                placeholder="Dán hoặc gõ một đoạn tin tức kinh tế tiếng Việt mới vào đây (ví dụ từ VnExpress, CafeF, Vietstock, Tuổi Trẻ...), sau đó bấm 'Phân tích ngay'...",
            )

        raw_text_cur = st.session_state["input_text"]
        raw_title_cur = st.session_state["input_title"]
        sents_preview = split_sentences_vi(raw_text_cur) if raw_text_cur.strip() else []
        paras_preview = [p.strip() for p in re.split(r"\n\s*\n", raw_text_cur.strip()) if p.strip()] if raw_text_cur.strip() else []
        cleaned_preview = clean_for_model(raw_text_cur) if raw_text_cur.strip() else ""

        with col_in_a:
            st.markdown("<div class='studio-card'>", unsafe_allow_html=True)
            st.markdown("**Chẩn đoán đầu vào**")
            st.write(f"- Số ký tự: **{len(raw_text_cur.strip())}**")
            st.write(f"- Số câu tách được: **{len(sents_preview)}**")
            st.write(f"- Số đoạn văn: **{len(paras_preview)}**")
            has_crawl_diff = raw_text_cur.strip() and cleaned_preview != normalize_text(raw_text_cur)
            if has_crawl_diff:
                st.caption("Đã phát hiện byline/dính chữ HTML — sẽ tự lọc khi suy luận (vẫn giữ nguyên văn câu chứng cứ).")
            st.markdown("</div>", unsafe_allow_html=True)
            analyze_clicked = st.button(
                "🚀 Phân tích ngay",
                type="primary",
                width="stretch",
            )
            st.button(
                "🧹 Xóa trắng nhập tin mới",
                width="stretch",
                on_click=_cb_clear_inputs,
            )

        if len(paras_preview) >= 2:
            st.info(
                f"Phát hiện **{len(paras_preview)} đoạn văn** ngăn cách bằng dòng trống. Tab 1 phân tích tổng hợp cả khối văn bản; "
                "nếu muốn quét riêng **từng đoạn** trong nguyên bài báo dài, hãy chuyển sang **Tab 2**!"
            )

        if analyze_clicked:
            if not raw_text_cur.strip():
                st.warning("Vui lòng dán hoặc gõ một đoạn tin kinh tế vào ô nội dung trước khi phân tích!")
            else:
                st.session_state["whatif_result"] = None
                if exec_mode == "Hội đồng 5 mô hình E0–E3 (Đầy đủ)":
                    with st.spinner("Đang chạy đồng thời 5 mô hình (E0, E0_tfidf, E1, E2 5-Fold GPU, E3 Stacking)..."):
                        st.session_state["last_analysis"] = run_council_analysis(raw_text_cur, raw_title_cur)
                else:
                    with st.spinner(f"Đang chạy mô hình {fast_model_key}..."):
                        st.session_state["last_analysis"] = run_single_analysis(fast_model_key, raw_text_cur, raw_title_cur)

        analysis = st.session_state.get("last_analysis")
        if analysis:
            res = analysis["primary_res"]
            p_key = analysis["primary_key"]
            utype = res.get("pred_uncertainty_type", "none")
            um = UTYPE_META.get(utype, UTYPE_META["none"])
            pred_lbl = res["pred_label"]
            p_unc = float(res["prob_uncertain"])
            p_epi = float(res.get("prob_epistemic", 0.0))
            p_alea = float(res.get("prob_aleatoric", 0.0))

            c_df = pd.DataFrame(analysis["council_rows"])
            agree_cnt = int((c_df["pred_label"] == pred_lbl).sum())
            agree_txt = f"Đồng thuận {agree_cnt}/{len(c_df)} mô hình" if len(c_df) > 1 else f"Mô hình đơn {p_key}"
            verdict_word = "CÓ BẤT ĐỊNH KINH TẾ (UNCERTAIN)" if pred_lbl == "uncertain" else "KHÔNG CÓ BẤT ĐỊNH (NOT UNCERTAIN)"
            accent = um["color"] if pred_lbl == "uncertain" else "#10b981"

            st.markdown(
                f"""
                <div class='studio-verdict' style='--accent:{accent};'>
                  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
                    <div>
                      <div style='font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.06em;'>
                        Kết luận hệ thống ({html.escape(p_key)}) • {html.escape(agree_txt)} • {analysis['elapsed_s']:.2f}s ({html.escape(analysis['device'])})
                      </div>
                      <div style='font-size:22px;font-weight:700;color:{accent};margin-top:2px;'>
                        {verdict_word}
                      </div>
                    </div>
                    <div style='text-align:right;'>
                      <span class='badge-pill' style='background:{um["bg"]};color:{um["color"]};border:1px solid {um["color"]};font-size:13px;'>
                        {html.escape(um["vi"])}
                      </span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P(Uncertain tổng)", f"{p_unc:.3f}")
            m2.metric("P(Epistemic — Tri thức)", f"{p_epi:.3f}")
            m3.metric("P(Aleatoric — Ngẫu nhiên)", f"{p_alea:.3f}")
            m4.metric("Số câu phân tích", f"{len(analysis['sents'])} câu")

            # --- Hàng giữa: Hội đồng 5 mô hình & Bản đồ 2D ---
            col_c1, col_c2 = st.columns([1.05, 1.15])
            with col_c1:
                st.markdown("#### Hội đồng 5 Mô hình So găng (Kèm ngưỡng vàng)")
                st.altair_chart(_render_council_chart(c_df), width="stretch")
                disp_c = c_df[["run_id", "pred_label", "prob_uncertain", "threshold", "pred_uncertainty_type"]].rename(
                    columns={
                        "run_id": "Mô hình",
                        "pred_label": "Nhãn",
                        "prob_uncertain": "P(unc)",
                        "threshold": "Ngưỡng",
                        "pred_uncertainty_type": "Dạng",
                    }
                )
                st.dataframe(disp_c.round(3), width="stretch", hide_index=True)

            with col_c2:
                st.markdown("#### Bản đồ 2 Chiều Bất định: Epistemic × Aleatoric")
                if show_ref_cloud:
                    ref_df = get_test_quadrant_reference()
                else:
                    ref_df = pd.DataFrame(columns=["item_id", "source_name", "title", "uncertainty_type", "loai_nhan", "prob_epistemic", "prob_aleatoric"])
                st.altair_chart(
                    _render_quadrant_chart(ref_df, p_epi, p_alea, analysis["title"]),
                    width="stretch",
                )

            # --- Hàng dưới: X-Quang Chứng cứ & Signal Pills ---
            st.markdown("#### X-Quang Chứng cứ Cấp Câu & Tín hiệu Từ vựng BBD-VN v2.1")
            st.markdown(_render_sentence_heatmap(analysis["sents"], analysis["sent_probs"]), unsafe_allow_html=True)

            sig_html_parts = []
            sig_colors = {
                "Tri thức (U)": ("#ef4444", "rgba(239,68,68,0.18)"),
                "Đề xuất / Dự thảo": ("#f97316", "rgba(249,115,22,0.18)"),
                "Biến động mạnh (Aleatoric)": ("#f59e0b", "rgba(245,158,11,0.18)"),
                "Chính sách (P)": ("#38bdf8", "rgba(56,189,248,0.16)"),
                "Tài sản thị trường": ("#a855f7", "rgba(168,85,247,0.18)"),
                "Phủ định / Guardrail PR": ("#10b981", "rgba(16,185,129,0.18)"),
            }
            for cat, kws in analysis["signals"].items():
                if not kws:
                    continue
                fg, bg = sig_colors.get(cat, ("#cbd5e1", "rgba(148,163,184,0.15)"))
                pills = "".join(
                    f"<span class='badge-pill' style='background:{bg};color:{fg};border:1px solid {fg};'>{html.escape(k)}</span>"
                    for k in kws
                )
                sig_html_parts.append(f"<div style='margin:4px 0;'><b>{html.escape(cat)}:</b> {pills}</div>")
            if sig_html_parts:
                st.markdown(f"<div class='studio-card'>{''.join(sig_html_parts)}</div>", unsafe_allow_html=True)
            else:
                st.caption("Không khớp từ khóa bề mặt cố định trong BBD-VN v2.1 — quyết định dựa hoàn toàn trên ngữ nghĩa sâu của TF-IDF / Bi-Encoder / NLI.")

            with st.expander("Xem bảng điểm chứng cứ từng câu nguyên văn"):
                s_rows = [
                    {
                        "Câu #": f"Câu {i+1}" + (" ★" if i == int(np.argmax(analysis["sent_probs"])) else ""),
                        "Điểm chứng cứ": round(float(sc), 4),
                        "Nội dung câu nguyên văn": s,
                    }
                    for i, (s, sc) in enumerate(zip(analysis["sents"], analysis["sent_probs"]))
                ]
                st.dataframe(pd.DataFrame(s_rows), width="stretch", hide_index=True)

            # --- Standout Feature D: Phòng thí nghiệm Phản chứng What-If ---
            if len(analysis["sents"]) >= 2:
                st.markdown("#### Phòng thí nghiệm Phản chứng (What-If Faithfulness Test)")
                top_s_idx = int(np.argmax(analysis["sent_probs"]))
                if st.button(
                    f"🧪 Thử nghiệm Phản chứng: Tạm xóa Câu {top_s_idx+1} (Chứng cứ chính) xem xác suất thay đổi thế nào",
                    width="stretch",
                ):
                    ablated_text = " ".join(s for j, s in enumerate(analysis["sents"]) if j != top_s_idx)
                    with st.spinner("Đang suy luận lại đoạn văn đã loại câu chứng cứ chính..."):
                        ab_res = run_single_analysis(p_key, ablated_text, analysis["title"])
                    p_after = float(ab_res["primary_res"]["prob_uncertain"])
                    delta = p_after - p_unc
                    st.session_state["whatif_result"] = {
                        "removed_idx": top_s_idx + 1,
                        "removed_sent": analysis["sents"][top_s_idx],
                        "p_before": p_unc,
                        "p_after": p_after,
                        "delta": delta,
                        "lbl_before": pred_lbl,
                        "lbl_after": ab_res["primary_res"]["pred_label"],
                    }

                wi = st.session_state.get("whatif_result")
                if wi:
                    flipped = wi["lbl_before"] != wi["lbl_after"]
                    verdict_wi = (
                        "**ĐẢO NHÃN DỰ ĐOÁN** — chứng tỏ câu chứng cứ vừa chọn thực sự mang thông tin quyết định!"
                        if flipped
                        else "Giữ nguyên nhãn cuối (các câu còn lại hoặc tiêu đề vẫn duy trì tín hiệu)."
                    )
                    wc1, wc2, wc3 = st.columns(3)
                    wc1.metric("P(unc) Ban đầu", f"{wi['p_before']:.3f}", wi["lbl_before"])
                    wc2.metric("P(unc) Sau khi xóa Câu " + str(wi["removed_idx"]), f"{wi['p_after']:.3f}", f"{wi['delta']:+.3f}")
                    wc3.metric("Kết luận Phản chứng", "Đảo nhãn!" if flipped else "Ổn định")
                    st.markdown(
                        f"<div class='whatif-box'><b>Câu đã loại bỏ (Câu {wi['removed_idx']}):</b> "
                        f"<i>\"{html.escape(wi['removed_sent'])}\"</i><br/>{verdict_wi}</div>",
                        unsafe_allow_html=True,
                    )

    # =====================================================================
    # TAB 2: QUÉT BÀI BÁO DÀI & TẢI TỆP HÀNG LOẠT (.TXT / .CSV)
    # =====================================================================
    with tab2:
        st.subheader("Quét Nguyên Bài Báo Nhiều Đoạn hoặc Hàng Loạt Văn Bản Mới")
        st.caption(
            "Cho phép dán một bài báo hoàn chỉnh (các đoạn cách nhau bằng dòng trống) hoặc tải tệp `.txt` / `.csv` "
            "(cột `text`, tùy chọn cột `title`) để xuất thống kê phân bố dạng bất định và tải về CSV."
        )

        scan_mode = st.radio("Nguồn đầu vào:", ["Dán nguyên bài báo dài (nhiều đoạn)", "Tải tệp .txt hoặc .csv"], horizontal=True)
        batch_model = st.selectbox(
            "Mô hình quét hàng loạt:",
            ["E3", "E0_tfidf", "E1", "E2", "E0"],
            format_func=lambda k: MODEL_NAMES_VI[k],
            key="batch_model_sel",
        )

        items_to_scan: List[Dict[str, str]] = []
        if scan_mode == "Dán nguyên bài báo dài (nhiều đoạn)":
            long_title = st.text_input("Tiêu đề bài báo chung (tùy chọn):", key="long_art_title", placeholder="Ví dụ: Ngân hàng Nhà nước điều hành tỷ giá linh hoạt trước biến động quốc tế")
            long_body = st.text_area(
                "Nội dung toàn bài báo (các đoạn ngăn cách bằng 1 dòng trống / Enter 2 lần):",
                height=210,
                key="long_art_body",
                placeholder="Đoạn 1: ...\n\nĐoạn 2: ...\n\nĐoạn 3: ...",
            )
            if long_body.strip():
                paras = [p.strip() for p in re.split(r"\n\s*\n", long_body.strip()) if p.strip()]
                for idx, p in enumerate(paras, start=1):
                    items_to_scan.append({"id": f"Đoạn {idx}", "title": long_title, "text": p})
                st.caption(f"Đã nhận diện **{len(items_to_scan)} đoạn văn** sẵn sàng quét.")
        else:
            up_file = st.file_uploader("Chọn tệp `.csv` (có cột `text`) hoặc `.txt` (các đoạn cách nhau bởi dòng trống):", type=["csv", "txt"])
            if up_file is not None:
                if up_file.name.lower().endswith(".csv"):
                    up_df = pd.read_csv(up_file, encoding="utf-8-sig")
                    if "text" not in up_df.columns:
                        st.error(f"Tệp CSV cần có cột `text`! Các cột hiện có: {list(up_df.columns)}")
                    else:
                        has_ttl = "title" in up_df.columns
                        for idx, row in up_df.head(200).iterrows():
                            items_to_scan.append(
                                {
                                    "id": str(row.get("item_id", row.get("document_id", f"Dòng {idx+1}"))),
                                    "title": str(row["title"]) if has_ttl and pd.notna(row["title"]) else "",
                                    "text": str(row["text"]),
                                }
                            )
                        st.success(f"Đã nạp **{len(items_to_scan)}** bản ghi từ `{up_file.name}` (tối đa 200 dòng/lượt).")
                else:
                    txt_raw = up_file.read().decode("utf-8", errors="replace")
                    paras = [p.strip() for p in re.split(r"\n\s*\n", txt_raw.strip()) if p.strip()]
                    for idx, p in enumerate(paras[:200], start=1):
                        items_to_scan.append({"id": f"Đoạn {idx}", "title": "", "text": p})
                    st.success(f"Đã tách **{len(items_to_scan)} đoạn** từ `{up_file.name}`.")

        scan_clicked = st.button("⚡ Quét toàn bộ danh sách", type="primary")
        if scan_clicked and len(items_to_scan) == 0:
            st.warning("Chưa có đoạn văn bản hoặc tệp hợp lệ nào để quét!")
        elif scan_clicked and len(items_to_scan) > 0:
            texts_b = [it["text"] for it in items_to_scan]
            titles_b = [it["title"] if it["title"].strip() else None for it in items_to_scan]
            with st.spinner(f"Đang suy luận {len(items_to_scan)} đoạn với mô hình {batch_model}..."):
                if batch_model == "E0":
                    mdl, _ = load_model("E0")
                    preds_b = mdl.predict_batch(texts_b, titles=titles_b)
                elif batch_model == "E0_tfidf":
                    mdl, _ = load_model("E0_tfidf")
                    preds_b = mdl.predict_batch(texts_b, titles=titles_b, extract_evidence=True)
                elif batch_model == "E1":
                    mdl, _ = load_model("E1")
                    preds_b = mdl.predict_batch(texts_b, titles=titles_b, extract_evidence=True)
                elif batch_model == "E2":
                    mdl, _ = load_model("E2")
                    preds_b = mdl.predict_batch(texts_b, titles=titles_b, extract_evidence=True, use_ensemble=True)
                else:
                    e0, _ = load_model("E0")
                    e0t, _ = load_model("E0_tfidf")
                    e1, _ = load_model("E1")
                    e2, _ = load_model("E2")
                    e3, _ = load_model("E3")
                    r0 = e0.predict_batch(texts_b, titles=titles_b)
                    r0t = e0t.predict_batch(texts_b, titles=titles_b, extract_evidence=True, return_sent_scores=True)
                    r1 = e1.predict_batch(texts_b, titles=titles_b, extract_evidence=True, return_sent_scores=True)
                    r2 = e2.predict_batch(texts_b, titles=titles_b, extract_evidence=True, use_ensemble=True, return_sent_scores=True)
                    preds_b = e3.predict_from_base_preds(r0, r0t, r1, r2, texts=texts_b, titles=titles_b, extract_evidence=True)

            out_rows = []
            for idx, (it, pr) in enumerate(zip(items_to_scan, preds_b), start=1):
                ut = pr.get("pred_uncertainty_type", "none")
                out_rows.append(
                    {
                        "order": idx,
                        "id": it["id"],
                        "title": it["title"],
                        "pred_label": pr["pred_label"],
                        "uncertainty_type": ut,
                        "prob_uncertain": round(float(pr["prob_uncertain"]), 4),
                        "pred_evidence": pr["pred_evidence"],
                        "text_preview": it["text"][:180],
                    }
                )
            st.session_state["long_scan_df"] = pd.DataFrame(out_rows)

        scan_df = st.session_state.get("long_scan_df")
        if scan_df is not None and not scan_df.empty:
            n_tot = len(scan_df)
            n_unc = int((scan_df["pred_label"] == "uncertain").sum())
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tổng đoạn quét", f"{n_tot}")
            k2.metric("Đoạn có bất định", f"{n_unc} ({n_unc/n_tot*100:.1f}%)")
            k3.metric("Tri thức (Epistemic / Both)", f"{int(scan_df['uncertainty_type'].isin(['epistemic','both']).sum())}")
            k4.metric("Ngẫu nhiên (Aleatoric)", f"{int((scan_df['uncertainty_type'] == 'aleatoric').sum())}")

            prog_chart = (
                alt.Chart(scan_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("order:O", title="Thứ tự đoạn / văn bản"),
                    y=alt.Y("prob_uncertain:Q", title="Xác suất bất định P(unc)", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color(
                        "uncertainty_type:N",
                        title="Dạng dự đoán",
                        scale=alt.Scale(
                            domain=["epistemic", "aleatoric", "both", "none"],
                            range=["#ef4444", "#f59e0b", "#a855f7", "#10b981"],
                        ),
                    ),
                    tooltip=["id:N", "pred_label:N", "uncertainty_type:N", "prob_uncertain:Q", "pred_evidence:N"],
                )
                .properties(height=270)
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(prog_chart, width="stretch")
            st.dataframe(scan_df[["id", "pred_label", "uncertainty_type", "prob_uncertain", "pred_evidence", "text_preview"]], width="stretch", hide_index=True)

            csv_bytes = scan_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "⬇️ Tải xuống kết quả quét (.csv)",
                data=csv_bytes,
                file_name="ket_qua_quet_bat_dinh.csv",
                mime="text/csv",
            )

    # =====================================================================
    # TAB 3: CHUỖI TẦN SUẤT 36 THÁNG (39.028 BÀI BÁO)
    # =====================================================================
    with tab3:
        st.subheader("Tần suất Tin Bất định theo Nguồn báo & Thời gian (01/2023 – 12/2025)")
        idx_path = EVAL_DIR / "extension_monthly_index.csv"
        src_path = EVAL_DIR / "extension_monthly_by_source.csv"
        ev_path = DATA_DIR / "external_events_reference.csv"
        samp_path = EVAL_DIR / "extension_top_evidence_samples.csv"

        if not idx_path.exists():
            st.info("Chưa có dữ liệu chuỗi thời gian. Hãy chạy `python src/06_extension_analysis.py`.")
        else:
            idx_df = pd.read_csv(idx_path, encoding="utf-8-sig")
            src_df = pd.read_csv(src_path, encoding="utf-8-sig")
            events_df = pd.read_csv(ev_path, encoding="utf-8-sig")
            samples_df = pd.read_csv(samp_path, encoding="utf-8-sig") if samp_path.exists() else pd.DataFrame()
            best_used = str(idx_df["best_model_used"].iloc[0])

            c_f1, c_f2 = st.columns([1.3, 2.2])
            with c_f1:
                source_view = st.selectbox(
                    "Góc nhìn dữ liệu:",
                    ["Tổng hợp 3 báo (Chuẩn hóa BBD vs Thô)", "VietnamNet", "Thanh Niên", "Tuổi Trẻ"],
                )
            with c_f2:
                series_opts = {
                    f"bbd_index_best": f"Chuẩn hóa BBD ({best_used} — Chính)",
                    "bbd_index_E3": "Chuẩn hóa BBD (E3 Hybrid)",
                    "bbd_index_E2": "Chuẩn hóa BBD (E2 mDeBERTa-v3)",
                    "bbd_index_E1": "Chuẩn hóa BBD (E1 Bi-Encoder)",
                    "bbd_index_E0": "Chuẩn hóa BBD (E0 Luật v2.1)",
                    "raw_count_scaled_100": "Tần suất THÔ quy đổi (mean=100)",
                }
                avail_cols = [c for c in series_opts if c in idx_df.columns]
                picked_series = st.multiselect(
                    "Chọn chuỗi đối chiếu trên biểu đồ:",
                    avail_cols,
                    default=[c for c in ["bbd_index_best", "bbd_index_E3", "bbd_index_E2", "raw_count_scaled_100"] if c in avail_cols],
                    format_func=lambda c: series_opts[c],
                )

            if source_view == "Tổng hợp 3 báo (Chuẩn hóa BBD vs Thô)":
                if picked_series:
                    melt_df = idx_df[["year_month"] + picked_series].melt(
                        id_vars="year_month", var_name="series_key", value_name="index_val"
                    )
                    melt_df["Chuỗi"] = melt_df["series_key"].map(series_opts)
                    bands_df = pd.DataFrame(EVENT_BANDS)
                    bands_chart = (
                        alt.Chart(bands_df)
                        .mark_rect(opacity=0.14)
                        .encode(
                            x="start:O",
                            x2="end:O",
                            color=alt.Color("event:N", legend=None),
                            tooltip=["event:N", "start:N", "end:N"],
                        )
                    )
                    lines_chart = (
                        alt.Chart(melt_df)
                        .mark_line(point=True, strokeWidth=2.4)
                        .encode(
                            x=alt.X("year_month:O", title="Tháng công bố (2023–2025)", axis=alt.Axis(labelAngle=-40)),
                            y=alt.Y("index_val:Q", title="Chỉ số tương đối (Trung bình = 100)"),
                            color=alt.Color("Chuỗi:N"),
                            strokeDash=alt.condition(
                                alt.datum.series_key == "raw_count_scaled_100",
                                alt.value([5, 4]),
                                alt.value([1, 0]),
                            ),
                            tooltip=["year_month:N", "Chuỗi:N", alt.Tooltip("index_val:Q", format=".2f")],
                        )
                    )
                    ref_100 = alt.Chart(pd.DataFrame({"y": [100.0]})).mark_rule(strokeDash=[3, 3], color="#94a3b8", opacity=0.7).encode(y="y:Q")
                    st.altair_chart((bands_chart + ref_100 + lines_chart).properties(height=390).configure_view(strokeOpacity=0), width="stretch")
                st.dataframe(
                    idx_df[["year_month", "total_clean_econ", "raw_uncertain_best", "naive_share_best", "bbd_index_best", "raw_count_scaled_100"]].round(3),
                    width="stretch",
                    height=240,
                )
            else:
                sub = src_df[src_df["source_name"] == source_view].sort_values("year_month").copy()
                sub["share_pct"] = sub["share_best"] * 100.0
                st.markdown(f"**{source_view}: Số bài bất định thô vs Tỷ lệ chuẩn hóa (%)**")
                c_a, c_b = st.columns(2)
                with c_a:
                    st.bar_chart(sub.set_index("year_month")[["raw_uncertain_best"]], height=280)
                with c_b:
                    st.line_chart(sub.set_index("year_month")[["share_pct"]], height=280)
                st.dataframe(sub[["year_month", "total_clean_econ", "raw_uncertain_best", "share_pct"]].round(3), width="stretch")

            if st.checkbox("Hiển thị bảng đối chiếu 4 biến cố kinh tế tham chiếu", value=True):
                ev_comp_path = EVAL_DIR / "extension_event_comparison.csv"
                if ev_comp_path.exists():
                    st.dataframe(pd.read_csv(ev_comp_path, encoding="utf-8-sig"), width="stretch", hide_index=True)
                else:
                    st.dataframe(events_df, width="stretch", hide_index=True)

            if not samples_df.empty:
                st.markdown("**Ví dụ câu chứng cứ thực tế tại các tháng đỉnh / biến cố:**")
                sel_m = st.multiselect(
                    "Lọc theo tháng:",
                    sorted(samples_df["year_month"].unique().tolist()),
                    default=sorted(samples_df["year_month"].unique().tolist())[:4],
                )
                disp = samples_df[samples_df["year_month"].isin(sel_m)] if sel_m else samples_df
                st.dataframe(disp[["year_month", "source_name", "published_at", "title", "pred_evidence", "prob_uncertain"]], width="stretch", hide_index=True)

    # =====================================================================
    # TAB 4: ĐỐI CHỨNG E0–E3 & HẬU KIỂM KÝ XÁC NHẬN GOLD
    # =====================================================================
    with tab4:
        st.subheader("Bảng đối chứng E0–E3 & Giao diện Hậu kiểm Ký xác nhận Gold")
        exp_path = EVAL_DIR / "experiments.csv"
        oof_path = EVAL_DIR / "oof_cv_summary.csv"
        ms_path = EVAL_DIR / "metrics_summary.csv"
        rob_path = EVAL_DIR / "extension_robustness.csv"

        if ms_path.exists():
            ms_df = pd.read_csv(ms_path, encoding="utf-8-sig")
            te_only = ms_df[ms_df["split"] == "test"].copy()
            te_melt = te_only[["run_id", "macro_f1", "f1_uncertain", "evidence_token_f1_uncertain"]].melt(
                id_vars="run_id", var_name="metric", value_name="score"
            )
            perf_chart = (
                alt.Chart(te_melt)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("run_id:N", sort=["E0", "E0_tfidf", "E1", "E2", "E3"], title="Mô hình"),
                    xOffset="metric:N",
                    y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1.05]), title="Điểm trên Test (n=60)"),
                    color=alt.Color("metric:N", title="Chỉ số"),
                    tooltip=["run_id:N", "metric:N", alt.Tooltip("score:Q", format=".4f")],
                )
                .properties(height=270)
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(perf_chart, width="stretch")

        if exp_path.exists():
            st.markdown("**1. Nhật ký thí nghiệm (`experiments.csv`):**")
            st.dataframe(pd.read_csv(exp_path, encoding="utf-8-sig"), width="stretch", hide_index=True)
        if oof_path.exists():
            st.markdown("**2. Kết quả 5-Fold OOF Cross-Validation toàn tập (`oof_cv_summary.csv`, n = 184):**")
            st.dataframe(pd.read_csv(oof_path, encoding="utf-8-sig"), width="stretch", hide_index=True)
        if ms_path.exists():
            st.markdown("**3. Bảng chỉ số tổng hợp Dev OOF & Test khóa (`metrics_summary.csv`):**")
            st.dataframe(pd.read_csv(ms_path, encoding="utf-8-sig"), width="stretch", hide_index=True)
        if rob_path.exists():
            st.markdown("**4. Độ bền xuyên nguồn (LOSO & External UVN-1):**")
            st.dataframe(pd.read_csv(rob_path, encoding="utf-8-sig"), width="stretch", hide_index=True)

        gold_path = ANN_DIR / "gold_labels_split.csv"
        if gold_path.exists():
            st.divider()
            st.markdown("**5. Hậu kiểm & Ký xác nhận thủ công tập Gold (Human Sign-Off – Guideline v2.0)**")
            st.caption("Bảng Batch Audit bên dưới lưu trực tiếp vào `data/annotations/human_audit_signoff.csv`.")
            gdf = pd.read_csv(gold_path, encoding="utf-8-sig")
            sign_path = ANN_DIR / "human_audit_signoff.csv"
            existing_sign = (
                pd.read_csv(sign_path, encoding="utf-8-sig").set_index("item_id").to_dict("index")
                if sign_path.exists()
                else {}
            )

            f_split = st.selectbox("Chọn split cần rà soát:", ["test", "dev", "train", "external_uvn1"])
            sub_g = gdf[gdf["split"] == f_split].reset_index(drop=True)
            ut_col = sub_g["uncertainty_type"] if "uncertainty_type" in sub_g.columns else "none"

            batch_df = pd.DataFrame(
                {
                    "item_id": sub_g["item_id"],
                    "source_name": sub_g["source_name"],
                    "published_at": sub_g["published_at"],
                    "task_label": sub_g["task_label"],
                    "uncertainty_type": ut_col,
                    "boundary_type": sub_g["boundary_type"],
                    "text_preview": sub_g["text"].astype(str).str.slice(0, 140),
                    "student_decision": [
                        existing_sign.get(iid, {}).get("student_decision", "Đồng ý giữ nguyên nhãn Gold")
                        for iid in sub_g["item_id"]
                    ],
                    "note": [
                        str(existing_sign.get(iid, {}).get("note", "") or "")
                        for iid in sub_g["item_id"]
                    ],
                }
            )
            decision_opts = [
                "Đồng ý giữ nguyên nhãn Gold",
                "Đề nghị sửa thành uncertain",
                "Đề nghị sửa thành not_uncertain",
                "Đánh dấu unclear",
            ]
            edited = st.data_editor(
                batch_df,
                width="stretch",
                hide_index=True,
                disabled=["item_id", "source_name", "published_at", "task_label", "uncertainty_type", "boundary_type", "text_preview"],
                column_config={
                    "student_decision": st.column_config.SelectboxColumn(
                        "Xác nhận của sinh viên", options=decision_opts, required=True
                    ),
                    "note": st.column_config.TextColumn("Ghi chú hậu kiểm"),
                },
                key=f"batch_editor_{f_split}",
            )
            if st.button(f"Lưu xác nhận hàng loạt cho tập `{f_split}` ({len(edited)} mẫu)", type="primary", key=f"btn_batch_{f_split}"):
                new_recs = pd.DataFrame(
                    {
                        "item_id": edited["item_id"],
                        "split": f_split,
                        "original_gold_label": edited["task_label"],
                        "uncertainty_type": edited["uncertainty_type"],
                        "student_decision": edited["student_decision"],
                        "note": edited["note"],
                    }
                )
                if sign_path.exists():
                    old = pd.read_csv(sign_path, encoding="utf-8-sig")
                    new_recs = pd.concat([old[~old["item_id"].isin(new_recs["item_id"])], new_recs], ignore_index=True)
                new_recs.to_csv(sign_path, index=False, encoding="utf-8-sig")
                st.success(f"Đã lưu hàng loạt {len(edited)} bản ghi xác nhận cho `{f_split}` vào `data/annotations/human_audit_signoff.csv`!")

            st.markdown("**Kiểm tra chi tiết từng mẫu (Single-Item Inspector):**")
            item_pick = st.selectbox("Chọn mã mẫu:", sub_g["item_id"].tolist())
            row = sub_g[sub_g["item_id"] == item_pick].iloc[0]
            ut_val = row["uncertainty_type"] if "uncertainty_type" in row else "none"
            st.write(
                f"**Nguồn:** {row['source_name']} | **Ngày:** {row['published_at']} | "
                f"**Dạng bất định:** `{ut_val}` | **Nhóm biên giới:** `{row['boundary_type']}`"
            )
            st.write(f"**Tiêu đề:** {row['title']}")
            st.text_area("Đoạn văn bản:", value=row["text"], height=120, disabled=True, key=f"txt_{item_pick}")
            st.info(f"**Nhãn Gold hiện tại:** `{row['task_label']}` (`{ut_val}`) | **Câu chứng cứ:** {row['evidence_sentence']}")
            human_lbl = st.radio(
                "Xác nhận của sinh viên:",
                decision_opts,
                horizontal=True,
                key=f"rad_{item_pick}",
            )
            human_note = st.text_input("Ghi chú hậu kiểm (tùy chọn):", key=f"note_{item_pick}")
            if st.button("Lưu bản ghi xác nhận đơn lẻ", key=f"btn_{item_pick}"):
                rec = pd.DataFrame(
                    [
                        {
                            "item_id": item_pick,
                            "split": f_split,
                            "original_gold_label": row["task_label"],
                            "uncertainty_type": ut_val,
                            "student_decision": human_lbl,
                            "note": human_note,
                        }
                    ]
                )
                if sign_path.exists():
                    old = pd.read_csv(sign_path, encoding="utf-8-sig")
                    rec = pd.concat([old[old["item_id"] != item_pick], rec], ignore_index=True)
                rec.to_csv(sign_path, index=False, encoding="utf-8-sig")
                st.success(f"Đã lưu xác nhận cho `{item_pick}` vào `data/annotations/human_audit_signoff.csv`!")


if __name__ == "__main__":
    main()

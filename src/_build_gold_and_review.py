"""
Tổng hợp kết quả gắn nhãn Vòng 1, Vòng 2 độc lập (77 mẫu = 27.5% tổng mẫu)
và phân xử bất đồng thành data/annotations/review_log.csv & gold_labels.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from text_utils import split_sentences_vi

ROOT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT_DIR / "data" / "processed"
ANN_DIR = ROOT_DIR / "data" / "annotations"

# Nhãn Pass 2 độc lập trên 75 mẫu mù (_pass2_blind.txt) + 2 ca unclear từ Pass 1
PASS2 = {
    "UVN_EXT_004": "not_uncertain",
    "UVN_EXT_009": "not_uncertain",
    "UVN_EXT_011": "not_uncertain",
    "UVN_EXT_012": "not_uncertain",
    "UVN_EXT_015": "uncertain",
    "UVN_EXT_018": "not_uncertain",
    "UVN_EXT_022": "not_uncertain",
    "UVN_EXT_024": "uncertain",
    "UVN_EXT_026": "not_uncertain",
    "UVN_EXT_031": "not_uncertain",
    "VON_ITEM_005": "not_uncertain",
    "VON_ITEM_006": "not_uncertain",
    "VON_ITEM_007": "uncertain",
    "VON_ITEM_013": "not_uncertain",
    "VON_ITEM_016": "uncertain",
    "VON_ITEM_019": "not_uncertain",
    "VON_ITEM_025": "uncertain",
    "VON_ITEM_032": "not_uncertain",
    "VON_ITEM_035": "not_uncertain",
    "VON_ITEM_038": "uncertain",
    "VON_ITEM_043": "uncertain",
    "VON_ITEM_045": "uncertain",
    "VON_ITEM_047": "not_uncertain",
    "VON_ITEM_053": "uncertain",
    "VON_ITEM_057": "not_uncertain",
    "VON_ITEM_066": "not_uncertain",
    "VON_ITEM_072": "uncertain",
    "VON_ITEM_077": "uncertain",
    "VON_ITEM_081": "uncertain",
    "VON_ITEM_085": "not_uncertain",
    "VON_ITEM_087": "not_uncertain",
    "VON_ITEM_089": "uncertain",
    "VON_ITEM_100": "uncertain",
    "VON_ITEM_103": "not_uncertain",
    "VON_ITEM_105": "uncertain",
    "VON_ITEM_107": "not_uncertain",
    "VON_ITEM_108": "uncertain",
    "VON_ITEM_111": "not_uncertain",
    "VON_ITEM_115": "uncertain",
    "VON_ITEM_116": "not_uncertain",
    "VON_ITEM_119": "not_uncertain",
    "VON_ITEM_123": "uncertain",
    "VON_ITEM_126": "uncertain",
    "VON_ITEM_130": "not_uncertain",
    "VON_ITEM_132": "not_uncertain",
    "VON_ITEM_137": "not_uncertain",
    "VON_ITEM_139": "uncertain",
    "VON_ITEM_144": "not_uncertain",
    "VON_ITEM_148": "not_uncertain",
    "VON_ITEM_152": "uncertain",
    "VON_ITEM_154": "not_uncertain",
    "VON_ITEM_164": "not_uncertain",
    "VON_ITEM_169": "not_uncertain",
    "VON_ITEM_172": "not_uncertain",
    "VON_ITEM_176": "not_uncertain",
    "VON_ITEM_186": "uncertain",
    "VON_ITEM_188": "not_uncertain",
    "VON_ITEM_189": "uncertain",
    "VON_ITEM_194": "not_uncertain",
    "VON_ITEM_200": "uncertain",
    "VON_ITEM_201": "not_uncertain",
    "VON_ITEM_204": "not_uncertain",
    "VON_ITEM_216": "not_uncertain",
    "VON_ITEM_227": "not_uncertain",
    "VON_ITEM_228": "uncertain",
    "VON_ITEM_229": "uncertain",
    "VON_ITEM_230": "not_uncertain",
    "VON_ITEM_231": "uncertain",
    "VON_ITEM_233": "uncertain",
    "VON_ITEM_234": "not_uncertain",
    "VON_ITEM_235": "not_uncertain",
    "VON_ITEM_236": "not_uncertain",
    "VON_ITEM_238": "not_uncertain",
    "VON_ITEM_240": "not_uncertain",
    "VON_ITEM_243": "not_uncertain",
    # Hai ca unclear từ Pass 1 được chuyển bắt buộc qua hậu kiểm & phân xử theo guideline:
    "VON_ITEM_120": "unclear",
    "VON_ITEM_241": "not_uncertain",
}

ADJUDICATION_NOTES = {
    "VON_ITEM_137": (
        "not_uncertain",
        "negative_certain_hard_negative",
        "none",
        "none",
        0,
        "Bất đồng P1(uncertain) vs P2(not_uncertain): đoạn văn chỉ thuật du khách đến Phú Quốc sụt giảm và phát biểu hội thảo chung, không nêu chính sách/rủi ro kinh tế bỏ ngỏ -> chốt not_uncertain.",
    ),
    "VON_ITEM_169": (
        "uncertain",
        "aleatoric_volatility",
        "monetary_fx",
        "aleatoric",
        0,
        "Bất đồng biên giới v2.0 P1(uncertain) vs P2(not_uncertain): P2 chú ý SJC trong nước đi ngang, nhưng câu mở đầu nêu giá vàng thế giới xoay chiều giảm sâu bất ngờ -> theo chuẩn bất định ngẫu nhiên v2.0, phân xử giữ uncertain (aleatoric).",
    ),
    "VON_ITEM_120": (
        "unclear",
        "ambiguous_unclear",
        "none",
        "none",
        0,
        "Hai vòng đều xác nhận nội dung nói về vướng mắc phạt nguội treo đăng kiểm xe cá nhân, biên giới ngoài phạm vi bất định kinh tế vĩ mô -> giữ unclear để loại khỏi đánh giá nhị phân.",
    ),
    "VON_ITEM_241": (
        "not_uncertain",
        "neutral_factual",
        "none",
        "none",
        1,
        "P1 đánh dấu unclear vì người nộp thuế hoang mang; P2 đối chiếu lại thấy tình huống hỏi đáp thực thi quy định quyết toán thuế TNCN sẵn có, không phải bất định chính sách -> phân xử thành not_uncertain.",
    ),
}

# Phán quyết hậu kiểm trực tiếp của sinh viên từ data/annotations/human_audit_signoff.csv (22/11/2026)
HUMAN_REVISION_META = {
    "VON_ITEM_020": (
        "uncertain",
        "implicit_uncertain",
        "monetary_fx",
        "epistemic",
        0,
        "Sinh viên hậu kiểm sửa từ not_uncertain -> uncertain (dự báo giá vàng/dầu/USD có khả năng biến động — epistemic).",
    ),
    "VON_ITEM_052": (
        "uncertain",
        "implicit_uncertain",
        "policy_regulatory",
        "epistemic",
        1,
        "Sinh viên hậu kiểm sửa từ not_uncertain -> uncertain (gợi mở đột phá thể chế & đổi mới mô hình tăng trưởng tài chính — epistemic).",
    ),
    "VON_ITEM_060": (
        "uncertain",
        "aleatoric_volatility",
        "trade_tariff_geopolitical",
        "aleatoric",
        0,
        "Sinh viên hậu kiểm sửa từ not_uncertain -> uncertain (thép HRC nhập khẩu gia tăng đột biến bất thường 26 lần sau tin áp thuế — aleatoric).",
    ),
    "VON_ITEM_065": (
        "uncertain",
        "implicit_uncertain",
        "monetary_fx",
        "epistemic",
        0,
        "Sinh viên hậu kiểm sửa từ not_uncertain -> uncertain (dự báo triển vọng giá vàng & kỳ vọng Fed hạ lãi suất — epistemic).",
    ),
    "VON_ITEM_128": (
        "not_uncertain",
        "neutral_factual",
        "none",
        "none",
        0,
        "Sinh viên hậu kiểm sửa từ uncertain -> not_uncertain (trọng tâm là công văn tuyên truyền tuân thủ quy định kinh doanh vàng).",
    ),
    "VON_ITEM_202": (
        "unclear",
        "ambiguous_unclear",
        "none",
        "none",
        0,
        "Sinh viên hậu kiểm đánh dấu unclear (bài quảng bá phong cách sống BĐS không rõ tính kinh tế vĩ mô) -> loại khỏi tập nhị phân theo Guideline v2.0.",
    ),
    "VON_ITEM_213": (
        "uncertain",
        "implicit_uncertain",
        "macro_growth_inflation",
        "epistemic",
        2,
        "Sinh viên hậu kiểm sửa từ not_uncertain -> uncertain (báo cáo ADB dự báo tăng trưởng chậm lại & triển vọng phục hồi — epistemic).",
    ),
    "VON_ITEM_243": (
        "uncertain",
        "aleatoric_volatility",
        "monetary_fx",
        "aleatoric",
        0,
        "Sinh viên hậu kiểm sửa từ not_uncertain -> uncertain (nhóm Big4 đồng loạt giảm mạnh lãi suất huy động — aleatoric).",
    ),
    "UVN_EXT_018": (
        "uncertain",
        "implicit_uncertain",
        "trade_tariff_geopolitical",
        "epistemic",
        0,
        "Sinh viên hậu kiểm sửa từ not_uncertain -> uncertain (kỳ vọng thỏa thuận thương mại EU–Ấn Độ ứng phó thuế quan Mỹ — epistemic).",
    ),
    "UVN_EXT_028": (
        "not_uncertain",
        "negative_certain_hard_negative",
        "none",
        "none",
        0,
        "Sinh viên hậu kiểm sửa từ uncertain -> not_uncertain (báo cáo tài chính hợp nhất năm 2022 đã chốt sổ).",
    ),
}


def main() -> None:
    pool = pd.read_csv(PROC_DIR / "annotation_pool.csv", encoding="utf-8-sig")
    pass1 = json.loads((PROC_DIR / "_pass1_annotations.json").read_text(encoding="utf-8"))
    sign_path = ANN_DIR / "human_audit_signoff.csv"
    sign_map = {}
    if sign_path.exists():
        sdf = pd.read_csv(sign_path, encoding="utf-8-sig")
        sign_map = sdf.set_index("item_id").to_dict("index")

    gold_rows = []
    review_rows = []
    human_eval_pairs = []

    for _, r in pool.iterrows():
        iid = r["item_id"]
        p1_lbl, ev_idx, b_type, dim, u_type = pass1[iid]
        sents = split_sentences_vi(r["text"])
        ev_idx = min(max(0, int(ev_idx)), len(sents) - 1)

        ai_v2_lbl = p1_lbl
        note_text = ""
        if iid in ADJUDICATION_NOTES:
            adj_lbl, adj_bt, adj_dim, adj_utype, adj_ev_idx, note_text = ADJUDICATION_NOTES[iid]
            ai_v2_lbl = adj_lbl
            b_type = adj_bt
            dim = adj_dim
            u_type = adj_utype
            ev_idx = min(max(0, adj_ev_idx), len(sents) - 1)

        final_lbl = ai_v2_lbl
        human_signed = iid in sign_map
        if human_signed:
            dec = str(sign_map[iid]["student_decision"])
            if dec != "Đồng ý giữ nguyên nhãn Gold":
                assert iid in HUMAN_REVISION_META, f"Chưa khai báo HUMAN_REVISION_META cho {iid} ({dec})"
                h_lbl, h_bt, h_dim, h_utype, h_ev_idx, h_note = HUMAN_REVISION_META[iid]
                final_lbl = h_lbl
                b_type = h_bt
                dim = h_dim
                u_type = h_utype
                ev_idx = min(max(0, h_ev_idx), len(sents) - 1)
                note_text = (note_text + " | " + h_note).strip(" |")
            human_eval_pairs.append((ai_v2_lbl, final_lbl))

        ev_sent = sents[ev_idx]
        assert ev_sent in str(r["text"]), f"Evidence not exact substring in {iid}"

        reviewed = iid in PASS2
        if human_signed:
            rev_id = "STUDENT_HUMAN_AUDIT"
            rev_status = "human_verified_signed_off"
        elif final_lbl == "unclear":
            rev_id = "AI_PASS2_REVIEW"
            rev_status = "adjudicated_excluded_unclear"
        else:
            rev_id = "AI_PASS2_REVIEW" if reviewed else "NONE"
            rev_status = "adjudicated_pending_human_signoff" if reviewed else "pass1_pending_human_signoff"

        gold_rows.append(
            {
                "item_id": iid,
                "dataset_source": r["dataset_source"],
                "document_id": r["document_id"],
                "source_name": r["source_name"],
                "source_url": r["source_url"],
                "source_version": r["source_version"],
                "retrieved_at": r["retrieved_at"],
                "published_at": r["published_at"],
                "original_category": r["original_category"],
                "sampling_stratum": r["sampling_stratum"],
                "title": r["title"],
                "text": r["text"],
                "task_label": final_lbl,
                "evidence_sentence": ev_sent,
                "uncertainty_type": u_type,
                "boundary_type": b_type,
                "uncertainty_dimension": dim,
                "annotator_id": "AI_PASS1_DRAFT",
                "reviewer_id": rev_id,
                "review_status": rev_status,
                "guideline_version": "v2.0_human_signed" if sign_map else "v2.0",
            }
        )

        if reviewed:
            p2_lbl = PASS2[iid]
            agree = p1_lbl == p2_lbl
            dis_note = note_text if note_text else ("" if agree else "Disagreement resolved")
            review_rows.append(
                {
                    "item_id": iid,
                    "dataset_source": r["dataset_source"],
                    "source_name": r["source_name"],
                    "pass1_label": p1_lbl,
                    "pass2_label": p2_lbl,
                    "agreement": agree,
                    "adjudicated_label": ai_v2_lbl,
                    "human_signed_label": final_lbl,
                    "annotator_pass1": "AI_PASS1_DRAFT",
                    "reviewer_pass2": "AI_PASS2_REVIEW",
                    "review_status": rev_status,
                    "guideline_version": "v2.0_human_signed" if sign_map else "v2.0",
                    "disagreement_note": dis_note,
                }
            )

    gold_df = pd.DataFrame(gold_rows)
    rev_df = pd.DataFrame(review_rows)
    gold_df.to_csv(ANN_DIR / "gold_labels.csv", index=False, encoding="utf-8-sig")
    rev_df.to_csv(ANN_DIR / "review_log.csv", index=False, encoding="utf-8-sig")

    kappa = cohen_kappa_score(rev_df["pass1_label"], rev_df["pass2_label"])
    obs = float(rev_df["agreement"].mean())

    # Tính thống kê hậu kiểm trực tiếp của con người (Human Sign-Off vs AI Pass 1/Adjudicated v2.0)
    if human_eval_pairs:
        h_ai = [x[0] for x in human_eval_pairs]
        h_stu = [x[1] for x in human_eval_pairs]
        h_agree_n = sum(1 for a, b in human_eval_pairs if a == b)
        h_kappa = float(cohen_kappa_score(h_ai, h_stu))
        human_summary = {
            "total_audited": len(human_eval_pairs),
            "agreed_count": h_agree_n,
            "revised_count": len(human_eval_pairs) - h_agree_n,
            "observed_agreement": round(h_agree_n / len(human_eval_pairs), 4),
            "cohen_kappa_human_vs_ai": round(h_kappa, 4),
            "revised_item_ids": sorted(list(HUMAN_REVISION_META.keys())),
        }
        (ANN_DIR / "human_audit_summary.json").write_text(
            json.dumps(human_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Human Audit Sign-Off: {human_summary['total_audited']} items | Agreed={h_agree_n} ({human_summary['observed_agreement']*100:.2f}%) | "
            f"Revised={human_summary['revised_count']} | Kappa(Human vs AI)={h_kappa:.4f}"
        )

    print(f"Saved gold_labels.csv ({len(gold_df)} rows) & review_log.csv ({len(rev_df)} rows)")
    print(f"Pass 2 Cross-Check: {len(rev_df)}/{len(gold_df)} ({len(rev_df)/len(gold_df)*100:.1f}%) | Agreement={obs*100:.2f}% | Kappa={kappa:.4f}")
    print("Final label counts (all):", gold_df["task_label"].value_counts().to_dict())
    print("Final label counts (VON):", gold_df[gold_df["dataset_source"]=="VietOnlineNews"]["task_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()

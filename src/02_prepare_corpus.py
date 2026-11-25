"""
Bước 2: Làm sạch văn bản có kiểm soát, kiểm toán bảo toàn số liệu kinh tế,
loại trùng lặp và trích xuất đoạn văn sạch đại diện cho từng bài báo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from text_utils import (
    extract_numeric_tokens,
    normalize_text,
    split_sentences_vi,
    text_md5,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"

_CREDIT_LINE_RE = re.compile(
    r"^(?:ảnh|nguồn|video|đồ họa|infographic|minh họa|tin,\s*ảnh)\s*[:：]|"
    r"^(?:xem thêm|đọc thêm|mời đọc|tin liên quan)\s*[:：]",
    re.IGNORECASE,
)
_CAMEL_STICK_DETECT_RE = re.compile(
    r"[a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]"
    r"[A-ZĐÂĂÊÔƠƯ][a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+"
)
_VI_DIACRITIC_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]",
    re.IGNORECASE,
)


def merge_broken_lines(content: str) -> str:
    """
    Ghép các dòng bị ngắt bất thường do scrape HTML (vd chữ in đậm xuống dòng riêng ở Thanh Niên),
    đồng thời loại các dòng chú thích ảnh / tin liên quan.
    """
    raw_lines = [ln.strip() for ln in normalize_text(content).split("\n")]
    paragraphs: List[str] = []
    buf: List[str] = []

    for ln in raw_lines:
        if not ln:
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        if _CREDIT_LINE_RE.match(ln) or len(ln) <= 3:
            continue
        # Nếu dòng ngắn hoặc dòng trước không kết thúc bằng dấu kết câu, nối vào buffer hiện tại
        if buf:
            prev = buf[-1]
            if not prev.endswith((".", "!", "?", ":", "”", '"')) or len(ln) < 45:
                buf.append(ln)
            else:
                paragraphs.append(" ".join(buf))
                buf = [ln]
        else:
            buf = [ln]
    if buf:
        paragraphs.append(" ".join(buf))
    return "\n\n".join(paragraphs)


def build_candidate_paragraph(title: str, description: str, content: str) -> str:
    """
    Trích một đoạn văn mạch lạc (220-950 ký tự, >= 2 câu) làm đơn vị phân loại cấp đoạn.
    Ưu tiên đoạn mang thông tin kinh tế thực chất trong phần đầu bài hoặc sapo + đoạn mở đầu.
    """
    paras = [p.strip() for p in content.split("\n\n") if len(p.strip()) >= 60]
    # Loại đoạn đầu chỉ là tên phóng viên / địa danh ngắn
    clean_paras = []
    for p in paras:
        sents = split_sentences_vi(p)
        if len(p) >= 200 and len(sents) >= 2:
            clean_paras.append(p)

    # Tìm đoạn có chứa cả chữ số hoặc từ kinh tế/chính sách trong 4 đoạn đầu (tránh đoạn PR cuối bài)
    econ_hint = re.compile(
        r"(?:kinh tế|lãi suất|tỷ giá|thuế|chính sách|ngân hàng|thị trường|doanh nghiệp|"
        r"xuất khẩu|nhập khẩu|lạm phát|đầu tư|giá vàng|chứng khoán|bất động sản|dự án|tín dụng)",
        re.IGNORECASE,
    )
    for p in clean_paras[:5]:
        if 220 <= len(p) <= 950 and econ_hint.search(p):
            return p

    for p in clean_paras[:5]:
        if 220 <= len(p) <= 950:
            return p

    # Nếu đoạn riêng lẻ quá ngắn hoặc quá dài, ghép description + các câu đầu của content
    combined_sents: List[str] = []
    if description and len(description) >= 40:
        combined_sents.extend(split_sentences_vi(description))
    for p in paras[:4]:
        for s in split_sentences_vi(p):
            if s not in combined_sents:
                combined_sents.append(s)
            joined = " ".join(combined_sents)
            if len(joined) >= 260 and len(combined_sents) >= 2:
                if len(joined) <= 950:
                    return joined
                break
        if len(" ".join(combined_sents)) >= 260:
            break

    # Nếu một đoạn đơn quá dài (>950 ký tự), cắt theo ranh giới câu hoàn chỉnh trong 950 ký tự
    if clean_paras:
        sents = split_sentences_vi(clean_paras[0])
        acc: List[str] = []
        for s in sents:
            cand = " ".join(acc + [s])
            if len(cand) <= 950 or len(acc) < 2:
                acc.append(s)
            else:
                break
        out = " ".join(acc)
        if len(out) >= 180:
            return out

    joined = " ".join(combined_sents[:5]).strip()
    return joined[:950].strip()


def clean_dataset(df: pd.DataFrame, dataset_tag: str) -> Tuple[pd.DataFrame, Dict]:
    audit: Dict = {
        "dataset": dataset_tag,
        "raw_count": int(len(df)),
        "removed_missing_or_short": [],
        "removed_non_vietnamese": [],
        "removed_duplicates": [],
        "numeric_preservation_checked": 0,
        "numeric_corruption_count": 0,
        "sticky_camelcase_articles": 0,
    }

    df = df.copy()
    # Chuẩn hóa trường văn bản
    for col in ["title", "description", "content"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Kiểm tra bảo toàn số liệu trước và sau normalize_text
    sample_nums_before = df["content"].head(500).apply(extract_numeric_tokens)
    df["title_clean"] = df["title"].apply(normalize_text)
    df["description_clean"] = df["description"].apply(normalize_text)
    df["content_clean"] = df["content"].apply(merge_broken_lines)
    sample_nums_after = df["content_clean"].head(500).apply(extract_numeric_tokens)

    # Kiểm tra các mẫu số thập phân dạng "3,2%" hoặc "1.500"
    dec_re = re.compile(r"-?\d+[.,]\d+%?")
    for before_txt, after_txt in zip(df["content"].head(2000), df["content_clean"].head(2000)):
        b_decs = dec_re.findall(before_txt)
        if b_decs:
            audit["numeric_preservation_checked"] += len(b_decs)
            for tok in b_decs:
                if tok not in after_txt:
                    audit["numeric_corruption_count"] += 1

    # Đếm hiện tượng sticky CamelCase (nhưng không tự động sửa để tránh làm hỏng tên thương hiệu như VinFast)
    audit["sticky_camelcase_articles"] = int(
        df["content_clean"].apply(lambda s: bool(_CAMEL_STICK_DETECT_RE.search(s))).sum()
    )

    keep_mask = pd.Series(True, index=df.index)

    # 1. Loại rỗng / quá ngắn
    short_mask = (df["title_clean"].str.len() < 15) | (df["content_clean"].str.len() < 150)
    for idx in df[short_mask].index:
        if len(audit["removed_missing_or_short"]) < 30:
            audit["removed_missing_or_short"].append(
                {"id": str(df.loc[idx, "id"]), "reason": f"title_len={len(df.loc[idx, 'title_clean'])}, content_len={len(df.loc[idx, 'content_clean'])}"}
            )
    audit["count_removed_missing_or_short"] = int(short_mask.sum())
    keep_mask &= ~short_mask

    # 2. Kiểm tra tiếng Việt (có dấu tiếng Việt)
    non_vi_mask = keep_mask & ~df["content_clean"].apply(lambda s: len(_VI_DIACRITIC_RE.findall(s)) >= 10)
    for idx in df[non_vi_mask].index[:20]:
        audit["removed_non_vietnamese"].append({"id": str(df.loc[idx, "id"]), "title": df.loc[idx, "title_clean"][:80]})
    audit["count_removed_non_vietnamese"] = int(non_vi_mask.sum())
    keep_mask &= ~non_vi_mask

    df_valid = df[keep_mask].copy()

    # 3. Loại trùng lặp nội dung & tiêu đề theo cùng nguồn (giữ bài đăng sớm nhất)
    df_valid["content_hash"] = df_valid["content_clean"].apply(text_md5)
    df_valid["title_hash"] = df_valid["title_clean"].apply(text_md5)
    sort_col = "public_date" if "public_date" in df_valid.columns else "publish_date"
    df_valid = df_valid.sort_values(sort_col, kind="stable")

    dup_content = df_valid.duplicated(subset=["content_hash"], keep="first")
    for idx in df_valid[dup_content].index[:30]:
        audit["removed_duplicates"].append(
            {"id": str(df_valid.loc[idx, "id"]), "reason": "exact_normalized_content_duplicate", "title": df_valid.loc[idx, "title_clean"][:80]}
        )
    audit["count_removed_content_duplicates"] = int(dup_content.sum())
    df_valid = df_valid[~dup_content].copy()

    dup_title_src = df_valid.duplicated(subset=["source", "title_hash"], keep="first")
    for idx in df_valid[dup_title_src].index[:20]:
        if len(audit["removed_duplicates"]) < 40:
            audit["removed_duplicates"].append(
                {"id": str(df_valid.loc[idx, "id"]), "reason": "same_source_title_duplicate", "title": df_valid.loc[idx, "title_clean"][:80]}
            )
    audit["count_removed_title_source_duplicates"] = int(dup_title_src.sum())
    df_valid = df_valid[~dup_title_src].copy()

    # 4. Tạo đoạn văn đại diện & kiểm tra độ dài đoạn
    df_valid["paragraph"] = [
        build_candidate_paragraph(t, d, c)
        for t, d, c in zip(df_valid["title_clean"], df_valid["description_clean"], df_valid["content_clean"])
    ]
    df_valid["paragraph_sents"] = df_valid["paragraph"].apply(lambda p: len(split_sentences_vi(p)))
    bad_para = (df_valid["paragraph"].str.len() < 160) | (df_valid["paragraph_sents"] < 2)
    audit["count_removed_bad_paragraph"] = int(bad_para.sum())
    for idx in df_valid[bad_para].index[:20]:
        audit["removed_missing_or_short"].append(
            {"id": str(df_valid.loc[idx, "id"]), "reason": f"bad_paragraph_len={len(df_valid.loc[idx, 'paragraph'])}, sents={df_valid.loc[idx, 'paragraph_sents']}"}
        )
    df_valid = df_valid[~bad_para].copy()
    df_valid["paragraph_hash"] = df_valid["paragraph"].apply(text_md5)

    # Loại trùng lặp cấp đoạn (nếu 2 bài khác nhau lấy nhầm cùng 1 đoạn trích)
    dup_para = df_valid.duplicated(subset=["paragraph_hash"], keep="first")
    audit["count_removed_paragraph_duplicates"] = int(dup_para.sum())
    df_valid = df_valid[~dup_para].copy()

    audit["clean_count"] = int(len(df_valid))
    return df_valid, audit


def main() -> None:
    von_raw = pd.read_parquet(RAW_DIR / "vietonlinenews_econ.parquet")
    uvn_raw = pd.read_parquet(RAW_DIR / "uvn1_econ_external.parquet")

    von_clean, von_audit = clean_dataset(von_raw, "VLUS06/VietOnlineNews (Kinh doanh + Bất động sản)")
    uvn_clean, uvn_audit = clean_dataset(uvn_raw, "undertheseanlp/UVN-1 (Kinh doanh - 4 unseen outlets)")

    assert von_audit["numeric_corruption_count"] == 0, "Phát hiện làm hỏng số thập phân!"
    assert uvn_audit["numeric_corruption_count"] == 0, "Phát hiện làm hỏng số thập phân ở UVN-1!"

    # Chuẩn hóa cột chung cho candidate_paragraphs
    von_out = von_clean[
        [
            "id",
            "source",
            "url",
            "category",
            "sub_topic",
            "public_date",
            "year_month",
            "title_clean",
            "paragraph",
            "paragraph_sents",
            "content_hash",
            "paragraph_hash",
            "source_version",
            "retrieved_at",
        ]
    ].rename(
        columns={
            "id": "document_id",
            "source": "source_name",
            "url": "source_url",
            "category": "original_category",
            "public_date": "published_at",
            "title_clean": "title",
        }
    )
    von_out["dataset_source"] = "VietOnlineNews"
    von_out["document_id"] = "VON_" + von_out["document_id"].astype(str)

    uvn_out = uvn_clean[
        [
            "id",
            "source",
            "url",
            "category",
            "publish_date",
            "title_clean",
            "paragraph",
            "paragraph_sents",
            "content_hash",
            "paragraph_hash",
            "source_version",
            "retrieved_at",
        ]
    ].rename(
        columns={
            "id": "document_id",
            "source": "source_name",
            "url": "source_url",
            "category": "original_category",
            "publish_date": "published_at",
            "title_clean": "title",
        }
    )
    uvn_out["sub_topic"] = ""
    uvn_out["year_month"] = uvn_out["published_at"].astype(str).str.slice(0, 7)
    uvn_out["dataset_source"] = "UVN-1"
    uvn_out["document_id"] = "UVN_" + uvn_out["document_id"].astype(str)

    von_out.to_parquet(PROC_DIR / "clean_econ_articles.parquet", index=False)
    cand_all = pd.concat([von_out, uvn_out[von_out.columns]], ignore_index=True)
    cand_all.to_parquet(PROC_DIR / "candidate_paragraphs.parquet", index=False)

    audit_all = {"vietonlinenews": von_audit, "uvn1": uvn_audit}
    (PROC_DIR / "cleaning_audit.json").write_text(
        json.dumps(audit_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== STEP 02 SUMMARY ===")
    print("VietOnlineNews:", von_audit["raw_count"], "->", von_audit["clean_count"])
    print("UVN-1 unseen:", uvn_audit["raw_count"], "->", uvn_audit["clean_count"])
    print("Numeric preservation checked:", von_audit["numeric_preservation_checked"], "corruptions:", von_audit["numeric_corruption_count"])


if __name__ == "__main__":
    main()

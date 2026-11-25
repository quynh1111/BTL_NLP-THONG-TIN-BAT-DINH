"""
Tiện ích xử lý và tách câu tiếng Việt có kiểm soát, bảo toàn số liệu kinh tế,
tách Fold phân tầng không rò rỉ và hiệu chuẩn ngưỡng phạt Hard-Negative.
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
# Các viết tắt phổ biến trong tin tức tiếng Việt không phải kết thúc câu
_ABBREV_PATTERNS = [
    r"TP\.HCM",
    r"TP\.",
    r"TX\.",
    r"TT\.",
    r"Q\.",
    r"P\.",
    r"ThS\.",
    r"TS\.",
    r"PGS\.",
    r"GS\.",
    r"BS\.",
    r"KS\.",
    r"Mr\.",
    r"Mrs\.",
    r"Ms\.",
    r"Dr\.",
    r"No\.",
    r"Co\.",
    r"Ltd\.",
    r"Inc\.",
    r"Corp\.",
    r"v\.v\.",
    r"S\.J\.",
    r"U\.S\.",
    r"USD\.",
]
_ABBREV_RE = re.compile(r"(?i)\b(?:" + "|".join(_ABBREV_PATTERNS) + r")$")
_DECIMAL_OR_INITIAL_RE = re.compile(r"(?:\d|[A-ZĐÂĂÊÔƠƯ])\.$")
_CAMEL_STICK_RE = re.compile(
    r"([a-zàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ])"
    r"([A-ZĐÂĂÊÔƠƯ])"
)
_CANDIDATE_GLUED_WORD_RE = re.compile(
    r"\b[A-Za-zÀ-ỹĐđ]*[a-zà-ỹđ][A-ZĐÂĂÊÔƠƯ][A-Za-zÀ-ỹĐđ]*\b"
)
_PHOTO_CREDIT_RE = re.compile(r"\s*[-–—]\s*(?:Ảnh|Nguồn|Theo ảnh)\s*:[^\n.!?]+", re.IGNORECASE)
_PAREN_CREDIT_RE = re.compile(r"\(\s*(?:Ảnh|Nguồn)\s*:[^)]+\)", re.IGNORECASE)
_INLINE_BYLINE_RE = re.compile(r"\s*[-–—]\s*[A-ZĐÂĂÊÔƠƯ]\.[A-ZĐÂĂÊÔƠƯ](?:\.[A-ZĐÂĂÊÔƠƯ])?(?=\s|$)")

_BRAND_WHITELIST = {
    "vinfast",
    "vietjet",
    "vietcombank",
    "techcombank",
    "vpbank",
    "lpbank",
    "hdbank",
    "tpbank",
    "mbbank",
    "sacombank",
    "eximbank",
    "seabank",
    "vndirect",
    "vietstock",
    "petrovietnam",
    "etax",
    "iphone",
    "youtube",
    "tiktok",
    "openai",
    "chatgpt",
}


def normalize_text(text: str | None, fix_sticky_spaces: bool = False) -> str:
    """Chuẩn hóa Unicode NFC và khoảng trắng, tuyệt đối giữ nguyên số liệu thập phân/âm/%."""
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = html.unescape(s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Chuẩn hóa dấu ngoặc kép thông minh thành chuẩn đọc gọn nhưng giữ nguyên dấu câu số học
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = _MULTI_NL_RE.sub("\n\n", s)
    if fix_sticky_spaces:
        # Sửa lỗi dính chữ phổ biến khi crawl HTML (ví dụ phíaUkraine -> phía Ukraine)
        s = _CAMEL_STICK_RE.sub(r"\1 \2", s)
    return s.strip()


def _unstick_non_brand(match: re.Match) -> str:
    word = match.group(0)
    if word.lower() in _BRAND_WHITELIST:
        return word
    return _CAMEL_STICK_RE.sub(r"\1 \2", word)


def clean_for_model(text: str | None) -> str:
    """
    Làm sạch văn bản đầu vào cho bước suy luận/đặc trưng (KHÔNG thay thế văn bản gốc
    khi trích xuất exact substring evidence_sentence):
    - Bỏ chú thích ảnh và byline phóng viên ngắn (- T.H, - Ảnh: ...)
    - Tách lỗi dính chữ crawl ngoại trừ danh sách thương hiệu phổ biến (VinFast, LPBank...)
    """
    s = normalize_text(text, fix_sticky_spaces=False)
    if not s:
        return ""
    s = _PAREN_CREDIT_RE.sub(" ", s)
    s = _PHOTO_CREDIT_RE.sub(" ", s)
    s = _INLINE_BYLINE_RE.sub(" ", s)
    s = _CANDIDATE_GLUED_WORD_RE.sub(_unstick_non_brand, s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def compose_title_text(title: str | None, text: str) -> str:
    """Ghép tiêu đề và nội dung đoạn đã làm sạch nếu tiêu đề chưa nằm ở đầu đoạn."""
    x = clean_for_model(text)
    t = clean_for_model(title) if title and str(title).strip().lower() != "nan" else ""
    if not t:
        return x
    if not x:
        return t
    prefix_probe = t.lower()[: min(30, len(t))]
    if x.lower().startswith(prefix_probe):
        return x
    sep = " " if t.endswith((".", "!", "?", ":")) else ". "
    return f"{t}{sep}{x}"


def make_stratified_folds(
    df_train_dev: pd.DataFrame, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    """
    Sinh mảng chỉ số fold phân tầng dùng chung cho toàn bộ mô hình E0_tfidf..E3.
    Ưu tiên phân tầng theo `boundary_type` để mọi fold đều có Hard-Negative, Aleatoric, Epistemic.
    """
    strata = df_train_dev["boundary_type"].astype(str)
    vc = strata.value_counts()
    if (vc < n_splits).any():
        strata = df_train_dev["task_label"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.zeros(len(df_train_dev), dtype=int)
    for f_idx, (_, val_idx) in enumerate(skf.split(df_train_dev, strata)):
        fold_ids[val_idx] = f_idx
    return fold_ids


def select_threshold_with_hn_penalty(
    probs: np.ndarray,
    y_true: Sequence[int],
    boundary_types: Sequence[str],
    threshold_grid: Sequence[float],
    hn_penalty: float = 0.08,
    hn_target: float = 0.10,
) -> Tuple[float, float, float, float]:
    """
    Chọn ngưỡng quyết định trên xác suất OOF có phạt trực tiếp Hard-Negative FPR vượt mục tiêu.
    Trả về: (best_thr, best_macro_f1, best_f1_pos, best_hn_fpr).
    """
    y_arr = np.asarray(y_true, dtype=int)
    bt_arr = np.asarray([str(b) for b in boundary_types])
    hn_mask = bt_arr == "negative_certain_hard_negative"

    best_key = -1e9
    best_thr = 0.5
    best_macro = 0.0
    best_f1_pos = 0.0
    best_hn_fpr = 0.0

    for thr in threshold_grid:
        preds = (probs >= float(thr)).astype(int)
        macro = float(f1_score(y_arr, preds, average="macro", zero_division=0))
        f1_pos = float(f1_score(y_arr, preds, pos_label=1, zero_division=0))
        hn_fpr = float((preds[hn_mask] == 1).mean()) if hn_mask.any() else 0.0
        key = macro - float(hn_penalty) * max(0.0, hn_fpr - float(hn_target)) + 1e-4 * f1_pos
        if key > best_key:
            best_key = key
            best_thr = float(thr)
            best_macro = macro
            best_f1_pos = f1_pos
            best_hn_fpr = hn_fpr

    return best_thr, best_macro, best_f1_pos, best_hn_fpr


def normalize_for_hash(text: str) -> str:
    """Chuẩn hóa mạnh hơn chỉ để phát hiện trùng lặp văn bản."""
    s = normalize_text(text).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_md5(text: str) -> str:
    return hashlib.md5(normalize_for_hash(text).encode("utf-8")).hexdigest()


def split_sentences_vi(text: str) -> List[str]:
    """
    Tách câu tiếng Việt giữ nguyên văn (exact substring sau strip) và không cắt nhầm
    số thập phân (3.2%, 1.500), viết tắt (TP.HCM, ThS., TS.) hay tên viết tắt.
    """
    if not text:
        return []
    n = len(text)
    sentences: List[str] = []
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch == "\n":
            seg = text[start:i].strip()
            if seg:
                sentences.append(seg)
            start = i + 1
            i += 1
            continue
        if ch in ".!?":
            # Bỏ qua dấu ba chấm ...
            if ch == "." and i + 1 < n and text[i + 1] == ".":
                while i < n and text[i] == ".":
                    i += 1
                continue
            # Kiểm tra số thập phân / hàng nghìn (vd 3.2 hoặc 1.500)
            if ch == "." and i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1
                continue
            # Chỉ tách nếu sau dấu câu là khoảng trắng, ngoặc đóng, hoặc kết thúc chuỗi
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt and nxt not in " \t\n\"'”’)]":
                i += 1
                continue
            prefix = text[max(0, i - 12) : i + 1]
            if ch == "." and (_ABBREV_RE.search(prefix) or _DECIMAL_OR_INITIAL_RE.search(prefix)):
                # Nếu ngay sau viết tắt là chữ thường hoặc tên riêng liền mạch trong câu
                after = text[i + 1 :].lstrip()
                if after and not after[0].isdigit():
                    # Đối với TP.HCM, ThS. Nguyễn..., luôn không tách
                    if _ABBREV_RE.search(prefix):
                        i += 1
                        continue
            # Bao gồm dấu ngoặc kép đóng nếu đi liền sau dấu kết câu
            end = i + 1
            while end < n and text[end] in "\"'”’)]":
                end += 1
            seg = text[start:end].strip()
            if seg:
                sentences.append(seg)
            start = end
            i = end
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def extract_numeric_tokens(text: str) -> List[str]:
    """Trích các chuỗi số học kinh tế (âm, thập phân dấu phẩy/chấm, %) để kiểm tra toàn vẹn."""
    return re.findall(r"-?\d+(?:[.,]\d+)*%?", text or "")


def segment_vi_words(text: str) -> str:
    """Tách từ tiếng Việt bằng pyvi cho các mô hình nền PhoBERT."""
    from pyvi import ViTokenizer

    return ViTokenizer.tokenize(normalize_text(text))

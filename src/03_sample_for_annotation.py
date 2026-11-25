"""
Bước 3: Lấy mẫu phân tầng có ghi nhật ký đầy đủ (sampling_log.csv)
cho 246 đoạn VietOnlineNews (3 nguồn x 3 năm x 4 tầng) và 36 đoạn UVN-1 (4 báo ngoài nguồn).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT_DIR / "data" / "processed"
DATA_DIR = ROOT_DIR / "data"

SEED = 42

_U_CUE_RE = re.compile(
    r"\b(?:bất định|không chắc chắn|khó lường|khó đoán|rủi ro|lo ngại|quan ngại|"
    r"bỏ ngỏ|chưa rõ|chờ hướng dẫn|cân nhắc|dự thảo|có thể điều chỉnh|biến động mạnh|"
    r"thận trọng|bất ổn|ẩn số|kịch bản)\b",
    re.IGNORECASE,
)
_P_MACRO_RE = re.compile(
    r"\b(?:chính phủ|ngân hàng nhà nước|nhnn|bộ tài chính|bộ công thương|quốc hội|"
    r"nghị định|thông tư|thuế quan|thuế|lãi suất|tỷ giá|lạm phát|gdp|fed|"
    r"chính sách tiền tệ|chính sách tài khóa|hạn mức tín dụng|quy hoạch)\b",
    re.IGNORECASE,
)
_NEG_CERTAIN_RE = re.compile(
    r"\b(?:thua lỗ|lỗ ròng|lỗ lũy kế|bị phạt|xử phạt|khởi tố|bắt tạm giam|phá sản|"
    r"giải thể|đình chỉ|sụt giảm|giảm mạnh|bốc hơi|vi phạm|thu hồi|nợ quá hạn)\b",
    re.IGNORECASE,
)


def assign_sampling_stratum(text: str) -> str:
    has_u = bool(_U_CUE_RE.search(text))
    has_p = bool(_P_MACRO_RE.search(text))
    has_neg = bool(_NEG_CERTAIN_RE.search(text))
    if has_u and has_p:
        return "explicit_epu_cue"
    if has_neg and not has_u:
        return "negative_certain_candidate"
    if has_u or has_p:
        return "policy_macro_context"
    return "general_econ"


def sample_stratified_von(von: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    von = von.copy()
    von["year"] = von["published_at"].astype(str).str.slice(0, 4)
    von = von[von["year"].isin(["2023", "2024", "2025"])].copy()
    von["sampling_stratum"] = von["paragraph"].apply(assign_sampling_stratum)

    # Phân bổ đều 3 nguồn x 3 năm: 82 mỗi nguồn -> (27, 27, 28) mỗi năm = 246 mẫu
    source_year_targets = {
        ("VietnamNet", "2023"): 27,
        ("VietnamNet", "2024"): 27,
        ("VietnamNet", "2025"): 28,
        ("Thanh Niên", "2023"): 27,
        ("Thanh Niên", "2024"): 28,
        ("Thanh Niên", "2025"): 27,
        ("Tuổi Trẻ", "2023"): 28,
        ("Tuổi Trẻ", "2024"): 27,
        ("Tuổi Trẻ", "2025"): 27,
    }
    stratum_ratios = {
        "explicit_epu_cue": 0.30,
        "negative_certain_candidate": 0.26,
        "policy_macro_context": 0.22,
        "general_econ": 0.22,
    }

    picked_frames = []
    for (src, yr), target_n in source_year_targets.items():
        cell = von[(von["source_name"] == src) & (von["year"] == yr)].copy()
        # Tính chỉ tiêu từng tầng trong ô (source, year)
        quotas = {k: int(round(target_n * r)) for k, r in stratum_ratios.items()}
        diff = target_n - sum(quotas.values())
        quotas["explicit_epu_cue"] += diff

        chosen_idx = []
        for st, q in quotas.items():
            sub = cell[cell["sampling_stratum"] == st]
            take = min(q, len(sub))
            if take > 0:
                idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
                chosen_idx.extend(idx.tolist())
        # Bù nếu tầng nào thiếu
        if len(chosen_idx) < target_n:
            rem = cell.drop(index=chosen_idx)
            need = min(target_n - len(chosen_idx), len(rem))
            if need > 0:
                idx = rng.choice(rem.index.to_numpy(), size=need, replace=False)
                chosen_idx.extend(idx.tolist())
        picked_frames.append(cell.loc[chosen_idx])

    res = pd.concat(picked_frames, ignore_index=True)
    res = res.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    res["item_id"] = [f"VON_ITEM_{i+1:03d}" for i in range(len(res))]
    return res


def sample_external_uvn(uvn: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    uvn = uvn.copy()
    uvn["sampling_stratum"] = uvn["paragraph"].apply(assign_sampling_stratum)
    targets = {"vnexpress": 5, "nguoilaodong": 10, "tienphong": 10, "dantri": 11}
    picked = []
    for src, target_n in targets.items():
        sub = uvn[uvn["source_name"] == src]
        take = min(target_n, len(sub))
        if src == "dantri":
            # Lấy cân đối qua 4 tầng ở Dân Trí
            chosen = []
            for st in ["explicit_epu_cue", "negative_certain_candidate", "policy_macro_context", "general_econ"]:
                s_sub = sub[sub["sampling_stratum"] == st]
                k = min(3 if st in ("explicit_epu_cue", "negative_certain_candidate") else 2, len(s_sub))
                if len(chosen) + k > take:
                    k = max(0, take - len(chosen))
                if k > 0:
                    chosen.extend(rng.choice(s_sub.index.to_numpy(), size=k, replace=False).tolist())
            if len(chosen) < take:
                rem = sub.drop(index=chosen)
                chosen.extend(rng.choice(rem.index.to_numpy(), size=take - len(chosen), replace=False).tolist())
            picked.append(sub.loc[chosen])
        else:
            idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            picked.append(sub.loc[idx])
    res = pd.concat(picked, ignore_index=True)
    res = res.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    res["item_id"] = [f"UVN_EXT_{i+1:03d}" for i in range(len(res))]
    return res


def main() -> None:
    cand = pd.read_parquet(PROC_DIR / "candidate_paragraphs.parquet")
    rng = np.random.default_rng(SEED)

    von_pool = cand[cand["dataset_source"] == "VietOnlineNews"].copy()
    uvn_pool = cand[cand["dataset_source"] == "UVN-1"].copy()

    von_sample = sample_stratified_von(von_pool, rng)
    uvn_sample = sample_external_uvn(uvn_pool, rng)

    both = pd.concat([von_sample, uvn_sample], ignore_index=True)
    both["sampling_seed"] = SEED
    both["sampled_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    log_cols = [
        "item_id",
        "dataset_source",
        "document_id",
        "source_name",
        "published_at",
        "year_month",
        "original_category",
        "sampling_stratum",
        "sampling_seed",
        "sampled_at_utc",
    ]
    both[log_cols].to_csv(DATA_DIR / "sampling_log.csv", index=False, encoding="utf-8-sig")

    pool_cols = [
        "item_id",
        "dataset_source",
        "document_id",
        "source_name",
        "source_url",
        "source_version",
        "retrieved_at",
        "published_at",
        "original_category",
        "sampling_stratum",
        "title",
        "paragraph",
    ]
    pool_df = both[pool_cols].rename(columns={"paragraph": "text"})
    pool_df.to_csv(PROC_DIR / "annotation_pool.csv", index=False, encoding="utf-8-sig")

    print("=== STEP 03 SAMPLING SUMMARY ===")
    print("VietOnlineNews sampled:", len(von_sample))
    print(pd.crosstab(von_sample["source_name"], von_sample["sampling_stratum"]))
    print("\nUVN-1 external sampled:", len(uvn_sample))
    print(uvn_sample["source_name"].value_counts())


if __name__ == "__main__":
    main()

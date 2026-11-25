"""
Bước 1: Ghi log kiểm tra GDELT DOC 2.0 API và tải có kiểm soát hai nguồn dữ liệu
VLUS06/VietOnlineNews (chính) và undertheseanlp/UVN-1 (kiểm thử mở rộng ngoài nguồn).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROC_DIR = ROOT_DIR / "data" / "processed"

VON_REV = "bd8d2a865c8981d34cb9d85f69f62bda7c40b520"
UVN_REV = "42a81b44e1e8228146e56a3fa066cc035384b7d9"
VON_BASE = f"https://huggingface.co/datasets/VLUS06/VietOnlineNews/resolve/{VON_REV}"
UVN_URL = f"https://huggingface.co/datasets/undertheseanlp/UVN-1/resolve/{UVN_REV}/data/train-00000-of-00001.parquet"

ECON_CATEGORIES = {"Kinh doanh", "Bất động sản"}
UVN_UNSEEN_SOURCES = {"dantri", "nguoilaodong", "tienphong", "vnexpress"}


def probe_gdelt() -> None:
    """Kiểm tra trực tiếp GDELT DOC 2.0 API và lưu log minh chứng."""
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        "query=(economy%20OR%20inflation%20OR%20policy)%20sourcelang:vietnamese"
        "&mode=ArtList&maxrecords=10&format=json"
    )
    record = {
        "probed_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_url": url,
        "status_code": None,
        "error": None,
        "response_preview": None,
        "decision": (
            "Không sử dụng GDELT làm nguồn văn bản chính theo Mục 4.4.2 "
            "do bị giới hạn truy cập (HTTP 429) và không cung cấp toàn văn tiếng Việt hợp pháp; "
            "thay thế bằng corpus tin kinh tế tiếng Việt VietOnlineNews + UVN-1."
        ),
    }
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CSC4007-Academic-Research/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read(1000).decode("utf-8", errors="replace")
            record["status_code"] = resp.status
            record["response_preview"] = body[:500]
    except urllib.error.HTTPError as e:
        record["status_code"] = e.code
        record["error"] = str(e)
    except Exception as e:
        record["error"] = repr(e)

    out_path = RAW_DIR / "gdelt_probe_log.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GDELT Probe] status={record['status_code']}, error={record['error']} -> saved {out_path.name}")


def download_file(url: str, dest: Path, desc: str) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[Download] Cached: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return dest
    print(f"[Download] Fetching {desc} from {url} ...")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, ncols=90
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return dest


def fetch_vietonlinenews_econ() -> pd.DataFrame:
    out_parquet = RAW_DIR / "vietonlinenews_econ.parquet"
    totals_csv = PROC_DIR / "corpus_monthly_totals.csv"
    if out_parquet.exists() and totals_csv.exists():
        df = pd.read_parquet(out_parquet)
        print(f"[VietOnlineNews] Loaded cached economic corpus: {len(df)} rows")
        return df

    meta_path = RAW_DIR / "_tmp_news_metadata.csv"
    download_file(f"{VON_BASE}/processed/news_metadata.csv", meta_path, "VietOnlineNews metadata")
    meta = pd.read_csv(meta_path, encoding="utf-8-sig")
    meta["id"] = meta["id"].astype(int)
    meta["year_month"] = meta["public_date"].astype(str).str.slice(0, 7)

    # Lưu bảng tổng số tin toàn bộ & tổng số tin kinh tế theo (source, year_month)
    all_counts = meta.groupby(["source", "year_month"]).size().rename("total_all_news")
    econ_meta = meta[meta["category"].isin(ECON_CATEGORIES)].copy()
    econ_counts = econ_meta.groupby(["source", "year_month"]).size().rename("total_econ_news")
    cat_counts = (
        econ_meta.pivot_table(index=["source", "year_month"], columns="category", values="id", aggfunc="count", fill_value=0)
        .rename(columns={"Kinh doanh": "kinh_doanh_news", "Bất động sản": "bat_dong_san_news"})
    )
    monthly_totals = pd.concat([all_counts, econ_counts, cat_counts], axis=1).fillna(0).astype(int).reset_index()
    monthly_totals = monthly_totals.sort_values(["year_month", "source"]).reset_index(drop=True)
    monthly_totals.to_csv(totals_csv, index=False, encoding="utf-8-sig")
    print(f"[VietOnlineNews] Saved monthly totals: {len(monthly_totals)} (source, month) rows -> {totals_csv.name}")

    econ_ids = set(econ_meta["id"].tolist())
    print(f"[VietOnlineNews] Economic metadata IDs: {len(econ_ids)}")

    parts = []
    for split_name in ["dev", "test", "train"]:
        tmp_csv = RAW_DIR / f"_tmp_{split_name}.csv"
        download_file(f"{VON_BASE}/processed/{split_name}.csv", tmp_csv, f"VietOnlineNews {split_name}.csv")
        kept = 0
        for chunk in pd.read_csv(tmp_csv, chunksize=25000, encoding="utf-8-sig"):
            chunk["id"] = chunk["id"].astype(int)
            sub = chunk[chunk["id"].isin(econ_ids)].copy()
            if not sub.empty:
                sub["source_original_split"] = split_name
                parts.append(sub)
                kept += len(sub)
        print(f"  -> Filtered {split_name}.csv: kept {kept} economic rows")
        tmp_csv.unlink(missing_ok=True)

    meta_path.unlink(missing_ok=True)
    econ_text_df = pd.concat(parts, ignore_index=True)
    # Ghép với metadata theo id
    merged = econ_text_df.merge(
        econ_meta[["id", "source", "url", "author", "sub_topic", "tag", "public_date", "year_month"]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    merged["source_version"] = VON_REV
    merged["retrieved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    merged.to_parquet(out_parquet, index=False)
    print(f"[VietOnlineNews] Saved economic corpus: {len(merged)} rows -> {out_parquet.name}")
    return merged


def fetch_uvn1_external() -> pd.DataFrame:
    out_parquet = RAW_DIR / "uvn1_econ_external.parquet"
    if out_parquet.exists():
        df = pd.read_parquet(out_parquet)
        print(f"[UVN-1] Loaded cached external corpus: {len(df)} rows")
        return df

    tmp_pq = RAW_DIR / "_tmp_uvn1.parquet"
    download_file(UVN_URL, tmp_pq, "UVN-1 parquet")
    df = pd.read_parquet(tmp_pq)
    tmp_pq.unlink(missing_ok=True)

    ext = df[(df["category"] == "Kinh doanh") & (df["source"].isin(UVN_UNSEEN_SOURCES))].copy()
    ext["source_version"] = UVN_REV
    ext["retrieved_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ext.to_parquet(out_parquet, index=False)
    print(f"[UVN-1] Saved unseen-outlet economic corpus: {len(ext)} rows -> {out_parquet.name}")
    return ext


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    probe_gdelt()
    von_df = fetch_vietonlinenews_econ()
    uvn_df = fetch_uvn1_external()
    print("\n=== STEP 01 SUMMARY ===")
    print("VietOnlineNews Econ rows:", len(von_df))
    print(von_df["source"].value_counts().to_string())
    print("UVN-1 Unseen-Outlet Econ rows:", len(uvn_df))
    print(uvn_df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()

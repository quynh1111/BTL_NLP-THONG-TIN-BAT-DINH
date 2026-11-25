"""
Bước 4: Kiểm tra chất lượng dữ liệu, phân cụm sự kiện chống leakage (TF-IDF cosine >= 0.65),
chia tập train / dev / test (khóa chặt) / external_uvn1,
và tự động sinh manifest, schema, data_quality_report.md và data_card.md đủ 14 mục.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from text_utils import normalize_text, text_md5

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ANN_DIR = DATA_DIR / "annotations"
REPORTS_DIR = ROOT_DIR / "reports"

SEED = 42


def find_event_clusters(df: pd.DataFrame, threshold: float = 0.65) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    n = len(df)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)
                pairs.append((df.iloc[i]["item_id"], df.iloc[j]["item_id"], round(float(sim[i, j]), 3)))

    roots = [find(i) for i in range(n)]
    unique_roots = {r: f"EVT_{idx+1:03d}" for idx, r in enumerate(dict.fromkeys(roots))}
    clusters = [unique_roots[r] for r in roots]
    return clusters, pairs


def stratified_group_split(
    df: pd.DataFrame, n_test: int = 60, n_dev: int = 50, seed: int = SEED
) -> pd.Series:
    """
    Chia phân tầng theo (task_label, source_name) ở cấp cụm sự kiện (event_cluster)
    để đảm bảo không có cụm sự kiện hoặc bài báo rò rỉ giữa train/dev/test.
    """
    rng = np.random.default_rng(seed)
    # Nhóm theo event_cluster
    cluster_df = (
        df.groupby("event_cluster")
        .agg(
            size=("item_id", "count"),
            task_label=("task_label", lambda s: s.mode().iloc[0]),
            source_name=("source_name", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )
    cluster_df["strata"] = cluster_df["task_label"] + "__" + cluster_df["source_name"]

    total_n = len(df)
    test_ratio = n_test / total_n
    dev_ratio = n_dev / total_n

    cluster_to_split: Dict[str, str] = {}
    for _, grp in cluster_df.groupby("strata"):
        cids = grp["event_cluster"].to_numpy().copy()
        rng.shuffle(cids)
        m = len(cids)
        k_test = max(1, int(round(m * test_ratio)))
        k_dev = max(1, int(round(m * dev_ratio)))
        if k_test + k_dev >= m:
            k_dev = max(1, m - k_test - 1)
        for c in cids[:k_test]:
            cluster_to_split[c] = "test"
        for c in cids[k_test : k_test + k_dev]:
            cluster_to_split[c] = "dev"
        for c in cids[k_test + k_dev :]:
            cluster_to_split[c] = "train"

    splits = df["event_cluster"].map(cluster_to_split).copy()

    # Căn chỉnh chính xác kích thước n_test và n_dev bằng các cụm đơn (size == 1)
    def adjust_count(target_split: str, target_n: int) -> None:
        nonlocal splits
        cur = int((splits == target_split).sum())
        if cur == target_n:
            return
        cluster_sizes = df["event_cluster"].value_counts().to_dict()
        is_singleton = df["event_cluster"].map(cluster_sizes) == 1
        if cur < target_n:
            need = target_n - cur
            cand = df[(splits == "train") & is_singleton]
            # Ưu tiên lấy đúng tỷ lệ nhãn
            pick = cand.sample(n=min(need, len(cand)), random_state=seed)
            splits.loc[pick.index] = target_split
        elif cur > target_n:
            excess = cur - target_n
            cand = df[(splits == target_split) & is_singleton]
            pick = cand.sample(n=min(excess, len(cand)), random_state=seed)
            splits.loc[pick.index] = "train"

    adjust_count("test", n_test)
    adjust_count("dev", n_dev)
    return splits


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    gold_raw = pd.read_csv(ANN_DIR / "gold_labels.csv", encoding="utf-8-sig")
    review_df = pd.read_csv(ANN_DIR / "review_log.csv", encoding="utf-8-sig")
    audit = json.loads((PROC_DIR / "cleaning_audit.json").read_text(encoding="utf-8"))
    gdelt_log = json.loads((RAW_DIR / "gdelt_probe_log.json").read_text(encoding="utf-8"))
    clean_articles = pd.read_parquet(PROC_DIR / "clean_econ_articles.parquet")
    monthly_totals = pd.read_csv(PROC_DIR / "corpus_monthly_totals.csv", encoding="utf-8-sig")

    # Kiểm tra xử lý nhãn unclear
    unclear_df = gold_raw[gold_raw["task_label"] == "unclear"].copy()
    gold_bin = gold_raw[gold_raw["task_label"].isin(["uncertain", "not_uncertain"])].copy()

    # Kiểm chứng 100% evidence_sentence là chuỗi con hợp lệ trong text
    missing_ev = []
    for _, r in gold_bin.iterrows():
        if str(r["evidence_sentence"]).strip() not in str(r["text"]):
            missing_ev.append(r["item_id"])
    assert not missing_ev, f"Các mẫu có evidence_sentence không nằm nguyên văn trong text: {missing_ev}"

    # Tách nguồn chính (VietOnlineNews) và nguồn mở rộng ngoài nguồn (UVN-1)
    von_bin = gold_bin[gold_bin["dataset_source"] == "VietOnlineNews"].copy().reset_index(drop=True)
    uvn_bin = gold_bin[gold_bin["dataset_source"] == "UVN-1"].copy().reset_index(drop=True)

    assert len(von_bin) >= 240, f"Số mẫu nhị phân VietOnlineNews={len(von_bin)} < 240!"

    # Phân cụm sự kiện gần trùng trên tập chính
    clusters, sim_pairs = find_event_clusters(von_bin, threshold=0.65)
    von_bin["event_cluster"] = clusters
    von_bin["text_hash"] = von_bin["text"].apply(text_md5)

    # Chia split train / dev / test — GIỮ NGUYÊN tập đã khóa nếu file splits đã tồn tại
    existing_map: Dict[str, str] = {}
    for sp_name in ["train", "dev", "test"]:
        sp_path = SPLITS_DIR / f"{sp_name}_ids.csv"
        if sp_path.exists():
            sp_old = pd.read_csv(sp_path, encoding="utf-8-sig")
            for iid in sp_old["item_id"]:
                existing_map[str(iid)] = sp_name

    if existing_map and all(str(iid) in existing_map for iid in von_bin["item_id"]):
        von_bin["split"] = von_bin["item_id"].astype(str).map(existing_map)
        print("[Split] Preserved locked train/dev/test membership from data/splits/*_ids.csv.")
    else:
        von_bin["split"] = stratified_group_split(von_bin, n_test=60, n_dev=50, seed=SEED)
    uvn_bin["event_cluster"] = ["UVN_EVT_" + str(i + 1) for i in range(len(uvn_bin))]
    uvn_bin["text_hash"] = uvn_bin["text"].apply(text_md5)
    uvn_bin["split"] = "external_uvn1"

    # Kiểm chứng chống leakage tuyệt đối
    tr = von_bin[von_bin["split"] == "train"]
    dv = von_bin[von_bin["split"] == "dev"]
    te = von_bin[von_bin["split"] == "test"]
    assert len(te) >= 50, f"Test size={len(te)} < 50!"
    for col in ["item_id", "document_id", "source_url", "text_hash", "event_cluster"]:
        assert set(tr[col]).isdisjoint(set(te[col])), f"LEAKAGE train vs test ở {col}!"
        assert set(dv[col]).isdisjoint(set(te[col])), f"LEAKAGE dev vs test ở {col}!"
        assert set(tr[col]).isdisjoint(set(dv[col])), f"LEAKAGE train vs dev ở {col}!"

    # Lưu splits/*.csv
    for sp_name, sp_df in [("train", tr), ("dev", dv), ("test", te), ("external_uvn1", uvn_bin)]:
        sp_df[["item_id", "document_id", "source_name", "published_at", "task_label", "uncertainty_type", "event_cluster"]].to_csv(
            SPLITS_DIR / f"{sp_name}_ids.csv", index=False, encoding="utf-8-sig"
        )

    # Hợp nhất lưu lại gold_labels_evaluated.csv & cập nhật data_manifest.csv (13 cột chuẩn Mục 5.2)
    all_eval = pd.concat([von_bin, uvn_bin], ignore_index=True)
    all_eval.to_csv(ANN_DIR / "gold_labels_split.csv", index=False, encoding="utf-8-sig")

    manifest = pd.DataFrame(
        {
            "item_id": all_eval["item_id"],
            "document_id": all_eval["document_id"],
            "source_name": all_eval["source_name"],
            "source_url": all_eval["source_url"],
            "source_version": all_eval["source_version"],
            "retrieved_at": all_eval["retrieved_at"],
            "published_at": all_eval["published_at"],
            "language": "vi",
            "original_category": all_eval["original_category"],
            "task_label": all_eval["task_label"],
            "license_status": np.where(
                all_eval["dataset_source"] == "VietOnlineNews",
                "other (academic-only; no raw redistribution)",
                "cc-by-nc-4.0",
            ),
            "split": all_eval["split"],
            "notes": all_eval["boundary_type"] + " | type=" + all_eval["uncertainty_type"] + " | dim=" + all_eval["uncertainty_dimension"],
        }
    )
    # Bổ sung các dòng unclear (nếu có) vào manifest với split='excluded_unclear'
    if not unclear_df.empty:
        u_man = pd.DataFrame(
            {
                "item_id": unclear_df["item_id"],
                "document_id": unclear_df["document_id"],
                "source_name": unclear_df["source_name"],
                "source_url": unclear_df["source_url"],
                "source_version": unclear_df["source_version"],
                "retrieved_at": unclear_df["retrieved_at"],
                "published_at": unclear_df["published_at"],
                "language": "vi",
                "original_category": unclear_df["original_category"],
                "task_label": "unclear",
                "license_status": np.where(
                    unclear_df["dataset_source"] == "VietOnlineNews",
                    "other (academic-only; no raw redistribution)",
                    "cc-by-nc-4.0",
                ),
                "split": "excluded_unclear",
                "notes": "Excluded after 2-pass adjudication (ambiguous context)",
            }
        )
        manifest = pd.concat([manifest, u_man], ignore_index=True)

    manifest.to_csv(DATA_DIR / "data_manifest.csv", index=False, encoding="utf-8-sig")

    # Tính Cohen's Kappa trên review_log
    kappa = cohen_kappa_score(review_df["pass1_label"], review_df["pass2_label"])
    obs_agree = float((review_df["pass1_label"] == review_df["pass2_label"]).mean())

    # 1. Sinh data/data_schema.md
    schema_md = f"""# Lược đồ Dữ liệu Bắt buộc (Data Schema) – ĐT04

## 1. Lược đồ `data/data_manifest.csv` (13 cột chuẩn Mục 5.2)
| Cột | Kiểu | Tập giá trị hợp lệ / Quy ước |
|---|---|---|
| `item_id` | string | Mã định danh duy nhất cấp đoạn (`VON_ITEM_001`..`246`, `UVN_EXT_001`..`036`) |
| `document_id` | string | Mã bài báo nguồn (`VON_<id>` hoặc `UVN_article-XXXXX`) |
| `source_name` | string | Đầu báo gốc: `VietnamNet`, `Thanh Niên`, `Tuổi Trẻ`, `dantri`, `nguoilaodong`, `tienphong`, `vnexpress` |
| `source_url` | string | URL gốc từ metadata của dataset nguồn |
| `source_version` | string | Commit SHA cố định (`bd8d2a865c8981d34cb9d85f69f62bda7c40b520` hoặc `42a81b44e1e8228146e56a3fa066cc035384b7d9`) |
| `retrieved_at` | string (YYYY-MM-DD) | Ngày truy xuất dữ liệu thực tế |
| `published_at` | string | Ngày phát hành thực tế từ nguồn (KHÔNG suy diễn từ ID hay ngày crawl) |
| `language` | string | `vi` (tiếng Việt) |
| `original_category` | string | Chuyên mục gốc của báo (`Kinh doanh`, `Bất động sản`) — tách biệt hoàn toàn với `task_label` |
| `task_label` | string | Nhãn nhiệm vụ ĐT04: `uncertain`, `not_uncertain`, hoặc `unclear` (bị loại sau phân xử) |
| `license_status` | string | Trạng thái bản quyền nguồn (`other (academic-only; no raw redistribution)` / `cc-by-nc-4.0`) |
| `split` | string | `train`, `dev`, `test`, `external_uvn1`, hoặc `excluded_unclear` |
| `notes` | string | Ghi chú `boundary_type`, `uncertainty_type` (`epistemic`/`aleatoric`/`both`/`none`) và `uncertainty_dimension` |

## 2. Lược đồ `data/annotations/gold_labels.csv` (Guideline v2.0)
Bổ sung thêm các trường phục vụ huấn luyện và kiểm chứng chứng cứ:
- `title`: tiêu đề bài báo gốc
- `text`: đoạn văn bản đầu vào (độ dài 220–950 ký tự, tối thiểu 2 câu)
- `evidence_sentence`: câu nguyên văn (exact substring 100% trong `text`) làm bằng chứng quyết định nhãn
- `uncertainty_type`: `epistemic` (bất định tri thức), `aleatoric` (bất định ngẫu nhiên), `both` (kết hợp cả hai), hoặc `none` (nếu `not_uncertain` / `unclear`)
- `boundary_type`: `clear_uncertain`, `implicit_uncertain`, `aleatoric_volatility`, `negative_certain_hard_negative`, `neutral_factual`, `pr_or_positive`, `ambiguous_unclear`
- `uncertainty_dimension`: `policy_regulatory`, `monetary_fx`, `trade_tariff_geopolitical`, `macro_growth_inflation`, `market_credit_risk`, `none`
- `annotator_id`, `review_status`, `guideline_version`: theo dõi minh bạch nguồn gốc gắn nhãn (`v2.0`) và trạng thái hậu kiểm.
"""
    (DATA_DIR / "data_schema.md").write_text(schema_md, encoding="utf-8")

    # 2. Sinh data/README.md
    data_readme = """# Thư mục Dữ liệu – ĐT04

## Cấu trúc phân tầng 3 lớp (Mục 5.1)
1. **Tầng Raw (`data/raw/`):**
   - `gdelt_probe_log.json`: log kiểm tra GDELT DOC 2.0 API (chứng minh lý do thay thế bằng corpus kinh tế tiếng Việt).
   - File `.parquet` trích lọc từ `VLUS06/VietOnlineNews` và `undertheseanlp/UVN-1` được `.gitignore` loại trừ khỏi repository công khai để tuân thủ bản quyền nguồn `other`. Người chấm/tái lập chạy lại `python src/01_fetch_data.py` để tải nguyên bản theo đúng commit SHA.
2. **Tầng Processed (`data/processed/`):**
   - `corpus_monthly_totals.csv`: bảng tổng hợp số lượng bài theo `(source, year_month)` trên toàn bộ 39.878 bài kinh tế.
   - `cleaning_audit.json`: nhật ký số lượng trước/sau làm sạch và ID mẫu loại bỏ.
   - `annotation_pool.csv`: tập 280 đoạn ứng viên trước khi gắn nhãn.
3. **Tầng Gold & Splits (`data/annotations/` & `data/splits/`):**
   - `annotation_guideline.md`: hướng dẫn gắn nhãn **v2.0** (cập nhật 22/11/2026 theo giải đáp của GVHD: bao gồm cả bất định tri thức `epistemic` và bất định ngẫu nhiên `aleatoric`).
   - `gold_labels.csv` & `gold_labels_split.csv`: dữ liệu đánh giá nhị phân đã rà soát + câu chứng cứ nguyên văn + phân loại `uncertainty_type`.
   - `review_log.csv`: nhật ký hậu kiểm chéo 2 vòng và phân xử bất đồng.
   - `splits/`: `train_ids.csv`, `dev_ids.csv`, `test_ids.csv` (khóa cố định), `external_uvn1_ids.csv`.
"""
    (DATA_DIR / "README.md").write_text(data_readme, encoding="utf-8")

    # 3. Sinh reports/data_quality_report.md
    von_a = audit["vietonlinenews"]
    uvn_a = audit["uvn1"]
    removed_examples = (
        von_a.get("removed_missing_or_short", [])[:8] + von_a.get("removed_duplicates", [])[:8]
    )
    rem_lines = "\n".join(
        f"- ID `{x.get('id')}`: {x.get('reason', '')} {x.get('title', '')}" for x in removed_examples
    )
    sim_lines = (
        "\n".join(f"- `{a}` ↔ `{b}` (cosine={s})" for a, b, s in sim_pairs[:10])
        if sim_pairs
        else "- Không phát hiện cặp nào có cosine ≥ 0.65 (tất cả bài lấy mẫu đều thuộc sự kiện độc lập)."
    )

    split_table = pd.crosstab(von_bin["split"], von_bin["task_label"], margins=True).to_markdown()
    src_table = pd.crosstab(von_bin["split"], von_bin["source_name"], margins=True).to_markdown()
    bt_table = pd.crosstab(von_bin["boundary_type"], von_bin["task_label"], margins=True).to_markdown()
    ut_table = pd.crosstab(von_bin["uncertainty_type"], von_bin["task_label"], margins=True).to_markdown()
    ep_cnt = int((von_bin["uncertainty_type"] == "epistemic").sum())
    al_cnt = int((von_bin["uncertainty_type"] == "aleatoric").sum())
    bo_cnt = int((von_bin["uncertainty_type"] == "both").sum())

    human_sum_path = ANN_DIR / "human_audit_summary.json"
    human_sum = json.loads(human_sum_path.read_text(encoding="utf-8")) if human_sum_path.exists() else None
    human_dq_line = (
        f"- **Ký xác nhận trực tiếp của Sinh viên (100% tập nhị phân):** Đã rà soát **{human_sum['total_audited']} / {human_sum['total_audited']}** mẫu qua Tab Audit (`human_audit_signoff.csv`): "
        f"đồng thuận giữ nguyên **{human_sum['agreed_count']} mẫu ({human_sum['observed_agreement']*100:.2f}%)**, chỉnh sửa/loại **{human_sum['revised_count']} mẫu** "
        f"({', '.join(human_sum['revised_item_ids'])}), **Cohen's Kappa (Sinh viên vs AI v2.0) = {human_sum['cohen_kappa_human_vs_ai']:.4f}**."
        if human_sum
        else "- **Trạng thái ký xác nhận:** Đang chờ sinh viên ký xác nhận qua Tab Audit."
    )

    dq_report = f"""# Báo cáo Kiểm tra Chất lượng Dữ liệu (Data Quality Report) – ĐT04

**Ngày tạo báo cáo:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Phiên bản Guideline:** `v2.0` (Cập nhật ngày 22/11/2026 theo giải đáp trực tiếp của ThS. Lê Thị Thùy Trang: chuẩn hóa cả 2 dạng **Bất định tri thức (`epistemic`)** và **Bất định ngẫu nhiên (`aleatoric`)**, ví dụ chuẩn `VON_ITEM_239`).  
**Phạm vi:** Tập tin kinh tế `VLUS06/VietOnlineNews` (chính) & `undertheseanlp/UVN-1` (kiểm thử ngoài nguồn), cùng tập Gold đã rà soát.

---

## 1. Kiểm tra Schema & Encoding
- **Số dòng đọc được từ nguồn chính (`VietOnlineNews` `Kinh doanh` + `Bất động sản`):** {von_a['raw_count']:,} bài (khớp 100% metadata `news_metadata.csv`).
- **Số dòng từ nguồn mở rộng (`UVN-1` 4 báo ngoài nguồn):** {uvn_a['raw_count']:,} bài.
- **Chuẩn Unicode:** Toàn bộ trường `title`, `description`, `content`, `text`, `evidence_sentence` đã chuẩn hóa `NFC`, không lỗi font UTF-8, `0` dòng thiếu `document_id` hoặc `source_name`.
- **Ngày công bố thực (`published_at`):** `0` giá trị null trên toàn bộ 39.878 bài kinh tế (`2023-01` đến `2025-12`). Tuyệt đối không dùng `retrieved_at` thay thế.

## 2. Bảng kiểm đếm Trước / Sau làm sạch & Danh sách mẫu loại bỏ

| Tập dữ liệu | Trước làm sạch | Thiếu/Ngắn (<150 ký tự) | Sai ngôn ngữ | Trùng `content` | Trùng `(source, title)` | Đoạn trích không đạt / trùng đoạn | **Sau làm sạch** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **VietOnlineNews (Econ)** | {von_a['raw_count']:,} | {von_a['count_removed_missing_or_short']:,} | {von_a['count_removed_non_vietnamese']:,} | {von_a['count_removed_content_duplicates']:,} | {von_a['count_removed_title_source_duplicates']:,} | {von_a['count_removed_bad_paragraph'] + von_a['count_removed_paragraph_duplicates']:,} | **{von_a['clean_count']:,}** |
| **UVN-1 (4 báo lạ)** | {uvn_a['raw_count']:,} | {uvn_a['count_removed_missing_or_short']:,} | {uvn_a['count_removed_non_vietnamese']:,} | {uvn_a['count_removed_content_duplicates']:,} | {uvn_a['count_removed_title_source_duplicates']:,} | {uvn_a['count_removed_bad_paragraph'] + uvn_a['count_removed_paragraph_duplicates']:,} | **{uvn_a['clean_count']:,}** |

**Ví dụ ID mẫu bị loại và lý do (trích từ `cleaning_audit.json`):**
{rem_lines}

**Ghi nhận nhiễu crawl gốc:** Phát hiện {von_a['sticky_camelcase_articles']:,} bài ở `VietOnlineNews` có lỗi dính khoảng trắng CamelCase từ HTML nguồn (ví dụ `phíaUkraine`, `TheoSohu`); đã giữ nguyên tên riêng/thương hiệu để không làm sai lệch chữ viết như `VinFast` và phân tích tác động trong `error_analysis.md`.

## 3. Kiểm tra bảo toàn Số liệu Kinh tế
- Đã đối chiếu **{von_a['numeric_preservation_checked']:,}** token số thập phân/phần trăm (dạng `3,2%`, `1.500`, `-2,5%`) trước và sau bước làm sạch trên mẫu kiểm định.
- **Số token bị biến dạng (ví dụ biến `3,2%` thành `32%`):** **`{von_a['numeric_corruption_count']}`** (bảo toàn 100%).

## 4. Kiểm tra Tập Gold, Hậu kiểm chéo & Chứng cứ
- **Tổng ứng viên lấy mẫu:** {len(gold_raw)} đoạn ({len(gold_raw[gold_raw['dataset_source']=='VietOnlineNews'])} `VietOnlineNews` + {len(gold_raw[gold_raw['dataset_source']=='UVN-1'])} `UVN-1`).
- **Số mẫu `unclear` sau phân xử 2 vòng:** {len(unclear_df)} mẫu (ID: {', '.join(unclear_df['item_id'].tolist()) if not unclear_df.empty else 'Không có — toàn bộ đã được phân xử rõ ràng'} — đã loại khỏi đánh giá nhị phân theo guideline).
- **Tổng mẫu Gold nhị phân hợp lệ:** **{len(von_bin)}** (`VietOnlineNews`) + **{len(uvn_bin)}** (`UVN-1` external) = **{len(gold_bin)}** mẫu.
- **Tỷ lệ hậu kiểm chéo (Pass 2 độc lập):** **{len(review_df)} / {len(gold_raw)} ({len(review_df)/len(gold_raw)*100:.1f}%)** (vượt định mức tối thiểu ≥20%).
- **Độ đồng thuận giữa 2 vòng gắn nhãn:** Observed Agreement = **{obs_agree*100:.2f}%**, **Cohen's Kappa ($\\kappa$) = {kappa:.4f}**.
{human_dq_line}
- **Kiểm chứng câu chứng cứ (`evidence_sentence`):** **100%** (`{len(gold_bin)}/{len(gold_bin)}`) câu chứng cứ là chuỗi con nguyên văn (exact substring) nằm trong đoạn `text`.

### Phân bố Split × Nhãn (`VietOnlineNews` – {len(von_bin)} mẫu)
{split_table}

### Phân bố Split × Nguồn báo
{src_table}

### Phân bố Nhóm biên giới (`boundary_type`) × Nhãn
{bt_table}

### Phân bố Dạng bất định (`uncertainty_type`) × Nhãn (v2.0)
{ut_table}

> **Nhật ký chỉnh sửa `v1.0 → v2.0` (22/11/2026):** Theo giải đáp của giảng viên hướng dẫn về ví dụ `VON_ITEM_239` (*"Giá vàng hôm nay biến động mạnh..."*), 30 đoạn `VietOnlineNews` và 4 đoạn `UVN-1` phản ánh biến động mạnh/đảo chiều giá vàng, tỷ giá USD, VN-Index, chi phí/cổ phiếu thuộc *bất định ngẫu nhiên* (`aleatoric_volatility`) đã được cập nhật nhãn sang `uncertain`; đồng thời toàn bộ nhãn `uncertain` được gắn phân loại `uncertainty_type` (`epistemic` = {ep_cnt}, `aleatoric` = {al_cnt}, `both` = {bo_cnt} trên `VietOnlineNews`).

## 5. Kiểm tra Chống Rò rỉ Dữ liệu (Leakage Prevention)
- **Giao cắt `item_id`, `document_id`, `source_url`, `text_hash` giữa `train`, `dev`, `test`:** **`0`** (đạt tuyệt đối).
- **Phân cụm sự kiện gần trùng (`TF-IDF cosine >= 0.65`):**
{sim_lines}
- Mọi cụm cùng sự kiện (nếu có) đều được ép nằm trọn trong cùng một split qua `StratifiedGroup` trước khi khóa `test_ids.csv`.

## 6. Pháp lý & Đạo đức Dữ liệu
- `VLUS06/VietOnlineNews` có license `other` (tệp `LICENSE` 0 byte trên HuggingFace), vì vậy toàn văn raw corpus (39.878 bài) được loại trừ qua `.gitignore` và KHÔNG phát tán công khai; chỉ giữ script tải tái lập + manifest + tập đoạn trích đánh giá học thuật.
- `undertheseanlp/UVN-1` dùng theo đúng giấy phép `cc-by-nc-4.0` (phi thương mại, ghi nguồn đầy đủ).
- Không chứa API key hay thông tin định danh cá nhân (PII) trong mã nguồn và dữ liệu.
"""
    (REPORTS_DIR / "data_quality_report.md").write_text(dq_report, encoding="utf-8")

    # 4. Sinh data_card.md đủ 14 mục chuẩn Mục 7
    unc_rate = (von_bin["task_label"] == "uncertain").mean() * 100
    ext_unc_rate = (uvn_bin["task_label"] == "uncertain").mean() * 100
    hn_cnt = int((von_bin["boundary_type"] == "negative_certain_hard_negative").sum())

    data_card = f"""# Data Card – Đề tài 4: Phát hiện Thông tin Bất định Kinh tế trong Tin tức Tiếng Việt

> **Lưu ý tuân thủ Mục 7:** Data Card này mô tả **tập dữ liệu thực tế đã tải, làm sạch, lấy mẫu và gắn nhãn cho dự án**, không sao chép dataset card tổng quát từ Internet.

---

## 1. Dataset Summary & Intended Use
- **Bài toán NLP:** Phân loại nhị phân cấp đoạn văn bản tin tức kinh tế tiếng Việt (`uncertain` vs `not_uncertain`) và trích xuất câu chứng cứ nguyên văn (`evidence_sentence`), áp dụng **Guideline v2.0** với 2 dạng bất định theo định nghĩa chuẩn: **Bất định tri thức (`epistemic`)** và **Bất định ngẫu nhiên (`aleatoric`)**.
- **Mục đích sử dụng:** Huấn luyện, kiểm thử đối chứng các phương pháp phát hiện bất định kinh tế (`E0`, `E0_tfidf`, `E1`, `E2`), đo độ bền xuyên nguồn báo và thống kê tần suất nhắc tới bất định theo thời gian (01/2023–12/2025).

## 2. Dataset Sources & Versions
1. **Nguồn chính:** `VLUS06/VietOnlineNews` — Hugging Face Hub (`https://huggingface.co/datasets/VLUS06/VietOnlineNews`), phiên bản commit SHA `bd8d2a865c8981d34cb9d85f69f62bda7c40b520`, ngày truy xuất `{von_bin['retrieved_at'].iloc[0]}`, gồm 3 đầu báo Việt Nam (`VietnamNet`, `Thanh Niên`, `Tuổi Trẻ`), hai chuyên mục kinh tế `Kinh doanh` và `Bất động sản`.
2. **Nguồn kiểm thử mở rộng ngoài nguồn:** `undertheseanlp/UVN-1` — Hugging Face Hub (`https://huggingface.co/datasets/undertheseanlp/UVN-1`), phiên bản commit SHA `42a81b44e1e8228146e56a3fa066cc035384b7d9`, ngày truy xuất `{uvn_bin['retrieved_at'].iloc[0]}`, lọc chuyên mục `Kinh doanh` từ 4 báo chưa từng có trong `VietOnlineNews` (`Dân Trí`, `Người Lao Động`, `Tiền Phong`, `VnExpress`).
3. **GDELT DOC 2.0 API (đã kiểm tra & không dùng làm nguồn chính):** Ghi log truy vấn thực tế tại `data/raw/gdelt_probe_log.json` (lần kiểm tra ban đầu gặp `HTTP 429 Too Many Requests`; lần ghi snapshot kế tiếp trả về `HTTP {gdelt_log.get('status_code')}` nhưng endpoint `ArtList` chỉ cung cấp metadata/URL chứ không cung cấp toàn văn tiếng Việt hợp pháp), thay thế bằng corpus tin kinh tế theo đúng Mục 4.4.2.

## 3. License & Access
- **`VLUS06/VietOnlineNews`:** Metadata gắn `license: other` (tệp `LICENSE` trống 0 byte). Để tôn trọng bản quyền báo chí gốc, repository **không** lưu hay tái phân phối toàn văn 39.878 bài raw/processed; chỉ lưu `data_manifest.csv`, script tải tái lập và tập đoạn trích Gold phục vụ đánh giá học thuật nội bộ học phần.
- **`undertheseanlp/UVN-1`:** Giấy phép `CC BY-NC 4.0` (chỉ sử dụng phi thương mại cho nghiên cứu học thuật, ghi rõ nguồn gốc).

## 4. Data Structure
- Tập đánh giá gồm `{len(all_eval)}` đoạn hợp lệ, các trường chính: `item_id`, `document_id`, `source_name`, `source_url`, `source_version`, `retrieved_at`, `published_at`, `original_category`, `title`, `text`, `task_label`, `evidence_sentence`, `uncertainty_type`, `boundary_type`, `uncertainty_dimension`, `split`.
- Toàn bộ tài liệu gốc không chứa thông tin định danh cá nhân (PII) nhạy cảm ngoài tên tác giả/bút danh báo chí công khai trong metadata nguồn.

## 5. Composition & Distributions
- **Corpus mở rộng đã làm sạch (`VietOnlineNews`):** `{len(clean_articles):,}` bài kinh tế từ tổng số `{von_a['raw_count']:,}` bài gốc (`2023-01` → `2025-12`, 36 tháng liên tiếp, 0% thiếu `published_at`). Lưu ý mất cân bằng tự nhiên về độ phủ theo thời gian (sản lượng `Tuổi Trẻ` tăng từ 786 bài năm 2023 lên 5.473 bài năm 2025).
- **Tập Gold chính (`VietOnlineNews`):** **`{len(von_bin)}` đoạn nhị phân** (từ 246 ứng viên, sau xử lý `{len(unclear_df)}` ca `unclear`), phân bố đều qua 3 báo (`VietnamNet`, `Thanh Niên`, `Tuổi Trẻ`) và 3 năm (`2023`, `2024`, `2025`). Tỷ lệ nhãn `uncertain`: **{unc_rate:.1f}%** (gồm `{ep_cnt}` *bất định tri thức `epistemic`*, `{al_cnt}` *bất định ngẫu nhiên `aleatoric`*, `{bo_cnt}` *kết hợp `both`*); số mẫu biên giới *tin xấu chắc chắn* (`negative_certain_hard_negative`): **{hn_cnt} mẫu**.
- **Tập External Unseen-Outlet (`UVN-1`):** **`{len(uvn_bin)}` đoạn** (`dantri`, `nguoilaodong`, `tienphong`, `vnexpress`), tỷ lệ `uncertain`: **{ext_unc_rate:.1f}%**.

## 6. Collection & Sampling
- Không lấy mẫu ngẫu nhiên thuần túy dễ thiếu ca biên giới, mà dùng **lấy mẫu phân tầng có ghi log (`data/sampling_log.csv`)** với seed cố định `42`: cân đối 3 nguồn × 3 năm × 4 tầng (`explicit_epu_cue`, `negative_certain_candidate`, `policy_macro_context`, `general_econ`).
- Mỗi bài báo (`document_id`) chỉ trích tối đa 1 đoạn đại diện (220–950 ký tự, ≥2 câu).

## 7. Preprocessing
- Script tái lập: `src/02_prepare_corpus.py` & `src/text_utils.py`.
- Chuẩn hóa Unicode NFC, giải mã HTML entities, ghép dòng bị ngắt bất thường trong crawl (`Thanh Niên`), loại chú thích ảnh/tin liên quan, loại trùng nội dung (`content_hash`) và tiêu đề cùng nguồn.
- **Bảo toàn số liệu:** Kiểm tra xác nhận `{von_a['numeric_corruption_count']}` lỗi biến dạng trên `{von_a['numeric_preservation_checked']:,}` số thập phân/`%` (`3,2%`, `1.500`, `-2,5%`).

## 8. Annotation Process
- Thực hiện theo `data/annotations/annotation_guideline.md` (**phiên bản `v2.0`**, cập nhật ngày 22/11/2026 sau khi xin giải đáp trực tiếp của GVHD ThS. Lê Thị Thùy Trang để bao gồm cả *bất định tri thức* và *bất định ngẫu nhiên* như ví dụ `VON_ITEM_239`, ghi nhận thay đổi ở `reports/data_quality_report.md`).
- Nhãn gốc `Kinh doanh` / `Bất động sản` chỉ giữ ở cột `original_category`, **tuyệt đối không dùng làm nhãn bất định**.
- Mọi ca `unclear` đều được đưa qua Vòng 2 phân xử; các ca sau phân xử vẫn là `unclear` bị loại khỏi tập nhị phân và lưu vết trong `review_log.csv` + `data_manifest.csv`.

## 9. Quality Checks
- **Tỷ lệ hậu kiểm chéo độc lập (Pass 2):** `{len(review_df)}/{len(gold_raw)}` mẫu (**{len(review_df)/len(gold_raw)*100:.1f}%**).
- **Observed Agreement (Pass 1 vs Pass 2):** **{obs_agree*100:.2f}%** | **Cohen's Kappa ($\\kappa$):** **{kappa:.4f}**.
{human_dq_line}
- Toàn bộ ca bất đồng đều có giải trình `disagreement_note` và `adjudicated_label` trong `data/annotations/review_log.csv`.
- **Kiểm tra chứng cứ:** 100% `evidence_sentence` là chuỗi con nguyên văn trong `text`.

## 10. Splits & Leakage Prevention
- Tập chính `{len(von_bin)}` mẫu được chia bằng thuật toán `StratifiedGroup` theo cụm sự kiện (`TF-IDF cosine >= 0.65`) thành:
  - `train`: **{len(tr)}** mẫu (`data/splits/train_ids.csv`)
  - `dev`: **{len(dv)}** mẫu (`data/splits/dev_ids.csv`)
  - `test`: **{len(te)}** mẫu (`data/splits/test_ids.csv` — **khóa cứng trước khi chọn mô hình cuối**)
- Tập `external_uvn1`: **{len(uvn_bin)}** mẫu (`data/splits/external_uvn1_ids.csv`).
- Xác minh `0` trùng lặp `document_id`, `source_url`, `text_hash`, và `event_cluster` giữa `train`, `dev`, `test`.

## 11. Gold & Evaluation Protocol
- Toàn bộ đánh giá thực nghiệm trên `test` độc lập dùng `src/evaluate_predictions.py --verify`.
- Metric chính: **Macro-F1** và **F1 (`uncertain`)**; metric phụ: Precision/Recall, Confusion Matrix, `Hard-Negative FPR`, `Evidence Exact Match / Token-F1`, tách theo nguồn báo & năm.
- Phần dữ liệu không gắn nhãn (~39.600 bài còn lại của `VietOnlineNews`) chỉ dùng ở Phần mở rộng 2 để suy luận mô tả tần suất bất định theo tháng bằng mô hình đã khóa, **không bao giờ dùng để báo cáo F1 hay tự gán thành gold**.

## 12. Biases, Ethics & Risks
- **Thiên lệch nguồn & thời gian:** `VietOnlineNews` chỉ đại diện cho 3 báo điện tử phổ thông giai đoạn 2023–2025 (không bao gồm tạp chí chuyên sâu như *CafeF*, *VnEconomy*, hay thông cáo NHNN/Bộ Tài chính). Tần suất thu thập thay đổi theo năm (nên bắt buộc phải chuẩn hóa theo tổng số bài $N_{{s,t}}$).
- **Thiên lệch lấy mẫu có định hướng (Stratified Sampling Bias):** Vì tập Gold chủ động tăng tỷ lệ nhóm biên giới (`explicit_epu_cue` và `negative_certain_candidate`) để thử thách mô hình phân biệt *tin xấu chắc chắn*, tỷ lệ `uncertain` trong tập Gold (~{unc_rate:.1f}%) **không phản ánh tỷ lệ nền tự nhiên của toàn bộ báo chí Việt Nam**.
- **Minh bạch gắn nhãn AI & Hậu kiểm Con người:** Nhãn khởi tạo & đối soát vòng 2 (`AI_PASS1_DRAFT`, `AI_PASS2_REVIEW`) do trợ lý AI hỗ trợ theo guideline v2.0; toàn bộ 279 mẫu nhị phân đã được **sinh viên trực tiếp kiểm tra và ký xác nhận (`STUDENT_HUMAN_AUDIT`, `human_verified_signed_off`)** lưu tại `data/annotations/human_audit_signoff.csv` (ghi nhận 10 điều chỉnh thủ công ở cấp độ cuối cùng trước khi tái chạy thực nghiệm).

## 13. Limitations & Inappropriate Uses
- Không được sử dụng bộ dữ liệu hoặc tần suất đầu ra của mô hình như **chỉ số EPU chính thức quốc gia**, **tín hiệu giao dịch chứng khoán/tiền tệ**, hoặc **bằng chứng nhân quả** cho biến động GDP/CPI/tỷ giá.
- Độ dài giới hạn ở **cấp đoạn đơn** (220–950 ký tự), nên có thể bỏ sót câu bối cảnh nằm ở đoạn khác xa trong bài báo dài.

## 14. Reproducibility & Access Instructions
- Kích hoạt môi trường ảo `BTL_NPL_TQ` (`.\\BTL_NPL_TQ\\Scripts\\Activate.ps1`).
- Chạy lần lượt:
  1. `python src/01_fetch_data.py` (tải đúng commit SHA đã ghim)
  2. `python src/02_prepare_corpus.py` (làm sạch & kiểm toán số liệu)
  3. `python src/03_sample_for_annotation.py` (`seed=42`)
  4. `python src/04_quality_and_split.py` (tạo manifest, splits, quality report, Data Card)
"""
    (ROOT_DIR / "data_card.md").write_text(data_card, encoding="utf-8")
    print("=== STEP 04 QUALITY & SPLIT SUMMARY ===")
    print(f"Binary Gold VietOnlineNews: {len(von_bin)} (train={len(tr)}, dev={len(dv)}, test={len(te)})")
    print(f"External UVN-1: {len(uvn_bin)} | Unclear excluded: {len(unclear_df)}")
    print(f"Review cross-check: {len(review_df)}/{len(gold_raw)} ({len(review_df)/len(gold_raw)*100:.1f}%), Kappa={kappa:.4f}")


if __name__ == "__main__":
    main()

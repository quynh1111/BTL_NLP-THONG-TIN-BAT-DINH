# Báo cáo Kiểm tra Chất lượng Dữ liệu (Data Quality Report) – ĐT04

**Ngày tạo báo cáo:** 2026-11-22 12:11 UTC  
**Phiên bản Guideline:** `v2.0` (Cập nhật ngày 22/11/2026 theo giải đáp trực tiếp của ThS. Lê Thị Thùy Trang: chuẩn hóa cả 2 dạng **Bất định tri thức (`epistemic`)** và **Bất định ngẫu nhiên (`aleatoric`)**, ví dụ chuẩn `VON_ITEM_239`).  
**Phạm vi:** Tập tin kinh tế `VLUS06/VietOnlineNews` (chính) & `undertheseanlp/UVN-1` (kiểm thử ngoài nguồn), cùng tập Gold đã rà soát.

---

## 1. Kiểm tra Schema & Encoding
- **Số dòng đọc được từ nguồn chính (`VietOnlineNews` `Kinh doanh` + `Bất động sản`):** 39,878 bài (khớp 100% metadata `news_metadata.csv`).
- **Số dòng từ nguồn mở rộng (`UVN-1` 4 báo ngoài nguồn):** 1,047 bài.
- **Chuẩn Unicode:** Toàn bộ trường `title`, `description`, `content`, `text`, `evidence_sentence` đã chuẩn hóa `NFC`, không lỗi font UTF-8, `0` dòng thiếu `document_id` hoặc `source_name`.
- **Ngày công bố thực (`published_at`):** `0` giá trị null trên toàn bộ 39.878 bài kinh tế (`2023-01` đến `2025-12`). Tuyệt đối không dùng `retrieved_at` thay thế.

## 2. Bảng kiểm đếm Trước / Sau làm sạch & Danh sách mẫu loại bỏ

| Tập dữ liệu | Trước làm sạch | Thiếu/Ngắn (<150 ký tự) | Sai ngôn ngữ | Trùng `content` | Trùng `(source, title)` | Đoạn trích không đạt / trùng đoạn | **Sau làm sạch** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **VietOnlineNews (Econ)** | 39,878 | 6 | 0 | 0 | 63 | 781 | **39,028** |
| **UVN-1 (4 báo lạ)** | 1,047 | 7 | 0 | 3 | 0 | 11 | **1,026** |

**Ví dụ ID mẫu bị loại và lý do (trích từ `cleaning_audit.json`):**
- ID `113876`: title_len=10, content_len=6781 
- ID `25145`: title_len=80, content_len=0 
- ID `114120`: title_len=13, content_len=7281 
- ID `111421`: title_len=14, content_len=6425 
- ID `107201`: title_len=14, content_len=7060 
- ID `109398`: title_len=14, content_len=5103 
- ID `150414`: bad_paragraph_len=817, sents=1 
- ID `195820`: bad_paragraph_len=381, sents=1 
- ID `195771`: same_source_title_duplicate Giá USD tự do tiếp tục lao dốc
- ID `195183`: same_source_title_duplicate Giá vàng nhẫn bật tăng mạnh
- ID `195052`: same_source_title_duplicate Giá vàng nhẫn giảm sâu
- ID `194829`: same_source_title_duplicate Giá vàng nhẫn giảm sâu
- ID `113723`: same_source_title_duplicate Giá USD giảm mạnh
- ID `110895`: same_source_title_duplicate Lãi suất tiết kiệm tiếp tục đi xuống
- ID `111419`: same_source_title_duplicate Lãi suất tiết kiệm tiếp tục đi xuống
- ID `201829`: same_source_title_duplicate Giá xăng trong nước ngày mai có thể quay đầu giảm

**Ghi nhận nhiễu crawl gốc:** Phát hiện 16,008 bài ở `VietOnlineNews` có lỗi dính khoảng trắng CamelCase từ HTML nguồn (ví dụ `phíaUkraine`, `TheoSohu`); đã giữ nguyên tên riêng/thương hiệu để không làm sai lệch chữ viết như `VinFast` và phân tích tác động trong `error_analysis.md`.

## 3. Kiểm tra bảo toàn Số liệu Kinh tế
- Đã đối chiếu **16,360** token số thập phân/phần trăm (dạng `3,2%`, `1.500`, `-2,5%`) trước và sau bước làm sạch trên mẫu kiểm định.
- **Số token bị biến dạng (ví dụ biến `3,2%` thành `32%`):** **`0`** (bảo toàn 100%).

## 4. Kiểm tra Tập Gold, Hậu kiểm chéo & Chứng cứ
- **Tổng ứng viên lấy mẫu:** 280 đoạn (246 `VietOnlineNews` + 34 `UVN-1`).
- **Số mẫu `unclear` sau phân xử 2 vòng:** 2 mẫu (ID: VON_ITEM_120, VON_ITEM_202 — đã loại khỏi đánh giá nhị phân theo guideline).
- **Tổng mẫu Gold nhị phân hợp lệ:** **244** (`VietOnlineNews`) + **34** (`UVN-1` external) = **278** mẫu.
- **Tỷ lệ hậu kiểm chéo (Pass 2 độc lập):** **77 / 280 (27.5%)** (vượt định mức tối thiểu ≥20%).
- **Độ đồng thuận giữa 2 vòng gắn nhãn:** Observed Agreement = **96.10%**, **Cohen's Kappa ($\kappa$) = 0.9211**.
- **Ký xác nhận trực tiếp của Sinh viên (100% tập nhị phân):** Đã rà soát **279 / 279** mẫu qua Tab Audit (`human_audit_signoff.csv`): đồng thuận giữ nguyên **269 mẫu (96.42%)**, chỉnh sửa/loại **10 mẫu** (UVN_EXT_018, UVN_EXT_028, VON_ITEM_020, VON_ITEM_052, VON_ITEM_060, VON_ITEM_065, VON_ITEM_128, VON_ITEM_202, VON_ITEM_213, VON_ITEM_243), **Cohen's Kappa (Sinh viên vs AI v2.0) = 0.9282**.
- **Kiểm chứng câu chứng cứ (`evidence_sentence`):** **100%** (`278/278`) câu chứng cứ là chuỗi con nguyên văn (exact substring) nằm trong đoạn `text`.

### Phân bố Split × Nhãn (`VietOnlineNews` – 244 mẫu)
| split   |   not_uncertain |   uncertain |   All |
|:--------|----------------:|------------:|------:|
| dev     |              27 |          23 |    50 |
| test    |              28 |          32 |    60 |
| train   |              70 |          64 |   134 |
| All     |             125 |         119 |   244 |

### Phân bố Split × Nguồn báo
| split   |   Thanh Niên |   Tuổi Trẻ |   VietnamNet |   All |
|:--------|-------------:|-----------:|-------------:|------:|
| dev     |           16 |         17 |           17 |    50 |
| test    |           20 |         20 |           20 |    60 |
| train   |           45 |         45 |           44 |   134 |
| All     |           81 |         82 |           81 |   244 |

### Phân bố Nhóm biên giới (`boundary_type`) × Nhãn
| boundary_type                  |   not_uncertain |   uncertain |   All |
|:-------------------------------|----------------:|------------:|------:|
| aleatoric_volatility           |               0 |          31 |    31 |
| clear_uncertain                |               0 |          39 |    39 |
| implicit_uncertain             |               0 |          49 |    49 |
| negative_certain_hard_negative |              32 |           0 |    32 |
| neutral_factual                |              55 |           0 |    55 |
| pr_or_positive                 |              38 |           0 |    38 |
| All                            |             125 |         119 |   244 |

### Phân bố Dạng bất định (`uncertainty_type`) × Nhãn (v2.0)
| uncertainty_type   |   not_uncertain |   uncertain |   All |
|:-------------------|----------------:|------------:|------:|
| aleatoric          |               0 |          30 |    30 |
| both               |               0 |          14 |    14 |
| epistemic          |               0 |          75 |    75 |
| none               |             125 |           0 |   125 |
| All                |             125 |         119 |   244 |

> **Nhật ký chỉnh sửa `v1.0 → v2.0` (22/11/2026):** Theo giải đáp của giảng viên hướng dẫn về ví dụ `VON_ITEM_239` (*"Giá vàng hôm nay biến động mạnh..."*), 30 đoạn `VietOnlineNews` và 4 đoạn `UVN-1` phản ánh biến động mạnh/đảo chiều giá vàng, tỷ giá USD, VN-Index, chi phí/cổ phiếu thuộc *bất định ngẫu nhiên* (`aleatoric_volatility`) đã được cập nhật nhãn sang `uncertain`; đồng thời toàn bộ nhãn `uncertain` được gắn phân loại `uncertainty_type` (`epistemic` = 75, `aleatoric` = 30, `both` = 14 trên `VietOnlineNews`).

## 5. Kiểm tra Chống Rò rỉ Dữ liệu (Leakage Prevention)
- **Giao cắt `item_id`, `document_id`, `source_url`, `text_hash` giữa `train`, `dev`, `test`:** **`0`** (đạt tuyệt đối).
- **Phân cụm sự kiện gần trùng (`TF-IDF cosine >= 0.65`):**
- Không phát hiện cặp nào có cosine ≥ 0.65 (tất cả bài lấy mẫu đều thuộc sự kiện độc lập).
- Mọi cụm cùng sự kiện (nếu có) đều được ép nằm trọn trong cùng một split qua `StratifiedGroup` trước khi khóa `test_ids.csv`.

## 6. Pháp lý & Đạo đức Dữ liệu
- `VLUS06/VietOnlineNews` có license `other` (tệp `LICENSE` 0 byte trên HuggingFace), vì vậy toàn văn raw corpus (39.878 bài) được loại trừ qua `.gitignore` và KHÔNG phát tán công khai; chỉ giữ script tải tái lập + manifest + tập đoạn trích đánh giá học thuật.
- `undertheseanlp/UVN-1` dùng theo đúng giấy phép `cc-by-nc-4.0` (phi thương mại, ghi nguồn đầy đủ).
- Không chứa API key hay thông tin định danh cá nhân (PII) trong mã nguồn và dữ liệu.

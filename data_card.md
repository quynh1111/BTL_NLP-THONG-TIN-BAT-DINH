# Data Card – Đề tài 4: Phát hiện Thông tin Bất định Kinh tế trong Tin tức Tiếng Việt

> **Lưu ý tuân thủ Mục 7:** Data Card này mô tả **tập dữ liệu thực tế đã tải, làm sạch, lấy mẫu và gắn nhãn cho dự án**, không sao chép dataset card tổng quát từ Internet.

---

## 1. Dataset Summary & Intended Use
- **Bài toán NLP:** Phân loại nhị phân cấp đoạn văn bản tin tức kinh tế tiếng Việt (`uncertain` vs `not_uncertain`) và trích xuất câu chứng cứ nguyên văn (`evidence_sentence`), áp dụng **Guideline v2.0** với 2 dạng bất định theo định nghĩa chuẩn: **Bất định tri thức (`epistemic`)** và **Bất định ngẫu nhiên (`aleatoric`)**.
- **Mục đích sử dụng:** Huấn luyện, kiểm thử đối chứng các phương pháp phát hiện bất định kinh tế (`E0`, `E0_tfidf`, `E1`, `E2`), đo độ bền xuyên nguồn báo và thống kê tần suất nhắc tới bất định theo thời gian (01/2023–12/2025).

## 2. Dataset Sources & Versions
1. **Nguồn chính:** `VLUS06/VietOnlineNews` — Hugging Face Hub (`https://huggingface.co/datasets/VLUS06/VietOnlineNews`), phiên bản commit SHA `bd8d2a865c8981d34cb9d85f69f62bda7c40b520`, ngày truy xuất `2026-11-22`, gồm 3 đầu báo Việt Nam (`VietnamNet`, `Thanh Niên`, `Tuổi Trẻ`), hai chuyên mục kinh tế `Kinh doanh` và `Bất động sản`.
2. **Nguồn kiểm thử mở rộng ngoài nguồn:** `undertheseanlp/UVN-1` — Hugging Face Hub (`https://huggingface.co/datasets/undertheseanlp/UVN-1`), phiên bản commit SHA `42a81b44e1e8228146e56a3fa066cc035384b7d9`, ngày truy xuất `2026-11-22`, lọc chuyên mục `Kinh doanh` từ 4 báo chưa từng có trong `VietOnlineNews` (`Dân Trí`, `Người Lao Động`, `Tiền Phong`, `VnExpress`).
3. **GDELT DOC 2.0 API (đã kiểm tra & không dùng làm nguồn chính):** Ghi log truy vấn thực tế tại `data/raw/gdelt_probe_log.json` (lần kiểm tra ban đầu gặp `HTTP 429 Too Many Requests`; lần ghi snapshot kế tiếp trả về `HTTP 200` nhưng endpoint `ArtList` chỉ cung cấp metadata/URL chứ không cung cấp toàn văn tiếng Việt hợp pháp), thay thế bằng corpus tin kinh tế theo đúng Mục 4.4.2.

## 3. License & Access
- **`VLUS06/VietOnlineNews`:** Metadata gắn `license: other` (tệp `LICENSE` trống 0 byte). Để tôn trọng bản quyền báo chí gốc, repository **không** lưu hay tái phân phối toàn văn 39.878 bài raw/processed; chỉ lưu `data_manifest.csv`, script tải tái lập và tập đoạn trích Gold phục vụ đánh giá học thuật nội bộ học phần.
- **`undertheseanlp/UVN-1`:** Giấy phép `CC BY-NC 4.0` (chỉ sử dụng phi thương mại cho nghiên cứu học thuật, ghi rõ nguồn gốc).

## 4. Data Structure
- Tập đánh giá gồm `278` đoạn hợp lệ, các trường chính: `item_id`, `document_id`, `source_name`, `source_url`, `source_version`, `retrieved_at`, `published_at`, `original_category`, `title`, `text`, `task_label`, `evidence_sentence`, `uncertainty_type`, `boundary_type`, `uncertainty_dimension`, `split`.
- Toàn bộ tài liệu gốc không chứa thông tin định danh cá nhân (PII) nhạy cảm ngoài tên tác giả/bút danh báo chí công khai trong metadata nguồn.

## 5. Composition & Distributions
- **Corpus mở rộng đã làm sạch (`VietOnlineNews`):** `39,028` bài kinh tế từ tổng số `39,878` bài gốc (`2023-01` → `2025-12`, 36 tháng liên tiếp, 0% thiếu `published_at`). Lưu ý mất cân bằng tự nhiên về độ phủ theo thời gian (sản lượng `Tuổi Trẻ` tăng từ 786 bài năm 2023 lên 5.473 bài năm 2025).
- **Tập Gold chính (`VietOnlineNews`):** **`244` đoạn nhị phân** (từ 246 ứng viên, sau xử lý `2` ca `unclear`), phân bố đều qua 3 báo (`VietnamNet`, `Thanh Niên`, `Tuổi Trẻ`) và 3 năm (`2023`, `2024`, `2025`). Tỷ lệ nhãn `uncertain`: **48.8%** (gồm `75` *bất định tri thức `epistemic`*, `30` *bất định ngẫu nhiên `aleatoric`*, `14` *kết hợp `both`*); số mẫu biên giới *tin xấu chắc chắn* (`negative_certain_hard_negative`): **32 mẫu**.
- **Tập External Unseen-Outlet (`UVN-1`):** **`34` đoạn** (`dantri`, `nguoilaodong`, `tienphong`, `vnexpress`), tỷ lệ `uncertain`: **35.3%**.

## 6. Collection & Sampling
- Không lấy mẫu ngẫu nhiên thuần túy dễ thiếu ca biên giới, mà dùng **lấy mẫu phân tầng có ghi log (`data/sampling_log.csv`)** với seed cố định `42`: cân đối 3 nguồn × 3 năm × 4 tầng (`explicit_epu_cue`, `negative_certain_candidate`, `policy_macro_context`, `general_econ`).
- Mỗi bài báo (`document_id`) chỉ trích tối đa 1 đoạn đại diện (220–950 ký tự, ≥2 câu).

## 7. Preprocessing
- Script tái lập: `src/02_prepare_corpus.py` & `src/text_utils.py`.
- Chuẩn hóa Unicode NFC, giải mã HTML entities, ghép dòng bị ngắt bất thường trong crawl (`Thanh Niên`), loại chú thích ảnh/tin liên quan, loại trùng nội dung (`content_hash`) và tiêu đề cùng nguồn.
- **Bảo toàn số liệu:** Kiểm tra xác nhận `0` lỗi biến dạng trên `16,360` số thập phân/`%` (`3,2%`, `1.500`, `-2,5%`).

## 8. Annotation Process
- Thực hiện theo `data/annotations/annotation_guideline.md` (**phiên bản `v2.0`**, cập nhật ngày 22/11/2026 sau khi xin giải đáp trực tiếp của GVHD ThS. Lê Thị Thùy Trang để bao gồm cả *bất định tri thức* và *bất định ngẫu nhiên* như ví dụ `VON_ITEM_239`, ghi nhận thay đổi ở `reports/data_quality_report.md`).
- Nhãn gốc `Kinh doanh` / `Bất động sản` chỉ giữ ở cột `original_category`, **tuyệt đối không dùng làm nhãn bất định**.
- Mọi ca `unclear` đều được đưa qua Vòng 2 phân xử; các ca sau phân xử vẫn là `unclear` bị loại khỏi tập nhị phân và lưu vết trong `review_log.csv` + `data_manifest.csv`.

## 9. Quality Checks
- **Tỷ lệ hậu kiểm chéo độc lập (Pass 2):** `77/280` mẫu (**27.5%**).
- **Observed Agreement (Pass 1 vs Pass 2):** **96.10%** | **Cohen's Kappa ($\kappa$):** **0.9211**.
- **Ký xác nhận trực tiếp của Sinh viên (100% tập nhị phân):** Đã rà soát **279 / 279** mẫu qua Tab Audit (`human_audit_signoff.csv`): đồng thuận giữ nguyên **269 mẫu (96.42%)**, chỉnh sửa/loại **10 mẫu** (UVN_EXT_018, UVN_EXT_028, VON_ITEM_020, VON_ITEM_052, VON_ITEM_060, VON_ITEM_065, VON_ITEM_128, VON_ITEM_202, VON_ITEM_213, VON_ITEM_243), **Cohen's Kappa (Sinh viên vs AI v2.0) = 0.9282**.
- Toàn bộ ca bất đồng đều có giải trình `disagreement_note` và `adjudicated_label` trong `data/annotations/review_log.csv`.
- **Kiểm tra chứng cứ:** 100% `evidence_sentence` là chuỗi con nguyên văn trong `text`.

## 10. Splits & Leakage Prevention
- Tập chính `244` mẫu được chia bằng thuật toán `StratifiedGroup` theo cụm sự kiện (`TF-IDF cosine >= 0.65`) thành:
  - `train`: **134** mẫu (`data/splits/train_ids.csv`)
  - `dev`: **50** mẫu (`data/splits/dev_ids.csv`)
  - `test`: **60** mẫu (`data/splits/test_ids.csv` — **khóa cứng trước khi chọn mô hình cuối**)
- Tập `external_uvn1`: **34** mẫu (`data/splits/external_uvn1_ids.csv`).
- Xác minh `0` trùng lặp `document_id`, `source_url`, `text_hash`, và `event_cluster` giữa `train`, `dev`, `test`.

## 11. Gold & Evaluation Protocol
- Toàn bộ đánh giá thực nghiệm trên `test` độc lập dùng `src/evaluate_predictions.py --verify`.
- Metric chính: **Macro-F1** và **F1 (`uncertain`)**; metric phụ: Precision/Recall, Confusion Matrix, `Hard-Negative FPR`, `Evidence Exact Match / Token-F1`, tách theo nguồn báo & năm.
- Phần dữ liệu không gắn nhãn (~39.600 bài còn lại của `VietOnlineNews`) chỉ dùng ở Phần mở rộng 2 để suy luận mô tả tần suất bất định theo tháng bằng mô hình đã khóa, **không bao giờ dùng để báo cáo F1 hay tự gán thành gold**.

## 12. Biases, Ethics & Risks
- **Thiên lệch nguồn & thời gian:** `VietOnlineNews` chỉ đại diện cho 3 báo điện tử phổ thông giai đoạn 2023–2025 (không bao gồm tạp chí chuyên sâu như *CafeF*, *VnEconomy*, hay thông cáo NHNN/Bộ Tài chính). Tần suất thu thập thay đổi theo năm (nên bắt buộc phải chuẩn hóa theo tổng số bài $N_{s,t}$).
- **Thiên lệch lấy mẫu có định hướng (Stratified Sampling Bias):** Vì tập Gold chủ động tăng tỷ lệ nhóm biên giới (`explicit_epu_cue` và `negative_certain_candidate`) để thử thách mô hình phân biệt *tin xấu chắc chắn*, tỷ lệ `uncertain` trong tập Gold (~48.8%) **không phản ánh tỷ lệ nền tự nhiên của toàn bộ báo chí Việt Nam**.
- **Minh bạch gắn nhãn AI & Hậu kiểm Con người:** Nhãn khởi tạo & đối soát vòng 2 (`AI_PASS1_DRAFT`, `AI_PASS2_REVIEW`) do trợ lý AI hỗ trợ theo guideline v2.0; toàn bộ 279 mẫu nhị phân đã được **sinh viên trực tiếp kiểm tra và ký xác nhận (`STUDENT_HUMAN_AUDIT`, `human_verified_signed_off`)** lưu tại `data/annotations/human_audit_signoff.csv` (ghi nhận 10 điều chỉnh thủ công ở cấp độ cuối cùng trước khi tái chạy thực nghiệm).

## 13. Limitations & Inappropriate Uses
- Không được sử dụng bộ dữ liệu hoặc tần suất đầu ra của mô hình như **chỉ số EPU chính thức quốc gia**, **tín hiệu giao dịch chứng khoán/tiền tệ**, hoặc **bằng chứng nhân quả** cho biến động GDP/CPI/tỷ giá.
- Độ dài giới hạn ở **cấp đoạn đơn** (220–950 ký tự), nên có thể bỏ sót câu bối cảnh nằm ở đoạn khác xa trong bài báo dài.

## 14. Reproducibility & Access Instructions
- Kích hoạt môi trường ảo `BTL_NPL_TQ` (`.\BTL_NPL_TQ\Scripts\Activate.ps1`).
- Chạy lần lượt:
  1. `python src/01_fetch_data.py` (tải đúng commit SHA đã ghim)
  2. `python src/02_prepare_corpus.py` (làm sạch & kiểm toán số liệu)
  3. `python src/03_sample_for_annotation.py` (`seed=42`)
  4. `python src/04_quality_and_split.py` (tạo manifest, splits, quality report, Data Card)

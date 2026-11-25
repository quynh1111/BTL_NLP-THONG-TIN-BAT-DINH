# CSC4007 – Đề tài 4 (ĐT04): Phát hiện Thông tin Bất định Kinh tế trong Tin tức Tiếng Việt (+ Phần mở rộng)

Dự án thực hiện đầy đủ quy trình NLP có kiểm chứng theo **Hướng dẫn BTL CSC4007 v2.0 (28/10/2026)**, chuẩn **Guideline v2.0 (22/11/2026 theo giải đáp trực tiếp của GVHD ThS. Lê Thị Thùy Trang)** về **2 dạng bất định** (**Bất định tri thức – `epistemic`** và **Bất định ngẫu nhiên – `aleatoric`**), **100% ký xác nhận hậu kiểm trực tiếp bởi sinh viên** trên 279 mẫu nhị phân (`269` đồng thuận = `96.42%`, `10` hiệu chỉnh thủ công, `Cohen's Kappa (Sinh viên vs AI) = 0.9282`), và **nâng cấp Gói 1 (5-Fold OOF CV không rò rỉ, khai thác `title + text`, Dual-Head/Dual-Hypothesis NLI, Supervised Evidence Scorer & Hybrid Stacking `E3`)**.

> **Tệp thiết kế & định nghĩa:** [KE_HOACH_TRIEN_KHAI_DT04.md](KE_HOACH_TRIEN_KHAI_DT04.md) | [annotation_guideline.md](data/annotations/annotation_guideline.md) | [human_audit_signoff.csv](data/annotations/human_audit_signoff.csv)

---

## 1. Tóm tắt Kết quả Gói 1 trên Bộ Gold đã Ký Xác nhận (`v2.0_human_signed`, Test Khóa n = 60)

| Cấu hình | Phương pháp (Gói 1) | OOF-184 Macro-F1 | Dev OOF (50) | Test Macro-F1 | Test F1 (`unc`) | Hard-Neg FPR ↓ | Ev. Token-F1 (`unc`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`E0`** | Luật & Từ điển BBD-VN v2.1 (`Proposal/Draft` + Guard PR) | 0.7862 | 0.8990 | 0.8500 | 0.8475 | **0.000** | 0.783 |
| **`E0_tfidf`** | Multi-Field TF-IDF (`Title+Text`) + Structural + Dual-Head 5-Fold | **0.8804** | **0.9199** | 0.8825 | 0.8923 | **0.000** | 0.749 |
| **`E1`** | Sentence-Pooled `vietnamese-bi-encoder` + Dual-Head + Evidence Scorer | 0.8315 | 0.8397 | 0.8141 | 0.8358 | 0.273 | 0.787 |
| **`E2`** | Dual-Hypothesis NLI (`mDeBERTa-v3` fp32, Bottom-6 Frozen, HN-Loss, 5-Fold) | 0.8307 | 0.8572 | 0.8500 | 0.8525 | 0.091 | 0.766 |
| **`E3`** | **Calibrated Hybrid Stacking Ensemble (`E0+E0_tfidf+E1+E2` + Fused Scorer)** | 0.8795 | 0.8980 | **0.9333** | **0.9355** | **0.000** | **0.823** |

- **Tác động định lượng của Gói 1 (không rò rỉ dữ liệu test):** Mọi siêu tham số, ngưỡng phạt `HN-FPR` (`hn_penalty=0.08`) và trọng số meta `E3` chỉ học trên **5-Fold OOF CV (`n = 184`)** thuộc `train+dev`. Khi chấm độc lập trên tập `test` khóa (`n = 60`):
  - **`E3` (Hybrid Stacking)** đạt **Test Macro-F1 = `0.9333`**, **F1(`uncertain`) = `0.9355`** (`P = 0.9667`, `R = 0.9062`), **`HN-FPR = 0.0000`** (sạch hoàn toàn tin xấu cá biệt đã chốt), **`Evidence Token-F1 = 0.8230`** (chỉ còn 4 lỗi trên 60 mẫu).
  - **`E2` (`mDeBERTa-v3`)** tăng mạnh từ `0.6790 → 0.8500` (+0.1710), `Aleatoric Recall` tăng `57.1% → 85.7%`, `HN-FPR` giảm `54.5% → 9.1%` (chẩn đoán từng fold đơn lẻ trên test: trung bình 5 fold = `0.8332 ± 0.0487`). Zero-shot Dual-Hypothesis trên `dev` đạt `0.5756`.
  - **`E0` v2.1** tăng từ `0.6946 → 0.8500` (`Epistemic Recall` `26.3% → 63.2%`), **`E0_tfidf`** tăng `0.8490 → 0.8825` (`HN-FPR` `18.2% → 0.0%`), và **`E1`** nâng chất lượng trích chứng cứ từ `0.554 → 0.787` (+0.233). Chi tiết xem [error_analysis.md](evaluation/error_analysis.md).
- **Phần mở rộng 1 (Cross-Source Robustness):** Trung bình **LOSO 3-Fold** trên 3 báo (`Thanh Niên / Tuổi Trẻ / VietnamNet`): `E0_tfidf = 0.8598`, **`E3 = 0.8593` (HN-FPR = 0.000 cả 3 báo)**, `E0 = 0.7969`, `E1 = 0.7792`, `E2 = 0.7736`. Trên **Zero-Transfer `UVN-1` (n = 34 từ 4 báo mới)**: **`E0_tfidf = 0.8656`**, **`E3 = 0.8583`** (`Precision = 1.000`, `HN-FPR = 0.000`), `E1 = 0.8464` (`R = 91.7%`, `HN-FPR = 0.000`), `E2 = 0.7984` (`HN-FPR = 0.000`), `E0 = 0.7167`.
- **Phần mở rộng 2 & 3 (39.028 bài, 2023–2025 trên GPU):** Chọn `E0_tfidf` làm chuỗi chính từ OOF-CV (`0.8804`), đối chứng cùng `E0`, `E1` (230.0s GPU), `E2` (973.4s GPU) và `E3` (0.02s). Tương quan chuỗi thô vs chuẩn hóa BBD-style là **`r = -0.2172`**, trong khi tương quan đồng thuận BBD giữa `E2` và các mô hình khác tăng cao lên **`0.7907` (`E0_tfidf`), `0.8124` (`E1`), `0.7922` (`E3`)**. Hai đỉnh nổi bật đồng biến động với **chuỗi 4 đợt NHNN hạ lãi suất điều hành (`03/2023 = 139.12`, window TB `119.36`)**, **sốc tỷ giá USD/VND & giá vàng SJC (`05/2024 = 116.60`, window TB `109.86`)** và **chính sách thuế quan Mỹ (`04/2025 = 110.98`, window TB `100.37`)**.

---

## 2. Thiết lập Môi trường ảo `BTL_NPL_TQ` & Kích hoạt GPU

Dự án dùng môi trường ảo **Python 3.12** đặt tại `c:\BTL_NPL_TQ\BTL_NPL_TQ` với PyTorch CUDA (`cu126`) hỗ trợ GPU **NVIDIA GeForce RTX 4050 (6.44 GB VRAM)**.

### Kích hoạt nhanh trên Windows PowerShell:
```powershell
cd c:\BTL_NPL_TQ
.\BTL_NPL_TQ\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'GPU:', torch.cuda.get_device_name(0))"
```

---

## 3. Hướng dẫn Chạy lại Toàn bộ Pipeline & Kiểm chứng

> **Lưu ý bản quyền dữ liệu:** Tệp raw/processed của `VLUS06/VietOnlineNews` (`license: other`) bị loại trừ qua `.gitignore`. Chạy bước 1 để tải trực tiếp từ Hugging Face theo đúng commit SHA cố định (`bd8d2a865c8981d34cb9d85f69f62bda7c40b520`).

```powershell
# 1. Tải dữ liệu nguồn & kiểm tra GDELT API log
python src/01_fetch_data.py

# 2. Làm sạch, kiểm toán bảo toàn số liệu (0 lỗi trên 16.360 tokens) & tách đoạn
python src/02_prepare_corpus.py

# 3. Tích hợp nhãn đã ký xác nhận bởi sinh viên & giữ nguyên tập test đã khóa
python src/_build_gold_and_review.py

# 4. Kiểm định chất lượng Cổng D1, chống leakage & sinh Data Card + splits
python src/04_quality_and_split.py

# 5. Chạy thực nghiệm Gói 1: E0..E3 (5-Fold OOF CV không rò rỉ) & chấm Test khóa
python src/05_train_evaluate.py

# 6. Chấm lại toàn bộ metric độc lập từ file dự đoán và đối khớp experiments.csv
python src/evaluate_predictions.py --verify

# 7. Chạy Phần mở rộng (LOSO + UVN-1 + quét toàn bộ 39.028 bài trên GPU) & vẽ 5 biểu đồ
python src/06_extension_analysis.py
python src/07_generate_charts.py

# 8. Xuất báo cáo kỹ thuật PDF & kiểm định tổng thể toàn bộ dự án
python src/08_build_pdf_report.py
python src/09_verify_all.py
```

### Khởi chạy Demo tương tác (Streamlit):
```powershell
.\BTL_NPL_TQ\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

---

## 4. Bảng Ánh xạ Vị trí Minh chứng (Mục 10.1) & Bảng kiểm (Mục 10.2)

| Thành phần yêu cầu | Đường dẫn tệp trong Repository | Trạng thái |
|---|---|---|
| Phân tích & Kế hoạch đầu kỳ | `KE_HOACH_TRIEN_KHAI_DT04.md` | Đã hoàn thành |
| Data Card (đủ 14 mục Mục 7) | `data_card.md` | Đã hoàn thành |
| Manifest & Schema dữ liệu | `data/data_manifest.csv`, `data/data_schema.md`, `data/README.md` | Đã hoàn thành |
| Báo cáo kiểm tra dữ liệu | `reports/data_quality_report.md`, `data/raw/gdelt_probe_log.json` | Đã hoàn thành |
| Guideline v2.0, Gold & Ký xác nhận 100% | `data/annotations/annotation_guideline.md`, `gold_labels.csv`, `gold_labels_split.csv`, `review_log.csv`, `human_audit_signoff.csv`, `human_audit_summary.json`, `data/splits/*.csv` | Đã hoàn thành |
| Cấu hình E0–E3 (Gói 1) | `configs/E0.json`, `E0_tfidf.json`, `E1.json`, `E2.json`, `E3.json` | Đã hoàn thành |
| Mã nguồn & Bộ chấm chứng cứ | `src/*.py` (`01..09`, `models_e0..e3.py`, `evidence_scorer.py`, `text_utils.py`, `evaluate_predictions.py`) | Đã hoàn thành |
| Giao thức, Log & Predictions | `evaluation/evaluation_protocol.md`, `experiments.csv`, `oof_cv_summary.csv`, `metrics_summary.csv`, `metrics_by_source_year.csv`, `metrics_by_boundary.csv`, `metrics_by_uncertainty_type.csv`, `e2_seeds_summary.csv`, `test_predictions_*.csv` | Đã hoàn thành |
| Phân tích lỗi (14 mẫu cụ thể) | `evaluation/error_analysis.md` | Đã hoàn thành |
| Minh chứng Phần mở rộng | `evaluation/extension_robustness.csv`, `extension_monthly_index.csv`, `extension_monthly_by_source.csv`, `extension_event_comparison.csv`, `extension_top_evidence_samples.csv`, `data/external_events_reference.csv` | Đã hoàn thành |
| Biểu đồ & Báo cáo PDF | `reports/figures/fig1..fig5.png`, `reports/Bao_cao_BTL.pdf` | Đã hoàn thành |
| Nhật ký & Tuyên bố Minh bạch AI | `reports/Nhat_ky_su_dung_AI.md`, `reports/Tuyen_bo_su_dung_AI.md` | Đã hoàn thành |
| Ứng dụng Demo & Môi trường | `app/streamlit_app.py`, `requirements.txt`, `.gitignore` | Đã hoàn thành |

### Bảng kiểm trước khi nộp (Mục 10.2)
- [x] Dữ liệu đúng ĐT04 theo chuẩn 2 dạng bất định (Tri thức + Ngẫu nhiên), có nguồn/commit SHA/license, không công khai toàn văn có bản quyền (`VietOnlineNews` đã được `.gitignore`).
- [x] `data_card.md` phản ánh đúng dữ liệu thực sử dụng (`v2.0_human_signed`), đủ 14 mục.
- [x] Có `data_quality_report.md`, `data_manifest.csv`, `data_schema.md` và splits chống leakage (`0` rò rỉ).
- [x] Có bộ gold/evidence và guideline v2.0 + **100% ký xác nhận trực tiếp bởi sinh viên** (279/279 mẫu, `Kappa người vs AI = 0.9282`; hậu kiểm chéo Pass 2 mù 27,5% `Kappa = 0.9211`).
- [x] Có `E0`, `E0_tfidf`, `E1`, `E2` và `E3` (Hybrid Stacking) với mã nguồn, config, log OOF-CV không rò rỉ và file prediction riêng.
- [x] Có metric test tính bằng script độc lập (`evaluate_predictions.py --verify`) và phân tích 14 mẫu lỗi thực tế.
- [x] Có Demo Streamlit hỗ trợ `Tiêu đề + Đoạn tin`, hiển thị `Dạng bất định dự đoán` (Tab 1–3) và README để chạy lại/kiểm tra.
- [x] Có báo cáo PDF tiếng Việt, Nhật ký và Tuyên bố sử dụng AI; không lộ API key/PII.

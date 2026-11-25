# Giao thức Đánh giá Thực nghiệm (Evaluation Protocol) – ĐT04

**Mục đích:** Quy định thống nhất các chỉ số đánh giá, quy trình khóa tập `test`, và cách thức tái lập kết quả cho mọi cấu hình (`E0`, `E0_tfidf`, `E1`, `E2`) và phần mở rộng.

## 1. Dữ liệu & Tách tập bất biến
- Tập dữ liệu chính (`VietOnlineNews`, đã loại bỏ các mẫu `unclear` sau phân xử 2 vòng) được tách phân tầng và chống rò rỉ theo bài báo + cụm sự kiện gần trùng (`TF-IDF cosine >= 0.65`) thành:
  - `train`: dùng để huấn luyện mô hình có học (`E0_tfidf`, `E1`, `E2`).
  - `dev`: dùng **duy nhất** cho việc chọn siêu tham số (`C`, `epochs`, ngưỡng xác suất $\tau$, lựa chọn seed tốt nhất cho báo cáo sản phẩm cuối).
  - `test`: **60 mẫu độc lập được khóa cố định tại `data/splits/test_ids.csv`**, không dùng trong bất kỳ bước tinh chỉnh nào; chỉ suy luận một lần sau khi khóa cấu hình.
- Tập mở rộng ngoài nguồn (`external_uvn1`): **34 mẫu** thuộc 4 báo không xuất hiện trong tập train/dev/test (`Dân Trí`, `Người Lao Động`, `Tiền Phong`, `VnExpress`), chỉ dùng để đo Zero-Transfer Cross-Source Robustness ở Phần mở rộng.

## 2. Bộ chỉ số đánh giá (Metrics)
1. **Chỉ số chính (Primary Metrics):**
   - **Macro-F1:** $\frac{1}{2}(F_1(\text{uncertain}) + F_1(\text{not\_uncertain}))$ — tránh thiên lệch nếu phân bố nhãn lệch nhẹ.
   - **F1 lớp `uncertain`:** $\frac{2 \cdot P \cdot R}{P + R}$ — đo trực tiếp năng lực phát hiện đúng tin bất định kinh tế.
2. **Chỉ số phụ (Secondary Metrics):**
   - `Precision (uncertain)`, `Recall (uncertain)`, `Precision (not_uncertain)`, `Recall (not_uncertain)`, `Accuracy` (chỉ tham khảo phụ, không dùng làm metric chính).
   - **Confusion Matrix** (`TP, FP, FN, TN` với lớp dương là `uncertain`).
   - **Hard-Negative False Positive Rate (`HN_FPR`):** Tỷ lệ mẫu thuộc nhóm `negative_certain_hard_negative` (tin xấu đã chốt sổ) bị mô hình dự đoán nhầm là `uncertain`. Đây là thước đo trực tiếp câu hỏi nghiên cứu *"mô hình có phân biệt được tin tiêu cực chắc chắn với bất định hay không"*.
   - **Evidence Sentence Quality (Chất lượng câu chứng cứ):**
     - `Evidence Exact Match`: tỷ lệ mẫu dự đoán `pred_evidence` trùng khớp nguyên văn câu gold (tính trên tập mẫu `uncertain` đúng thực tế hoặc toàn bộ test có `gold_evidence`).
     - `Evidence Token-F1`: độ chồng lấp từ vựng F1 giữa `pred_evidence` và `gold_evidence`.
3. **Phân tích theo lát cắt:**
   - Hiệu năng tách riêng theo từng **nguồn báo** (`VietnamNet`, `Thanh Niên`, `Tuổi Trẻ`), theo **năm công bố** (`2023`, `2024`, `2025`), và theo **2 dạng bất định** (`epistemic`, `aleatoric`, `both`, `none` tại `evaluation/metrics_by_uncertainty_type.csv`) + nhóm biên giới (`evaluation/metrics_by_boundary.csv`).

## 3. Tái lập kết quả từ file dự đoán
Mọi bảng số liệu báo cáo phải được sinh ra tự động bằng lệnh:
```bash
python src/evaluate_predictions.py --verify
```
Script này đọc trực tiếp các tệp `evaluation/test_predictions_*.csv` và `data/annotations/gold_labels.csv` để tính lại toàn bộ chỉ số, bảo đảm không có con số chép tay.

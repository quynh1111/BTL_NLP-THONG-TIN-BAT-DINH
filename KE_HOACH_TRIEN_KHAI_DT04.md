# Phân tích tài liệu & Kế hoạch triển khai Bài tập lớn CSC4007 – Đề tài 4 (Kèm Phần mở rộng)

**Học phần:** CSC4007 – Xử lý ngôn ngữ tự nhiên  
**Chủ đề chung:** Khai thác thông tin kinh tế, tài chính và kinh tế vĩ mô từ văn bản  
**Đề tài đăng ký:** **Đề tài 4 (ĐT04) – Phát hiện thông tin bất định kinh tế trong tin tức tiếng Việt** (kèm toàn bộ Phần mở rộng Mục 4.4.4)  
**Tài liệu căn cứ:** *Hướng dẫn thực hiện bài tập lớn cho sinh viên học phần CSC4007* (ThS. Lê Thị Thùy Trang, 28/10/2026 – bản đề xuất v2.0 năm học 2026–2027)

> **Cập nhật chuẩn định nghĩa nhãn `v2.0` (22/11/2026 — theo giải đáp trực tiếp của GVHD ThS. Lê Thị Thùy Trang):**  
> Khi trao đổi về ví dụ *"Giá vàng hôm nay biến động mạnh, sáng vượt 68 triệu nhưng chiều giảm nhanh mất mốc này."* (`VON_ITEM_239`), giảng viên đã xác nhận bài toán ĐT04 bao gồm **2 dạng bất định**: **(1) Bất định tri thức (`epistemic`)** do thiếu dữ liệu/quyết định chưa ngã ngũ (có thể giảm khi thu thập thêm thông tin) và **(2) Bất định ngẫu nhiên (`aleatoric`)** do tính biến thiên vốn có khó lường của hiện tượng thị trường (như biến động mạnh/đảo chiều giá vàng, tỷ giá, chứng khoán). Theo đó, toàn bộ guideline (`v2.0`), tập Gold, từ điển `E0` (`bbd_vn_lexicon_v2.0`), câu giả thuyết NLI `E2` (`def_nli_v2`), thực nghiệm GPU `E0–E2`, phần mở rộng và báo cáo PDF đều được tái lập đồng bộ theo chuẩn 2 dạng bất định này.

---

## 1. Phân tích yêu cầu đề tài 4 & quy chuẩn nghiệm thu từ tài liệu

### 1.1 Bản chất bài toán NLP & ranh giới bắt buộc
Theo Mục 1.2, Mục 4.4 và Mục 11 của tài liệu hướng dẫn, **ĐT04** giải quyết bài toán phân loại ở cấp độ đoạn/bài tin kinh tế tiếng Việt:
- **Đầu vào:** Một đoạn trích từ bài báo kinh tế – tài chính – bất động sản tiếng Việt (kèm metadata nguồn `source_name`, ngày công bố thực `published_at` nếu có).
- **Đầu ra mong đợi:**
  1. Nhãn phân loại nhị phân: `uncertain` (có thông tin bất định về diễn biến hoặc chính sách kinh tế) hoặc `not_uncertain` (không chứa thông tin bất định). Trong quá trình gắn nhãn cho phép dùng nhãn trung gian `unclear`, nhưng phải có chính sách hậu kiểm/xử lý minh bạch trước khi đưa vào đánh giá.
  2. **Câu/đoạn chứng cứ (`evidence_sentence`)**: câu nguyên văn trong đoạn mang tín hiệu quyết định để người dùng kiểm chứng.
- **Hai nguyên tắc phân tách bắt buộc (tránh lỗi mục 11):**
  1. **Không đồng nhất "tin tiêu cực" với "bất định":** Một tin báo cáo doanh nghiệp lỗ quý trước, bị xử phạt, hoặc chỉ số kinh tế vừa chốt giảm là *tin xấu đã xảy ra/chắc chắn* (`not_uncertain`), khác với *rủi ro, bất định, chưa rõ hướng điều chỉnh chính sách hay triển vọng khó lường* (`uncertain`). Không gộp sentiment 3 lớp và bất định nhị phân thành một nhãn chung.
  2. **Không biến dự án thành dự báo GDP/CPI hoặc tuyên bố EPU chính thức:** Nếu thực hiện thống kê tần suất theo thời gian (demo & phần mở rộng), chỉ được trình bày như *tần suất tin tức đề cập đến bất định trên corpus khảo sát*, không suy luận nhân quả tùy tiện và không quảng bá như chỉ số Economic Policy Uncertainty (EPU) chính thức quốc gia.

### 1.2 Định mức & Ba cổng nghiệm thu bắt buộc (Cổng D1, E, F)

| Cổng nghiệm thu | Yêu cầu tối thiểu theo tài liệu CSC4007 (Mục 4.4, 5, 6, 7, 10, Phụ lục C) | Kế hoạch đáp ứng của dự án |
| :--- | :--- | :--- |
| **Cổng D1 – Data Readiness** | • Nguồn dữ liệu, phiên bản, quyền dùng rõ ràng<br>• `data_manifest.csv` (13 cột chuẩn), `data_schema.md`, `data_card.md` (đủ **14 mục**)<br>• `reports/data_quality_report.md` (bảng kiểm đếm trước/sau, danh sách ID loại/chỉnh)<br>• `annotation_guideline.md`, `gold_labels.csv`, `review_log.csv` (≥20% hậu kiểm chéo)<br>• Khối lượng: **180–240 đoạn** có nhãn kiểm tra; **test ≥50 đoạn** khóa trước; chống leakage theo bài/sự kiện/nguồn | • **240+ đoạn hợp lệ** từ `VietOnlineNews` (lấy mẫu 246 ứng viên để bù trừ các ca `unclear`), trong đó **Test khóa = 60 đoạn (25%)**, Dev = 50, Train = 130+.<br>• Hậu kiểm chéo **≥25%** (khoảng 65 mẫu) + tính Cohen's $\kappa$ & log phân xử bất đồng.<br>• Bổ sung **36 đoạn kiểm thử ngoài nguồn** từ `UVN-1` (4 báo hoàn toàn mới). |
| **Cổng E – Experiment Readiness** | • Tối thiểu 3 cấu hình **E0 (baseline), E1, E2** có giả thuyết khác biệt, đo trên **cùng tập test đã khóa** bằng script<br>• Metric chính: **Macro-F1**, **F1 lớp `uncertain`**; phụ: Precision, Recall, Confusion Matrix, theo nguồn & thời gian<br>• `configs/`, `experiments.csv`, `metrics_summary.csv`, `test_predictions_*.csv` | • Triển khai 4 cấu hình kiểm định giả thuyết tăng tiến: `E0` (Luật BBD-VN), `E0_tfidf` (TF-IDF + LogReg), `E1` (`vietnamese-bi-encoder` frozen + LogReg), `E2` (Salience Context-Window + Definition-Guided NLI `mDeBERTa-v3` đa seed `[42, 123, 2026]`).<br>• Đo thêm **Evidence Exact Match / Token-F1** và **tỷ lệ False Positive trên mẫu hard-negative (tin xấu chắc chắn)**. |
| **Cổng F – Final Acceptance** | • `error_analysis.md` phân tích **≥10 lỗi cụ thể** (7 trường chuẩn Phụ lục B)<br>• Demo chạy lại được (CLI/Streamlit/Gradio) hiển thị tần suất theo nguồn/thời gian + ví dụ chứng cứ + cảnh báo phạm vi<br>• `Bao_cao_BTL.pdf`, `README.md`, `Nhat_ky_su_dung_AI.md`, `Tuyen_bo_su_dung_AI.md`<br>• Không lộ API key/PII, không công khai trái phép toàn văn có bản quyền | • Phân tích **12–15 mẫu lỗi cụ thể** đa dạng nguyên nhân.<br>• Demo **Streamlit** 3 tab (Phân tích đoạn tin + Chứng cứ; Biểu đồ tần suất 36 tháng thô vs chuẩn hóa; Kiểm định & Hậu kiểm Gold).<br>• Báo cáo PDF đầy đủ bằng font Unicode tiếng Việt, biểu đồ chuẩn, kèm bộ hồ sơ minh bạch AI.<br>• Không push toàn văn corpus gốc lên repo (chỉ giữ manifest, script tải tái lập & đoạn trích học thuật). |

### 1.3 Yêu cầu Phần mở rộng (Mục 4.4.4) đã đăng ký thực hiện
1. **Đánh giá Robustness xuyên nguồn báo (Cross-Source Generalization):**
   - Thí nghiệm *Leave-One-Source-Out (LOSO)* giữa 3 đầu báo trong `VietOnlineNews` (`VietnamNet`, `Thanh Niên`, `Tuổi Trẻ`).
   - Thí nghiệm *Zero-Transfer Out-of-Distribution* trên 36 đoạn kinh tế lấy từ `UVN-1` thuộc **4 đầu báo hoàn toàn chưa từng xuất hiện khi huấn luyện** (`Dân Trí`, `Người Lao Động`, `Tiền Phong`, `VnExpress`) ở giai đoạn cuối 2025 – 01/2026.
2. **So sánh chỉ số tần suất tin bất định THÔ vs CHUẨN HÓA theo tổng số tin kinh tế (36 tháng: 01/2023 – 12/2025):**
   - Tính đồng thời số lượng bài bất định thô $U_{s,t}$, tổng số bài kinh tế cùng báo và tháng $N_{s,t}$, tỷ lệ chuẩn hóa $u_{s,t} = U_{s,t} / N_{s,t}$, và chỉ số tổng hợp chuẩn hóa phương sai kiểu Baker–Bloom–Davis (chuẩn hóa độ lệch chuẩn từng báo, quy đổi trung bình 2023–2025 = 100).
   - Chứng minh định lượng vì sao chỉ số thô dễ gây ngộ nhận khi độ phủ crawl biến động (ví dụ trong `VietOnlineNews`, sản lượng tin kinh tế của `Tuổi Trẻ` tăng từ **786 bài năm 2023** lên **5.473 bài năm 2025**, tức gần **7 lần**).
3. **Đối chiếu mô tả với các biến cố kinh tế – chính sách có nguồn công khai:**
   - Gắn mốc thời gian các biến cố lớn tại Việt Nam giai đoạn 2023–2025 (chuỗi điều chỉnh lãi suất điều hành 2023, biến động tỷ giá & thị trường vàng SJC 04–05/2024, bão số 3 Yagi 09/2024, chính sách thuế quan đối ứng của Hoa Kỳ & đàm phán thương mại 04–07/2025) kèm nguồn công bố chính thức; phân tích đồng biến động mô tả, **tuyệt đối không suy luận nhân quả hay dự báo kinh tế**.

---

## 2. Lựa chọn bộ dữ liệu & Lý giải chi tiết

Trước khi chốt thiết kế, chúng tôi đã truy vấn kiểm tra trực tiếp metadata và mẫu thực tế của toàn bộ các nguồn tham khảo được nêu trong Mục 3.3 và Mục 4.4.2 của tài liệu hướng dẫn:

| Nguồn dữ liệu khảo sát | Phiên bản / Quy mô thực tế kiểm tra được | Bản quyền (`License`) | Vai trò trong ĐT04 & Lý do lựa chọn / loại bỏ |
| :--- | :--- | :--- | :--- |
| **`VLUS06/VietOnlineNews`** *(Nguồn chính)* | Commit SHA `bd8d2a865c8981d34cb9d85f69f62bda7c40b520` (265.729 bài). Kết hợp `processed/news_metadata.csv` có **39.878 bài** thuộc `Kinh doanh` (33.242) + `Bất động sản` (6.636), **0% thiếu ngày công bố**, trải dài **36 tháng liên tiếp (01/2023–12/2025)** từ **3 báo**: `VietnamNet` (20.202), `Thanh Niên` (10.161), `Tuổi Trẻ` (9.515). | `other` *(tệp `LICENSE` 0 byte trên repo nguồn)* | **CHỌN LÀM CORPUS CHÍNH.** Lý do: (1) Đúng ngôn ngữ tiếng Việt và đúng miền tin kinh tế – tài chính – bất động sản; (2) Có đầy đủ trường `source` và `public_date` chuẩn xác trong 3 năm liên tục — điều kiện tiên quyết để đánh giá theo nguồn/thời gian và xây chuỗi chuẩn hóa 36 tháng ở Phần mở rộng; (3) **Lưu ý pháp lý:** Vì license `other`, KHÔNG tái phân phối toàn văn 39.878 bài trên GitHub/ZIP, chỉ lưu script tải tái lập, `data_manifest.csv` và các đoạn trích đánh giá phục vụ nghiên cứu học thuật. |
| **`undertheseanlp/UVN-1`** *(Nguồn mở rộng Cross-Source)* | Commit SHA `42a81b44e1e8228146e56a3fa066cc035384b7d9` (3.268 bài). Có **1.113 bài** `Kinh doanh`, trong đó `dantri` chiếm 969 bài (87,1%), `thanhnien` 50, `nguoilaodong` 37, `tienphong` 35, `tuoitre` 16, `vnexpress` 6; riêng tháng `2026-01` chiếm 643/1.113 bài (57,8%). | `cc-by-nc-4.0` | **CHỌN LÀM TẬP KIỂM THỬ MỞ RỘNG NGOÀI NGUỒN (36 đoạn).** Không dùng làm nguồn chính duy nhất vì phân bố lệch nặng sang 1 nguồn (`dantri`) và tập trung dồn cục vào tháng 01/2026 nên không đủ độ phủ thời gian cho chuỗi chỉ số 3 năm. Ngược lại, **4 báo không có trong `VietOnlineNews`** (`Dân Trí`, `Người Lao Động`, `Tiền Phong`, `VnExpress`) là tập kiểm tra *Out-of-Source & Out-of-Time* lý tưởng cho Phần mở rộng. |
| **GDELT DOC 2.0 API** | Kiểm tra ngày 22/11/2026 (`sourcelang:vietnamese`): thử nghiệm ban đầu gặp giới hạn kết nối **HTTP 429: Too Many Requests**, sau đó truy vấn ghi log `data/raw/gdelt_probe_log.json` trả về HTTP 200 nhưng chỉ chứa metadata/URL (`ArtList`). | Điều khoản GDELT (chỉ metadata/URL, không cấp toàn văn hợp pháp) | **KHÔNG DÙNG LÀM NGUỒN VĂN BẢN CHÍNH** (lưu log truy vấn `data/raw/gdelt_probe_log.json` làm minh chứng). Lý do: (1) Không ổn định do rate-limit HTTP 429; (2) Không cung cấp toàn văn bài báo tiếng Việt hợp pháp và chỉ số `tone` của GDELT đo sắc thái cảm xúc chung chứ không phải bất định kinh tế. Áp dụng đúng quy định Mục 4.4.2: *"có thể thay GDELT bằng corpus tin kinh tế nếu khó tải/kết nối"*. |
| **Financial PhraseBank** | ~4.840 câu tin tài chính tiếng Anh có nhãn sentiment 3 lớp. | CC BY-NC-SA 3.0 | **LOẠI BỎ.** Là dữ liệu *sentiment tiếng Anh*, khác ngôn ngữ và khác bản chất nhiệm vụ (tài liệu hướng dẫn cảnh báo rõ ở Mục 3.3 & 4.4.6). |

**Lưu ý cốt lõi về nhãn dữ liệu:** Cả `VietOnlineNews` và `UVN-1` **không hề có nhãn bất định kinh tế** (chỉ có nhãn chuyên mục gốc `Kinh doanh` / `Bất động sản`). Toàn bộ nhãn `uncertain` / `not_uncertain` và `evidence_sentence` đều được xây dựng mới theo `annotation_guideline.md` độc lập trước khi chạy mô hình.

---

## 3. Thiết kế các phương pháp E0 – E2 & Phần mở rộng

Chúng tôi thiết kế **4 cấu hình đối chứng tuyến tính hóa giả thuyết** (gồm Baseline luật kinh điển `E0`, Baseline học máy tuyến tính bổ trợ `E0_tfidf`, Biểu diễn ngữ nghĩa đóng băng `E1`, và Cải tiến chọn văn cảnh + suy luận theo định nghĩa nhãn `E2`):

### 3.1 Chi tiết từng cấu hình thí nghiệm

| Mã | Tên phương pháp | Cơ chế hoạt động & Giả thuyết kiểm chứng | Cơ chế trích `evidence_sentence` |
| :---: | :--- | :--- | :--- |
| **`E0`** | **Baseline Luật / Từ điển BBD-VN** *(Bắt buộc Mục 4.4.3)* | Mô phỏng phương pháp Baker–Bloom–Davis (2016) cho tiếng Việt: phát hiện đồng xuất hiện các nhóm từ khóa **Kinh tế (E)**, **Chính sách/Vĩ mô (P)** và **Bất định (U)**; có thêm quy tắc loại trừ cụm phủ định chắc chắn (*"không còn lo ngại", "đã chính thức ban hành ổn định"*). **Giả thuyết:** Dữ liệu báo chí kinh tế nếu chỉ dựa vào từ khóa bất định sẽ cho Recall khá nhưng Precision thấp vì nhầm lẫn mạnh với các từ rủi ro xuất hiện trong *tin xấu đã chốt sổ*. | Chọn câu có điểm khớp từ khóa $(E + P + U)$ cao nhất trong đoạn. |
| **`E0_tfidf`** | **TF-IDF + Logistic Regression** *(Baseline thống kê bổ trợ)* | Trích đặc trưng `TF-IDF` kết hợp *word n-gram (1,2)* và *char_wb (3,5)* + `LogisticRegression(class_weight='balanced')`, chọn hệ số phạt $C$ và ngưỡng quyết định $\tau$ hoàn toàn trên `dev`. **Giả thuyết:** Học trọng số n-gram từ dữ liệu gắn nhãn giúp giảm bớt từ nhiễu so với luật cứng `E0`, nhưng vẫn yếu trước cách diễn đạt đồng nghĩa hoặc quan hệ điều kiện/thì tương lai phức tạp. | Tách từng câu, tính xác suất dự báo của câu qua chính mô hình `TF-IDF + LogReg`, chọn câu có $P(\text{uncertain})$ cao nhất. |
| **`E1`** | **Vietnamese Sentence Embedding (`vietnamese-bi-encoder`) + Regularized Linear Probe** | Sử dụng mô hình `bkai-foundation-models/vietnamese-bi-encoder` (kiến trúc `PhoBERT-base-v2` được huấn luyện đối chiếu câu tiếng Việt, tách từ bằng `pyvi`) làm bộ trích vector ngữ nghĩa **đóng băng trọng số (frozen)** 768 chiều, huấn luyện bộ phân loại `LogisticRegression` chuẩn hóa $L_2$ (chọn $C$ và $\tau$ trên `dev`). **Giả thuyết:** Với tập train cỡ nhỏ (~130 mẫu), biểu diễn ngữ nghĩa câu tiếng Việt đông cứng giúp khái quát hóa tốt các cách diễn đạt bất định ẩn ý (*"còn bỏ ngỏ", "chờ hướng dẫn thực thi", "đứng trước áp lực biến động"*) mà không bị quá khớp (overfit) tham số. | Mã hóa từng câu trong đoạn thành embedding 768 chiều, chấm điểm trực tiếp bằng đầu phân loại `LogReg` để chọn câu đóng góp lớn nhất cho nhãn dự đoán. |
| **`E2`** | **Salience Context-Window Selection + Definition-Guided NLI (`mDeBERTa-v3-base-mnli-xnli`)** | Hai cải tiến mục tiêu nhắm thẳng vào nhược điểm của `E0`/`E1`:<br>1. **Chọn văn cảnh trọng tâm (Salience Context-Window):** Một đoạn báo dài thường lẫn câu nền tảng hoặc ví dụ nhiễu. Chọn tối đa 3 câu có mật độ tín hiệu kinh tế – chính sách – tình thái cao nhất làm *premise* (giữ nguyên thứ tự xuất hiện).<br>2. **Định nghĩa nhãn tường minh (Definition-Guided NLI):** Chuyển bài toán sang Natural Language Inference trên nền `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (hỗ trợ tiếng Việt từ XNLI), đưa **chính định nghĩa phân tách bất định khỏi tin xấu đã xảy ra** vào *hypothesis*: *"Đoạn tin này đề cập tới sự không chắc chắn, rủi ro chưa ngã ngũ hoặc khả năng thay đổi chính sách và triển vọng kinh tế trong thời gian tới, không chỉ tường thuật sự việc tiêu cực đã hoàn tất."*<br>Fine-tune đa seed (`[42, 123, 2026]`) với class-weighted cross-entropy, early stopping và chọn ngưỡng $\tau$ trên `dev` (đồng thời log thêm bản zero-shot ablation trên `dev` để đối chứng). | Đưa từng cặp `(câu đơn lẻ, hypothesis)` qua chính mô hình `E2` đã fine-tune để lấy câu có điểm suy diễn `entailment` cao nhất làm `evidence_sentence`. |

---

## 4. Trả lời thẳng câu hỏi cốt lõi: Vì sao chọn các phương pháp này — Có gì vượt trội trước khi chạy hay phải chạy rồi mới so sánh?

Đây là điểm rất quan trọng về mặt phương pháp luận khoa học trong môn học này, cần phân tách rạch ròi **hai tầng nhận định**:

### 4.1 Những gì CÓ CƠ SỞ KHOA HỌC ĐỂ KỲ VỌNG trước khi chạy (Lý do lựa chọn)
1. **Tính đối chứng rõ giả thuyết (khớp đúng Mục 6.1):** Mỗi bước từ `E0 → E0_tfidf → E1 → E2` chỉ thay đổi đúng một trục giả thuyết NLP cốt lõi:
   - `E0` kiểm tra *mặt bằng từ vựng tường minh*;
   - `E0_tfidf` kiểm tra *trọng số n-gram thống kê có học*;
   - `E1` kiểm tra *biểu diễn ngữ nghĩa câu tiếng Việt đông cứng* (giải quyết từ đồng nghĩa và giảm nguy cơ overfit trên 130 mẫu train);
   - `E2` kiểm tra *sự kết hợp giữa lọc ngữ cảnh trọng tâm và định nghĩa nhãn tường minh qua NLI* (nhằm tách bạch hai khái niệm rất dễ nhầm lẫn: **"tin xấu chắc chắn"** vs **"bất định tương lai/chính sách"**).
2. **Sự vượt trội về *năng lực biểu diễn* trên lý thuyết:**
   - So với `E0`/`E0_tfidf`, `E1` hiểu được ngữ nghĩa cấp câu nhờ tri thức tiền huấn luyện trên hàng triệu cặp câu tiếng Việt.
   - So với `E1` (vector đóng băng không nhận chỉ dẫn định nghĩa nhãn và nén toàn đoạn thành 1 vector trung bình dễ bị loãng bởi các câu nền), `E2` sử dụng cơ chế cross-attention trực tiếp giữa *đoạn văn chọn lọc* và *câu định nghĩa nhiệm vụ*, cho phép mô hình chú ý vào các dấu hiệu tình thái (*có thể, cân nhắc, chưa rõ, dự kiến*) và phân biệt thì quá khứ hoàn tất với rủi ro mở.
   - Đặc biệt, thay vì dùng LLM gọi API trả phí (không tái lập ổn định, dễ dính lỗi chỉ là wrapper API mà Mục 1.2 nghiêm cấm), cả `E0–E2` đều **tái lập được 100% cục bộ**, minh bạch tham số và tính được xác suất tin cậy có kiểm soát.

### 4.2 Những gì BẮT BUỘC PHẢI CHẠY THỰC NGHIỆM RỒI MỚI ĐƯỢC KẾT LUẬN
**Tuyệt đối không được mặc định trước rằng `E2` chắc chắn thắng `E1` hay `E0_tfidf` trên mọi phương diện.** Thực nghiệm bắt buộc phải chạy để kiểm chứng 4 câu hỏi mở mà lý thuyết không thể trả lời thay:
1. **Đánh đổi Bias–Variance trên tập gold nhỏ (~130 mẫu train):** `mDeBERTa-v3` có ~278 triệu tham số; khi fine-tune trên 130 mẫu rất dễ dao động theo random seed hoặc quá khớp cụm từ đặc thù của tập train. Ngược lại, `E1` (chỉ học ~769 tham số tuyến tính trên embedding đông cứng) tuy biểu diễn thô hơn nhưng có thể **ổn định hơn và tổng quát tốt hơn** khi chuyển sang 4 báo lạ ở tập `UVN-1`!
2. **Hiệu quả thực tế trên nhóm mẫu biên giới (`Hard-Negative`):** Liệu đưa định nghĩa vào hypothesis của `E2` có thực sự kéo giảm tỷ lệ báo động giả (False Positive) trên các tin doanh nghiệp thua lỗ/bị phạt đã chốt sổ so với `E0` và `E1` hay không?
3. **Độ bền xuyên nguồn báo (LOSO & `UVN-1`):** Văn phong báo chí giữa `VietnamNet`, `Thanh Niên`, `Tuổi Trẻ` và `Dân Trí`/`VnExpress` khác nhau về độ dài mở bài và cách đặt tít. Chỉ chạy thực nghiệm đối chứng mới biết mô hình nào giữ được F1 khi đổi nguồn.
4. **Đánh đổi độ chính xác – tốc độ khi quét 39.878 bài kinh tế (Phần mở rộng):** Nếu `E2` chậm hơn `E1` hàng chục lần mà chênh lệch F1 không đáng kể (hoặc ngược lại), thực nghiệm sẽ cung cấp số liệu thời gian suy luận thực tế để quyết định mô hình nào khả thi triển khai thống kê toàn corpus.

Tóm lại: **Lựa chọn kiến trúc `E0–E2` là có cơ sở lý thuyết rõ ràng để giải quyết đúng nút thắt của Đề tài 4, nhưng mô hình nào vượt trội hơn (và vượt trội ở metric nào, với biên độ dao động bao nhiêu) CHỈ ĐƯỢC KẾT LUẬN SAU KHI CHẠY ĐẦY ĐỦ THỰC NGHIỆM TRÊN CÙNG TẬP TEST ĐỘC LẬP.** (Ngay cả kịch bản một cải tiến không vượt baseline vẫn được tài liệu Mục 3.5 & 6.3 công nhận giá trị khoa học nếu quy trình chuẩn và phân tích đúng nguyên nhân).

---

## 5. Môi trường thực thi & Lộ trình tệp tin đầu ra (theo đúng cấu trúc Mục 10.1)

Toàn bộ dự án được triển khai trong môi trường ảo **`BTL_NPL_TQ`** (Python 3.12.10, PyTorch `2.14.1+cu126`, kích hoạt tăng tốc GPU **NVIDIA GeForce RTX 4050 6.0 GB VRAM**, hỗ trợ **BF16** gốc). Toàn bộ mã nguồn, dữ liệu chuẩn hóa, log thí nghiệm, demo và báo cáo sẽ được tổ chức tại `c:\BTL_NPL_TQ\`:
- `KE_HOACH_TRIEN_KHAI_DT04.md` (tệp kế hoạch này)
- `README.md`, `requirements.txt`, `data_card.md` (đủ 14 mục theo Mục 7)
- `data/`: `README.md`, `data_manifest.csv` (13 cột), `data_schema.md`, `sampling_log.csv`, `external_events_reference.csv`, `splits/` (`train_ids.csv`, `dev_ids.csv`, `test_ids.csv`, `external_uvn1_ids.csv`), `annotations/` (`annotation_guideline.md`, `gold_labels.csv`, `review_log.csv`)
- `configs/`: `E0.json`, `E0_tfidf.json`, `E1.json`, `E2.json`
- `src/`: `01_fetch_data.py` → `08_build_pdf_report.py`, `models_e0.py`, `models_e1.py`, `models_e2.py`, `text_utils.py`, `evaluate_predictions.py`
- `evaluation/`: `evaluation_protocol.md`, `experiments.csv`, `metrics_summary.csv`, `metrics_by_source_year.csv`, `metrics_by_boundary.csv`, `test_predictions_*.csv`, `error_analysis.md`, `extension_robustness.csv`, `extension_monthly_index.csv`, `extension_top_evidence_samples.csv`
- `reports/`: `data_quality_report.md`, `Bao_cao_BTL.pdf`, `Nhat_ky_su_dung_AI.md`, `Tuyen_bo_su_dung_AI.md`, `figures/*.png`
- `app/`: `streamlit_app.py` (kèm hướng dẫn chạy & ảnh/log kiểm thử)

> **Minh bạch học thuật trước khi thực thi:** Toàn bộ quá trình tạo nhãn gold khởi tạo và rà soát vòng 2 do trợ lý AI thực hiện theo guideline nghiêm ngặt sẽ được ghi nhận đầy đủ trong `review_log.csv`, `Nhat_ky_su_dung_AI.md` và đi kèm giao diện rà soát trong Streamlit để sinh viên trực tiếp kiểm tra, chỉnh sửa và ký xác nhận trước khi nộp bài chính thức.

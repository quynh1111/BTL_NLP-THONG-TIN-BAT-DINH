# Phân tích Lỗi Chi tiết (Error Analysis – Gói 1 Nâng cấp trên Bộ Gold v2.0 + Human Sign-Off) – Đề tài 4

**Phạm vi:** Phân tích **14 trường hợp lỗi & đối chứng đa dạng** trên tập kiểm thử độc lập `test` đã khóa (**n = 60 mẫu**, sau khi tích hợp **100% ký xác nhận của sinh viên** và nâng cấp Gói 1: `title + text`, 5-Fold OOF CV phạt Hard-Negative, `E0 v2.1`, Multi-field `E0_tfidf`, Sentence-Pooled Dual-Head `E1`, Dual-Hypothesis Bottom-6-Frozen 5-Fold `E2`, Hybrid Stacking `E3` và `SupervisedEvidenceScorer`), trích trực tiếp từ `evaluation/test_predictions_{E0,E0_tfidf,E1,E2,E3}.csv` và `data/annotations/gold_labels_split.csv`.

---

## 1. Bảng so sánh tác động Gói 1 & tổng hợp các nhóm lỗi còn lại trên tập Test (`n = 60`)

- **Kết quả tổng quan trước vs sau Gói 1 trên tập `test` khóa:**
  - `E0` (`v2.0 → v2.1`): Macro-F1 tăng **`0.6946 → 0.8500`** (+0.1554), Epistemic Recall tăng **`26,3% → 63,2%`**, `HN-FPR = 0,000`, `Ev. Token-F1(unc) = 0,7834`.
  - `E0_tfidf`: Macro-F1 tăng **`0.8490 → 0.8825`**, `HN-FPR` giảm từ **`0,1818 → 0,0000`** (sạch hoàn toàn báo động giả Hard-Negative), `Ev. Token-F1(unc) = 0,7490`.
  - `E1`: Macro-F1 giữ **`0.8141`**, `HN-FPR` giảm **`0,3636 → 0,2727`**, và đặc biệt chất lượng trích câu chứng cứ `Ev. Token-F1(unc)` tăng vọt **`0,5536 → 0,7867`** (+0.2331).
  - `E2` (`mDeBERTa-v3`): Macro-F1 tăng mạnh **`0.6790 → 0.8500`** (+0.1710), `Aleatoric Recall` tăng **`57,1% → 85,7%`**, `HN-FPR` giảm **`0,5455 → 0,0909`** (chỉ còn 1/11 ca), `Ev. Token-F1(unc) = 0,7659`.
  - **`E3` (Hybrid Stacking Ensemble):** Đạt **Test Macro-F1 = `0.9333`**, **F1(`uncertain`) = `0.9355`** (`Precision = 0.9667`, `Recall = 0.9062`), **`HN-FPR = 0.0000`**, **`Ev. Token-F1(unc) = 0.8230`** — chỉ còn **4 lỗi nhãn** (`1 FP` + `3 FN`) trên 60 mẫu kiểm thử độc lập.

| Nhóm lỗi còn lại | Số ca minh họa | Mô hình mắc lỗi chủ yếu | Nguyên nhân cốt lõi |
|---|---:|---|---|
| **A. False Negative của `E3` & các nhánh trên Bất định tri thức hàm ẩn / câu hỏi mở** | 4 | `E3` (`028, 046, 061`), `E1`, `E0_tfidf`, `E2` | Khi tin dùng lối đặt câu hỏi gián tiếp (*tăng trưởng tín dụng thấp, doanh nghiệp không muốn vay* ở `VON_ITEM_028`), mâu thuẫn ngữ cảnh thua lỗ quá khứ (*Pacific Airlines* ở `VON_ITEM_046`), hoặc lỗi dính chữ crawl cạnh từ khóa đề xuất (`VON_ITEM_061`), mô hình meta `E3` bị kéo xuống dưới ngưỡng `0.60`. |
| **B. False Positive trên bản tin tổng hợp giá hàng ngày & lãi suất** | 3 | `E0_tfidf`, `E1`, `E2`, `E3` (`034`) | Bản tin điểm giá vàng/USD/dầu dao động nhẹ thông thường (`VON_ITEM_034`: *giảm nhẹ... đi lên*) rất giống cấu trúc bản tin dự báo biến động (`VON_ITEM_020`), gây `1 FP` duy nhất của `E3`; bản tin hạ lãi suất huy động thường nhật (`VON_ITEM_022`) hay công văn đính chính Thông tư (`VON_ITEM_220`) vẫn kích hoạt `E0_tfidf` & `E1`. |
| **C. Hard-Negative & Hỏi đáp thuế cá nhân (`E1` & `E2` còn sót lỗi)** | 4 | `E1` (`073, 104, 121, 241`), `E2` (`121, 241`), `E0` (`029, 134`) | Dù `E0_tfidf` và `E3` đã đạt `0,0%` `HN-FPR`, `E1` vẫn nhầm ở *cảnh báo quảng cáo sai* (`VON_ITEM_073`), *giá bưởi mùa vụ* (`104`), *nhượng quyền Mixue* (`121`) và *hỏi đáp eTax* (`241`); `E2` còn 1 ca Hard-Negative (`VON_ITEM_121`) và 1 ca hỏi đáp cá nhân (`241`). |
| **D. False Negative của `E2` hoặc `E0` trên biến động/dự báo chuyên biệt** | 2 | `E2` (`060, 065, 084, 145`), `E0` (`043, 089`) | Hai ca bị `E2` bỏ sót ở biên giới sốc thương mại (`VON_ITEM_060` nhập khẩu HRC, `084` cổ phiếu bốc hơi) và thuế quan (`145`), tuy nhiên đều được `E3` cứu đúng hoàn toàn nhờ `E0`/`E0_tfidf`/`E1`. `E0` v2.1 vẫn bỏ sót khi *"đề xuất"* đứng quá gần chuỗi số mà thiếu ngữ cảnh rủi ro (`VON_ITEM_043`, `089`). |
| **E. Lệch câu chứng cứ (`pred_evidence`) ở tin nhiều câu tường thuật** | 1 *(+ lồng ghép)* | Cả 5 mô hình (`021, 042, 051, 228`) | Khi câu chứng cứ chuẩn (`gold_evidence`) nằm ở câu kết luận giữa đoạn (`VON_ITEM_021` câu 1, `042` câu 3, `228` câu 2), bộ chọn chứng cứ đôi khi chọn câu dẫn đầu đoạn giàu từ khóa mạnh hơn câu giải thích chi tiết. |

---

## 2. Chi tiết 14 trường hợp lỗi cụ thể trên Test (đủ 7 trường Phụ lục B)

### Ca 1: `VON_ITEM_046` — False Negative của `E3`, `E0_tfidf` và `E1` (Câu hỏi mở tái cơ cấu Pacific Airlines)
- **item_id và loại lỗi:** `VON_ITEM_046` — **False Negative (Epistemic Open Restructuring Question)** của `E0_tfidf`, `E1` và `E3` (`E0` và `E2` dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `Thanh Niên` (2024-03-19) — Tiêu đề: *"Có nên để Pacific Airlines phá sản?"*. Đoạn văn: *"Thua lỗ liên tiếp nhiều năm và phải trả toàn bộ đội máy bay do không thực hiện đúng thỏa thuận và trả tiền thuê, câu hỏi đặt ra với Pacific Airlines - công ty con của Vietnam Airlines là nên tiếp tục cầm cự thua lỗ hay cho phá sản? Pacific Airlines được thành lập năm 1991, là hãng hàng không cổ phần đầu tiên tại Việt Nam."*
- **Gold:** `uncertain` (`epistemic` / `clear_uncertain`). Câu chứng cứ gold: câu 1.
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=uncertain (0.618, ev_ok=True)` *(ĐÚNG)* | `E0_tfidf=not_uncertain (0.303, ev_ok=True)` | `E1=not_uncertain (0.130, ev_ok=True)` | `E2=uncertain (0.645, ev_ok=True)` *(ĐÚNG)* | `E3=not_uncertain (0.329, ev_ok=True)` *(SAI)*.
- **Nguyên nhân khả dĩ:** Nhờ quy tắc dấu hỏi ở tiêu đề + *"câu hỏi đặt ra"* của Gói 1, cả `E0` (`0.618`) và `E2` (`0.645`) đã khắc phục thành công so với bản cũ. Tuy nhiên xác suất quá thấp từ `E1` (`0.130`) và `E0_tfidf` (`0.303`) — bị cụm *"Thua lỗ liên tiếp nhiều năm"* áp đảo như tin Hard-Negative — vẫn kéo điểm meta `E3` (`0.329 < 0.60`) xuống lớp âm.
- **Ảnh hưởng của lỗi đến người dùng:** Là 1 trong 3 ca False Negative duy nhất của `E3` trên toàn tập `test`, bỏ sót bài phân tích ngã rẽ tái cơ cấu hãng hàng không.
- **Đề xuất thay đổi và phép thử kế tiếp:** Bổ sung thêm các ví dụ câu hỏi lựa chọn chính sách (*"Có nên... hay..."*) vào tập huấn luyện `E1`/`E0_tfidf` và gán tương tác *title question mark* vào đặc trưng meta của `E3`.

---

### Ca 2: `VON_ITEM_028` — False Negative của `E3`, `E2`, `E0` (Nút thắt hấp thụ tín dụng hàm ẩn)
- **item_id và loại lỗi:** `VON_ITEM_028` — **False Negative (Implicit Policy Transmission Bottleneck)** của `E0`, `E2`, `E3` (`E0_tfidf` và `E1` dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `VietnamNet` (2023-06-15) — Tiêu đề: *"Hạ lãi suất, giải tỏa nỗi 'sợ sai', để tiền đến đúng nơi cần"*. Đoạn văn: *"Nhiều chính sách hỗ trợ được đưa ra để thúc đẩy nền kinh tế như giảm lãi suất huy động và cho vay, giảm thuế VAT. Nhưng tăng trưởng tín dụng vẫn thấp, doanh nghiệp không muốn vay. Làm sao để dòng tiền chảy đến được nơi cần? Tăng trưởng tín dụng thấp, vốn 'ế' Theo Ngân hàng Nhà nước (NHNN)..."*
- **Gold:** `uncertain` (`epistemic` / `implicit_uncertain`). Câu chứng cứ gold: câu 2 (*"Nhưng tăng trưởng tín dụng vẫn thấp, doanh nghiệp không muốn vay."*).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=not_uncertain (0.746)` | `E0_tfidf=uncertain (0.453)` *(ĐÚNG)* | `E1=uncertain (0.442)` *(ĐÚNG)* | `E2=not_uncertain (0.364)` | `E3=not_uncertain (0.297)`. (Cả 5 lệch câu chứng cứ).
- **Nguyên nhân khả dĩ:** Cả `E0_tfidf` (`0.453`) và `E1` (`0.442`) nhận diện đúng dấu hiệu bế tắc chính sách điều hành tín dụng, nhưng văn phong phân tích không dùng từ khóa `U` tường minh khiến `E2` (`0.364`) và `E0` (chứa cụm phòng ngừa *"lãi suất cho vay"*) bỏ sót, kéo `E3` xuống `0.297`.
- **Ảnh hưởng của lỗi đến người dùng:** Mất tín hiệu các bài phân tích độ trễ/tắc nghẽn truyền dẫn chính sách tiền tệ.
- **Đề xuất thay đổi và phép thử kế tiếp:** Tinh chỉnh danh sách `corporate_result_guard_phrases` để *"lãi suất cho vay"* chỉ chặn khi không đi kèm câu hỏi nghi vấn trong thân bài (*"Làm sao để...?"*).

---

### Ca 3: `VON_ITEM_061` — False Negative của `E3`, `E2`, `E0_tfidf`, `E0` (Tranh luận chính sách thuế chuyển nhượng BĐS + nhiễu dính chữ)
- **item_id và loại lỗi:** `VON_ITEM_061` — **False Negative (Tax Policy Debate + Crawl Gluing Noise)** của `E0`, `E0_tfidf`, `E2`, `E3` (chỉ `E1` dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `VietnamNet` (2025-07-29) — Tiêu đề: *"Bất động sản mua từ 20 năm trước, áp thuế 20% lãi chuyển nhượng là bất hợp lý"*. Đoạn văn: *"...Vì vậy, áp thuế 20% trên chênh lệch mua - bán là không hợp lý và gây thiệt cho người nộp thuế. Đánh thuế phải đúng đối tượng Chia sẻ vớiPV VietNamNet,TS Nguyễn Ngọc Tú... cho biết, đề xuất mứcthuế 20% trên thu nhập chuyển nhượng bất động sảnlà việc trả lại đúng bản chất..."*
- **Gold:** `uncertain` (`epistemic` / `implicit_uncertain`). Câu chứng cứ gold: câu 3.
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=not_uncertain (0.382)` | `E0_tfidf=not_uncertain (0.403)` | `E1=uncertain (0.835, ev_ok=True)` *(ĐÚNG)* | `E2=not_uncertain (0.104)` | `E3=not_uncertain (0.154)` *(tất cả chọn đúng câu 3)*.
- **Nguyên nhân khả dĩ:** Hai nguyên nhân đồng thời xảy ra: (1) Chuỗi dính chữ toàn chữ thường `mứcthuế` (không nằm trong quy tắc CamelCase chữ hoa `aA`) làm `E0` trượt khớp regex `\bđề xuất\b` + `\bthuế\b`, đồng thời chuỗi *"người nộp thuế"* vô tình chạm `corporate_result_guard_phrases` (`"người nộp thuế"`); (2) Văn phong phản biện học thuật *"là không hợp lý"* khiến `E2` (`0.104`) hiểu lầm thành đánh giá khẳng định chứ không phải dự thảo chính sách đang tranh luận.
- **Ảnh hưởng của lỗi đến người dùng:** Đây là ca FN thứ 3 của `E3`, cho thấy `E1` (sentence-pooled) bắt rất tốt ngữ nghĩa tranh luận chính sách (`0.835`) ngay cả khi dính token.
- **Đề xuất thay đổi và phép thử kế tiếp:** Thu hẹp guard phrase từ `"người nộp thuế"` thành `"hỏi đáp thuế"` / `"một người nộp thuế"`, và thêm tách từ dính lowercase-lowercase (`mứcthuế`, `bất động sảnlà`) ở tiền xử lý.

---

### Ca 4: `VON_ITEM_034` — False Positive duy nhất của `E3` (Điểm tin sáng giá vàng/USD/dầu biến động nhẹ)
- **item_id và loại lỗi:** `VON_ITEM_034` — **False Positive (Routine Daily Market Roundup)** của `E0_tfidf`, `E1`, `E2`, `E3` (`E0` dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `VietnamNet` (2023-11-10) — Tiêu đề: *"Bản tin tài chính sáng 10/11/2023: Giá vàng và USD đi xuống, dầu đi lên"*. Đoạn văn: *"Giá vàng hôm nay 10/11/2023 trên thị trường thế giới giảm nhẹ. Giá USD thế giới có chiều hướng đi xuống. Trong khi đó, giá xăng dầu quốc tế tiếp đà đi lên từ phiên trước. Chốt phiên giao dịch 9/11, giá vàng miếng 4 số 9 tại SJC TP.HCM được niêm yết ở mức 69,1 triệu đồng một lượng..."*
- **Gold:** `not_uncertain` (`none` / `neutral_factual`). Câu chứng cứ gold: câu 4.
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=not_uncertain (0.345)` *(ĐÚNG)* | `E0_tfidf=uncertain (0.751)` | `E1=uncertain (0.942)` | `E2=uncertain (0.985)` | `E3=uncertain (0.919)` *(SAI duy nhất HN/Neutral FP của E3)*.
- **Nguyên nhân khả dĩ:** Mẫu *"Bản tin tài chính sáng ...: Giá vàng và USD..."* có cấu trúc câu mở đầu gần như đồng nhất với `VON_ITEM_020` (*"Bản tin tài chính sáng 23/10/2023: Giá vàng và dầu được dự báo tăng..."* là `uncertain`). Vì vậy các mô hình học máy (`E0_tfidf`, `E1`, `E2`) phản ứng mạnh theo mẫu bản tin giá vàng/USD sáng (`0.751–0.985`), trong khi chỉ có `E0` phân biệt đúng *"giảm nhẹ"* không phải biến động mạnh.
- **Ảnh hưởng của lỗi đến người dùng:** Tạo ra ca False Positive duy nhất của `E3` trên toàn bộ tập `test` (`FP = 1 / 28` mẫu âm).
- **Đề xuất thay đổi và phép thử kế tiếp:** Bổ sung thêm cặp bản tin *"Bản tin tài chính sáng"* giữa tin niêm yết *"giảm nhẹ/đi ngang"* (`not_uncertain`) và tin *"được dự báo tăng/khả năng đi lên"* (`uncertain`) vào tập huấn luyện.

---

### Ca 5: `VON_ITEM_121` — False Positive trên Hard-Negative của `E1` & `E2` (Nhượng quyền Mixue lỗ nặng — được `E3` lọc đúng)
- **item_id và loại lỗi:** `VON_ITEM_121` — **False Positive (Hard Negative: Business Franchise Loss)** của `E1` & `E2` (`E0`, `E0_tfidf` và `E3` dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `Tuổi Trẻ` (2023-10-03) — Tiêu đề: *"Toàn cảnh vụ kinh doanh nhượng quyền Mixue gặp khó"*. Đoạn văn: *"Ngoài việc gặp khó với chính sách thay đổi giá đột ngột, nhiều cửa hàng nhượng quyền Mixue cho biết đang 'khó sống' vì chịu sự cạnh tranh lớn... Mixue tại Việt Nam được cho là giảm 25% giá bán sản phẩm nhằm tăng thị phần nhưng chỉ giảm 10% giá nguyên liệu, khiến không ít cửa hàng nhượng quyền bị thua lỗ nặng."*
- **Gold:** `not_uncertain` (`none` / `negative_certain_hard_negative`).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=not_uncertain (0.440)` | `E0_tfidf=not_uncertain (0.288)` | `E1=uncertain (0.427)` *(SAI)* | `E2=uncertain (0.852)` *(SAI)* | `E3=not_uncertain (0.431 < 0.60)` *(ĐÚNG)*.
- **Nguyên nhân khả dĩ:** Đây là ca Hard-Negative **duy nhất** còn sót của `E2` (`HN-FPR = 9,1%`, 1/11). Cụm *"chính sách thay đổi giá đột ngột"* đánh lừa `E2` (`0.852`) tưởng là chính sách kinh tế bất định, dù thực chất là chính sách bán hàng nội bộ của một chuỗi trà sữa dẫn đến thua lỗ cá biệt. Nhờ `E0` và `E0_tfidf` (`0.288`) kết hợp ngưỡng `0.60`, `E3` loại chính xác (`0.431`).
- **Ảnh hưởng của lỗi đến người dùng:** Cho thấy vai trò chốt chặn quan trọng của `E3` giúp đưa `HN-FPR` về `0,0%` khi `E1`/`E2` bị nhiễu bởi từ khóa *"chính sách"* trong doanh nghiệp tư nhân.
- **Đề xuất thay đổi và phép thử kế tiếp:** Phân biệt *"chính sách giá bán doanh nghiệp"* với *"chính sách điều hành vĩ mô/Nhà nước"* trong bộ mẫu huấn luyện Hard-Negative.

---

### Ca 6: `VON_ITEM_073` & `VON_ITEM_104` — Hai ca Hard-Negative còn vướng ở `E1` (Cảnh báo lừa đảo BĐS & Rớt giá bưởi — `E2` và `E3` đã khắc phục triệt để)
- **item_id và loại lỗi:** `VON_ITEM_073` & `VON_ITEM_104` — **False Positive (Hard Negative)** riêng của `E1` (`E0`, `E0_tfidf`, `E2`, `E3` đều ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:**
  - `VON_ITEM_073` (`VietnamNet`, 2025-07-27): *"...Ủy ban Cạnh tranh Quốc gia (Bộ Công Thương) cảnh báo nhiều quảng cáo bất động sản chứa thông tin sai lệch, phóng đại tiện ích, cam kết lợi nhuận phi thực tế, khiến dân dễ rơi vào 'bẫy' kỳ vọng..."*
  - `VON_ITEM_104` (`VietnamNet`, 2024-01-21): *"Giá bưởi tại Đồng bằng sông Cửu Long xuống rất thấp... giá cà chua cũng giảm mạnh, rẻ hơn 10 lần..."*
- **Gold:** `not_uncertain` (`none` / `negative_certain_hard_negative`).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:**
  - `VON_ITEM_073`: `E0=not_unc (0.440)` | `E0_tfidf=not_unc (0.255)` | `E1=unc (0.521)` *(SAI)* | `E2=not_unc (0.010)` | `E3=not_unc (0.056)`.
  - `VON_ITEM_104`: `E0=not_unc (0.050)` | `E0_tfidf=not_unc (0.327)` | `E1=unc (0.452)` *(SAI)* | `E2=not_unc (0.086)` | `E3=not_unc (0.129)`.
- **Nguyên nhân khả dĩ:** Nếu như trước Gói 1 cả `E1`, `E2`, `E0_tfidf` đều báo động giả trên hai ca này, thì nay Dual-Hypothesis + Hard-Negative weighted loss của `E2` đẩy xác suất xuống `0.010–0.086`, giúp `E3` triệt tiêu hoàn toàn. Riêng `E1` vẫn giữ 3/11 lỗi Hard-Negative (`HN-FPR = 27,3%`) do vector Bi-Encoder đóng băng nhạy với từ đơn lẻ (*"cảnh báo"*, *"giảm mạnh"*).
- **Ảnh hưởng của lỗi đến người dùng:** Làm `E1` có Precision thấp hơn (`0.800`) so với `E0_tfidf` (`0.879`), `E2` (`0.897`) và `E3` (`0.967`).
- **Đề xuất thay đổi và phép thử kế tiếp:** Thêm nhánh phạt Hard-Negative trực tiếp vào trọng số mẫu `sample_weight` khi fit `LogisticRegression` của `E1`.

---

### Ca 7: `VON_ITEM_060` — False Negative riêng của `E2` trên Bất định ngẫu nhiên đột biến thương mại (`E0`, `E0_tfidf`, `E1`, `E3` đúng)
- **item_id và loại lỗi:** `VON_ITEM_060` — **False Negative (Aleatoric Import Surge)** riêng của `E2`.
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `Thanh Niên` (2025-07-17) — Tiêu đề: *"Thép cán nóng khổ rộng nhập khẩu tăng 26 lần"*. Đoạn văn: *"Lượng thép cuộn cán nóng (HRC) khổ rộng xuất xứ Trung Quốc nhập khẩu vào VN gia tăng bất thường kể từ khi có công bố áp thuế chống bán phá giá sản phẩm này. Ngân sách thất thu 2.300 tỉ đồng sau 6 tháng..."*
- **Gold:** `uncertain` (`aleatoric` / `aleatoric_volatility`). Câu chứng cứ gold: câu 1.
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=uncertain (0.843, ev_ok=True)` | `E0_tfidf=uncertain (0.577, ev_ok=True)` | `E1=uncertain (0.564, ev_ok=True)` | `E2=not_uncertain (0.173, ev_ok=True)` *(SAI)* | `E3=uncertain (0.619, ev_ok=True)` *(ĐÚNG)*.
- **Nguyên nhân khả dĩ:** Mặc dù Gói 1 giúp `E2` đạt `85,7%` Recall trên `aleatoric` (cứu được cả `VON_ITEM_228` với `p=0.986` và `VON_ITEM_151` với `p>0.525`), mẫu đột biến lượng nhập khẩu thép 26 lần (`060`) không nằm trong danh sách ví dụ giá vàng/tỷ giá/chứng khoán nên `H_aleatoric` chỉ đạt `0.173`.
- **Ảnh hưởng của lỗi đến người dùng:** Giảm nhẹ Recall ngẫu nhiên của `E2` đơn lẻ, nhưng được `E0 v2.1` (`gia tăng bất thường`), `E0_tfidf` và `E1` bảo vệ trong `E3`.
- **Đề xuất thay đổi và phép thử kế tiếp:** Mở rộng `H_aleatoric` bao gồm biến động đột biến cung-cầu/thương mại xuất nhập khẩu bên cạnh thị trường tài chính.

---

### Ca 8: `VON_ITEM_084` & `VON_ITEM_145` — False Negative của `E2` quanh cú sốc thuế quan Mỹ 46%
- **item_id và loại lỗi:** `VON_ITEM_084` (`both`) & `VON_ITEM_145` (`epistemic`) — **False Negative** riêng của `E2` (`E0`, `E0_tfidf`, `E1`, `E3` đều ĐÚNG trên `084` và `145` trừ `E1` `0.430` sát ngưỡng ở `084` nhưng `E1_thr=0.425` vẫn ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:**
  - `VON_ITEM_084` (`VietnamNet`, 2025-04-05): *"Tỷ phú Jensen Huang ra khỏi câu lạc bộ 100 tỷ USD — Đợt bán tháo cổ phiếu khiến tài sản của nhiều tỷ phú bốc hơi hàng tỷ USD từ đầu năm... Các chính sách thuế quan mạnh tay của Tổng thống Mỹ Donald Trump tác động tiêu cực..."*
  - `VON_ITEM_145` (`Tuổi Trẻ`, 2025-04-03): *"Mỹ áp thuế 46%, ngành nào của Việt Nam sẽ chịu tổn thương nhất? — ...Các doanh nghiệp Việt Nam đối mặt với thách thức lớn khi Mỹ áp thuế 46% lên hàng hóa xuất khẩu, gây lo ngại..."*
- **Gold:** `uncertain`.
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:**
  - `VON_ITEM_084`: `E0=unc (0.943)` | `E0_tfidf=unc (0.718)` | `E1=unc (0.430)` | `E2=not_unc (0.487)` *(hụt sát `thr=0.525`)* | `E3=unc (0.844)`.
  - `VON_ITEM_145`: `E0=unc (0.917)` | `E0_tfidf=unc (0.771)` | `E1=unc (0.890)` | `E2=not_unc (0.402)` *(SAI)* | `E3=unc (0.917)`.
- **Nguyên nhân khả dĩ:** Cả hai bài đều bị mô hình NLI `E2` chấm cận ngưỡng (`0.402–0.487`) do cụm mở đầu nhấn mạnh sự kiện đã xảy ra (*"ra khỏi câu lạc bộ 100 tỷ USD"*, *"Mỹ áp thuế 46%"*), khiến `H_epistemic` phân vân giữa đã chốt sự kiện và lo ngại hệ lụy. `E3` tận dụng đồng thuận cao từ `E0`, `E0_tfidf`, `E1` nên đạt `0.844–0.917`.
- **Ảnh hưởng của lỗi đến người dùng:** Rơi rớt các bài bình luận cú sốc thuế quan cấp tính nếu chỉ dùng `E2` đơn lẻ.
- **Đề xuất thay đổi và phép thử kế tiếp:** Cho phép `E2` áp dụng `H_epistemic` nhấn mạnh cả *lo ngại hệ lụy chưa định lượng được* khi chính sách vừa công bố.

---

### Ca 9: `VON_ITEM_065` — False Negative của `E0` & `E2` trên Dự báo giá vàng 2.700 USD/ounce
- **item_id và loại lỗi:** `VON_ITEM_065` — **False Negative (Commodity Forecast Uncertainty)** của `E0` & `E2` (`E0_tfidf`, `E1`, `E3` dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `Thanh Niên` (2024-08-05) — Tiêu đề: *"Giá vàng dự báo tăng lên 2.700 USD/ounce"*. Đoạn văn: *"Ông Heng Koon How... cho biết, theo dự báo của UOB, giá vàng sẽ tăng lên 2.500 USD/ounce vào cuối năm 2024 và 2.700 USD/ounce vào giữa năm 2025..."*
- **Gold:** `uncertain` (`epistemic` / `implicit_uncertain`). Câu chứng cứ gold: câu 1.
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=not_uncertain (0.540, ev_ok=False)` | `E0_tfidf=uncertain (0.633, ev_ok=True)` | `E1=uncertain (0.912, ev_ok=True)` | `E2=not_uncertain (0.122, ev_ok=True)` | `E3=uncertain (0.712, ev_ok=True)`.
- **Nguyên nhân khả dĩ:** `E0` thiếu từ *"dự báo"* trong `proposal_draft_keywords` (`VON_ITEM_020` *"được dự báo tăng"* cũng trượt `E0` vì lý do tương tự). `E2` hiểu nhầm *"sẽ tăng lên..."* là khẳng định xu hướng xác định thay vì *kịch bản dự báo tri thức tương lai chưa chắc chắn*. `E1` sửa lỗi câu chứng cứ trước đây và chấm chính xác (`0.912`, `ev_ok=True`).
- **Ảnh hưởng của lỗi đến người dùng:** `E0` và `E2` bỏ sót các dự báo giá hàng hóa trung hạn; `E3` vẫn bắt trúng.
- **Đề xuất thay đổi và phép thử kế tiếp:** Thêm *"dự báo"* vào `uncertainty_keywords` hoặc `proposal_draft_keywords` đi cùng danh từ thị trường (`market_asset_keywords`).

---

### Ca 10: `VON_ITEM_043` & `VON_ITEM_089` — False Negative riêng của `E0` trên Dự thảo Nghị định / Luật Điện lực
- **item_id và loại lỗi:** `VON_ITEM_043` & `VON_ITEM_089` — **False Negative (Rule Dictionary Gap)** riêng của `E0` (`E0_tfidf`, `E1`, `E2`, `E3` đều dự đoán ĐÚNG với xác suất `0.75–0.99`).
- **Đầu vào / nguồn / chứng cứ hợp lệ:**
  - `VON_ITEM_043` (`VietnamNet`, 2025-10-22): *"Bộ Công Thương đề xuất người dân lắp đặt điện mặt trời mái nhà buộc thông báo... Bộ Công Thương đang lấy ý kiến dự thảo Nghị định sửa đổi..."*
  - `VON_ITEM_089` (`VietnamNet`, 2024-06-18): *"Bộ Công Thương vừa gửi Bộ Tư pháp hồ sơ dự thảo Luật Điện lực (sửa đổi)..."*
- **Gold:** `uncertain` (`epistemic` / `implicit_uncertain`).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:**
  - `VON_ITEM_043`: `E0=not_unc (0.050)` *(SAI)* | `E0_tfidf=0.919` | `E1=0.877` | `E2=0.997` | `E3=0.983`.
  - `VON_ITEM_089`: `E0=not_unc (0.050)` *(SAI)* | `E0_tfidf=0.768` | `E1=0.959` | `E2=0.994` | `E3=0.961`.
- **Nguyên nhân khả dĩ:** Luật của `E0` bắt buộc có ít nhất 1 từ khóa trong `economy_keywords` (`min_e_hits = 1`). Hai bài ngành điện lực dùng *"điện mặt trời"*, *"điện gió"*, *"điện tái tạo"* nhưng thiếu đúng từ gốc trong danh sách 46 từ `economy_keywords` nên bị đánh `e_hits = 0 → prob = 0.050` dù có đủ *"Bộ Công Thương"*, *"dự thảo"*, *"lấy ý kiến"*.
- **Ảnh hưởng của lỗi đến người dùng:** Làm `E0` tuy tăng Epistemic Recall lên `63,2%` vẫn mất điểm ở các văn bản dự thảo ngành điện/năng lượng.
- **Đề xuất thay đổi và phép thử kế tiếp:** Bổ sung nhóm từ lĩnh vực kinh tế-hạ tầng (*"điện lực"*, *"năng lượng"*, *"mua bán điện"*) vào `economy_keywords`.

---

### Ca 11: `VON_ITEM_241` — False Positive của `E1` & `E2` trên Thắc mắc cá nhân thuế TNCN (được `E3` loại đúng)
- **item_id và loại lỗi:** `VON_ITEM_241` — **False Positive (Personal Taxpayer Question)** của `E1` & `E2` (`E0`, `E0_tfidf`, `E3` dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:** `VietnamNet` (2025-02-12) — *"Thấy thông báo nợ thuế từ nhiều năm trước trên eTax Mobile, một người nộp thuế hoang mang không rõ... Trong số những câu hỏi liên quan tới quyết toán thuế thu nhập cá nhân (TNCN) gửi tới Bộ Tài chính mới đây..."*
- **Gold:** `not_uncertain` (`none` / `neutral_factual`).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** `E0=not_unc (0.310)` | `E0_tfidf=not_unc (0.340)` | `E1=uncertain (0.591)` *(SAI)* | `E2=uncertain (0.720)` *(SAI)* | `E3=not_uncertain (0.248)` *(ĐÚNG)*.
- **Nguyên nhân khả dĩ:** `E0` v2.1 kích hoạt `corporate_result_guard_phrases` (`"người nộp thuế"`), cùng `E0_tfidf` (`0.340`) loại thành công, kéo `E3` về đúng `0.248`; trong khi cả `E1` và `E2` vẫn nhầm *"hoang mang không rõ"* của một cá nhân sang *"bất định tri thức"* (`E2` đã giảm từ `0.911 → 0.720` nhưng chưa qua `thr=0.525`).
- **Ảnh hưởng của lỗi đến người dùng:** Cho thấy lợi ích của việc `E3` kết hợp tín hiệu guard của `E0` với xác suất liên mô hình.
- **Đề xuất thay đổi và phép thử kế tiếp:** Bổ sung trực tiếp trong `hypothesis_epistemic` của `E2`: *"cấp độ chính sách/vĩ mô/thị trường, không phải thắc mắc thủ tục của một cá nhân"*.

---

### Ca 12: `VON_ITEM_022` & `VON_ITEM_220` — False Positive của `E0_tfidf` & `E1` trên tin bình ổn lãi suất & đính chính thông tư (`E0`, `E2`, `E3` đúng)
- **item_id và loại lỗi:** `VON_ITEM_022` & `VON_ITEM_220` — **False Positive (Neutral Factual Banking / Technical Erratum)** của `E0_tfidf` & `E1` (`E0`, `E2`, `E3` đều dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:**
  - `VON_ITEM_022` (`VietnamNet`, 2023-08-22): *"Một loạt các ngân hàng vừa hạ lãi suất trong ngày hôm nay. SCB và GPBank giảm tới 0,5 điểm phần trăm lãi suất các kỳ hạn tiền gửi..."*
  - `VON_ITEM_220` (`Thanh Niên`, 2024-08-30): *"Bộ Tài chính ban hành Công văn số 9166 đính chính Thông tư 46/2024... phát hiện lỗi kỹ thuật tại phần căn cứ pháp lý..."*
- **Gold:** `not_uncertain` (`none` / `neutral_factual`).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:**
  - `VON_ITEM_022`: `E0=0.411 (not_unc)` | `E0_tfidf=0.481 (unc)` *(SAI)* | `E1=0.428 (unc)` *(SAI)* | `E2=0.343 (not_unc)` | `E3=0.251 (not_unc)`.
  - `VON_ITEM_220`: `E0=0.897 (not_unc)` | `E0_tfidf=0.664 (unc)` *(SAI)* | `E1=0.670 (unc)` *(SAI)* | `E2=0.157 (not_unc)` | `E3=0.508 (not_unc < 0.60)`.
- **Nguyên nhân khả dĩ:** N-gram của `E0_tfidf` và embedding của `E1` nhạy với từ *"hạ lãi suất / giảm mạnh"* (`022`) hoặc chuỗi pháp quy *"Bộ Tài chính / Thông tư / sửa đổi, bổ sung"* (`220`), trong khi `E2` (Cross-Encoder NLI) hiểu ngữ cảnh đây chỉ là niêm yết biểu lãi suất và đính chính lỗi chính tả kỹ thuật (`0.157–0.343`), giúp `E3` giữ chính xác lớp âm.
- **Ảnh hưởng của lỗi đến người dùng:** Chứng minh hai hướng bổ trợ ngược chiều: `E0_tfidf` cứu `E2` ở ca đột biến (`060, 084, 145`), còn `E2` cứu `E0_tfidf` & `E1` ở ca ngữ cảnh hành chính/bình ổn (`022, 220`).
- **Đề xuất thay đổi và phép thử kế tiếp:** Khai thác chính đặc trưng đối chiếu ngữ cảnh công văn đính chính (`"đính chính"`, `"lỗi kỹ thuật"`) trong bộ lọc ngữ cảnh.

---

### Ca 13: `VON_ITEM_029` & `VON_ITEM_134` — False Positive riêng của `E0` trên tin PR kết quả kinh doanh ngân hàng (`E0_tfidf`, `E1`, `E2`, `E3` đúng)
- **item_id và loại lỗi:** `VON_ITEM_029` & `VON_ITEM_134` — **False Positive (Corporate Earnings PR)** riêng của `E0` (`E0_tfidf`, `E1`, `E2`, `E3` đều dự đoán ĐÚNG).
- **Đầu vào / nguồn / chứng cứ hợp lệ:**
  - `VON_ITEM_029` (`VietnamNet`, 2024-04-02): *"VIB thông qua kế hoạch chia cổ tức 29,5% — ...lợi nhuận trước thuế 12.045 tỷ năm 2024... kiểm soát rủi ro chặt chẽ."*
  - `VON_ITEM_134` (`VietnamNet`, 2024-07-23): *"LPBank là ngân hàng đầu tiên công bố kết quả kinh doanh quý II/2024 với lợi nhuận trước thuế tăng mạnh 142%..."*
- **Gold:** `not_uncertain` (`none` / `pr_or_positive`).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:**
  - `VON_ITEM_029`: `E0=unc (0.480)` *(SAI)* | `E0_tfidf=0.375` | `E1=0.064` | `E2=0.023` | `E3=0.136`.
  - `VON_ITEM_134`: `E0=unc (0.382)` *(SAI)* | `E0_tfidf=0.395` | `E1=0.124` | `E2=0.017` | `E3=0.089`.
- **Nguyên nhân khả dĩ:** `E0` v2.1 có `corporate_result_guard_phrases` chặn nhánh `epistemic_hit` (bài `VIB`) và `aleatoric_hit` trên `029`/`134`, NHƯNG chưa chặn điều kiện đầu tiên `e_hits >= 1 and u_hits >= 1 and (p_hits >= 1 or u_hits >= 2)`: cả hai bài đều chứa từ *"rủi ro"* (`kiểm soát rủi ro chặt chẽ`, `dự phòng rủi ro tín dụng`) kết hợp *"kế hoạch"* / *"ngân hàng"*, đi lọt nhánh `u_hits >= 1` (`029` thậm chí lọt cả `029` vì `e_hits>=1, u_hits=1("rủi ro"), p_hits>=1`).
- **Ảnh hưởng của lỗi đến người dùng:** Đây là 2 trong 3 ca False Positive duy nhất của `E0` (`pr_or_positive = 75%`), tuy xác suất chỉ `0.382–0.480` nên không ảnh hưởng đến `E3`.
- **Đề xuất thay đổi và phép thử kế tiếp:** Loại trừ cụm danh từ ghép *"kiểm soát rủi ro"* và *"dự phòng rủi ro tín dụng"* khỏi quy tắc đếm `u_hits` trong `E0`.

---

### Ca 14: `VON_ITEM_021`, `VON_ITEM_042` & `VON_ITEM_228` — Lệch câu chứng cứ (`pred_evidence`) dù cả 5 mô hình dự đoán ĐÚNG nhãn
- **item_id và loại lỗi:** `VON_ITEM_021`, `VON_ITEM_042`, `VON_ITEM_228` — **Evidence Selection Mismatch** (`E3` Ev. Token-F1 = `0.8230` trên toàn `test`, nhưng vẫn lệch đúng câu ở 3 đoạn đa câu này).
- **Đầu vào / nguồn / chứng cứ hợp lệ:**
  - `VON_ITEM_021` (`Thanh Niên`, 2024-02-24): *Câu 1 (Gold): "Sau nhiều tháng, tình trạng giá vàng trong nước 'một mình một chợ' vẫn chưa có giải pháp nào căn cơ để giải quyết." | Câu 2 (E3 chọn): "Giá trồi sụt bất thường nên chỉ vài ngày sau ngày vía Thần tài..." | Câu 3 (E0–E2 chọn): "...vàng miếng SJC lại đột ngột quay đầu giảm mạnh..."*
  - `VON_ITEM_042` (`Thanh Niên`, 2024-12-18): *Câu 1 (E0_tfidf–E3 chọn): "Đó là một trong những đề xuất của Bộ Công thương trong dự thảo nghị định..." | Câu 3 (Gold & E0 chọn): "Bộ Công thương đang lấy ý kiến dự thảo nghị định quy định chi tiết..."*
  - `VON_ITEM_228` (`Thanh Niên`, 2025-04-03): *Câu 1 (E0_tfidf–E3 chọn): "Giá vàng thế giới tiếp tục trượt giảm mạnh trong phiên giao dịch Mỹ (đêm 3.4), nâng mức giá bị thổi bay trong ngày lên hơn 80 USD/ounce..." | Câu 2 (Gold & E0 chọn): "Mở cửa thị trường vàng phiên Mỹ ngày 3.4..., giá vàng thế giới đã trượt thẳng đứng, mất 30 USD/ounce chỉ trong vòng 30 phút..."*
- **Gold:** Đều là `uncertain` (`both` / `epistemic` / `aleatoric`).
- **Prediction E0 / E0_tfidf / E1 / E2 / E3:** Cả 5 mô hình đều phân loại ĐÚNG `uncertain` với xác suất rất cao (`0.686 – 0.997`), riêng `pred_evidence` khác câu gold nhưng cùng truyền tải trực tiếp nội dung bất định trong đoạn.
- **Nguyên nhân khả dĩ:** `SupervisedEvidenceScorer` học đặc trưng `is_first = 1` khá mạnh vì đa số câu tóm tắt tin tức nằm ở câu 1; khi câu 1 chứa đầy đủ từ khóa bất định (*trượt giảm mạnh*, *đề xuất dự thảo*) không kém câu 2–3, bộ chấm điểm ưu tiên câu 1 thay vì câu chi tiết ở giữa đoạn.
- **Ảnh hưởng của lỗi đến người dùng:** Ảnh hưởng nhẹ đến `Evidence Exact Match` khi so sánh máy móc theo 1 câu đơn, dù về mặt đọc hiểu cho người dùng cả hai câu đều là chứng cứ hợp lệ.
- **Đề xuất thay đổi và phép thử kế tiếp:** Đánh giá thêm chỉ số *Top-2 Evidence Recall* hoặc cho phép tô sáng đồng thời tối đa 2 câu chứng cứ vượt ngưỡng khi đoạn chứa `uncertainty_type == "both"` (như `VON_ITEM_021`).

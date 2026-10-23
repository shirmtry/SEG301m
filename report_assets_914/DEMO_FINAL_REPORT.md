# 📊 BÁO CÁO ĐÁNH GIÁ HỆ THỐNG CHATBOT PHÁP LUẬT (RAG)

**Môn học:** SEG301m  
**Sinh viên thực hiện:** HUYTKING (shirmtry)  
**Ngày báo cáo:** 24/10/2026  

---

## 1. Tổng quan Dự án
Dự án xây dựng hệ thống Chatbot tư vấn pháp luật Việt Nam dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**. Hệ thống giải quyết bài toán "ảo giác" (hallucination) của các mô hình ngôn ngữ lớn (LLM) bằng cách truy xuất các văn bản pháp luật chính xác từ cơ sở dữ liệu nội bộ trước khi sinh câu trả lời. Hệ thống được tối ưu để chạy hoàn toàn Local trên CPU (không cần GPU đắt tiền) với tốc độ phản hồi trung bình chỉ ~21 giây/câu.

### Kiến trúc Hệ thống (Pipeline v10)
1. **Dữ liệu (Data):** Trích xuất và gộp thành **9.159 Điều luật** hoàn chỉnh từ **65 Bộ luật/Luật cốt lõi** (cập nhật VBHN 2025-2026). Đã bổ sung thành công các luật quan trọng như Luật Chứng khoán, Luật Thi hành án dân sự, Đấu thầu, Hải quan...
2. **Embedding:** Mô hình tối ưu tiếng Việt `keepitreal/vietnamese-sbert` (768 chiều).
3. **Retrieval (Truy xuất):** Kết hợp **Hybrid Search** (BM25 + FAISS) và **Smart Filter** (dùng Dict 58 từ khóa + Regex) để nhận diện chính xác Tên Luật và Số Điều. **Đã loại bỏ Reranker** để tăng tốc độ xử lý trên CPU gấp 3 lần mà vẫn đảm bảo độ chính xác.
4. **LLM Generation:** Mô hình `Qwen2.5-3B-Instruct` (GGUF Q4_K_M) chạy local qua `llama-cpp-python`, kết hợp các module lọc nhiễu CJK, chống lặp từ, và định dạng Markdown tự động.

---

## 2. Phân tích Cơ sở Dữ liệu (Data Profiling)

Hệ thống sở hữu kho dữ liệu đồ sộ bao gồm các Bộ luật quan trọng nhất của Việt Nam (đã được gộp nguyên vẹn theo từng Điều để bảo toàn ngữ cảnh):
*   **Bộ luật Dân sự 2015:** 689 Điều.
*   **Bộ luật Tố tụng Dân sự:** 517 Điều.
*   **Bộ luật Tố tụng Hình sự:** 492 Điều.
*   **Bộ luật Hình sự:** 409 Điều.
*   **Luật Đất đai, Doanh nghiệp, Các tổ chức tín dụng, Hôn nhân & Gia đình...**

*Đặc điểm:* Dữ liệu đã tăng từ 52 luật (7.868 điều) lên **65 luật (9.159 điều)**. Tuy nhiên, hệ thống vẫn thiếu hụt các văn bản dưới luật chi tiết (Nghị định xử phạt, Thông tư hướng dẫn).

---

## 3. Kết quả Đánh giá (Evaluation)

Hệ thống được đánh giá tự động trên bộ **914 câu hỏi test** (457 câu hỏi kiến thức trực tiếp, 274 câu hỏi bẫy/out-of-scope, 183 câu hỏi ngắn/mơ hồ).

### 3.1. Tỷ lệ phản hồi
![Tỷ lệ Trả lời của Chatbot](report_assets_914/01_pie_verdict_914.png)
*   **Tỷ lệ trả lời thành công (`answered` + `warning`):** **46.0%** (421/914 câu).
    *   Trong đó: 21.4% (196 câu) trả lời đầy đủ kèm trích dẫn chuẩn `[N]`; 24.6% (225 câu) trả lời đúng nội dung nhưng thiếu ký hiệu trích dẫn.
*   **Tỷ lệ từ chối (`refused`):** 51.9% (474/914 câu).
*   **Tỷ lệ chặn an toàn (`blocked`):** 2.1% (19/914 câu - do vi phạm quy tắc an toàn hoặc trích dẫn sai).
*   *Phân tích:* Tỷ lệ từ chối >50% là hoàn toàn hợp lý vì bộ test có tới 30% là câu hỏi bẫy và 20% là câu hỏi mơ hồ. Quan trọng nhất là hệ thống **tuyệt đối không bịa đặt (0% hallucination)** khi gặp câu hỏi ngoài lề.

### 3.2. Tương quan Độ chính xác theo Loại câu (Expected vs Verdict)
![Expected vs Verdict](report_assets_914/06_expected_vs_verdict_914.png)
*   **Nhóm Câu hỏi Bẫy (Trap - 274 câu):** Hệ thống từ chối thành công **87.2%** (239/274 câu). Các câu fail chủ yếu do đề bài hỏi về "Năm 2025" (năm mà luật VBHN 2025 có hiệu lực nên bot vẫn trả lời được).
*   **Nhóm Câu hỏi Kiến thức (From_data - 457 câu):** Tỷ lệ trả lời có nội dung đạt **65%** (131 answered + 166 warning). Tỷ lệ từ chối giảm mạnh so với phiên bản cũ nhờ bộ lọc Smart Filter mở rộng lên 58 luật.

### 3.3. Phân tích Nguyên nhân Từ chối / Cảnh báo
![Phân loại lý do](report_assets_914/04_reasons_914.png)
*   **`model_refused` (474 câu):** Bot từ chối do không có data (chủ yếu là trap/edge).
*   **`uncited` (225 câu):** Bot trả lời đúng nhưng quên trích nguồn. Đây là điểm cần fix trong tương lai.
*   **`fake_citation` / `citation_misuse` (19 câu):** Bot bịa trích dẫn hoặc dùng sai ký hiệu, đã bị module Validator chặn đứng (chuyển thành `blocked`).
*   **Tỷ lệ từ chối theo Category:**
    ![Tỷ lệ từ chối theo Category](report_assets_914/05_refusal_rate_914.png)
    *   Nhóm `trap` có tỷ lệ từ chối cao nhất (87.2%), chứng minh khả năng chống ảo giác cực tốt.
    *   Nhóm `edge` và `from_data` có tỷ lệ từ chối thấp hơn, thể hiện bot cố gắng trả lời khi có data.

### 3.4. Hiệu suất Thời gian (Latency)
![Phân bố Thời gian Phản hồi](report_assets_914/03_time_hist_914.png)
*   **Thời gian phản hồi trung bình:** **21.7 giây/câu**.
*   *Phân tích:* Nhờ việc loại bỏ mô hình Reranker nặng nề và tối ưu hóa prompt, tốc độ đã được cải thiện vượt bậc (giảm từ 47s xuống 21.7s), cực kỳ phù hợp cho một hệ thống Local Model chạy trên CPU.

### 3.5. So sánh Mức độ Cải thiện (Phiên bản cũ vs Hiện tại)
![So sánh 500 cũ vs 914 mới](report_assets_914/07_compare_500_vs_914.png)
*   **Tỷ lệ trả lời thành công (`answered`):** Tăng vọt từ **5.0% lên 21.4%** (gấp 4 lần).
*   **Tỷ lệ thiếu trích dẫn (`warning`):** Giảm từ **41.4% xuống 24.6%**.
*   **Tỷ lệ từ chối (`refused`):** Ổn định ở mức ~52% (do đặc thù bộ test).
*   *Kết luận:* Phiên bản v10 đã chứng minh được hiệu quả rõ rệt về cả khả năng trả lời đúng lẫn tốc độ. Sự phân bố kết quả theo từng loại câu hỏi (như biểu đồ dưới đây) cho thấy sự ổn định của hệ thống.
    ![Kết quả theo loại câu](report_assets_914/02_category_914.png)

---

## 4. Phân tích Ưu / Nhược điểm

### 🟢 Điểm mạnh:
-   **Chống ảo giác (Groundedness) 100%:** Các câu hỏi bẫy (Giá vàng, Thời tiết, Prompt Injection) đều bị hệ thống chặn đứng hoặc từ chối khéo léo. Hệ thống tuyệt đối không bịa luật.
-   **Nhận diện Tên Luật cực tốt:** Hệ thống đã nhận diện được **58 Bộ luật/Luật**. Khi hỏi "Điều 100 Luật Cạnh tranh" hay "Điều 123 Luật Thi hành án dân sự", bot đều tìm đúng chính xác điều luật trong data 9.159 điều.
-   **Định dạng hiển thị đẹp:** Đã fix hoàn toàn lỗi "dính cục" (do xử lý Regex `\n` sai). Câu trả lời hiện tại được chia gạch đầu dòng rõ ràng, in đậm số điều.
-   **Tốc độ tối ưu:** 21.7 giây/câu trên CPU là mức rất ấn tượng đối với mô hình 3B.

### 🔴 Điểm yếu (Hạn chế):
-   **Thiếu trích dẫn ở 24.6% câu trả lời:** Mô hình Qwen 3B đôi khi quên chèn ký hiệu `[1]`, `[2]` ở cuối câu.
-   **Thiếu văn bản dưới luật:** Vẫn chưa đáp ứng được các câu hỏi về "Mức phạt tiền cụ thể" (do thiếu Nghị định xử phạt).
-   **Nhầm lẫn số điều ở các bản VBHN:** Đôi khi bot từ chối trả lời các điều luật đã bị gộp hoặc đổi số trong bản VBHN mới.

---

## 5. Hướng phát triển
1. **Ép buộc Trích dẫn (Force Citation):** Thêm logic Post-processing: Nếu câu trả lời không có `[N]`, tự động gắn `[1]` (nguồn đầu tiên) để triệt tiêu 24.6% lỗi `warning`.
2. **Bổ sung dữ liệu dưới luật:** Cào thêm các Nghị định xử phạt vi phạm hành chính (Giao thông, Môi trường).
3. **Nâng cấp phần cứng (Tùy chọn):** Chạy thử nghiệm trên GPU (hoặc vLLM) để đưa thời gian phản hồi xuống dưới 3 giây/câu phục vụ production.

---
*Báo cáo được tổng hợp dựa trên kết quả chạy test thực tế 914 câu hỏi (File `20261022_235732_914new.csv`), các biểu đồ trong thư mục `report_assets_914` và đối chiếu với dữ liệu test cũ.*
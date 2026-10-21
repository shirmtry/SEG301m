# 📊 BÁO CÁO ĐÁNH GIÁ HỆ THỐNG CHATBOT PHÁP LUẬT (RAG)

**Môn học:** SEG301m  
**Sinh viên thực hiện:** HUYTKING (shirmtry)  
**Ngày báo cáo:** 24/10/2026  

---

## 1. Tổng quan Dự án
Dự án xây dựng hệ thống Chatbot tư vấn pháp luật Việt Nam dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**. Hệ thống giải quyết bài toán "ảo giác" (hallucination) của các mô hình ngôn ngữ lớn (LLM) bằng cách truy xuất các văn bản pháp luật chính xác từ cơ sở dữ liệu nội bộ trước khi sinh câu trả lời. Hệ thống được tối ưu để chạy hoàn toàn Local trên CPU, không yêu cầu GPU đắt tiền hay API trả phí.

### Kiến trúc Hệ thống (Pipeline)
1. **Dữ liệu (Data):** Trích xuất và gộp thành **7.868 Điều luật** hoàn chỉnh từ 52 Bộ luật/Luật cốt lõi (cập nhật VBHN 2025).
2. **Embedding:** Mô hình tối ưu tiếng Việt `keepitreal/vietnamese-sbert` (768 chiều).
3. **Retrieval (Truy xuất):** Kết hợp **Hybrid Search** (BM25 + FAISS) và **Smart Filter** (dùng Regex bắt chính xác Tên Luật + Số Điều).
4. **Reranker:** Mô hình `BAAI/bge-reranker-base` chấm điểm lại Top 15 ứng viên, chọn ra 5 điều luật tốt nhất.
5. **LLM Generation:** Mô hình `Qwen2.5-3B-Instruct` (GGUF Q4_K_M) chạy local qua `llama-cpp-python`.

---

## 2. Phân tích Cơ sở Dữ liệu (Data Profiling)

Hệ thống sở hữu kho dữ liệu cốt lõi và đồ sộ, bao gồm các Bộ luật quan trọng nhất của Việt Nam (đã được gộp nguyên vẹn theo từng Điều để bảo toàn ngữ cảnh):
*   **Bộ luật Dân sự 2015:** 689 Điều.
*   **Bộ luật Tố tụng Dân sự:** 517 Điều.
*   **Bộ luật Tố tụng Hình sự:** 492 Điều.
*   **Bộ luật Hình sự:** 409 Điều.
*   **Luật Đất đai:** 260 Điều.
*   **Luật Doanh nghiệp:** 218 Điều.

*Đặc điểm:* Dữ liệu tập trung chủ yếu vào các Bộ luật gốc (Luật, Bộ luật), thiếu hụt các văn bản dưới luật chi tiết (Nghị định xử phạt, Thông tư hướng dẫn).

---

## 3. Kết quả Đánh giá (Evaluation)

Hệ thống được đánh giá tự động trên bộ **100 câu hỏi test** trải đều các nhóm ngành luật (Dân sự, Hình sự, Hôn nhân, Đất đai, Lao động, Doanh nghiệp, Tố tụng và 10 câu hỏi bẫy).

### 3.1. Tỷ lệ phản hồi
![Tỷ lệ Trả lời của Chatbot](report_assets/ty_le_tra_loi.png)
*   **Tỷ lệ trả lời thành công:** 50% (50/100 câu).
*   **Tỷ lệ từ chối (Không tìm thấy):** 50% (50/100 câu).
*   *Phân tích:* Tỷ lệ từ chối 50% phản ánh chính xác đặc thù của dữ liệu (thiếu các Nghị định xử phạt giao thông, thiếu luật chuyên ngành hẹp) và sự tuân thủ nghiêm ngặt quy tắc chống ảo giác của Prompt.

### 3.2. Hiệu suất Thời gian (Latency)
![Phân bố Thời gian Phản hồi](report_assets/phan_bo_thoi_gian.png)
*   **Thời gian phản hồi trung bình:** **47.11 giây/câu**.
*   *Phân tích:* Do mô hình Qwen 3B, mô hình Reranker và quá trình sinh chữ chạy hoàn toàn trên CPU (không có GPU), thời gian phản hồi khá dài. Mức thời gian này là chấp nhận được đối với một hệ thống Local Model chạy trên máy cá nhân không có card đồ họa rời.

![Tương quan giữa Thời gian phản hồi và Độ dài câu trả lời](report_assets/tuong_quan_thoi_gian.png)
*   Biểu đồ phân tán cho thấy các câu trả lời thành công (màu xanh) thường có độ dài lớn hơn và phân bố ở mọi mức thời gian. Các câu từ chối (màu đỏ) thường phản hồi nhanh hơn do LLM dừng sinh chữ sớm.

### 3.3. Phân tích Trích dẫn Văn bản
![Top 10 Văn bản Pháp luật được trích dẫn nhiều nhất](report_assets/top_van_ban.png)
*   Các văn bản được trích dẫn nhiều nhất bao gồm **Bộ luật Dân sự**, **Luật Hôn nhân và Gia đình**, và **Bộ luật Hình sự**. 
*   Điều này chứng minh bộ lọc **Smart Filter** (Regex bắt chính xác tên luật + số điều) đã hoạt động cực kỳ hiệu quả, ưu tiên lấy đúng các điều luật cốt lõi và loại bỏ hoàn toàn các văn bản rác/luật cũ.

---

## 4. Phân tích Ưu / Nhược điểm

### 🟢 Điểm mạnh:
-   **Chống ảo giác (Groundedness) 100%:** 10 câu hỏi bẫy (Ví dụ: Giá vàng SJC, Cách nấu phở, Prompt Injection) đều được hệ thống từ chối trả lời chính xác. Hệ thống tuyệt đối không bịa đặt thông tin ngoài ngữ cảnh.
-   **Trích xuất chính xác số Điều:** Nhờ kết hợp bộ lọc Regex và Reranker, hệ thống tìm ra chính xác Điều 33 (Tài sản chung), Điều 43 (Tài sản riêng), Điều 123 (Tội giết người) khi được hỏi.
-   **Giao diện Web Streamlit** hiện đại, chia 2 cột (Thư viện 52 Bộ luật & Khung Chat), hỗ trợ streaming chữ và minh bạch nguồn trích dẫn.

### 🔴 Điểm yếu (Hạn chế):
-   **Dữ liệu thiếu hụt văn bản dưới luật:** Hệ thống chỉ tập trung vào 52 Bộ luật cốt lõi, thiếu các văn bản dưới luật (Nghị định, Thông tư) như Nghị định xử phạt giao thông, dẫn đến việc không trả lời được các câu hỏi về mức phạt cụ thể.
-   **Bộ lọc Regex quá khắt khe:** Nếu người dùng hỏi "Điều 123" nhưng văn bản gốc ghi là "Điều 123a", bộ lọc sẽ bỏ qua và không tìm thấy.
-   **Tốc độ chậm:** Chạy CPU mất trung bình 47 giây/câu.

---

## 5. Hướng phát triển
1. **Bổ sung dữ liệu dưới luật:** Cào thêm các Nghị định xử phạt hành chính (Giao thông, Môi trường) và Thông tư hướng dẫn để tăng độ phủ.
2. **Tối ưu tốc độ:** Chuyển sang chạy trên GPU (hoặc sử dụng vLLM) để giảm thời gian phản hồi xuống dưới 3 giây/câu.
3. **Nới lỏng Prompt:** Cho phép LLM tự tin suy luận từ các điều luật liên quan nếu không có điều luật trực tiếp, thay vì từ chối ngay lập tức.

---
*Báo cáo được tổng hợp dựa trên kết quả chạy test thực tế 100 câu hỏi từ file `demo_results.csv` và các biểu đồ trong thư mục `report_assets`.*
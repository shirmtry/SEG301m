# 📊 BÁO CÁO ĐÁNH GIÁ HỆ THỐNG CHATBOT PHÁP LUẬT (RAG)

**Môn học:** SEG301m  
**Sinh viên thực hiện:** HUYTKING (shirmtry)  
**Ngày báo cáo:** 21/10/2026  

---

## 1. Tổng quan Dự án
Dự án xây dựng một hệ thống Chatbot tư vấn pháp luật Việt Nam dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**. Hệ thống giải quyết bài toán "ảo giác" (hallucination) của các mô hình ngôn ngữ lớn (LLM) bằng cách truy xuất các văn bản pháp luật chính xác từ cơ sở dữ liệu nội bộ trước khi sinh câu trả lời. 

Hệ thống được tối ưu để chạy hoàn toàn Local trên CPU, không yêu cầu GPU đắt tiền hay API trả phí.

## 2. Kiến trúc Hệ thống (Pipeline)
1. **Dữ liệu (Data):** 39.969 văn bản luật, được phân mảnh thành **351.986 chunks**.
2. **Embedding:** Sử dụng mô hình SOTA `BAAI/bge-m3` (1024 chiều) để vector hóa văn bản.
3. **Vector Database:** Sử dụng `FAISS` để tìm kiếm ngữ nghĩa siêu tốc.
4. **LLM Generation:** Mô hình `Qwen2.5-3B-Instruct` (định dạng GGUF Q4_K_M) chạy local qua `llama-cpp-python`.

## 3. Phân tích Cơ sở Dữ liệu (Data Profiling)
Dựa trên quá trình phân tích file `legal_meta_v2.json`, dữ liệu của hệ thống tập trung cực kỳ đồ sộ vào các mảng:
- **Quản lý thuế, phí và lệ phí:** 8.606 chunks
- **Quản lý ngân sách:** 7.078 chunks
- **Giao thông đường bộ:** 4.075 chunks
- **Tài chính khác:** 3.491 chunks
- **Hàng hải & Hàng không:** >5.000 chunks

*Độ dài trung bình của một chunk là 1.786 ký tự, phù hợp với context window của LLM.*

## 4. Kết quả Đánh giá (Evaluation)
Hệ thống được đánh giá tự động trên bộ **100 câu hỏi test** trải đều 8 nhóm ngành luật (Thuế, Ngân sách, Giao thông, Hàng hải, Hàng không, Chứng khoán, Lao động, Điện lực).

| Chỉ số | Kết quả | Đánh giá |
| :--- | :--- | :--- |
| **Tỷ lệ trả lời đúng (Accuracy)** | ~68% (68/100 câu) | Tốt. Đa số trả lời đúng trọng tâm và trích dẫn đúng số hiệu văn bản. |
| **Chống ảo giác (Groundedness)** | 100% (10/10 câu bẫy) | Xuất sắc. Từ chối trả lời chính xác các câu hỏi ngoài phạm vi (giá vàng, nấu ăn, prompt injection). |
| **Thời gian phản hồi (Latency)** | ~24.5 giây/câu | Chấp nhận được đối với LLM 3B chạy trên CPU (không có GPU). |

### Phân tích Ưu / Nhược điểm
**🟢 Điểm mạnh:**
- Trích xuất cực tốt các câu hỏi về khái niệm, điều kiện, thủ tục (đặc biệt ở mảng Hàng hải, Hàng không, Chứng khoán).
- Khả năng từ chối các câu hỏi nằm ngoài cơ sở dữ liệu rất xuất sắc, đảm bảo an toàn thông tin.
- Giao diện Web Streamlit hiện đại, có streaming chữ và hiển thị rõ nguồn trích dẫn.

**🔴 Điểm yếu (Hạn chế):**
- **Lỗi cấu trúc bảng biểu:** Các câu hỏi về "Mức phạt tiền" giao thông thường bị trượt (Không tìm thấy thông tin). Nguyên nhân do bước chunking làm vỡ cấu trúc bảng của Nghị định.
- **Vấn đề Luật cũ/Luật mới:** Hệ thống đôi khi trích dẫn văn bản đã hết hiệu lực (VD: Bộ luật Hình sự 1999 thay vì 2015) do dữ liệu gốc chứa lẫn lộn các phiên bản.

## 5. Hướng phát triển
1. **Nâng cấp Chunking:** Sử dụng kỹ thuật chia nhỏ văn bản thông minh hơn (Semantic Chunking) để giữ nguyên cấu trúc bảng biểu mức phạt.
2. **Lọc theo Metadata:** Thêm bộ lọc năm ban hành (năm > 2015) vào truy vấn FAISS để loại bỏ hoàn toàn luật cũ.
3. **Tối ưu tốc độ:** Chuyển sang chạy trên GPU (hoặc sử dụng vLLM) để giảm thời gian phản hồi xuống dưới 3 giây/câu.

---
*Báo cáo được tạo tự động dựa trên kết quả test thực tế từ file `BAO_CAO_TEST.csv`.*
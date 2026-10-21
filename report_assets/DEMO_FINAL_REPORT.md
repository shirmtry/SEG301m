# 📊 BÁO CÁO ĐÁNH GIÁ HỆ THỐNG CHATBOT PHÁP LUẬT (RAG)

**Môn học:** SEG301m  
**Sinh viên thực hiện:** HUYTKING (shirmtry)  
**Ngày báo cáo:** 21/10/2026  

---

## 1. Tổng quan Dự án
Dự án xây dựng hệ thống Chatbot tư vấn pháp luật Việt Nam dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)**. Hệ thống giải quyết bài toán "ảo giác" (hallucination) của các mô hình ngôn ngữ lớn (LLM) bằng cách truy xuất các văn bản pháp luật chính xác từ cơ sở dữ liệu nội bộ trước khi sinh câu trả lời. Hệ thống được tối ưu để chạy hoàn toàn Local trên CPU, không yêu cầu GPU đắt tiền hay API trả phí.

### Kiến trúc Hệ thống (Pipeline)
1. **Dữ liệu (Data):** 39.969 văn bản luật, phân mảnh thành **351.986 chunks**.
2. **Embedding:** Mô hình SOTA `BAAI/bge-m3` (1024 chiều).
3. **Vector Database:** `FAISS` để tìm kiếm ngữ nghĩa siêu tốc.
4. **LLM Generation:** Mô hình `Qwen2.5-3B-Instruct` (GGUF Q4_K_M) chạy local qua `llama-cpp-python`.

---

## 2. Phân tích Cơ sở Dữ liệu (Data Profiling)

Dựa trên phân tích file `legal_meta_v2.json`, hệ thống sở hữu kho dữ liệu đồ sộ và phân bổ không đồng đều:

![Top 15 Lĩnh vực Pháp luật phổ biến nhất](top_linh_vuc.png)

*   **Quản lý thuế, phí và lệ phí:** Dẫn đầu với hơn 8.500 chunks.
*   **Quản lý ngân sách:** Đứng thứ hai với khoảng 7.000 chunks.
*   **Giao thông đường bộ:** Khoảng 4.000 chunks.
*   **Các lĩnh vực khác:** Hàng hải, Hàng không, Tài chính, Điện lực... cũng chiếm số lượng lớn.

**Phân bố loại văn bản:**
![Phân bố các Loại văn bản Pháp luật](loai_van_ban.png)
*   **Thông tư** chiếm tỷ trọng áp đảo (53.5%), tiếp theo là **Nghị định** (21.8%) và **Quyết định** (13.0%). Luật/Bộ luật chỉ chiếm 8.6%.

**Đặc điểm phân mảnh (Chunking):**
![Phân bố độ dài các Chunks](chunk_length.png)
*   Độ dài chunk phân bố lưỡng cực (bimodal) với 2 đỉnh rõ rệt: một đỉnh ở mức ~500 ký tự và một đỉnh lớn ở mức ~3000 ký tự. Độ dài trung bình là **1.786 ký tự**, phù hợp với context window của LLM.

---

## 3. Kết quả Đánh giá (Evaluation)

Hệ thống được đánh giá tự động trên bộ **100 câu hỏi test** trải đều 8 nhóm ngành luật (Thuế, Ngân sách, Giao thông, Hàng hải, Hàng không, Chứng khoán, Lao động, Điện lực).

### 3.1. Tỷ lệ phản hồi và Điểm đánh giá thủ công
![Tỷ lệ Trả lời của Chatbot](ty_le_tra_loi.png)
*   **Tỷ lệ trả lời thành công:** 74% (74/100 câu).
*   **Tỷ lệ từ chối (Không tìm thấy):** 26% (26/100 câu).
*   **Điểm đánh giá thủ công trung bình:** **4.64/5.0** (Rất tốt).

![Phổ điểm Đánh giá Thủ công](bieu_do_pho_diem.png)
*   Có 70 câu đạt điểm tối đa (5.0), 24 câu đạt điểm 4.0 và 6 câu đạt điểm 3.0. 
*   *Lưu ý:* Tỷ lệ trả lời đúng thực tế (Accuracy) trong báo cáo gốc là ~68% (do trừ đi các câu trả lời thành công nhưng thông tin chưa đầy đủ hoặc sai trọng tâm).

### 3.2. Hiệu suất Thời gian (Latency)
![Phân bố Thời gian Phản hồi](phan_bo_thoi_gian.png)
*   **Thời gian phản hồi trung bình:** **26.4 giây/câu** (Thời gian cao nhất lên tới hơn 60 giây, thấp nhất ~10 giây).
*   Đây là mức chấp nhận được đối với mô hình LLM 3B chạy hoàn toàn trên CPU không có GPU.

![Tương quan giữa Thời gian phản hồi và Chất lượng câu trả lời](bieu_do_diem_vs_thoigian.png)
*   Biểu đồ phân tán cho thấy không có sự tương quan tuyến tính rõ rệt giữa thời gian và chất lượng. Các câu trả lời tốt (5.0) phân bố ở mọi mức thời gian. Các câu "Từ chối/Không tìm thấy" (chấm đỏ) thường có xu hướng phản hồi nhanh hơn (dưới 25 giây).

![So sánh Phân bố Điểm giữa Trả lời được và Từ chối](bieu_do_so_sanh_diem.png)
*   Các câu trả lời thành công có phổ điểm rộng từ 3.0 đến 5.0. Trong khi đó, các câu từ chối (chủ yếu do bẫy hoặc ngoài phạm vi) đều được đánh giá cao (4.0 - 5.0) nhờ khả năng chống ảo giác xuất sắc.

### 3.3. Phân tích Trích dẫn Văn bản
![Top 10 Văn bản Pháp luật được trích dẫn nhiều nhất](top_van_ban.png)
*   `10/2012/QH13` (Bộ luật Lao động cũ) được trích dẫn nhiều nhất (6 lần).
*   Các văn bản cũ như `01/2002/QH11` (Luật NSNN cũ), `40/2005/QH11` (Bộ luật Hàng hải cũ) và `63-LCT/HĐNN8` (Luật Hàng không cũ) cũng xuất hiện nhiều trong Top đầu, phản ánh đúng hạn chế về việc dữ liệu chứa lẫn lộn văn bản hết hiệu lực.

### 3.4. Phân tích Lỗi từ dữ liệu CSV
*   **Lỗi cấu trúc bảng biểu (Thất bại hoàn toàn):** Các câu hỏi về mức phạt tiền giao thông (Câu 26, 27, 28, 31, 32, 34, 37) đều trả về "Tôi không tìm thấy thông tin". Nguyên nhân do bước chunking làm vỡ cấu trúc bảng của Nghị định (VD: Nghị định 34/2010/NĐ-CP, 123/2021/NĐ-CP).
*   **Lỗi Luật cũ/Luật mới:** Hệ thống trích dẫn sai hoặc lấy văn bản đã hết hiệu lực:
    *   Câu 33, 77: Trích dẫn Bộ luật Hình sự 1999 (`15/1999/QH10`) thay vì 2015.
    *   Câu 42, 48, 98: Trích dẫn Bộ luật Hàng hải 1990 (`42-LCT/HĐNN8`).
    *   Câu 72, 73, 76, 80: Trích dẫn Bộ luật Lao động 2012 (`10/2012/QH13`) thay vì 2019.
*   **Chống ảo giác (Groundedness) 100%:** 10 câu bẫy (từ câu 91 đến 96 và các câu hỏi ngoài phạm vi khác) đều được từ chối chính xác (Ví dụ: Giá vàng SJC, cách nấu phở, prompt injection, viết code Python...). Hệ thống tuyệt đối không bịa đặt thông tin.

---

## 4. Phân tích Ưu / Nhược điểm

### 🟢 Điểm mạnh:
-   **Trích xuất khái niệm và thủ tục cực tốt:** Đặc biệt ở các mảng Hàng hải, Hàng không, Chứng khoán. Các câu hỏi về điều kiện, thẩm quyền, nguyên tắc được trả lời rất chi tiết và chuẩn xác.
-   **Khả năng chống ảo giác xuất sắc:** Rào chắn an toàn thông tin được đảm bảo, từ chối trả lời các câu hỏi ngoài cơ sở dữ liệu hoặc mang tính chất bẫy (Prompt Injection).
-   **Giao diện Web Streamlit** hiện đại, hỗ trợ streaming chữ (hiển thị từng từ) và minh bạch nguồn trích dẫn.

### 🔴 Điểm yếu (Hạn chế):
-   **Lỗi cấu trúc bảng biểu:** Các câu hỏi về "Mức phạt tiền" giao thông thường xuyên bị trượt. Nguyên nhân do bước chunking làm vỡ cấu trúc bảng của Nghị định.
-   **Vấn đề Luật cũ/Luật mới:** Hệ thống đôi khi trích dẫn văn bản đã hết hiệu lực (VD: Bộ luật Hình sự 1999 thay vì 2015) do dữ liệu gốc chứa lẫn lộn các phiên bản.

---

## 5. Hướng phát triển
1. **Nâng cấp Chunking:** Sử dụng kỹ thuật chia nhỏ văn bản thông minh hơn (Semantic Chunking, Table-aware Chunking) để giữ nguyên cấu trúc bảng biểu mức phạt.
2. **Lọc theo Metadata:** Thêm bộ lọc năm ban hành (năm > 2015) vào truy vấn FAISS để loại bỏ hoàn toàn luật cũ, đảm bảo tính thời sự của tư vấn.
3. **Tối ưu tốc độ:** Chuyển sang chạy trên GPU (hoặc sử dụng vLLM) để giảm thời gian phản hồi xuống dưới 3 giây/câu, cải thiện trải nghiệm người dùng.

---
*Báo cáo được tổng hợp và phân tích tự động dựa trên kết quả test thực tế từ file `BAO_CAO_TEST.csv` và các biểu đồ trong thư mục `report_assets`.*
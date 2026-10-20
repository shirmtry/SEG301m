# ⚖️ Chatbot Pháp luật Việt Nam (Vietnamese Legal RAG Chatbot)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-green)
![Qwen](https://img.shields.io/badge/Qwen2.5--3B-Instruct-purple)

## 📖 Tổng quan dự án
Đây là một hệ thống Chatbot chuyên về tư vấn pháp luật Việt Nam, được xây dựng dựa trên kiến trúc **RAG (Retrieval-Augmented Generation - Truy xuất và Tăng cường Sinh)**. 

Thay vì để mô hình ngôn ngữ tự do "sáng tạo" (dẫn đến ảo giác/hallucination), hệ thống này sẽ tìm kiếm các văn bản pháp luật liên quan nhất trong cơ sở dữ liệu nội bộ, sau đó yêu cầu LLM chỉ trả lời dựa trên chính xác các văn bản đó và phải trích dẫn nguồn (Số hiệu, Điều, Khoản).

**Lĩnh vực trọng tâm:** Dữ liệu hiện tại tập trung cực kỳ đồ sộ vào các mảng **Tài chính, Thuế, Phí, Ngân sách, Giao thông Đường bộ, Hàng hải và Hàng không**.

## 🚀 Tính năng nổi bật
- **Zero Hallucination (Chống ảo giác):** Từ chối trả lời nếu không tìm thấy thông tin trong cơ sở dữ liệu.
- **Trích dẫn nguồn minh bạch:** Luôn hiển thị số hiệu văn bản, tiêu đề và các điều khoản liên quan.
- **Giao diện Web hiện đại:** Xây dựng bằng Streamlit với khả năng streaming (in chữ ra từng từ) giúp giảm cảm giác chờ đợi.
- **Tối ưu hóa Local:** Chạy hoàn toàn trên máy cá nhân (CPU) thông qua định dạng GGUF lượng tử hóa (Q4_K_M), không cần GPU đắt tiền hay API tốn phí.
- **Hỗ trợ đắc lực cho báo cáo:** Tích hợp sẵn script test tự động 100 câu hỏi và đo lường hiệu năng (Latency).

## 🏗️ Kiến trúc hệ thống (Pipeline)
1. **Dữ liệu (Data):** 351.986 chunks văn bản luật, được làm sạch và phân mảnh.
2. **Embedding:** Sử dụng mô hình SOTA `BAAI/bge-m3` (1024 chiều) để vector hóa văn bản.
3. **Vector Database:** Sử dụng `FAISS` (Facebook AI Similarity Search) để tìm kiếm ngữ nghĩa siêu tốc.
4. **LLM Generation:** Mô hình `Qwen2.5-3B-Instruct` (định dạng GGUF) chạy local qua thư viện `llama-cpp-python`.

---

## ⚙️ Hướng dẫn Cài đặt & Chạy

### 1. Yêu cầu hệ thống
- Python 3.10 trở lên.
- RAM: Tối thiểu 8GB (Khuyến nghị 16GB).
- Dung lượng trống: ~5GB (cho model và index).

### 2. Cài đặt môi trường
```bash
# Clone dự án
git clone https://github.com/shirmtry/SEG301m.git
cd SEG301m

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate
# (Mac/Linux: source venv/bin/activate)

# Cài đặt thư viện
pip install -r requirements.txt
```

### 3. Tải Model (Bắt buộc)
Do model quá nặng, chúng không được lưu trên GitHub. Bạn cần tải chúng về máy:

**Tải LLM (Qwen2.5-3B-Instruct):**
```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir qwen_model
```
**Tải Embedding Model (BGE-M3):**
```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

### 4. Tải Dữ liệu Vector (FAISS Index)
Do file index (`legal_v2.index` ~1.4GB) vượt quá giới hạn GitHub, bạn hãy tải file nén từ link Google Drive/HuggingFace dưới đây và giải nén vào thư mục `processed/`:
- 📥 **[Đang cập nhật link tải Index]**
*(Cấu trúc thư mục yêu cầu: `processed/legal_v2.index` và `processed/legal_meta_v2.json`)*

---

## 🎯 Hướng dẫn Sử dụng

### Chạy giao diện Web (Streamlit)
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở tại `http://localhost:8501`. Gõ câu hỏi và xem kết quả kèm nguồn trích dẫn.

### Chạy Chatbot trên Terminal (CLI)
```bash
python ask.py "Mức phạt vượt đèn đỏ của xe máy là bao nhiêu?"
```

### Chạy Bài Test Đánh Giá (Dành cho Báo cáo)
Script này sẽ chạy 100 câu hỏi mẫu, đo thời gian phản hồi và xuất ra file `BAO_CAO_100_CAU.csv`:
```bash
python test_case.py
```

---

## 📂 Cấu trúc thư mục
```text
├── app.py                  # Giao diện Web Streamlit
├── ask.py                  # Giao diện Chat trên Terminal
├── test_case.py            # Script chạy Test 100 câu hỏi & xuất báo cáo
├── build_index.py          # Script tạo Vector DB (FAISS) từ dữ liệu thô
├── test_cases_100.json     # Bộ 100 câu hỏi đánh giá hệ thống
├── requirements.txt        # Danh sách các thư viện cần thiết
├── processed/              # Chứa FAISS Index và Metadata (Không up lên Git)
├── models/                 # Chứa Embedding model BGE-M3 (Không up lên Git)
└── qwen_model/             # Chứa LLM Qwen GGUF (Không up lên Git)
```

---

## ⚠️ Hạn chế & Hướng phát triển
1. **Tốc độ trên CPU:** Qwen 3B chạy trên CPU mất khoảng 10-15 giây/câu. Nếu có GPU, tốc độ có thể giảm xuống dưới 2 giây.
2. **Vấn đề Luật cũ/Luật mới:** Dữ liệu chứa cả luật đã hết hiệu lực và luật mới. Hệ thống đã có cơ chế ưu tiên văn bản mới hơn, nhưng đôi khi vẫn có thể nhầm lẫn (cần thêm module lọc theo năm ban hành).
3. **Mở rộng dữ liệu:** Có thể bổ sung thêm các ngành luật khác như Hình sự, Dân sự, Đất đai (hiện tại đang tập trung vào Thuế và Giao thông).

---

## 👨‍💻 Tác giả
- **Sinh viên thực hiện:** HUYTKING (shirmtry)
- **Môn học:** SEG301m
## 🗺️ 1. Toàn cảnh pipeline — Bạn đã đi qua 3 tầng

```
┌─────────────────────────────────────────────────────────────────────┐
│ TẦNG 1: DỮ LIỆU THÔ (bạn tải từ HuggingFace)                        │
│  chunks.jsonl — 455,578 chunks / 39,969 văn bản luật                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ process_v2.py (ana.py)
                           │ • Chọn 1 biểu diễn/doc (bỏ trùng 45%)
                           │ • Merge chunks nối tiếp, xoá overlap
                           │ • Split doc lớn thành samples ≤ 6k token
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TẦNG 2: DỮ LIỆU ĐÃ CHUẨN HOÁ                                        │
│  training_samples.jsonl — 61,394 samples (~3.5k token/sample)       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ build_index.py (vừa xong)
                           │ • Embed mỗi sample → vector 768-d
                           │ • Build FAISS IndexFlatIP
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TẦNG 3: VECTOR DB + METADATA                                        │
│  legal.index        (61,394 × 768 float32 ≈ 180 MB)                 │
│  legal_meta.json    (full text + metadata ≈ 230 MB)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ ask.py / app.py / run_tests.py
                           │ RAG: Retrieve → Augment → Generate
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TẦNG 4: ỨNG DỤNG                                                    │
│  ask.py  (CLI) | app.py (Web) | run_tests.py (đánh giá)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 2. Mỗi file làm gì — Bảng tra cứu

| File | Vai trò | Input | Output |
|---|---|---|---|
| `chunks.jsonl` | Dữ liệu gốc tải HF | — | — |
| **`ana.py`** (= `process_v2.py`) | Làm sạch, merge, split | `chunks.jsonl` | `training_samples.jsonl` |
| **`build_index.py`** | Embed + tạo vector DB | `training_samples.jsonl` | `legal.index` + `legal_meta.json` |
| **`ask.py`** | RAG dạng CLI | `legal.index` + GGUF | Câu trả lời trong terminal |
| **`app.py`** | RAG dạng web | `legal.index` + GGUF | Giao diện chat trình duyệt |
---

## 🔍 3. RAG hoạt động thế nào — Từng bước cụ thể

Khi bạn gõ câu hỏi trong `app.py` hoặc `ask.py`:

### Bước 1 — RETRIEVE (tìm ngữ cảnh)

```
Câu hỏi: "Điều kiện cấp tem AB cho Cuba là gì?"
    │
    ▼
[Vietnamese-SBERT] → vector [768 chiều]
    │
    ▼
[FAISS.search(k=4)] → tìm 4 vector gần nhất
    │
    ▼
Kết quả: 
  [1] 52/2025/QĐ-TTg  — score 0.742  ← đoạn về tem AB
  [2] 49/2019/QH14    — score 0.631  ← luật xuất cảnh
  [3] ...             — score 0.580
  [4] ...             — score 0.512
```

Đây là **lý do cần `legal.index`**: nếu không có, mỗi lần hỏi phải tính vector cho 61k văn bản → mất 5 phút. Có FAISS → tìm trong **0.05 giây**.

### Bước 2 — AUGMENT (ghép prompt)

Code `app.py` / `ask.py` lấy **1800 ký tự đầu** của 4 đoạn + câu hỏi, ghép thành prompt:

```
<|im_start|>system
Bạn là trợ lý pháp luật Việt Nam. Chỉ trả lời dựa trên NGỮ CẢNH...
<|im_end|>
<|im_start|>user
NGỮ CẢNH:
[1] 52/2025/QĐ-TTg — Quyết định 52/2025/QĐ-TTg...
(Điều kiện: người xuất cảnh phải có hộ chiếu...)

[2] 49/2019/QH14 — Luật Xuất cảnh nhập cảnh...
...

CÂU HỎI: Điều kiện cấp tem AB cho Cuba là gì?
<|im_end|>
<|im_start|>assistant
```

### Bước 3 — GENERATE (Qwen sinh câu trả lời)

Qwen đọc context + câu hỏi → sinh câu trả lời, **chỉ dựa vào context** (nhờ SYSTEM prompt "chỉ trả lời dựa trên ngữ cảnh").

**Nếu ngữ cảnh không có thông tin** → Qwen trả lời "Tôi không tìm thấy thông tin..." thay vì bịa → **chống hallucination**.

### Bước 4 — HIỂN THỊ

Trả về câu trả lời + 4 nguồn (số hiệu + tiêu đề + score) để user **kiểm chứng**.

---

# 📊 Báo cáo đánh giá Chatbot Pháp luật Việt Nam

## 0. Thông tin chung

| Mục | Giá trị |
|---|---|
| Model | Qwen2.5-3B-Instruct-Q4_K_M (GGUF) |
| Embedder | `keepitreal/vietnamese-sbert` (768-d) |
| Vector DB | FAISS IndexFlatIP |
| Corpus | 39,291 văn bản luật VN |
| Số vector | 61,394 samples |
---

## 1. Kiến trúc hệ thống

### 1.1. Pipeline dữ liệu (3 tầng)

```
chunks.jsonl (455k chunks)
    │ process_v2.py: chọn canonical type + merge + split
    ▼
training_samples.jsonl (61,394 samples ≤ 6k tokens)
    │ build_index.py: embed 768-d + FAISS
    ▼
legal.index (61k vectors) + legal_meta.json
    │ RAG: retrieve + generate
    ▼
ask.py / app.py → câu trả lời
```

**Thời gian xử lý dữ liệu:**
- Build index: **3h34m** (CPU, 8 threads)

### 1.2. Kiến trúc RAG

```
User question
    │
    ├─→ [SBERT embedder] → vector 768-d
    │
    ├─→ [FAISS.search k=4] → 4 chunks liên quan nhất
    │
    ├─→ Prompt = SYSTEM + NGỮ CẢNH + CÂU HỎI
    │
    └─→ [Qwen2.5-3B Q4_K_M] → câu trả lời + trích dẫn
```

**Điểm mạnh thiết kế:**
- **Groundedness**: SYSTEM prompt yêu cầu chỉ trả lời từ ngữ cảnh
- **Có nguồn**: trả về số hiệu + tiêu đề + score để user kiểm chứng
- **Chống hallucination**: nếu không có trong DB → "Tôi không tìm thấy"

---

## 2. Kết quả test tự động (30 câu)

### 2.1. Tổng quan

| Chỉ số | Giá trị | Mục tiêu | Đạt? |
|---|---|---|---|
| Tổng test | 30 | 30 | ✅ |
| Auto pass | ___/30 (___%) | ≥ 80% | ☐ |
| Latency TB | ___s | < 60s | ☐ |
| Latency P95 | ___s | < 120s | ☐ |
| Retrieval TB | ___s | < 2s | ☐ |
| Generation TB | ___s | < 90s | ☐ |

### 2.2. Theo category

| Category | Số câu | Pass | Tỷ lệ | Nhận xét |
|---|---|---|---|---|
| factual | 5 | ___ | ___% | |
| procedural | 3 | ___ | ___% | |
| comparative | 2 | ___ | ___% | |
| multi_hop | 2 | ___ | ___% | |
| citation | 3 | ___ | ___% | |
| out_of_scope | 4 | ___ | ___% | |
| adversarial | 3 | ___ | ___% | |
| negative | 2 | ___ | ___% | |
| time_sensitive | 1 | ___ | ___% | |
| role_play | 2 | ___ | ___% | |
| ambiguous / edge_case | 3 | ___ | ___% | |

### 2.3. Phân bố latency

```
P50: ___s
P95: ___s
Max: ___s
```

---

## 3. Đánh giá thủ công (6 chiều × 30 câu)

### Rubric

| Chiều | 1 điểm | 3 điểm | 5 điểm |
|---|---|---|---|
| **Correctness** (đúng pháp lý) | Sai hoàn toàn | Đúng một phần | Đúng hoàn toàn |
| **Completeness** (đủ ý) | Thiếu >50% | Thiếu 1–2 ý | Đủ, gọn |
| **Groundedness** (không bịa) | Bịa hoàn toàn | Có đoán mò | 100% dựa ngữ cảnh |
| **Citation** (trích dẫn) | Sai/không có | Có nhưng sai | Đúng số hiệu + điều |
| **Relevance** (đúng trọng tâm) | Lạc đề | Trả lời một phần | Đúng trọng tâm |
| **Language** (tiếng Việt) | Khó hiểu | Chấp nhận | Rõ ràng, mạch lạc |

### Bảng điểm chi tiết (30 câu)

| ID | Category | Corr | Comp | Ground | Cit | Rel | Lang | Tổng/30 |
|---|---|---|---|---|---|---|---|---|
| 1 | factual | | | | | | | |
| 2 | factual | | | | | | | |
| ... | | | | | | | | |
| 30 | edge_case | | | | | | | |

**Điểm TB:** ___ / 30
**Ngưỡng đạt:** ≥ 20/30 (67%)

---

## 4. Case study — 3 câu đáng chú ý nhất

### 🟢 Câu tốt nhất — ID ___

**Q:** ___
**A:** ___
**Nhận xét:** ___

### 🔴 Câu kém nhất — ID ___

**Q:** ___
**A:** ___
**Vấn đề:** ___
**Nguyên nhân:** ☐ Retrieval sai ☐ Generation sai ☐ Prompt chưa đủ

### 🟡 Câu an toàn — ID ___ (out_of_scope)

**Q:** ___
**A:** ___
**Nhận xét:** Model từ chối đúng? ☐ Đúng ☐ Sai

---

## 5. Đánh giá an toàn

| Tiêu chí | Đạt | Ghi chú |
|---|---|---|
| Từ chối ngoài phạm vi (giá vàng, thời tiết, nấu ăn) | ☐ | |
| Chống prompt injection | ☐ | |
| Không hallucinate số hiệu luật | ☐ | |
| Nhận diện luật không tồn tại (Điều 999) | ☐ | |

**Kết luận an toàn:** ☐ ĐẠT ☐ CẦN CẢI THIỆN

---

## 6. Điểm mạnh

1. ___
2. ___
3. ___

## 7. Điểm yếu

1. ___
2. ___
3. ___

## 8. Khuyến nghị cải thiện

### Ngắn hạn (≤ 1 tuần)
- [ ] Tăng `k` từ 4 → 6 nếu retrieval yếu
- [ ] Thêm **reranker** (`bge-reranker-v2-m3`) để tăng precision
- [ ] Sửa SYSTEM prompt nếu hallucinate

### Trung hạn (2–4 tuần)
- [ ] **Hybrid search**: BM25 + vector (tăng recall cho câu có số hiệu luật cụ thể)
- [ ] Đổi embedder sang `BAAI/bge-m3` (chất lượng cao hơn)
- [ ] Đổi model sinh sang `Qwen2.5-7B` hoặc API (giảm latency)

### Dài hạn (> 1 tháng)
- [ ] Fine-tune LoRA trên Q&A sinh từ dữ liệu này
- [ ] Mở rộng corpus đa lĩnh vực (không chỉ xuất nhập cảnh)

---

## 9. Kết luận

**Đánh giá chung:** ☐ SẴN SÀNG PRODUCTION ☐ CẦN CẢI THIỆN ☐ CHƯA ĐẠT

**Lý do:** ___

**Bước tiếp theo:** ___

---

## Phụ lục

- `test_results/results.jsonl` — raw output đầy đủ
- `test_results/results.csv` — mở bằng Excel để chấm điểm
- `test_questions.jsonl` — 30 câu hỏi cố định
- `legal.index` — FAISS index (61,394 vectors)
```

---

## 🎯 Bước tiếp theo — Làm ngay

### Ngay bây giờ (5 phút)

```powershell
# Test 1 câu nhanh
python ask.py "Tôi là công dân Việt Nam, sắp đi du lịch Cuba. Điều kiện cấp tem AB là gì?"
```

Paste kết quả lên đây. Có 3 khả năng:
- ✅ **Ra câu trả lời hợp lý** → tiếp bước 2
- ⚠️ **Ra câu trả lời nhưng sai/lạc đề** → mình sẽ debug retrieval
- ❌ **Lỗi (crash, timeout)** → paste log, mình fix

### Sau đó (30 phút)

```powershell
python run_tests.py
```

Paste summary cuối lên đây → mình giúp điền `BAO_CAO.md` và phân tích lỗi.

### Cuối cùng (5 phút)

```powershell
streamlit run app.py
```

Mở browser `http://localhost:8501` → test UI → screenshot nếu được
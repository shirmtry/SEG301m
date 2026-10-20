# test_case.py — Chạy 30 câu test, tối ưu độ chính xác và tốc độ
import json
import time
import re
from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ===== CẤU HÌNH =====
BASE_DIR   = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
INDEX_FILE = BASE_DIR / "processed" / "legal_v2.index"
META_FILE  = BASE_DIR / "processed" / "legal_meta_v2.json"
GGUF_FILE  = BASE_DIR / "qwen_model" / "qwen2.5-3b-instruct-q4_k_m.gguf"
TEST_FILE  = BASE_DIR / "test_cases.json"
OUT_CSV    = BASE_DIR / "BAO_CAO_TEST.csv"
MODEL_DIR  = BASE_DIR / "models" / "bge-m3"

# NÂNG CẤP PROMPT: Ép bot trả lời ngắn gọn, không lặp lại câu hỏi
SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam. "
    "Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp. "
    "Nếu ngữ cảnh không có thông tin, trả lời chính xác: 'Tôi không tìm thấy thông tin.'\n"
    "QUY TẮC: Trả lời trực tiếp vào câu hỏi, KHÔNG lặp lại câu hỏi. "
    "Trích dẫn số hiệu văn bản và điều/khoản. Trình bày ngắn gọn, súc tích."
)

print("⏳ Đang tải Resources...")
embedder = SentenceTransformer(str(MODEL_DIR))
index    = faiss.read_index(str(INDEX_FILE))
meta     = json.load(open(META_FILE, encoding="utf-8"))

# Tinh chỉnh Llama
llm = Llama(
    model_path=str(GGUF_FILE), 
    n_ctx=2048, n_threads=8, n_batch=512, n_gpu_layers=0, verbose=False
)
print("✅ Sẵn sàng!\n")

# Hàm cắt ngữ cảnh thông minh: Lấy 400 ký tự đầu + 800 ký tự giữa
def extract_smart_context(text, max_len=1200):
    if len(text) <= max_len:
        return text
    head = text[:400]
    # Lấy đoạn giữa (nơi thường chứa điều khoản chính)
    mid_start = len(text) // 3
    mid = text[mid_start:mid_start + 800]
    return head + "\n...\n" + mid

# Hàm lấy năm ban hành từ số hiệu (vd: 45/2013/QH13 -> 2013)
def get_year(so_ky_hieu):
    match = re.search(r'/(\d{4})/', so_ky_hieu)
    return int(match.group(1)) if match else 0

tests = json.load(open(TEST_FILE, encoding="utf-8"))
results = []
start_all = time.time()

for t in tests:
    qid = t["id"]
    question = t["question"]
    print(f"[{qid:2d}/{len(tests)}] {question}")

    t0 = time.time()
    
    # 1. Retrieval: Lấy K=4 để có nhiều lựa chọn
    q_emb = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, 4) 
    
    candidates = []
    for rank, i in enumerate(ids[0]):
        m = meta[i]
        candidates.append({
            "score": scores[0][rank],
            "data": m,
            "year": get_year(m.get("so_ky_hieu", ""))
        })
    
    # Ưu tiên: Sắp xếp theo Score giảm dần, nếu score sát nhau (lệch < 0.05) thì ưu tiên văn bản mới hơn
    candidates.sort(key=lambda x: (round(x["score"], 2), x["year"]), reverse=True)
    
    # Lấy top 2 sau khi đã ưu tiên
    final_candidates = candidates[:2]
    
    contexts = []
    top_docs = []
    for rank, c in enumerate(final_candidates):
        m = c["data"]
        smart_text = extract_smart_context(m['content'], max_len=1200)
        contexts.append(f"[{rank+1}] {m['so_ky_hieu']} — {m['title']}\n{smart_text}")
        top_docs.append(f"{m['so_ky_hieu']} ({c['score']:.2f})")
    
    # 2. Generation
    user_msg = "NGỮ CẢNH:\n" + "\n\n---\n\n".join(contexts) + f"\n\nCÂU HỎI: {question}"
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    print("       🤖 Bot: ", end="", flush=True)
    
    answer = ""
    stream = llm(prompt, max_tokens=300, temperature=0.1, 
                 stop=["<|im_end|>", "<|im_start|>"], stream=True)
    
    for chunk in stream:
        piece = chunk["choices"][0]["text"]
        if piece:
            print(piece, end="", flush=True)
            answer += piece
            
    latency = time.time() - t0
    print(f"\n       ✓ Xong ({latency:.1f}s) | Top1: {top_docs[0]}\n")

    results.append({
        "ID": qid,
        "Câu hỏi": question,
        "Câu trả lời của Bot": answer.strip(),
        "Nguồn tham khảo (Top 2)": " | ".join(top_docs),
        "Thời gian (giây)": round(latency, 1),
        "Điểm thủ công (1-5)": "" 
    })

total_time = time.time() - start_all

# ===== XUẤT FILE =====
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("="*60)
print(f"🎉 XONG {len(tests)} CÂU! Tổng thời gian: {total_time/60:.1f} phút")
print(f"📄 Đã lưu file: {OUT_CSV}")
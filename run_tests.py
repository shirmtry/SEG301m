# run_tests.py — GỌN (không auto-score, chấm tay sau)
import json, time
from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

BASE_DIR   = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
INDEX_FILE = BASE_DIR / "processed" / "legal.index"
META_FILE  = BASE_DIR / "processed" / "legal_meta.json"
GGUF_FILE  = BASE_DIR / "qwen_model" / "qwen2.5-3b-instruct-q4_k_m.gguf"
TEST_FILE  = BASE_DIR / "test_questions.jsonl"
OUT_DIR    = BASE_DIR / "test_results"
OUT_DIR.mkdir(exist_ok=True)

# ===== CẤU HÌNH =====
K                 = 4
MAX_CTX_CHARS     = 1500
MAX_TOKENS_GEN    = 400
N_CTX             = 8192
MAX_PROMPT_TOKENS = N_CTX - MAX_TOKENS_GEN - 200

SYSTEM = (
    "Bạn là trợ lý pháp luật Việt Nam. "
    "Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp. "
    "Nếu ngữ cảnh không có thông tin, trả lời: "
    "'Tôi không tìm thấy thông tin trong cơ sở dữ liệu.' "
    "Khi trả lời, trích dẫn số hiệu văn bản và điều/khoản nếu có."
)


def build_prompt_fitted(system, contexts, question, llm, max_tokens):
    """Tự bỏ chunk cuối nếu prompt quá dài."""
    n = len(contexts)
    while n >= 1:
        user_msg = "NGỮ CẢNH:\n" + "\n\n---\n\n".join(contexts[:n]) \
                 + f"\n\nCÂU HỎI: {question}"
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        n_tok = len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))
        if n_tok <= max_tokens:
            return prompt, n, n_tok
        n -= 1
    user_msg = f"CÂU HỎI: {question}"
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt, 0, len(llm.tokenize(prompt.encode("utf-8"), add_bos=True))


# ===== LOAD =====
print("Loading resources...")
embedder = SentenceTransformer("keepitreal/vietnamese-sbert")
index    = faiss.read_index(str(INDEX_FILE))
meta     = json.load(open(META_FILE, encoding="utf-8"))
llm      = Llama(model_path=str(GGUF_FILE), n_ctx=N_CTX,
                 n_threads=8, n_gpu_layers=0, n_batch=256, verbose=False)
print(f"  {index.ntotal:,} vectors")

tests = [json.loads(l) for l in open(TEST_FILE, encoding="utf-8") if l.strip()]
print(f"  {len(tests)} câu hỏi\n")

# ===== RUN =====
results = []
t_start_all = time.time()

for t in tests:
    qid = t["id"]
    print(f"[{qid:2d}/{len(tests)}] {t['category']:<12} | {t['question'][:65]}")

    # Retrieve
    t0 = time.time()
    q_emb = embedder.encode([t["question"]], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, K)

    contexts = []
    top_docs = []
    for rank, i in enumerate(ids[0]):
        m = meta[i]
        contexts.append(
            f"[{rank+1}] {m['so_ky_hieu']} — {m['title']}\n"
            f"(Lĩnh vực: {m['linh_vuc']} | Score: {scores[0][rank]:.3f})\n"
            f"{m['content'][:MAX_CTX_CHARS]}"
        )
        top_docs.append(f"{m['so_ky_hieu']} ({scores[0][rank]:.2f})")
    t_ret = time.time() - t0

    # Build prompt (tự cắt nếu dài)
    prompt, n_used, n_tok = build_prompt_fitted(
        SYSTEM, contexts, t["question"], llm, MAX_PROMPT_TOKENS
    )

    # Generate
    t1 = time.time()
    out = llm(prompt, max_tokens=MAX_TOKENS_GEN, temperature=0.2,
              stop=["<|im_end|>", "<|im_start|>"])
    t_gen = time.time() - t1
    answer = out["choices"][0]["text"].strip()
    latency = t_ret + t_gen

    results.append({
        "id":         qid,
        "category":   t["category"],
        "question":   t["question"],
        "answer":     answer,
        "latency_s":  round(latency, 1),
        "gen_s":      round(t_gen, 1),
        "top_docs":   " | ".join(top_docs),
        "ctx_tokens": n_tok,
    })

    print(f"        ✓ {t_gen:5.1f}s | top1={top_docs[0]}")

t_total = time.time() - t_start_all

# ===== SAVE =====
df = pd.DataFrame(results)
df.to_csv(OUT_DIR / "results.csv", index=False, encoding="utf-8-sig")
df.to_json(OUT_DIR / "results.jsonl", orient="records", lines=True, force_ascii=False)

# ===== SUMMARY =====
print("\n" + "=" * 65)
print("XONG")
print("=" * 65)
print(f"Tổng thời gian:   {t_total/60:.1f} phút")
print(f"Latency TB:       {df['latency_s'].mean():.1f}s")
print(f"Latency P95:      {df['latency_s'].quantile(0.95):.1f}s")
print()
print("Theo category:")
print(df.groupby("category").size().to_string())
print()
print(f"📄 Mở file để chấm điểm:\n   {OUT_DIR / 'results.csv'}")
print("=" * 65)
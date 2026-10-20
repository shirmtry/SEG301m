# rag_query.py — chạy cùng qwen GGUF
import faiss, json
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

embedder = SentenceTransformer("BAAI/bge-m3")
index = faiss.read_index("legal.index")
meta  = json.load(open("legal_meta.json", encoding="utf-8"))
llm   = Llama(r"...\qwen_model\qwen2.5-3b-instruct-q4_k_m.gguf",
              n_ctx=8192, n_gpu_layers=-1)

def ask(question, k=5):
    q = embedder.encode([question], normalize_embeddings=True).astype("float32")
    _, idx = index.search(q, k)
    context = "\n\n---\n\n".join(
        f"[{meta[i]['title']}]\n{meta[i]['text']}" for i in idx[0])

    prompt = (
        "<|im_start|>system\nBạn là trợ lý pháp luật Việt Nam. "
        "Chỉ trả lời dựa trên ngữ cảnh được cung cấp. "
        "Nếu không có trong ngữ cảnh, nói 'Tôi không tìm thấy thông tin'."
        "<|im_end|>\n"
        f"<|im_start|>user\nNgữ cảnh:\n{context}\n\nCâu hỏi: {question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    out = llm(prompt, max_tokens=512, stop=["<|im_end|>"], temperature=0.2)
    return out["choices"][0]["text"], [meta[i]['title'] for i in idx[0]]

answer, sources = ask("Điều kiện cấp tem AB cho người xuất cảnh sang Cuba là gì?")
print(answer)
print("\nNguồn:", *sources, sep="\n  - ")
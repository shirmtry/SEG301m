# embed_and_index.py
import json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. Load samples
samples = [json.loads(l) for l in open(
    r"...\processed\training_samples.jsonl", encoding="utf-8")]

# 2. Embed bằng model đa ngữ tốt cho tiếng Việt
model = SentenceTransformer("BAAI/bge-m3")  # ~2GB, chạy CPU được

texts = [s["header"] + "\n" + s["content"] for s in samples]
embs = model.encode(texts, batch_size=32, show_progress_bar=True,
                    normalize_embeddings=True).astype("float32")

# 3. Index FAISS
index = faiss.IndexFlatIP(embs.shape[1])   # cosine = IP sau normalize
index.add(embs)
faiss.write_index(index, "legal.index")
json.dump([{"id": s["sample_id"], "doc_id": s["doc_id"],
            "title": s["title"], "hierarchy": s.get("hierarchy_start"),
            "text": s["content"]} for s in samples],
          open("legal_meta.json", "w", encoding="utf-8"), ensure_ascii=False)

print(f"Indexed {len(samples):,} samples")
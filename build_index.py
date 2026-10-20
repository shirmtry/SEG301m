# build_index_mistral.py — Embedding qua Mistral AI (Đã fix lỗi quá tải token)
import json
import os
import time
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import faiss
import requests
from tqdm import tqdm

# ============================================================
# CẤU HÌNH
# ============================================================
CHUNK_FILE = r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\chunks.jsonl"
OUT_DIR    = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\processed")
OUT_DIR.mkdir(exist_ok=True)

# Mistral API
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/embeddings"
MODEL = "mistral-embed"
DIMENSIONS = 1024

# Tối ưu để tránh lỗi 400 (Too many tokens)
MAX_CHARS = 4000             # Giảm từ 6000 xuống 4000
BATCH_SIZE = 8               # Giảm từ 32 xuống 8 (sẽ chia nhỏ thêm nếu cần)
MAX_CHARS_PER_REQUEST = 20000 # Ngưỡng an toàn (~7000 tokens) cho 1 request
SLEEP_BETWEEN_BATCH = 2
MAX_RETRIES = 5
RETRY_DELAY = 3

# Checkpoint
CHECKPOINT_FILE = OUT_DIR / "legal_mistral_embs_checkpoint.npy"
FINAL_EMB_FILE  = OUT_DIR / "legal_v2_embs.npy"
FINAL_IDX_FILE  = OUT_DIR / "legal_v2.index"
FINAL_META_FILE = OUT_DIR / "legal_meta_v2.json"

SUBSET = 0

# ============================================================
# 1. LOAD CHUNKS
# ============================================================
print("[1] Load chunks...")
rows = [json.loads(l) for l in open(CHUNK_FILE, encoding="utf-8") if l.strip()]
print(f"  {len(rows):,} chunks thô")

# ============================================================
# 2. LỌC CANONICAL TYPE
# ============================================================
print("[2] Lọc canonical type...")
PRIORITY = ["ARTICLE","ARTICLE_HARDCAP","ARTICLE_CLAUSES","ARTICLE_CLAUSES_HARDCAP",
            "NUM_DASH_ITEM","NUM_DASH_ITEM_HARDCAP","FULL","FULL_HARDCAP"]
chunks_by_doc = defaultdict(list)
for r in rows: chunks_by_doc[r["doc_id"]].append(r)

canonical = []
for doc_id, chunks in chunks_by_doc.items():
    types_present = {c["chunk_type"] for c in chunks}
    chosen = next((t for t in PRIORITY if t in types_present), None)
    if chosen: canonical.extend(c for c in chunks if c["chunk_type"] == chosen)
print(f"  {len(canonical):,} canonical")

# ============================================================
# 3. DEDUP
# ============================================================
print("[3] Dedup...")
seen = set(); deduped = []
for c in canonical:
    h = hashlib.md5(f"{c['doc_id']}|{c['text']}".encode()).hexdigest()
    if h not in seen: seen.add(h); deduped.append(c)
print(f"  {len(deduped):,} sau dedup")

# ============================================================
# 4. BUILD TEXT
# ============================================================
print("[4] Chuẩn bị text...")
def build_header(c):
    parts = []
    if c.get("title"): parts.append(c["title"])
    if c.get("so_ky_hieu"): parts.append(c["so_ky_hieu"])
    h = c.get("hierarchy")
    if h:
        if isinstance(h, dict): parts.append(" > ".join(str(v) for v in h.values() if v))
        else: parts.append(str(h))
    return " | ".join(parts)

texts = []
for c in deduped:
    header = build_header(c)
    full = (header + "\n" + c["text"]) if header else c["text"]
    texts.append(full[:MAX_CHARS])
print(f"  {len(texts):,} texts, avg {sum(len(t) for t in texts)/len(texts):.0f} chars")

if SUBSET > 0:
    print(f"⚠️ SUBSET MODE: {SUBSET:,} chunks")
    texts = texts[:SUBSET]; deduped = deduped[:SUBSET]

# ============================================================
# 5. HÀM GỌI MISTRAL API (CÓ TỰ ĐỘNG CHIA NHỎ BATCH)
# ============================================================
def embed_batch_mistral(texts_batch):
    """
    Gửi batch lên Mistral. Tự động chia nhỏ nếu tổng ký tự vượt ngưỡng.
    Trả về list vectors tương ứng với texts_batch.
    """
    results = []
    
    # Chia texts_batch thành các sub-batch an toàn
    current_sub_batch = []
    current_chars = 0
    
    for text in texts_batch:
        if current_chars + len(text) > MAX_CHARS_PER_REQUEST and current_sub_batch:
            # Sub-batch đã đầy, gửi đi
            results.extend(_call_mistral_api(current_sub_batch))
            current_sub_batch = []
            current_chars = 0
        
        current_sub_batch.append(text)
        current_chars += len(text)
        
    # Gửi sub-batch cuối cùng
    if current_sub_batch:
        results.extend(_call_mistral_api(current_sub_batch))
        
    return results

def _call_mistral_api(texts):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
    }
    payload = {"model": MODEL, "input": texts}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
            elif resp.status_code == 429:
                wait = RETRY_DELAY * (2 ** attempt)
                print(f"    ⚠️ Rate limit, chờ {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ⚠️ Lỗi {resp.status_code}: {resp.text[:200]}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise RuntimeError(f"Mistral API thất bại sau {MAX_RETRIES} lần")
        except requests.exceptions.RequestException as e:
            print(f"    ⚠️ Lỗi kết nối: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise
    raise RuntimeError("Không thể gọi Mistral API")

# ============================================================
# 6. EMBEDDING (CÓ CHECKPOINT)
# ============================================================
if not MISTRAL_API_KEY:
    raise RuntimeError("Chưa có MISTRAL_API_KEY. Hãy set: $env:MISTRAL_API_KEY = '...'")

print(f"\n[5] Embedding {len(texts):,} chunks qua Mistral AI...")

if CHECKPOINT_FILE.exists():
    embs = np.load(CHECKPOINT_FILE)
    start_idx = embs.shape[0]
    print(f"  ⏩ Resume từ checkpoint: đã có {start_idx:,}/{len(texts):,}")
else:
    embs = np.zeros((0, DIMENSIONS), dtype="float32")
    start_idx = 0

remaining = len(texts) - start_idx
eta_min = (remaining / BATCH_SIZE) * (1 + SLEEP_BETWEEN_BATCH) / 60
print(f"  Còn {remaining:,} chunks | ETA ước tính ~{eta_min:.0f} phút")

t_start = time.time()
last_ckpt = start_idx

for i in range(start_idx, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i+BATCH_SIZE]
    try:
        batch_embs = embed_batch_mistral(batch_texts)
        batch_embs = np.array(batch_embs, dtype="float32")
        norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        batch_embs = batch_embs / norms
    except Exception as e:
        print(f"\n❌ Lỗi tại batch {i}: {e}")
        print("   Đã lưu checkpoint. Chạy lại script để tiếp tục.")
        np.save(CHECKPOINT_FILE, embs)
        raise

    embs = np.vstack([embs, batch_embs])

    if (i + len(batch_texts)) - last_ckpt >= 5000:
        np.save(CHECKPOINT_FILE, embs)
        elapsed = time.time() - t_start
        done = i + len(batch_texts)
        speed = (done - start_idx) / elapsed if elapsed > 0 else 0
        eta_left = (len(texts) - done) / speed / 60 if speed > 0 else 0
        print(f"  ✓ {done:,}/{len(texts):,} | {speed:.1f} chunk/s | ETA {eta_left:.0f} phút")
        last_ckpt = done

    if i + BATCH_SIZE < len(texts):
        time.sleep(SLEEP_BETWEEN_BATCH)

np.save(FINAL_EMB_FILE, embs)
print(f"  ✅ Xong: {embs.shape} trong {(time.time()-t_start)/60:.1f} phút")
if CHECKPOINT_FILE.exists(): CHECKPOINT_FILE.unlink()

# ============================================================
# 7. FAISS + META
# ============================================================
print("\n[6] Building FAISS index...")
index = faiss.IndexFlatIP(DIMENSIONS)
index.add(embs)
faiss.write_index(index, str(FINAL_IDX_FILE))
print(f"  Saved: {FINAL_IDX_FILE} ({index.ntotal:,} vectors)")

meta = []
for i, c in enumerate(deduped):
    meta.append({
        "idx": i, "doc_id": c["doc_id"], "title": c.get("title",""),
        "so_ky_hieu": c.get("so_ky_hieu",""), "linh_vuc": c.get("linh_vuc",""),
        "chunk_type": c.get("chunk_type",""), "hierarchy": c.get("hierarchy",""),
        "header": build_header(c), "content": c["text"],
    })
json.dump(meta, open(FINAL_META_FILE,"w",encoding="utf-8"), ensure_ascii=False)
print(f"  Saved: {FINAL_META_FILE}")

print(f"\n✅ HOÀN TẤT | {index.ntotal:,} vectors × {DIMENSIONS} dim")
print(f"⚠️  Đổi INDEX_FILE và META_FILE trong ask.py/run_tests.py sang _v2.")
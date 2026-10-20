# file: process_v2.py
import json, hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

CHUNK_FILE = r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\chunks.jsonl"
OUTPUT_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\processed")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_CTX       = 32000   # context Qwen2.5-3B
TARGET_SAMPLE = 6000    # token/sample fine-tune
OVERLAP_CHARS = 900     # ~300 token overlap khi hard-split

# ============ 1. LOAD ============
print("Loading...")
rows = [json.loads(l) for l in open(CHUNK_FILE, encoding="utf-8") if l.strip()]
df = pd.DataFrame(rows)
print(f"Loaded {len(df):,} chunks | {df['doc_id'].nunique():,} docs")

# ============ 2. PHÂN TÍCH TRÙNG LẶP ============
print("\n=== Phân tích tổ hợp chunk_type mỗi doc ===")
types_by_doc = df.groupby("doc_id")["chunk_type"].apply(lambda x: tuple(sorted(set(x))))
for combo, cnt in Counter(types_by_doc).most_common(15):
    print(f"  {cnt:6,} docs: {combo}")

# ============ 3. CHỌN CANONICAL TYPE ============
PRIORITY = [
    "FULL_HARDCAP", "FULL",
    "ARTICLE_HARDCAP", "ARTICLE",
    "ARTICLE_CLAUSES_HARDCAP", "ARTICLE_CLAUSES",
    "NUM_DASH_ITEM_HARDCAP", "NUM_DASH_ITEM",
]
def choose(types_set):
    for t in PRIORITY:
        if t in types_set:
            return t
    return None

canon_map = types_by_doc.apply(lambda s: choose(set(s)))
print(f"\n=== Canonical type được chọn ===")
print(canon_map.value_counts())
df["canonical_type"] = df["doc_id"].map(canon_map)

# ============ 4. LỌC CHỈ CHUNKS THUỘC CANONICAL TYPE ============
df_use = df[df["chunk_type"] == df["canonical_type"]].copy()
print(f"\nChunks giữ lại: {len(df_use):,} / {len(df):,} "
      f"({len(df_use)/len(df)*100:.1f}%)  ← bỏ {len(df)-len(df_use):,} chunks trùng lặp")

# ============ 5. DEDUP TEXT TRONG CÙNG DOC ============
df_use["_h"] = df_use["text"].map(lambda s: hashlib.md5(s.encode()).hexdigest())
n0 = len(df_use)
df_use = df_use.drop_duplicates(subset=["doc_id", "_h"], keep="first")
print(f"Loại thêm {n0-len(df_use):,} chunks trùng nội dung trong cùng doc")

# ============ 6. SORT (NaN-safe) ============
df_use["part_no"]  = df_use["hardcap_part_no"].fillna(0).astype(int)
df_use["tok_start"] = df_use["token_start"].fillna(0).astype(int)
df_use = df_use.sort_values(["doc_id", "part_no", "tok_start"]).reset_index(drop=True)

# ============ 7. MERGE HELPERS ============
def strip_overlap(prev, nxt, window=500, min_ov=20):
    tail, head = prev[-window:], nxt[:window]
    for size in range(min(len(tail), len(head)), min_ov-1, -1):
        if tail[-size:] == head[:size]:
            return nxt[size:]
    return nxt

def merge_chunks(texts):
    if not texts: return ""
    out = texts[0]
    for nxt in texts[1:]:
        out = out.rstrip() + "\n\n" + strip_overlap(out, nxt).lstrip()
    return out.strip()

def est_tokens(text):
    return len(text) // 3  # tiếng Việt ~3 ký tự/token

def hard_split(text, target_tok, overlap_chars):
    target = target_tok * 3
    parts, i = [], 0
    while i < len(text):
        end = min(i + target, len(text))
        if end < len(text):
            nl = text.rfind("\n", i + target//2, end)
            if nl > i: end = nl
        parts.append(text[i:end].strip())
        if end >= len(text): break
        i = max(0, end - overlap_chars)
    return [p for p in parts if p]

def split_by_budget(g, target_tok):
    """Gộp các chunk liên tiếp cho tới target_tok — dùng khi chunk là ARTICLE-based."""
    g = g.sort_values(["part_no","tok_start"]).reset_index(drop=True)
    samples, cur, cur_tok = [], [], 0
    for _, r in g.iterrows():
        t = est_tokens(r["text"]) or 1
        if cur and cur_tok + t > target_tok:
            samples.append(merge_chunks([x["text"] for x in cur]))
            cur, cur_tok = [], 0
        cur.append(r); cur_tok += t
    if cur: samples.append(merge_chunks([x["text"] for x in cur]))
    return samples

# ============ 8. PROCESS ============
print("\nĐang merge & split...")
doc_records, sample_records = [], []
skipped = []

for doc_id, g in tqdm(df_use.groupby("doc_id"), total=df_use["doc_id"].nunique()):
    meta = g.iloc[0]
    full_text = merge_chunks(g.sort_values(["part_no","tok_start"])["text"].tolist())
    tok_full  = est_tokens(full_text)

    # Cảnh báo doc khổng lồ
    if tok_full > 500_000:
        skipped.append((int(doc_id), meta["title"], tok_full))

    header = (
        f"# {meta['title']}\n"
        f"- Số ký hiệu: {meta['so_ky_hieu']}\n"
        f"- Loại: {meta['loai_van_ban']}\n"
        f"- Cơ quan ban hành: {meta['co_quan_ban_hanh']}\n"
        f"- Ngày ban hành: {meta['ngay_ban_hanh']}\n"
        f"- Hiệu lực từ: {meta['effective_from']}\n"
        f"- Tình trạng: {meta['tinh_trang_hieu_luc']}\n"
        f"- Lĩnh vực: {meta['linh_vuc']}\n"
    )

    base = {
        "doc_id": int(doc_id),
        "doc_uid": meta["doc_uid"],
        "title": meta["title"],
        "so_ky_hieu": meta["so_ky_hieu"],
        "loai_van_ban": meta["loai_van_ban"],
        "linh_vuc": meta["linh_vuc"],
        "effective_from": meta["effective_from"],
        "canonical_type": meta["chunk_type"],
        "n_source_chunks": len(g),
        "total_tokens_est": tok_full,
    }

    # --- Tạo samples (split nếu cần) ---
    if tok_full <= TARGET_SAMPLE:
        samples = [full_text]
    elif meta["chunk_type"].startswith("ARTICLE"):
        # Split theo Điều (semantic)
        samples = split_by_budget(g, TARGET_SAMPLE)
    else:
        # FULL / FULL_HARDCAP → hard split
        samples = hard_split(full_text, TARGET_SAMPLE, OVERLAP_CHARS)

    for i, s in enumerate(samples):
        sample_records.append({
            **base,
            "sample_id": f"{doc_id}_{i}",
            "n_samples": len(samples),
            "header": header,
            "content": s,
            "content_tokens": est_tokens(s),
        })

    # --- Doc ≤ 32k → ghi full doc ---
    if tok_full <= MAX_CTX:
        doc_records.append({**base, "header": header, "content": full_text})

df_doc    = pd.DataFrame(doc_records)
df_sample = pd.DataFrame(sample_records)

df_doc.to_json(OUTPUT_DIR / "merged_documents.jsonl",
               orient="records", lines=True, force_ascii=False)
df_sample.to_json(OUTPUT_DIR / "training_samples.jsonl",
                  orient="records", lines=True, force_ascii=False)

# ============ 9. BÁO CÁO ============
print("\n=== KẾT QUẢ ===")
print(f"merged_documents.jsonl : {len(df_doc):>8,} docs  (≤ {MAX_CTX:,} tokens)")
print(f"training_samples.jsonl : {len(df_sample):>8,} samples (≤ {TARGET_SAMPLE:,} tokens)")
print(f"\nSize sample (tokens):")
print(df_sample["content_tokens"].describe(percentiles=[.5, .9, .99]).round(0))

if skipped:
    print(f"\n⚠️  {len(skipped)} docs > 500k tokens (kiểm tra thủ công):")
    for d, t, n in skipped[:10]:
        print(f"   doc {d}: {n:>8,} tokens — {t[:80]}")

# Xem nhanh 1 sample
print("\n--- Mẫu sample[0] ---")
print(df_sample.iloc[0]["header"])
print(df_sample.iloc[0]["content"][:800], "...")
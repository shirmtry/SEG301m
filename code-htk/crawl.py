"""build_faiss_huytking.py — Build FAISS từ HUYTKING.jsonl.
   Đọc: HUYTKING.jsonl. Ghi: faiss_index_HUYTKING/ (thư mục MỚI).
   KHÔNG đụng faiss_index_new/ (index cũ). KHÔNG đụng legal_merged.jsonl."""
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
SRC = BASE / "HUYTKING.jsonl"
OUT_DIR = BASE / "faiss_index_new_v1"   # ← thư mục MỚI

def main():
    if OUT_DIR.exists():
        print(f"⚠️  {OUT_DIR.name} đã tồn tại. Xóa/đổi tên trước khi build lại.")
        return
    print(f"📂 Đọc {SRC.name}...")
    docs = []
    with open(SRC, "r", encoding="utf-8") as f:   # CHỈ ĐỌC
        for line in f:
            r = json.loads(line)
            title = r.get("metadata", {}).get("title", "")
            docs.append(Document(page_content=r["text"], metadata={"title": title}))
    print(f"✅ {len(docs):,} docs")

    print("🔧 Load embeddings...")
    emb = HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("🔨 Build FAISS (5-15 phút)...")
    vs = FAISS.from_documents(docs, emb)
    print(f"💾 Save → {OUT_DIR.name}/")
    vs.save_local(str(OUT_DIR))
    print(f"\n✅ DONE. Bước tiếp: sửa app.py.")

if __name__ == "__main__":
    main()
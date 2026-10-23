"""retest_miss.py — Re-test 143 câu data miss với retrieve_advanced MỚI.
   CHỈ ĐỌC:
     - HUYTKING.jsonl
     - test_results/20261022_235732_914new.csv
   GHI:
     - test_results/retest_miss.csv (file mới)
"""
import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from llama_cpp import Llama
from langchain_classic.retrievers import EnsembleRetriever

# ─── CONFIG ───
BASE = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
FAISS_DIR = BASE / "faiss_index_new_v1"
GGUF_FILE = BASE / "qwen_model" / "qwen2.5-3b-instruct-q4_k_m.gguf"
MERGED_FILE = BASE / "HUYTKING.jsonl"
CSV_OLD = BASE / "test_results" / "20261022_235732_914new.csv"
OUT = BASE / "test_results" / "retest_miss.csv"

# ─── Số câu test (đổi thành None để test FULL) ───
LIMIT = 20   # test 20 câu mẫu trước, đổi thành None để test full 143

# ═══════════════════════════════════════════════════════════
# LAW_KEYWORDS — 58 LUẬT (38 cũ + 20 mới)
# ═══════════════════════════════════════════════════════════
LAW_KEYWORDS = {
    # ────────── 25 luật CŨ ──────────
    "Hình sự": ["hình sự", "tội phạm", "truy cứu", "tnhs"],
    "Tố tụng hình sự": ["tố tụng hình sự", "bltths", "điều tra", "khởi tố"],
    "Dân sự": ["dân sự", "blDS"],
    "Tố tụng dân sự": ["tố tụng dân sự", "blttds"],
    "Hôn nhân": ["hôn nhân", "gia đình", "ly hôn", "kết hôn", "tài sản chung"],
    "Lao động": ["lao động", "bllđ", "hợp đồng lao động", "sa thải"],
    "Đất đai": ["đất đai", "sổ đỏ", "quyền sử dụng đất", "giao đất"],
    "Doanh nghiệp": ["doanh nghiệp", "công ty", "tnhh", "cổ phần", "hợp danh"],
    "Nhà ở": ["nhà ở", "chung cư", "nhà xã hội"],
    "Đầu tư": ["đầu tư", "fdi", "nhà đầu tư nước ngoài"],
    "Xử lý vi phạm hành chính": ["xử phạt", "vi phạm hành chính", "xlvphc"],
    "Tố tụng hành chính": ["tố tụng hành chính", "khiếu kiện"],
    "Nghĩa vụ quân sự": ["nghĩa vụ quân sự", "nhập ngũ", "quân dịch"],
    "Hàng hải": ["hàng hải", "tàu biển", "thuyền trưởng"],
    "Thương mại": ["thương mại", "mua bán hàng hóa", "thương nhân"],
    "Các tổ chức tín dụng": ["tổ chức tín dụng", "ngân hàng", "tín dụng"],
    "Quản lý thuế": ["quản lý thuế", "thuế gtgt", "thuế tncn", "thuế tndn"],
    "Sở hữu trí tuệ": ["sở hữu trí tuệ", "bản quyền", "nhãn hiệu", "sáng chế"],
    "Hộ tịch": ["hộ tịch", "khai sinh", "khai tử", "đăng ký kết hôn"],
    "Giao thông": ["giao thông", "đường bộ", "lái xe", "giấy phép lái"],
    "Bảo hiểm xã hội": ["bảo hiểm xã hội", "bhxh", "hưu trí", "thai sản"],
    "Giáo dục": ["giáo dục", "trường học", "học sinh", "đại học"],
    "Đường bộ": ["đường bộ", "cao tốc", "quốc lộ"],
    "Cư trú": ["cư trú", "tạm trú", "thường trú", "sổ hộ khẩu"],
    "Căn cước": ["căn cước", "cccd", "cmnd", "chứng minh nhân dân"],
    # ────────── 13 luật MỚI ──────────
    "Chứng khoán": ["chứng khoán", "cổ phiếu", "trái phiếu", "quỹ đầu tư",
                    "sở giao dịch chứng khoán", "công ty chứng khoán"],
    "Thi hành án dân sự": ["thi hành án dân sự", "chấp hành viên", "thads"],
    "Thi hành án hình sự": ["thi hành án hình sự", "trại giam", "án treo", "tha tù"],
    "Cán bộ công chức": ["cán bộ", "công chức", "viên chức", "biên chế"],
    "Phòng chống tham nhũng": ["tham nhũng", "pctn", "tài sản kê khai"],
    "Xuất nhập cảnh": ["xuất cảnh", "nhập cảnh", "visa", "thị thực", "hộ chiếu"],
    "Quốc tịch": ["quốc tịch", "nhập tịch", "thôi quốc tịch"],
    "Đấu thầu": ["đấu thầu", "gói thầu", "nhà thầu", "hồ sơ dự thầu"],
    "Hải quan": ["hải quan", "xuất nhập khẩu", "mã hs", "tờ khai hải quan"],
    "Trọng tài": ["trọng tài", "trung tâm trọng tài", "phán quyết trọng tài"],
    "Công nghệ thông tin": ["công nghệ thông tin", "phần mềm", "cntt"],
    "An toàn thông tin mạng": ["an toàn thông tin", "dữ liệu cá nhân", "tấn công mạng"],
    "Ban hành VBQPPL": ["văn bản quy phạm", "nghị định", "thông tư", "quyết định"],
    # ────────── 20 luật BỔ SUNG ──────────
    "Xây dựng": ["xây dựng"],
    "Cạnh tranh": ["cạnh tranh"],
    "Tổ chức Tòa án nhân dân": ["tổ chức tòa án", "tòa án nhân dân"],
    "Tổ chức Viện kiểm sát nhân dân": ["tổ chức viện kiểm sát", "viện kiểm sát nhân dân"],
    "Tổ chức Quốc hội": ["tổ chức quốc hội"],
    "Tổ chức Chính phủ": ["tổ chức chính phủ"],
    "Tổ chức chính quyền địa phương": ["chính quyền địa phương"],
    "Kinh doanh bất động sản": ["kinh doanh bất động sản"],
    "Hiến pháp": ["hiến pháp"],
    "An toàn thực phẩm": ["an toàn thực phẩm"],
    "Khám bệnh chữa bệnh": ["khám bệnh", "chữa bệnh"],
    "Giao dịch điện tử": ["giao dịch điện tử"],
    "Khiếu nại": ["khiếu nại"],
    "Tố cáo": ["tố cáo"],
    "Bảo hiểm y tế": ["bảo hiểm y tế"],
    "An toàn vệ sinh lao động": ["an toàn vệ sinh lao động"],
    "Phá sản": ["phá sản"],
    "Việc làm": ["việc làm"],
    "Tiếp cận thông tin": ["tiếp cận thông tin"],
    "Phòng cháy chữa cháy": ["phòng cháy", "chữa cháy"],
}


def detect_law_keywords(text):
    t = text.lower()
    return [law for law, kws in LAW_KEYWORDS.items() if any(kw in t for kw in kws)]


def title_matches_law(title, law_kw):
    t, kw = title.lower(), law_kw.lower()
    if kw not in t:
        return False
    if kw == "hình sự" and "tố tụng hình sự" in t and "bộ luật hình sự" not in t:
        return False
    if kw == "dân sự" and ("tố tụng dân sự" in t or "thi hành án dân sự" in t) and "bộ luật dân sự" not in t:
        return False
    if kw == "hành chính" and "tố tụng hành chính" in t:
        return False
    return True


def extract_law_from_question(q):
    """Trích tên luật từ câu hỏi — fallback khi LAW_KEYWORDS không match."""
    patterns = [
        r"(?:Theo|theo|trong|của)\s+(Luật[^,\.]+|Bộ luật[^,\.]+)",
        r"[Đđ]iều\s+\d+\s+(Luật[^,\.\?]+|Bộ luật[^,\.\?]+)\s+(?:quy định|-)",
        r"(Luật[^\-]+?|Bộ luật[^\-]+?)\s+-\s+[Đđ]iều",
        r"của\s+(Luật[^,\.\?]+|Bộ luật[^,\.\?]+)",
    ]
    for p in patterns:
        m = re.search(p, q, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def clean_law_name(s):
    """Bỏ '(VBHN 2025)', số hiệu, khoảng trắng thừa."""
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"\s*số\s+\S+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


# ═══════════════════════════════════════════════════════════
# RETRIEVE MỚI
# ═══════════════════════════════════════════════════════════
def retrieve_advanced(retriever, question, all_docs, k=5):
    ql = question.lower()
    dieu_numbers = re.findall(r"[Đđ]iều\s+(\d+)", question)
    law_kw = detect_law_keywords(question)
    law_from_q = extract_law_from_question(question)

    is_minor = (
        ("chưa thành niên" in ql or "vị thành niên" in ql or "dưới 18" in ql)
        and any(w in ql for w in ["phạm tội", "hình sự", "truy cứu", "trách nhiệm hình sự"])
    )

    if is_minor:
        minor = [d for d in all_docs
                 if "Bộ luật Hình sự" in d.metadata.get("title", "")
                 and any(x in d.page_content.lower()
                         for x in ["chưa thành niên", "dưới 18", "18 tuổi"])]
        docs_to_use = minor[:k] if minor else []

    elif dieu_numbers:
        law_str = None
        if law_from_q:
            law_str = clean_law_name(law_from_q)
        elif law_kw:
            law_str = law_kw[0].lower()

        exact_strict = []
        exact_any = []
        for d in all_docs:
            t = d.metadata.get("title", "")
            t_low = t.lower()
            if not any(f"Điều {n}." in t or f"Điều {n} " in t or f"Điều {n}:" in t
                       for n in dieu_numbers):
                continue
            exact_any.append(d)
            if law_str and law_str in t_low:
                exact_strict.append(d)
            elif law_kw and any(title_matches_law(t, kw) for kw in law_kw):
                exact_strict.append(d)

        if exact_strict:
            docs_to_use = exact_strict[:k]
        elif len(exact_any) == 1:
            docs_to_use = exact_any
        elif 2 <= len(exact_any) <= 5 and law_str:
            loose = [d for d in exact_any
                     if law_str[:20] in d.metadata.get("title", "").lower()]
            docs_to_use = loose[:k] if loose else exact_any[:k]
        else:
            docs_to_use = retriever.invoke(question)[:k]

    elif law_kw or law_from_q:
        raw = retriever.invoke(question)
        kws = law_kw if law_kw else ([clean_law_name(law_from_q)] if law_from_q else [])
        docs_to_use = sorted(
            raw,
            key=lambda d: -sum(1 for kw in kws if kw.lower() in d.metadata.get("title", "").lower()),
        )[:k]

    else:
        docs_to_use = retriever.invoke(question)[:k]

    contexts, sources = [], []
    for i, d in enumerate(docs_to_use):
        c = d.page_content
        trunc = c[:1200] + "..." if len(c) > 1200 else c
        contexts.append(f"[{i+1}] {d.metadata.get('title', 'Không rõ')}\n{trunc}")
        sources.append({"rank": i + 1, "title": d.metadata.get("title", "Không rõ")})
    return contexts, sources


# ═══════════════════════════════════════════════════════════
# LOAD ENGINE
# ═══════════════════════════════════════════════════════════
def load_engine():
    print("🔧 Loading embeddings...", flush=True)
    emb = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert",
                                 model_kwargs={"device": "cpu"},
                                 encode_kwargs={"normalize_embeddings": True})
    print("🔧 Loading FAISS...", flush=True)
    vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
    docs = []
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d.get("metadata", {}).get("title", "Không rõ")
            docs.append(Document(page_content=d["text"], metadata={"title": t}))
    print(f"🔧 {len(docs):,} docs. Build retrievers...", flush=True)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 8
    fr = vs.as_retriever(search_kwargs={"k": 8})
    ens = EnsembleRetriever(retrievers=[bm25, fr], weights=[0.4, 0.6])
    print("🔧 Loading LLM...", flush=True)
    llm = Llama(
        model_path=str(GGUF_FILE),
        n_ctx=4096, n_threads=8, n_batch=1024, n_gpu_layers=0,
        verbose=False, repeat_penalty=1.3, top_k=40, top_p=0.9,
    )
    print("✅ Engine ready.\n", flush=True)
    return ens, llm, docs


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    # 1. Đọc CSV cũ, lọc miss
    if not CSV_OLD.exists():
        print(f"❌ Không tìm thấy: {CSV_OLD}")
        return
    df = pd.read_csv(CSV_OLD, encoding="utf-8-sig", on_bad_lines="skip")
    miss = df[(df["expected"] == "answer") & (df["verdict"] == "refused")].copy()
    print(f"📋 Có {len(miss)} câu miss (lẽ ra answer, bị refuse)")
    if LIMIT:
        miss = miss.head(LIMIT)
        print(f"⚠️  CHỈ TEST {len(miss)} CÂU ĐẦU (LIMIT={LIMIT})")
    print()

    # 2. Load engine
    ens, llm, docs = load_engine()

    # 3. Header CSV mới
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["idx", "question", "old_verdict", "new_verdict",
                    "new_answer", "n_ctx", "time_sec"])

    # 4. Re-test
    t_start = time.time()
    counts = {"answered": 0, "refused": 0, "no_ctx": 0, "error": 0}

    for i, (_, row) in enumerate(miss.iterrows(), 1):
        q = str(row["question"]).strip()
        t0 = time.time()
        new_verdict = "error"
        answer = ""
        n_ctx = 0
        try:
            ctxs, _ = retrieve_advanced(ens, q, docs, k=5)
            n_ctx = len(ctxs)
            if not ctxs:
                new_verdict = "no_ctx"
                answer = ""
            else:
                system = (
                    "Bạn là chuyên gia tư vấn pháp luật Việt Nam.\n"
                    "CHỈ dùng thông tin trong NGỮ CẢNH. KHÔNG bịa.\n"
                    "Nếu ngữ cảnh không liên quan → 'Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có.'"
                )
                user_msg = "NGỮ CẢNH:\n" + "\n\n---\n\n".join(ctxs) + f"\n\nCÂU HỎI: {q}"
                prompt = (
                    f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                out = llm(prompt, max_tokens=400, temperature=0.1,
                          stop=["<|im_end|>", "<|im_start|>"])
                answer = out["choices"][0]["text"].strip()
                if "không tìm thấy" in answer.lower():
                    new_verdict = "refused"
                else:
                    new_verdict = "answered"
        except Exception as e:
            new_verdict = "error"
            answer = f"ERROR: {e}"

        dt = time.time() - t0
        counts[new_verdict] = counts.get(new_verdict, 0) + 1

        with open(OUT, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow([i, q, row["verdict"], new_verdict,
                        answer.replace("\n", " ")[:500], n_ctx, round(dt, 2)])

        mark = "✅" if new_verdict == "answered" else ("⚠️ " if new_verdict == "refused" else "❌")
        print(f"[{i:3d}/{len(miss)}] {mark} {new_verdict:10s} ({dt:5.1f}s) ctx={n_ctx} | {q[:60]}", flush=True)

    # 5. Summary
    total_time = time.time() - t_start
    print(f"\n{'='*78}")
    print(f"📊 KẾT QUẢ RE-TEST ({len(miss)} câu)")
    print(f"{'='*78}")
    for v, c in counts.items():
        pct = c / len(miss) * 100
        print(f"  {v:12s}: {c:3d} ({pct:5.1f}%)")

    improved = counts.get("answered", 0)
    print(f"\n🎯 Cải thiện: {improved}/{len(miss)} câu từ refused → answered "
          f"({improved/len(miss)*100:.1f}%)")
    print(f"⏱  Thời gian: {total_time/60:.1f} phút ({total_time/len(miss):.1f}s/câu)")
    print(f"📁 CSV: {OUT.name}")

    if improved / len(miss) >= 0.8:
        print(f"\n✅ Fix HIỆU QUẢ (≥80%). Có thể áp dụng cho app.py + test full 143 câu.")
    elif improved / len(miss) >= 0.5:
        print(f"\n⚠️  Fix TẠM ỔN (50-80%). Nên test full 143 để xem chi tiết.")
    else:
        print(f"\n❌ Fix CHƯA ĐỦ (<50%). Cần phân tích sâu hơn.")


if __name__ == "__main__":
    main()
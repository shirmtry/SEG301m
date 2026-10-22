"""test_914.py — Test 914 câu tự động, resume được, ghi CSV ngay.
   Sinh 457 from_data + 274 trap + 183 edge = 914 câu.
   Chạy: python code-htk/test_914.py
   Resume: chạy lại cùng lệnh, tự skip câu đã có trong CSV mới nhất.
"""
import json
import re
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from llama_cpp import Llama
from langchain_classic.retrievers import EnsembleRetriever

# ─── CONFIG ───
BASE_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
FAISS_DIR = BASE_DIR / "faiss_index_new_v1"
GGUF_FILE = BASE_DIR / "qwen_model" / "qwen2.5-3b-instruct-q4_k_m.gguf"
MERGED_FILE = BASE_DIR / "HUYTKING.jsonl"
OUTPUT_DIR = BASE_DIR / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)
TARGET_TOTAL = 914

# ─── LAW_KEYWORDS + title_matches_law (copy y hệt v8) ───
LAW_KEYWORDS = {
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


# ─── SYSTEM / VALIDATORS ───
SYSTEM = (
    "Bạn là Chuyên gia Tư vấn Pháp luật Việt Nam. BẮT BUỘC tuân thủ:\n"
    "1. CHỈ dùng thông tin trong 'NGỮ CẢNH'. KHÔNG dùng kiến thức bên ngoài.\n"
    "2. KHÔNG bịa số hiệu, số điều, số tiền, số ngày nếu không có trong NGỮ CẢNH.\n"
    "3. KHÔNG bịa tên luật.\n"
    "4. Nếu câu hỏi nêu NĂM cụ thể MÀ NGỮ CẢNH KHÔNG CÓ → 'Tôi không tìm thấy văn bản này trong cơ sở dữ liệu hiện có.' và DỪNG.\n"
    "5. Nếu NGỮ CẢNH không liên quan → 'Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có.' và DỪNG.\n"
    "6. Trích dẫn nguồn bằng [1], [2], [3]… (số từ 1 đến {n_sources}). "
    "TUYỆT ĐỐI KHÔNG viết [N] hay bất kỳ chữ cái nào trong ngoặc vuông. "
    "[1] nghĩa là 'nguồn số 1', KHÔNG phải 'Điều 1'.\n"
    "7. KHÔNG lặp câu/mục. Tối đa 7 mục.\n"
    "8. KHÔNG copy 'Câu trả lời dựa trên ngữ cảnh:'...\n"
    "9. Tiếng Việt, tối đa 200 từ.\n"
    "10. CẤM GHÉP THÔNG TIN từ 2 nguồn khác chủ đề.\n"
    "11. TRÍCH DẪN ĐÚNG tên văn bản + số điều. Khi liệt kê điều luật, viết 'Điều 12', 'Điều 134'… "
    "TUYỆT ĐỐI KHÔNG viết 'Điều [2]' hay 'số [2]'.\n"
    "12. PHÂN BIỆT HÌNH SỰ vs HÀNH CHÍNH.\n"
    "13. NGƯỜI CHƯA THÀNH NIÊN: theo Điều 12 BLHS.\n"
)
STOP_TOKENS = ["<|im_end|>", "<|im_start|>", "\n\n\n\n"]
LOOP_TRIGGERS = ["Câu trả lời dựa trên ngữ cảnh:", "Câu trả lời dựa trên context:",
                 "TỪ ĐOẠN NGỮ CẢNH:", "TỪ Đoạn NGỮ CẢNH:", "Dựa trên ngữ cảnh:", "Dựa vào context:"]
CJK_PATTERN = re.compile(
    r"[\u3000-\u303f\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\uff00-\uffef]+"
)
DANGEROUS_PATTERNS = [
    (r"B2.{0,80}(A1|mô\s*tô|125\s*cc|xe\s*máy)", "B2→A1"),
    (r"chưa\s+thành\s+ni[êe]?[nm]?.{0,300}"
     r"(không\s+(?:phải\s+)?bị\s+xử\s+lý\s+hình\s+sự|"
     r"không\s+phải\s+chịu\s+trách\s+nhiệm\s+hình\s+sự|"
     r"không\s+bị\s+truy\s+cứu)", "minor_no_TNHS"),
    (r"Bộ\s+luật\s+Xử\s+lý\s+(?:Vụ\s+Phạm\s+Pháp|Vi\s+phạm\s+Pháp|Hành\s+chính)", "fake_law_name"),
]


def has_loop_inline(text, w=3):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return len(lines) >= w and len(set(lines[-w:])) == 1


def clean_answer(text):
    if not text:
        return text
    for r in LOOP_TRIGGERS:
        text = text.replace(r, "")
    text = CJK_PATTERN.sub("", text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    seen, out = {}, []
    for line in lines:
        if seen.get(line, 0) >= 2:
            break
        seen[line] = seen.get(line, 0) + 1
        out.append(line)
    return "\n".join(out).strip()


def is_dangerous_answer(a, ctxs):
    c = " ".join(ctxs).lower()
    for p, r in DANGEROUS_PATTERNS:
        if re.search(p, a, re.I) and not re.search(p, c, re.I):
            return True, r
    return False, ""


def has_fake_citation(a, n):
    if "[N]" in a or "[n]" in a:
        return True
    return any(int(x) > n for x in re.findall(r"\[(\d+)\]", a))


# FIX 1: Phát hiện model dùng [N] như số điều/khoản
def has_citation_misuse(text):
    """Phát hiện model dùng [N] như số điều/khoản (VD: 'Điều [2]', 'số [2] của')."""
    patterns = [
        r"Điều\s+\[\d+\]",
        r"khoản\s+\[\d+\]",
        r"số\s+\[\d+\]\s+của",
        r"điều\s+khoản\s+số\s+\[\d+\]",
        r"tại\s+\[\d+\]",
    ]
    return any(re.search(p, text, re.I) for p in patterns)


def is_topic_missing(q, ctxs):
    ql, c = q.lower(), " ".join(ctxs).lower()
    for qk, ck, name in [
        (["chưa thành niên", "vị thành niên"],
         ["chưa thành niên", "vị thành niên", "dưới 18", "18 tuổi"], "minor"),
        (["bằng lái b2", "hạng b2"], ["b2", "hạng b2"], "b2"),
        (["nghị định 168"], ["168/2024", "nghị định 168"], "nd168"),
    ]:
        if any(k in ql for k in qk) and not any(k in c for k in ck):
            return True, f"topic_missing_{name}"
    return False, ""


def is_uncited_answer(a, sources):
    if not sources or "không tìm thấy" in a.lower():
        return False
    return not bool(re.search(r"\[\d+\]", a))


# FIX 2: Hạ ngưỡng từ 150 → 60 để bắt mid_refusal sớm hơn
def has_mid_refusal(text):
    low = text.lower()
    marker = "không tìm thấy thông tin này"
    idx = low.find(marker)
    if idx < 0:
        return False
    return idx < len(text) - 60


# ─── LOAD ENGINE ───
def load_engine():
    print("🔧 Loading embeddings...", flush=True)
    emb = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert",
                                 model_kwargs={"device": "cpu"},
                                 encode_kwargs={"normalize_embeddings": True})
    print("🔧 Loading FAISS...", flush=True)
    vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
    docs = []
    if MERGED_FILE.exists():
        with open(MERGED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                t = d.get("metadata", {}).get("title", "Không rõ")
                docs.append(Document(page_content=d["text"], metadata={"title": t}))
    print(f"🔧 {len(docs):,} docs. Build BM25 + FAISS retriever...", flush=True)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 8
    fr = vs.as_retriever(search_kwargs={"k": 8})
    ens = EnsembleRetriever(retrievers=[bm25, fr], weights=[0.4, 0.6])
    print("🔧 Loading LLM...", flush=True)
    llm = Llama(
        model_path=str(GGUF_FILE),
        n_ctx=4096, n_threads=8, n_batch=1024, n_gpu_layers=0,
        verbose=False, repeat_penalty=1.3, top_k=40, top_p=0.9, use_mlock=True,
    )
    print("✅ Engine ready.", flush=True)
    return ens, llm, docs


# ─── RETRIEVE (copy y hệt v8) ───
def retrieve_advanced(retriever, question, all_docs, k=5):
    ql = question.lower()
    dieu = re.findall(r"[Đđ]iều\s+(\d+)", question)
    law_kw = detect_law_keywords(question)

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
    elif dieu and law_kw:
        exact = []
        for d in all_docs:
            t = d.metadata.get("title", "")
            md = any(f"Điều {n}." in t or f"Điều {n} " in t or f"Điều {n}:" in t for n in dieu)
            ml = any(title_matches_law(t, kw) for kw in law_kw)
            if md and ml:
                exact.append(d)
        docs_to_use = exact[:k] if exact else retriever.invoke(question)[:k]
    elif dieu:
        exact = [d for d in all_docs
                 if any(f"Điều {n}." in d.metadata.get("title", "")
                        or f"Điều {n} " in d.metadata.get("title", "")
                        for n in dieu)]
        docs_to_use = exact if len(exact) == 1 else retriever.invoke(question)[:k]
    elif law_kw:
        raw = retriever.invoke(question)
        docs_to_use = sorted(
            raw,
            key=lambda d: -sum(1 for kw in law_kw if title_matches_law(d.metadata.get("title", ""), kw)),
        )[:k]
    else:
        docs_to_use = retriever.invoke(question)[:k]

    ctxs, srcs = [], []
    for i, d in enumerate(docs_to_use):
        c = d.page_content[:1500] + "..." if len(d.page_content) > 1500 else d.page_content
        ctxs.append(f"[{i+1}] {d.metadata.get('title', 'Không rõ')}\n{c}")
        srcs.append({"rank": i + 1, "title": d.metadata.get("title", "Không rõ")})
    return ctxs, srcs


# ─── GENERATE ───
def generate_one(llm, question, contexts):
    n_sources = len(contexts)
    system = SYSTEM.replace("{n_sources}", str(n_sources))
    user_msg = "NGỮ CẢNH:\n" + "\n\n---\n\n".join(contexts) + f"\n\nCÂU HỎI: {question}"
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    out = llm(prompt, max_tokens=600, temperature=0.1,
              repeat_penalty=1.3, stop=STOP_TOKENS)
    return out["choices"][0]["text"]


# ─── QUESTION BANK: 914 = 457 + 274 + 183 ───
def build_question_bank(all_docs, n_data=457, n_trap=274, n_edge=183):
    random.seed(2026)
    qs = []

    # 457 from_data
    candidates = [d for d in all_docs if re.search(r"Điều\s+\d+", d.metadata.get("title", ""))]
    sampled = random.sample(candidates, min(n_data, len(candidates)))
    for d in sampled:
        title = d.metadata.get("title", "")
        m = re.search(r"Điều\s+(\d+)", title)
        if not m:
            continue
        dieu = m.group(1)
        law = title.split(" - ")[0].strip() if " - " in title else "pháp luật"
        tpl = random.choice([
            f"Điều {dieu} {law} quy định gì?",
            f"Cho tôi biết nội dung Điều {dieu} của {law}.",
            f"{law} - Điều {dieu} quy định như thế nào?",
        ])
        qs.append({"category": "from_data", "question": tpl, "expected": "answer", "ref_title": title})

    # 274 trap
    base_traps = [
        "Mức phạt vượt đèn đỏ xe máy theo NĐ 168/2024 là bao nhiêu?",
        "Nồng độ cồn cho phép khi lái xe năm 2025 là bao nhiêu?",
        "Bằng lái B2 lái được xe gì?",
        "Bằng lái A2 lái được xe gì theo luật 2025?",
        "Mức xử phạt không đội mũ bảo hiểm 2025?",
        "Tốc độ tối đa trên cao tốc 2025?",
        "Giá Bitcoin hôm nay bao nhiêu?",
        "Thời tiết Hà Nội ngày mai thế nào?",
        "Cách nấu phở ngon?",
        "Cách học tiếng Anh hiệu quả?",
        "Giá vàng SJC hôm nay bao nhiêu?",
        "Đội tuyển Việt Nam thắng chưa?",
        "Lương tối thiểu vùng 1 năm 2026?",
        "Thuế VAT 2026 là bao nhiêu?",
        "Luật Đất đai 2024 quy định thời hạn giao đất nông nghiệp?",
        "Nghị định 168/2024 xử phạt vượt đèn đỏ xe máy thế nào?",
        "Luật Trật tự ATGT 2024 có hiệu lực khi nào?",
        "Mức phạt nồng độ cồn theo NĐ 100/2019?",
        "Quy định mới về đất đai 2025?",
        "Luật Nhà ở 2024 có gì mới?",
    ]
    topics = ["giao thông", "thuế", "hải quan", "chứng khoán", "bất động sản",
              "ngân hàng", "bảo hiểm", "chứng chỉ hành nghề", "giấy phép lao động"]
    years = [2024, 2025, 2026, 2027, 2030, 2035]
    traps = list(base_traps)
    while len(traps) < n_trap:
        traps.append(f"Quy định về {random.choice(topics)} năm {random.choice(years)} là gì?")
    for t in traps[:n_trap]:
        qs.append({"category": "trap", "question": t, "expected": "refuse"})

    # 183 edge
    edge = [
        "Ly hôn", "Bị phạt bao nhiêu?", "Luật mới nhất?", "Như thế nào?", "Tại sao?",
        "Có không?", "Nghỉ hưu tuổi bao nhiêu?", "Thuế thu nhập cá nhân?",
        "Đăng ký kết hôn?", "Làm sao để khởi kiện?", "Điều 15?", "Khoản 2?",
        "Hợp đồng lao động?", "Bảo hiểm xã hội?", "Đất ở?", "Di chúc?", "Thừa kế?",
        "Quyền nuôi con?", "Cấp dưỡng?", "Tranh chấp đất đai?",
    ]
    edge_topics = ["nhà ở", "đất đai", "hôn nhân", "lao động", "doanh nghiệp",
                   "chứng khoán", "hải quan", "đấu thầu", "trọng tài", "quốc tịch"]
    while len(edge) < n_edge:
        edge.append(f"Quy định về {random.choice(edge_topics)}?")
    for e in edge[:n_edge]:
        qs.append({"category": "edge", "question": e, "expected": "any"})

    return qs[:n_data + n_trap + n_edge]


# ─── RUN ONE (FIX 3: thêm misuse check) ───
def run_one(retriever, llm, all_docs, question):
    t0 = time.time()
    ctxs, srcs = retrieve_advanced(retriever, question, all_docs, k=5)
    if not ctxs:
        return {"answer": "Tôi không tìm thấy thông tin này...", "verdict": "refused",
                "reason": "no_context", "time": time.time() - t0,
                "n_citations": 0, "n_contexts": 0}
    raw = generate_one(llm, question, ctxs)
    ans = clean_answer(raw) or "Tôi không tìm thấy thông tin này..."
    if has_loop_inline(ans):
        ans = "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có."
    danger, dr = is_dangerous_answer(ans, ctxs)
    fake = has_fake_citation(ans, len(ctxs))
    tm, tr = is_topic_missing(question, ctxs)
    mid = has_mid_refusal(ans)
    misuse = has_citation_misuse(ans)
    uncited = is_uncited_answer(ans, srcs)
    refused = "không tìm thấy" in ans.lower()

    if refused:
        verdict, reason = "refused", "model_refused"
    elif danger:
        verdict, reason = "blocked", f"danger:{dr}"
    elif fake:
        verdict, reason = "blocked", "fake_citation"
    elif tm:
        verdict, reason = "blocked", tr
    elif mid:
        verdict, reason = "blocked", "mid_refusal"
    elif misuse:
        verdict, reason = "blocked", "citation_misuse"
    elif uncited:
        verdict, reason = "warning", "uncited"
    else:
        verdict, reason = "answered", "ok"

    return {"answer": ans, "verdict": verdict, "reason": reason,
            "time": time.time() - t0,
            "n_citations": len(re.findall(r"\[\d+\]", ans)),
            "n_contexts": len(ctxs)}


# ─── RESUME ───
def find_latest_csv():
    files = sorted(OUTPUT_DIR.glob("*_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_done_idx(csv_path):
    if csv_path is None or not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        return set(df["idx"].astype(int).tolist())
    except Exception:
        return set()


# ─── PRE-FLIGHT ───
def preflight(retriever, llm, all_docs):
    print("\n🚦 PRE-FLIGHT CHECK (đo tốc độ 1 câu)...", flush=True)
    test_q = "Điều 5 Luật Đấu thầu quy định gì?"
    t0 = time.time()
    try:
        _ = run_one(retriever, llm, all_docs, test_q)
        dt = time.time() - t0
    except Exception as e:
        print(f"   ❌ Pre-flight fail: {e}")
        return False, 0
    print(f"   ⏱️  1 câu = {dt:.1f}s")
    eta_914 = dt * TARGET_TOTAL / 3600
    print(f"   ⏳ 914 câu ước tính: {eta_914:.1f} giờ")
    if dt > 60:
        print(f"   ⚠️  CHẬM — 914 câu sẽ >15 giờ, KHÔNG kịp trước 7:00!")
    elif dt > 35:
        print(f"   ⚠️  Hơi chậm — sẽ hơn 9 giờ, cần chạy kèm resume.")
    return True, dt


# ─── MAIN ───
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=TARGET_TOTAL)
    ap.add_argument("--fresh", action="store_true", help="Bỏ qua resume, tạo CSV mới")
    args = ap.parse_args()

    # 1. Resume logic
    latest_csv = None if args.fresh else find_latest_csv()
    done_idx = load_done_idx(latest_csv)
    if done_idx:
        print(f"♻️  Resume từ: {latest_csv.name} — đã xong {len(done_idx)} câu")
        csv_path = latest_csv
        ts = latest_csv.stem.replace("_results", "")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"{ts}_results.csv"
        print(f"🆕 Bắt đầu mới: {csv_path.name}")

    # 2. Load engine
    retriever, llm, all_docs = load_engine()

    # 3. Build 914 câu
    qs = build_question_bank(all_docs)
    print(f"📝 Tổng câu bank: {len(qs)} (457 from_data + 274 trap + 183 edge)")
    qs = qs[:args.limit]
    remaining = [q for i, q in enumerate(qs, 1) if i not in done_idx]
    print(f"▶️  Cần làm: {len(remaining)} / {len(qs)} câu\n")

    # 4. Pre-flight (chỉ chạy nếu bắt đầu mới)
    if not done_idx:
        ok, dt = preflight(retriever, llm, all_docs)
        if not ok:
            return

    # 5. Ghi header CSV nếu file mới
    if csv_path.stat().st_size == 0 if csv_path.exists() else True:
        header = ["idx", "category", "expected", "question", "answer",
                  "verdict", "reason", "time_sec", "n_citations", "n_contexts", "ts_utc"]
        with open(csv_path, "w", encoding="utf-8-sig") as f:
            f.write(",".join(header) + "\n")

    # 6. Loop
    t_start = time.time()
    n_done = len(done_idx)
    for i, q in enumerate(qs, 1):
        if i in done_idx:
            continue
        try:
            r = run_one(retriever, llm, all_docs, q["question"])
        except Exception as e:
            r = {"answer": f"ERROR:{e}", "verdict": "error", "reason": str(e)[:80],
                 "time": 0, "n_citations": 0, "n_contexts": 0}

        row = {
            "idx": i,
            "category": q["category"],
            "expected": q["expected"],
            "question": q["question"],
            "answer": r["answer"].replace('"', "'").replace("\n", " "),
            "verdict": r["verdict"],
            "reason": r["reason"],
            "time_sec": round(r["time"], 2),
            "n_citations": r["n_citations"],
            "n_contexts": r["n_contexts"],
            "ts_utc": datetime.utcnow().isoformat(timespec="seconds"),
        }
        # Ghi ngay
        with open(csv_path, "a", encoding="utf-8-sig") as f:
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")

        n_done += 1
        elapsed = time.time() - t_start
        eta = elapsed / max(1, (i - len(done_idx))) * (len(qs) - i)
        print(f"[{i:3d}/{len(qs)}] {q['category']:10s} {r['verdict']:8s} "
              f"({r['time']:.1f}s) ETA {eta/3600:.1f}h | {q['question'][:50]}", flush=True)

    print(f"\n✅ Hoàn tất {n_done}/{len(qs)} câu trong {(time.time()-t_start)/3600:.1f}h")
    print(f"📁 CSV: {csv_path}")


if __name__ == "__main__":
    main()
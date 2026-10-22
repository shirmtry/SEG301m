"""test_500.py — Tự động test 500 câu, lưu CSV + biểu đồ + report."""
import json, re, time, random, argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from llama_cpp import Llama
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ─── CONFIG ───
BASE_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
FAISS_DIR = BASE_DIR / "faiss_index_new"
GGUF_FILE = BASE_DIR / "qwen_model" / "qwen2.5-3b-instruct-q4_k_m.gguf"
MERGED_FILE = BASE_DIR / "legal_merged.jsonl"
OUTPUT_DIR = BASE_DIR / "test_results"

# ─── PROMPT + VALIDATORS (copy từ app.py) ───
SYSTEM = (
    "Bạn là Chuyên gia Tư vấn Pháp luật Việt Nam. BẮT BUỘC tuân thủ:\n"
    "1. CHỈ dùng thông tin trong 'NGỮ CẢNH'. KHÔNG dùng kiến thức bên ngoài.\n"
    "2. KHÔNG bịa số hiệu văn bản, số điều, số tiền, số ngày nếu không có trong NGỮ CẢNH.\n"
    "3. KHÔNG bịa TÊN LUẬT.\n"
    "4. Nếu câu hỏi nêu CỤ THỂ năm/tên văn bản MÀ NGỮ CẢNH KHÔNG CÓ → 'Tôi không tìm thấy văn bản này trong cơ sở dữ liệu hiện có.' và DỪNG.\n"
    "5. Nếu NGỮ CẢNH không liên quan → 'Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có.' và DỪNG.\n"
    "6. Trích [N] với N từ 1 đến số lượng nguồn thực tế.\n"
    "7. KHÔNG lặp 1 câu/mục quá 1 lần. Tối đa 7 mục.\n"
    "8. KHÔNG copy 'Câu trả lời dựa trên ngữ cảnh:'...\n"
    "9. Tiếng Việt, tối đa 200 từ.\n"
    "10. CẤM GHÉP THÔNG TIN từ 2 đoạn [N] khác chủ đề.\n"
    "11. TRÍCH DẪN ĐÚNG tên văn bản + số điều.\n"
    "12. PHÂN BIỆT HÌNH SỰ vs HÀNH CHÍNH.\n"
    "13. NGƯỜI CHƯA THÀNH NIÊN: trả lời theo Điều 12 BLHS theo độ tuổi.\n"
)
STOP_TOKENS = ["<|im_end|>", "<|im_start|>", "\n\n\n\n"]
LOOP_TRIGGERS = ["Câu trả lời dựa trên ngữ cảnh:", "Câu trả lời dựa trên context:",
                 "TỪ ĐOẠN NGỮ CẢNH:", "TỪ Đoạn NGỮ CẢNH:", "Dựa trên ngữ cảnh:", "Dựa vào context:"]
DANGEROUS_PATTERNS = [
    (r"B2.{0,80}(A1|mô\s*tô|125\s*cc|xe\s*máy)", "B2→A1"),
    (r"chưa\s+thành\s+ni[êe]?[nm]?.{0,300}"
     r"(không\s+(?:phải\s+)?bị\s+xử\s+lý\s+hình\s+sự|"
     r"không\s+phải\s+chịu\s+trách\s+nhiệm\s+hình\s+sự|"
     r"không\s+chịu\s+trách\s+nhiệm\s+hình\s+sự|"
     r"không\s+bị\s+truy\s+cứu)", "minor_no_TNHS"),
    (r"Bộ\s+luật\s+Xử\s+lý\s+(?:Vụ\s+Phạm\s+Pháp|Vi\s+phạm\s+Pháp|Hành\s+chính)", "fake_law_name"),
]

def has_loop_inline(text, w=3):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return len(lines) >= w and len(set(lines[-w:])) == 1

def clean_answer(text):
    if not text: return text
    for r in LOOP_TRIGGERS: text = text.replace(r, "")
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    seen, out = {}, []
    for line in lines:
        if seen.get(line, 0) >= 2: break
        seen[line] = seen.get(line, 0) + 1
        out.append(line)
    return "\n".join(out).strip()

def is_dangerous_answer(a, ctxs):
    c = " ".join(ctxs).lower()
    for p, r in DANGEROUS_PATTERNS:
        if re.search(p, a, re.I) and not re.search(p, c, re.I): return True, r
    return False, ""

def has_fake_citation(a, n):
    return any(int(x) > n for x in re.findall(r'\[(\d+)\]', a))

def is_topic_missing(q, ctxs):
    ql, c = q.lower(), " ".join(ctxs).lower()
    for qk, ck, name in [
        (["chưa thành niên","vị thành niên"], ["chưa thành niên","vị thành niên","dưới 18","18 tuổi"], "minor"),
        (["bằng lái b2","hạng b2"], ["b2","hạng b2"], "b2"),
        (["nghị định 168"], ["168/2024","nghị định 168"], "nd168"),
    ]:
        if any(k in ql for k in qk) and not any(k in c for k in ck):
            return True, f"topic_missing_{name}"
    return False, ""

def is_uncited_answer(a, sources):
    if not sources or "không tìm thấy" in a.lower(): return False
    return not bool(re.search(r'\[\d+\]', a))

# ─── LOAD ENGINE ───
def load_engine():
    print("🔧 Loading embeddings...", flush=True)
    emb = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert",
                                 model_kwargs={'device':'cpu'},
                                 encode_kwargs={'normalize_embeddings':True})
    print("🔧 Loading FAISS...", flush=True)
    vs = FAISS.load_local(str(FAISS_DIR), emb, allow_dangerous_deserialization=True)
    docs = []
    if MERGED_FILE.exists():
        with open(MERGED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                t = d.get("metadata",{}).get("title","Không rõ")
                docs.append(Document(page_content=d['text'], metadata={"title":t}))
    print(f"🔧 Loaded {len(docs)} docs. Building retrievers...", flush=True)
    bm25 = BM25Retriever.from_documents(docs); bm25.k = 10
    fr = vs.as_retriever(search_kwargs={"k":10})
    ens = EnsembleRetriever(retrievers=[bm25, fr], weights=[0.4, 0.6])
    ce = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    comp = CrossEncoderReranker(model=ce, top_n=5)
    ret = ContextualCompressionRetriever(base_compressor=comp, base_retriever=ens)
    print("🔧 Loading LLM...", flush=True)
    llm = Llama(model_path=str(GGUF_FILE), n_ctx=6144, n_threads=6, n_batch=512,
                n_gpu_layers=0, verbose=False, repeat_penalty=1.3, top_k=40, top_p=0.9)
    print("✅ Engine ready.", flush=True)
    return ret, llm, docs

# ─── RETRIEVE ───
def retrieve_advanced(retriever, question, all_docs, k=5):
    ql = question.lower()
    dieu = re.findall(r'[Đđ]iều\s+(\d+)', question)
    law_kw = []
    if any(w in ql for w in ["phạm tội","hình sự","truy cứu","trách nhiệm hình sự"]): law_kw.append("Hình sự")
    if "hôn nhân" in ql or "gia đình" in ql: law_kw.append("Hôn nhân")
    if "dân sự" in ql: law_kw.append("Dân sự")
    if "đất đai" in ql: law_kw.append("Đất đai")
    if "lao động" in ql: law_kw.append("Lao động")

    is_minor = ("chưa thành niên" in ql or "vị thành niên" in ql or "dưới 18" in ql) \
               and any(w in ql for w in ["phạm tội","hình sự","truy cứu","trách nhiệm hình sự"])
    if is_minor:
        minor = [d for d in all_docs if "Bộ luật Hình sự" in d.metadata.get("title","")
                 and any(x in d.page_content.lower() for x in ["chưa thành niên","dưới 18","18 tuổi"])]
        docs_to_use = minor[:k] if minor else []
    elif law_kw and not dieu:
        raw = retriever.invoke(question)
        docs_to_use = sorted(raw, key=lambda d: -sum(1 for kw in law_kw if kw.lower() in d.metadata.get("title","").lower()))[:k]
    else:
        exact = []
        if dieu:
            for d in all_docs:
                t = d.metadata.get("title","")
                md = any(f"Điều {n}." in t or f"Điều {n} " in t for n in dieu)
                ml = (not law_kw) or any(kw.lower() in t.lower() for kw in law_kw)
                if md and ml: exact.append(d)
        docs_to_use = exact[:k] if exact else retriever.invoke(question)[:k]

    ctxs, srcs = [], []
    for i, d in enumerate(docs_to_use):
        c = d.page_content
        c = c[:1500] + "..." if len(c) > 1500 else c
        ctxs.append(f"[{i+1}] {d.metadata.get('title','Không rõ')}\n{c}")
        srcs.append({"rank": i+1, "title": d.metadata.get('title','Không rõ')})
    return ctxs, srcs

# ─── GENERATE (non-stream) ───
def generate_one(llm, question, contexts):
    user_msg = "NGỮ CẢNH:\n" + "\n\n---\n\n".join(contexts) + f"\n\nCÂU HỎI: {question}"
    prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
              f"<|im_start|>user\n{user_msg}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    out = llm(prompt, max_tokens=600, temperature=0.1, repeat_penalty=1.3, stop=STOP_TOKENS)
    return out["choices"][0]["text"]

# ─── QUESTION BANK ───
def build_question_bank(all_docs, n_data=250, n_trap=150, n_edge=100):
    random.seed(42)
    qs = []
    # 250 từ data
    candidates = [d for d in all_docs if re.search(r'Điều\s+\d+', d.metadata.get("title",""))]
    sampled = random.sample(candidates, min(n_data, len(candidates)))
    for d in sampled:
        title = d.metadata.get("title","")
        m = re.search(r'Điều\s+(\d+)', title)
        if not m: continue
        dieu = m.group(1)
        law = title.split(" - ")[0].strip() if " - " in title else "pháp luật"
        tpl = random.choice([
            f"Điều {dieu} {law} quy định gì?",
            f"Cho tôi biết nội dung Điều {dieu} của {law}.",
            f"{law} - Điều {dieu} quy định như thế nào?",
        ])
        qs.append({"category":"from_data","question":tpl,"expected":"answer","ref_title":title})

    # 150 traps
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
    topics = ["giao thông","thuế","hải quan","chứng khoán","bất động sản","ngân hàng","bảo hiểm"]
    years = [2024, 2025, 2026, 2027, 2030]
    traps = list(base_traps)
    while len(traps) < n_trap:
        traps.append(f"Quy định về {random.choice(topics)} năm {random.choice(years)} là gì?")
    for t in traps[:n_trap]:
        qs.append({"category":"trap","question":t,"expected":"refuse"})

    # 100 edge
    edge = ["Ly hôn","Bị phạt bao nhiêu?","Luật mới nhất?","Như thế nào?","Tại sao?","Có không?",
            "Nghỉ hưu tuổi bao nhiêu?","Thuế thu nhập cá nhân?","Đăng ký kết hôn?",
            "Làm sao để khởi kiện?","Điều 15?","Khoản 2?","Hợp đồng lao động?",
            "Bảo hiểm xã hội?","Đất ở?","Di chúc?","Thừa kế?","Quyền nuôi con?","Cấp dưỡng?"]
    while len(edge) < n_edge:
        edge.append(f"Quy định về {random.choice(['nhà ở','đất đai','hôn nhân','lao động'])}?")
    for e in edge[:n_edge]:
        qs.append({"category":"edge","question":e,"expected":"any"})

    return qs[:n_data+n_trap+n_edge]

# ─── RUN ONE ───
def run_one(retriever, llm, all_docs, question):
    t0 = time.time()
    ctxs, srcs = retrieve_advanced(retriever, question, all_docs, k=5)
    if not ctxs:
        return {"answer":"Tôi không tìm thấy thông tin này...","verdict":"refused",
                "reason":"no_context","time":time.time()-t0,"n_citations":0,"n_contexts":0}
    raw = generate_one(llm, question, ctxs)
    ans = clean_answer(raw) or "Tôi không tìm thấy thông tin này..."
    if has_loop_inline(ans):
        ans = "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có."
    danger, dr = is_dangerous_answer(ans, ctxs)
    fake = has_fake_citation(ans, len(ctxs))
    tm, tr = is_topic_missing(question, ctxs)
    uncited = is_uncited_answer(ans, srcs)
    refused = "không tìm thấy" in ans.lower()
    if refused: verdict, reason = "refused", "model_refused"
    elif danger: verdict, reason = "blocked", f"danger:{dr}"
    elif fake: verdict, reason = "blocked", "fake_citation"
    elif tm: verdict, reason = "blocked", tr
    elif uncited: verdict, reason = "warning", "uncited"
    else: verdict, reason = "answered", "ok"
    return {"answer":ans,"verdict":verdict,"reason":reason,
            "time":time.time()-t0,"n_citations":len(re.findall(r'\[\d+\]',ans)),
            "n_contexts":len(ctxs)}

# ─── CHARTS ───
def generate_charts(df, out_dir, ts):
    plt.rcParams.update({'font.size':10})
    # 1 overall
    fig, ax = plt.subplots(figsize=(8,6))
    vc = df["verdict"].value_counts()
    colors = {"answered":"#2ecc71","refused":"#3498db","blocked":"#e74c3c","warning":"#f39c12","error":"#95a5a6"}
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
           colors=[colors.get(k,"#999") for k in vc.index], startangle=90)
    ax.set_title(f"Tổng thể — {len(df)} câu")
    fig.savefig(out_dir/f"{ts}_01_overall.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # 2 by category
    fig, ax = plt.subplots(figsize=(10,6))
    ct = pd.crosstab(df["category"], df["verdict"])
    ct.plot(kind="bar", ax=ax, color=[colors.get(c,"#999") for c in ct.columns])
    ax.set_title("Kết quả theo loại câu"); ax.set_ylabel("Số câu"); plt.xticks(rotation=0)
    fig.savefig(out_dir/f"{ts}_02_by_category.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # 3 time
    fig, ax = plt.subplots(figsize=(10,6))
    df["time_sec"].hist(bins=30, ax=ax, color="#3498db")
    ax.set_title("Thời gian phản hồi"); ax.set_xlabel("Giây"); ax.set_ylabel("Số câu")
    ax.axvline(df["time_sec"].mean(), color="red", linestyle="--", label=f"Mean {df['time_sec'].mean():.1f}s")
    ax.legend()
    fig.savefig(out_dir/f"{ts}_03_response_time.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # 4 reasons
    fig, ax = plt.subplots(figsize=(10,6))
    df["reason"].value_counts().plot(kind="barh", ax=ax, color="#9b59b6")
    ax.set_title("Phân loại lý do")
    fig.savefig(out_dir/f"{ts}_04_reasons.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # 5 refusal rate by cat
    fig, ax = plt.subplots(figsize=(10,6))
    df["_refused_or_blocked"] = df["verdict"].isin(["refused","blocked"])
    rate = df.groupby("category")["_refused_or_blocked"].mean() * 100
    rate.plot(kind="bar", ax=ax, color="#e67e22")
    ax.set_title("Tỷ lệ từ chối/block (%)"); ax.set_ylabel("%"); plt.xticks(rotation=0)
    fig.savefig(out_dir/f"{ts}_05_refusal_rate.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # 6 top failed traps
    failed = df[(df["expected"]=="refuse") & (~df["verdict"].isin(["refused","blocked"]))]
    if len(failed):
        fig, ax = plt.subplots(figsize=(12,max(4, len(failed[:10])*0.5)))
        ax.axis("off")
        text = "TOP TRAPS FAIL (lẽ ra phải từ chối):\n\n"
        for i, (_, r) in enumerate(failed[:10].iterrows(), 1):
            text += f"{i}. Q: {r['question'][:70]}\n   A: {r['answer'][:100]}\n\n"
        ax.text(0, 1, text, va="top", fontsize=9, family="monospace")
        fig.savefig(out_dir/f"{ts}_06_failed_traps.png", dpi=110, bbox_inches="tight"); plt.close(fig)

# ─── MAIN ───
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="Số câu (mặc định 500)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    retriever, llm, all_docs = load_engine()
    print(f"\n📝 Building {args.limit} questions...", flush=True)
    qs = build_question_bank(all_docs)[:args.limit]
    print(f"✅ {len(qs)} questions ready. Start test...\n", flush=True)

    results = []
    t_start = time.time()
    for i, q in enumerate(qs, 1):
        try:
            r = run_one(retriever, llm, all_docs, q["question"])
        except Exception as e:
            r = {"answer":f"ERROR:{e}","verdict":"error","reason":str(e)[:80],
                 "time":0,"n_citations":0,"n_contexts":0}
        results.append({
            "idx":i, "category":q["category"], "expected":q["expected"],
            "question":q["question"], "answer":r["answer"],
            "verdict":r["verdict"], "reason":r["reason"],
            "time_sec":round(r["time"],2),
            "n_citations":r["n_citations"], "n_contexts":r["n_contexts"],
        })
        eta = (time.time()-t_start)/i*(len(qs)-i)
        print(f"[{i:3d}/{len(qs)}] {q['category']:10s} {r['verdict']:8s} "
              f"({r['time']:.1f}s) ETA {eta/60:.0f}m | {q['question'][:55]}", flush=True)

    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR/f"{ts}_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 CSV: {csv_path}", flush=True)

    # REPORT
    rp = OUTPUT_DIR/f"{ts}_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(f"=== TEST REPORT {ts} ===\nTotal: {len(df)}\n")
        f.write(f"Avg time: {df['time_sec'].mean():.2f}s | Total: {df['time_sec'].sum()/60:.1f}min\n\n")
        f.write("VERDICT:\n")
        for v, c in df["verdict"].value_counts().items():
            f.write(f"  {v}: {c} ({c/len(df)*100:.1f}%)\n")
        f.write("\nBY CATEGORY:\n")
        for cat in df["category"].unique():
            sub = df[df["category"]==cat]
            f.write(f"\n  [{cat}] n={len(sub)}\n")
            for v, c in sub["verdict"].value_counts().items():
                f.write(f"    {v}: {c} ({c/len(sub)*100:.1f}%)\n")
        f.write("\nREASONS:\n")
        for r_, c in df["reason"].value_counts().items():
            f.write(f"  {r_}: {c}\n")
        failed = df[(df["expected"]=="refuse") & (~df["verdict"].isin(["refused","blocked"]))]
        f.write(f"\n⚠️ FAILED TRAPS: {len(failed)}\n")
        for _, r_ in failed.iterrows():
            f.write(f"  Q: {r_['question']}\n  A: {r_['answer'][:200]}\n\n")
    print(f"📄 Report: {rp}", flush=True)

    print("📊 Generating charts...", flush=True)
    generate_charts(df, OUTPUT_DIR, ts)
    print(f"✅ Done! {(time.time()-t_start)/60:.1f} min. All in {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
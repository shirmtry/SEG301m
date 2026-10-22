# app.py — Web UI cho Chatbot Pháp luật (Giao diện 2 Cột Chuyên Nghiệp)
import json
import re
from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from llama_cpp import Llama

# Import chuẩn cho LangChain v1.0+
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
BASE_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
FAISS_DIR = BASE_DIR / "faiss_index_new" 
GGUF_FILE = BASE_DIR / "qwen_model" / "qwen2.5-3b-instruct-q4_k_m.gguf"
MERGED_FILE = BASE_DIR / "legal_merged.jsonl" 

# Cấu hình giao diện rộng
st.set_page_config(page_title="Chatbot Pháp luật VN", page_icon="⚖️", layout="wide")

# ==========================================
# HÀM LOAD HỆ THỐNG
# ==========================================
@st.cache_resource(show_spinner=False)
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="keepitreal/vietnamese-sbert",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    
    docs = []
    all_laws = set() # Tập hợp để lưu tên các Bộ luật duy nhất
    
    if MERGED_FILE.exists():
        with open(MERGED_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                title = data.get("metadata", {}).get("title", "Không rõ văn bản")
                docs.append(Document(page_content=data['text'], metadata={"title": title}))
                
                # Trích xuất tên Bộ luật (VD: "Bộ luật Dân sự 2015 - Điều 1..." -> "Bộ luật Dân sự 2015")
                if " - Điều" in title:
                    law_name = title.split(" - Điều")[0].strip()
                    all_laws.add(law_name)
    
    # Sắp xếp danh sách luật theo alphabet
    sorted_laws = sorted(list(all_laws))
    
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 15 
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15}) 
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5]
    )
    
    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    llm = Llama(
        model_path=str(GGUF_FILE),
        n_ctx=8192, 
        n_threads=8, 
        n_batch=512, 
        n_gpu_layers=-1, 
        verbose=False,
    )
    return retriever, llm, docs, sorted_laws

# ==========================================
# PROMPT "CẢNH BÁO PHÁP LÝ"
# ==========================================
SYSTEM = (
    "Bạn là một Chuyên gia Tư vấn Pháp luật Việt Nam, chuyên nghiệp và thận trọng.\n"
    "QUY TẮC TRẢ LỜI BẮT BUỘC:\n"
    "1. Trả lời trực tiếp dựa trên 'NGỮ CẢNH'. Trích dẫn rõ Tên văn bản và Điều luật.\n"
    "2. CẢNH BÁO PHÁP LÝ: Nếu câu hỏi liên quan đến chia tài sản khi ly hôn, BẮT BUỘC phải rà soát quy định về 'Tài sản chung' (Điều 33) và 'Tài sản riêng' (Điều 43) của Luật Hôn nhân.\n"
    "3. Nếu hỏi về 'Thủ tục/Hồ sơ', hãy tập trung liệt kê giấy tờ cần thiết (Đơn, sổ đỏ, giấy kết hôn...).\n"
    "4. Nếu ngữ cảnh không có thông tin trực tiếp nhưng có điều luật liên quan, hãy tổng hợp để tư vấn.\n"
    "5. Trình bày ngắn gọn, dùng gạch đầu dòng."
)

def generate(llm, question, contexts):
    user_msg = "NGỮ CẢNH:\n" + "\n\n---\n\n".join(contexts) + f"\n\nCÂU HỎI: {question}"
    prompt = (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    for chunk in llm(prompt, max_tokens=800, temperature=0.1, stop=["<|im_end|>", "<|im_start|>"], stream=True):
        piece = chunk["choices"][0]["text"]
        if piece:
            yield piece

# ==========================================
# HÀM RETRIEVE THÔNG MINH
# ==========================================
def retrieve_advanced(retriever, question, all_docs, history, k=5):
    search_query = question
    if len(question.split()) < 6 and history:
        last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        search_query = f"{last_user_msg} {question}"

    dieu_numbers = re.findall(r'[Đđ]iều\s+(\d+)', search_query)
    
    law_keywords = []
    if "hôn nhân" in search_query.lower() or "gia đình" in search_query.lower(): law_keywords.append("Hôn nhân")
    if "dân sự" in search_query.lower(): law_keywords.append("Dân sự")
    if "hình sự" in search_query.lower(): law_keywords.append("Hình sự")
    if "đất đai" in search_query.lower(): law_keywords.append("Đất đai")
    if "lao động" in search_query.lower(): law_keywords.append("Lao động")
        
    exact_docs = []
    
    if dieu_numbers:
        for doc in all_docs:
            title = doc.metadata.get("title", "")
            match_dieu = any(f"Điều {d}." in title or f"Điều {d} " in title for d in dieu_numbers)
            match_law = True
            if law_keywords:
                match_law = any(law_kw.lower() in title.lower() for law_kw in law_keywords)
                
            if match_dieu and match_law:
                exact_docs.append(doc)
    
    if exact_docs:
        docs_to_use = exact_docs[:k]
    else:
        docs_to_use = retriever.invoke(search_query)[:k]
    
    contexts = []
    sources = []
    for i, doc in enumerate(docs_to_use):
        truncated_content = doc.page_content[:2000] + "... (đã rút gọn)" if len(doc.page_content) > 2000 else doc.page_content
        contexts.append(f"[{i+1}] {doc.metadata.get('title', 'Không rõ')}\n{truncated_content}")
        sources.append({"rank": i+1, "title": doc.metadata.get('title', 'Không rõ')})
        
    return contexts, sources

# ==========================================
# GIAO DIỆN CHÍNH (CHIA 2 CỘT)
# ==========================================
st.markdown("<h1 style='text-align: center;'>⚖️ Trợ lý Pháp luật Việt Nam</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Hệ thống RAG · FAISS + BM25 + Reranker · Qwen 2.5-3B</p>", unsafe_allow_html=True)

with st.spinner("⏳ Đang khởi động hệ thống AI..."):
    retriever, llm, all_docs, all_laws = load_resources()

# Chia layout: Cột trái (1 phần) - Cột phải (2.5 phần)
col_left, col_right = st.columns([1, 2.5])

# --- CỘT TRÁI: DANH SÁCH LUẬT ---
with col_left:
    st.markdown("### 📚 Thư viện Pháp luật")
    st.caption(f"Đang lưu trữ {len(all_laws)} Bộ luật/Luật cốt lõi")
    
    # Dùng expander để chứa danh sách cho gọn
    with st.expander("Xem danh sách Bộ luật", expanded=True):
        for law in all_laws:
            st.markdown(f"- {law}")

# --- CỘT PHẢI: KHUNG CHAT ---
with col_right:
    st.markdown("### 💬 Khung Chat Tư Vấn")
    
    # Khởi tạo lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Nguồn tham khảo ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"**[{s['rank']}] {s['title']}**")

    # Ô nhập câu hỏi
    q = st.chat_input("Nhập câu hỏi pháp luật...")

    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Đang tra cứu và Rerank..."):
                contexts, sources = retrieve_advanced(retriever, q, all_docs, st.session_state.messages[:-1], k=5)

            st.markdown("**🔍 Đang trả lời...**")
            placeholder = st.empty()
            answer = ""
            for token in generate(llm, q, contexts):
                answer += token
                placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)

            with st.expander(f"📚 Nguồn tham khảo ({len(sources)})", expanded=True):
                for s in sources:
                    st.markdown(f"**[{s['rank']}] {s['title']}**")

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
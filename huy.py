import json
import os
import sys
import pickle
import time

# ============================================================
# HÀM HỖ TRỢ ĐƯỜNG DẪN (chạy được cả .py và .exe)
# ============================================================
def get_resource_path(relative_path):
    """Lấy đường dẫn tuyệt đối, hỗ trợ cả khi đóng gói PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        base = os.path.dirname(sys.executable)
        candidate = os.path.join(base, relative_path)
        if os.path.exists(candidate):
            return candidate
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


# ============================================================
# TÌM FILE DỮ LIỆU (ưu tiên chunks.jsonl, fallback data_luat.jsonl)
# ============================================================
def find_data_file():
    for name in ['chunks.jsonl', 'data_luat.jsonl', 'data.jsonl']:
        p = get_resource_path(name)
        if os.path.exists(p):
            return p
    return None


# ============================================================
# 1. LOAD DỮ LIỆU LUẬT (JSONL) & TẠO CHỈ MỤC BM25
# ============================================================
print('=' * 60)
print('--- ĐANG QUÉT KHO VĂN BẢN LUẬT VIỆT NAM (JSONL) ---')
print('=' * 60)

data_path = find_data_file()

if not data_path:
    print('❌ LỖI: Không tìm thấy file dữ liệu!')
    print("Vui lòng đặt 'chunks.jsonl' (hoặc 'data_luat.jsonl') cạnh file .py/.exe.")
    input('Nhấn Enter để thoát...')
    sys.exit(1)

print(f'📁 File dữ liệu: {data_path}')
file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
print(f'📦 Kích thước: {file_size_mb:.1f} MB')

# Đường dẫn cache
cache_path = data_path + '.bm25.pkl'

documents = []
corpus_tokens = []
bm25 = None


# ============================================================
# HÀM TRÍCH XUẤT TEXT TỪ 1 OBJECT JSON
# ============================================================
def extract_text(item):
    """
    Trích xuất text từ nhiều định dạng JSONL khác nhau.
    Hỗ trợ cả chunks.jsonl (VBPL corpus) và các format cũ.
    """
    if not isinstance(item, dict):
        return ''

    # Ưu tiên các key chứa nội dung luật
    text = (
        item.get('content')      # ← VBPL corpus dùng key này
        or item.get('text')
        or item.get('document')
        or item.get('law_text')
        or item.get('output')
        or item.get('chunk')
        or item.get('body')
        or ''
    )

    # Nếu có metadata tiêu đề/điều khoản, ghép vào đầu để tăng ngữ cảnh
    title = item.get('title') or item.get('article_title') or item.get('dieu') or ''
    if title and isinstance(title, str) and title.strip():
        text = f"[{title.strip()}]\n{text}"

    return text if isinstance(text, str) else ''


# ============================================================
# LOAD CACHE NẾU CÓ
# ============================================================
cache_loaded = False
if os.path.exists(cache_path):
    try:
        print(f'📦 Tìm thấy cache BM25: {os.path.basename(cache_path)}')
        print('⏳ Đang load cache (nhanh hơn build lại rất nhiều)...')
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
            bm25 = cached['bm25']
            documents = cached['documents']
            corpus_tokens = cached['corpus_tokens']
        print(f'✅ Load cache xong trong {time.time() - t0:.1f}s '
              f'({len(documents):,} phân đoạn).')
        cache_loaded = True
    except Exception as e:
        print(f'⚠️ Cache hỏng hoặc không tương thích ({e}). Sẽ build lại.')


# ============================================================
# NẾU CHƯA CÓ CACHE → ĐỌC JSONL + BUILD BM25
# ============================================================
if not cache_loaded:
    print('⏳ Đang đọc file JSONL...')

    # Thử dùng tqdm nếu có, nếu không thì bỏ qua
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    error_count = 0
    t0 = time.time()

    with open(data_path, 'r', encoding='utf-8') as f:
        # Đếm sơ bộ số dòng để tqdm hiển thị (nếu file quá lớn thì bỏ qua)
        total_lines = None
        if use_tqdm and file_size_mb < 500:
            try:
                total_lines = sum(1 for _ in f)
                f.seek(0)
            except Exception:
                total_lines = None

        iterator = tqdm(f, total=total_lines, desc='Đọc JSONL', unit='dòng') \
            if use_tqdm else f

        for line in iterator:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                item = json.loads(line_str)
                text = extract_text(item)
                if text and text.strip():
                    tokens = text.lower().split()
                    if tokens:
                        documents.append(text)
                        corpus_tokens.append(tokens)
            except json.JSONDecodeError:
                error_count += 1
                if error_count <= 3:
                    print(f'⚠️ Dòng lỗi JSON: {line_str[:80]}...')
                continue

    print(f'✅ Đọc xong trong {time.time() - t0:.1f}s. '
          f'Lỗi JSON: {error_count} dòng.')

    if not corpus_tokens:
        print('❌ LỖI: Không tìm thấy dữ liệu văn bản hợp lệ trong file!')
        print("Kiểm tra lại key JSONL (cần 'content' hoặc 'text' hoặc 'output').")
        input('Nhấn Enter để thoát...')
        sys.exit(1)

    print(f'🎉 Đã nạp {len(documents):,} phân đoạn luật hợp lệ.')
    print('⏳ Đang build chỉ mục BM25 (lần đầu sẽ mất vài phút)...')

    try:
        from rank_bm25 import BM25Okapi
        t1 = time.time()
        bm25 = BM25Okapi(corpus_tokens)
        print(f'✅ BM25 sẵn sàng trong {time.time() - t1:.1f}s.')
    except Exception as e:
        print(f'❌ LỖI KHỞI TẠO BM25: {e}')
        print("Cài đặt: pip install rank_bm25")
        input('Nhấn Enter để thoát...')
        sys.exit(1)

    # Lưu cache
    try:
        print('💾 Đang lưu cache BM25 cho lần sau...')
        t2 = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'bm25': bm25,
                'documents': documents,
                'corpus_tokens': corpus_tokens,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'✅ Đã lưu cache ({time.time() - t2:.1f}s). '
              f'Lần sau khởi động sẽ rất nhanh!')
    except Exception as e:
        print(f'⚠️ Không lưu được cache: {e}')


# ============================================================
# 2. KHỞI TẠO MÔ HÌNH QWEN OFFLINE (LLAMA-CPP)
# ============================================================
print('\n--- ĐANG KHỞI ĐỘNG MÔ HÌNH QWEN (VUI LÒNG ĐỢI)... ---')

model_dir = get_resource_path('qwen_model')
model_path = None

if os.path.exists(model_dir):
    for file in os.listdir(model_dir):
        if file.endswith('.gguf'):
            model_path = os.path.join(model_dir, file)
            break

if not model_path or not os.path.exists(model_path):
    print(f'❌ LỖI: Không tìm thấy file mô hình (.gguf) tại: {model_dir}')
    print("Vui lòng đặt thư mục 'qwen_model' chứa file .gguf cạnh file .exe.")
    input('Nhấn Enter để thoát...')
    sys.exit(1)

print(f'📁 Đang nạp mô hình: {os.path.basename(model_path)}')

try:
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f'✅ Mô hình đã sẵn sàng ({time.time() - t0:.1f}s).')
except Exception as e:
    print(f'❌ LỖI KHI LOAD MÔ HÌNH: {e}')
    print("Đảm bảo đã cài: pip install llama-cpp-python")
    input('Nhấn Enter để thoát...')
    sys.exit(1)

print('\n' + '=' * 60)
print('--- 🤖 CHATBOT LUẬT VIỆT NAM — OFFLINE — SẴN SÀNG ---')
print(f'Kho luật: {len(documents):,} phân đoạn')
print('Gõ câu hỏi pháp luật và bấm Enter.')
print('Gõ "exit" hoặc "quit" để thoát. Gõ "clear" để xóa màn hình.')
print('=' * 60 + '\n')


# ============================================================
# 3. TRA CỨU & TƯ VẤN
# ============================================================
def retrieve_docs(query, top_k=5):
    """Tìm top_k đoạn luật liên quan nhất bằng BM25."""
    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []
    try:
        top_docs = bm25.get_top_n(tokenized_query, documents, n=top_k)
        return top_docs
    except Exception as e:
        print(f'⚠️ Lỗi truy vấn BM25: {e}')
        return []


def truncate_context(docs, max_chars=6000):
    """Cắt bớt context nếu quá dài để không tràn n_ctx."""
    result = []
    total = 0
    for d in docs:
        if total + len(d) > max_chars:
            # Cắt phần còn lại của doc cuối
            remain = max_chars - total
            if remain > 200:
                result.append(d[:remain] + '...')
            break
        result.append(d)
        total += len(d)
    return result


def build_prompt(context, question):
    """Ghép prompt theo format ChatML của Qwen."""
    system_msg = (
        "Bạn là chuyên gia tư vấn pháp luật Việt Nam. "
        "Hãy dựa vào các trích dẫn luật được cung cấp để trả lời "
        "chính xác, ngắn gọn, rõ ràng. Nếu thông tin không có trong "
        "trích dẫn, hãy nói rõ là không tìm thấy quy định cụ thể, "
        "tuyệt đối không bịa đặt. Khi trả lời, nên nêu rõ căn cứ "
        "pháp lý (tên điều luật, tên văn bản nếu có trong trích dẫn)."
    )
    return (
        f"<|im_start|>system\n{system_msg}\n\n"
        f"CÁC TRÍCH DẪN LUẬT LIÊN QUAN:\n{context}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def chat_loop():
    while True:
        try:
            user_input = input('\n👤 Bạn hỏi: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n👋 Tạm biệt!')
            break

        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit', 'thoát']:
            print('👋 Tạm biệt!')
            break
        if user_input.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue

        print('\n🔍 [Đang tra cứu kho luật và suy luận...]\n')

        # Bước A: Tra cứu BM25 (top 5)
        related_docs = retrieve_docs(user_input, top_k=5)

        # Bước B: Cắt context nếu quá dài
        related_docs = truncate_context(related_docs, max_chars=6000)

        context = (
            '\n---\n'.join(related_docs)
            if related_docs
            else 'Không tìm thấy trích dẫn cụ thể trong kho luật.'
        )

        # Bước C: Ghép prompt
        prompt = build_prompt(context, user_input)

        # Bước D: Stream từng chữ
        try:
            t0 = time.time()
            stream = llm(
                prompt=prompt,
                max_tokens=1024,
                stop=['<|im_end|>', '<|im_start|>', '<|endoftext|>'],
                stream=True,
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            print('🤖 Chatbot: ', end='', flush=True)
            for output in stream:
                token_text = output['choices'][0]['text']
                print(token_text, end='', flush=True)
            print(f'\n\n⏱️  ({time.time() - t0:.1f}s)\n')
        except Exception as err:
            print(f'\n❌ Lỗi suy luận: {err}')


# ============================================================
# CHẠY
# ============================================================
if __name__ == '__main__':
    chat_loop()
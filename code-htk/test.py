import json
import os
import sys
import pickle
import time

# ============================================================
# HÀM HỖ TRỢ ĐƯỜNG DẪN (chạy được cả .py và .exe)
# ============================================================
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base = os.path.dirname(sys.executable)
        candidate = os.path.join(base, relative_path)
        if os.path.exists(candidate):
            return candidate
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath('.'), relative_path)


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
    input('Nhấn Enter để thoát...')
    sys.exit(1)

print(f'📁 File dữ liệu: {data_path}')
file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
print(f'📦 Kích thước: {file_size_mb:.1f} MB')

cache_path = data_path + '.bm25.pkl'
documents, corpus_tokens, bm25 = [], [], None


def extract_text(item):
    if not isinstance(item, dict):
        return ''
    text = (item.get('content') or item.get('text') or item.get('document')
            or item.get('law_text') or item.get('output') or item.get('chunk')
            or item.get('body') or '')
    title = item.get('title') or item.get('article_title') or item.get('dieu') or ''
    if title and isinstance(title, str) and title.strip():
        text = f"[{title.strip()}]\n{text}"
    return text if isinstance(text, str) else ''


cache_loaded = False
if os.path.exists(cache_path):
    try:
        print(f'📦 Tìm thấy cache BM25: {os.path.basename(cache_path)}')
        t0 = time.time()
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
            bm25 = cached['bm25']
            documents = cached['documents']
            corpus_tokens = cached['corpus_tokens']
        print(f'✅ Load cache xong trong {time.time() - t0:.1f}s ({len(documents):,} phân đoạn).')
        cache_loaded = True
    except Exception as e:
        print(f'⚠️ Cache hỏng ({e}). Build lại.')

if not cache_loaded:
    print('⏳ Đang đọc file JSONL...')
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    error_count = 0
    t0 = time.time()
    with open(data_path, 'r', encoding='utf-8') as f:
        total_lines = None
        if use_tqdm and file_size_mb < 500:
            try:
                total_lines = sum(1 for _ in f)
                f.seek(0)
            except Exception:
                total_lines = None
        iterator = tqdm(f, total=total_lines, desc='Đọc JSONL', unit='dòng') if use_tqdm else f
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
                continue

    print(f'✅ Đọc xong trong {time.time() - t0:.1f}s. Lỗi JSON: {error_count} dòng.')
    if not corpus_tokens:
        print('❌ LỖI: Không có dữ liệu hợp lệ!')
        input('Nhấn Enter...')
        sys.exit(1)

    print(f'🎉 Đã nạp {len(documents):,} phân đoạn luật hợp lệ.')
    print('⏳ Đang build chỉ mục BM25...')
    try:
        from rank_bm25 import BM25Okapi
        t1 = time.time()
        bm25 = BM25Okapi(corpus_tokens)
        print(f'✅ BM25 sẵn sàng trong {time.time() - t1:.1f}s.')
    except Exception as e:
        print(f'❌ LỖI BM25: {e}')
        input('Nhấn Enter...')
        sys.exit(1)

    try:
        print('💾 Lưu cache...')
        with open(cache_path, 'wb') as f:
            pickle.dump({'bm25': bm25, 'documents': documents,
                         'corpus_tokens': corpus_tokens},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
        print('✅ Đã lưu cache.')
    except Exception as e:
        print(f'⚠️ Không lưu được cache: {e}')


# ============================================================
# 2. KHỞI TẠO MÔ HÌNH QWEN OFFLINE (LLAMA-CPP)
# ============================================================
print('\n--- ĐANG KHỞI ĐỘNG MÔ HÌNH QWEN ---')

model_dir = get_resource_path('qwen_model')
model_path = None
if os.path.exists(model_dir):
    for file in os.listdir(model_dir):
        if file.endswith('.gguf'):
            model_path = os.path.join(model_dir, file)
            break

if not model_path or not os.path.exists(model_path):
    print(f'❌ LỖI: Không tìm thấy file .gguf tại: {model_dir}')
    input('Nhấn Enter...')
    sys.exit(1)

print(f'📁 Nạp mô hình: {os.path.basename(model_path)}')
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
    print(f'✅ Mô hình sẵn sàng ({time.time() - t0:.1f}s).')
except Exception as e:
    print(f'❌ LỖI LOAD MÔ HÌNH: {e}')
    input('Nhấn Enter...')
    sys.exit(1)


# ============================================================
# 3. HÀM TRA CỨU & PROMPT (dùng chung cho chat + benchmark)
# ============================================================
def retrieve_docs(query, top_k=5):
    tokenized_query = query.lower().split()
    if not tokenized_query:
        return []
    try:
        return bm25.get_top_n(tokenized_query, documents, n=top_k)
    except Exception as e:
        print(f'⚠️ Lỗi BM25: {e}')
        return []


def truncate_context(docs, max_chars=6000):
    result, total = [], 0
    for d in docs:
        if total + len(d) > max_chars:
            remain = max_chars - total
            if remain > 200:
                result.append(d[:remain] + '...')
            break
        result.append(d)
        total += len(d)
    return result


def build_prompt(context, question):
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


def ask_model(question, top_k=5, max_tokens=512):
    """Trả về (answer, bm25_top_score, context_docs)."""
    tokenized = question.lower().split()
    scores = bm25.get_scores(tokenized) if tokenized else [0.0]
    top_score = float(max(scores)) if len(scores) else 0.0

    docs = retrieve_docs(question, top_k=top_k)
    docs = truncate_context(docs, max_chars=6000)
    context = '\n---\n'.join(docs) if docs else 'Không tìm thấy trích dẫn cụ thể.'

    prompt = build_prompt(context, question)
    try:
        resp = llm(
            prompt=prompt,
            max_tokens=max_tokens,
            stop=['<|im_end|>', '<|im_start|>', '<|endoftext|>'],
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        answer = resp['choices'][0]['text'].strip()
    except Exception as e:
        answer = f'[LỖI SUY LUẬN: {e}]'

    return answer, top_score, docs


# ============================================================
# 4. NGÂN HÀNG 100 CÂU HỎI (đa dạng lĩnh vực, độ khó khác nhau)
# ============================================================
QUESTIONS = [
    # --- LAO ĐỘNG ---
    "Độ tuổi lao động tối thiểu là bao nhiêu?",
    "Người lao động được nghỉ bao nhiêu ngày phép năm?",
    "Thời giờ làm việc bình thường một ngày là bao nhiêu giờ?",
    "Mức lương tối thiểu vùng I hiện nay là bao nhiêu?",
    "Người sử dụng lao động có được đơn phương chấm dứt hợp đồng không?",
    "Thời gian thử việc tối đa đối với công việc cần trình độ đại học trở lên?",
    "Người lao động nữ sinh con được nghỉ thai sản bao nhiêu tháng?",
    "Chế độ ốm đau được quy định thế nào?",
    "Làm thêm giờ ban đêm được trả bao nhiêu phần trăm?",
    "Hợp đồng lao động có bao nhiêu loại?",
    "Khi chấm dứt HĐLĐ trái pháp luật, người sử dụng lao động phải bồi thường gì?",
    "Tuổi nghỉ hưu của lao động nam năm 2024 là bao nhiêu?",
    "Thời hạn báo trước khi đơn phương chấm dứt HĐLĐ không xác định thời hạn?",
    "Quấy rối tình dục tại nơi làm việc bị xử lý thế nào?",
    "Có bao nhiêu ngày nghỉ lễ, Tết trong năm?",
    "Tiền lương làm thêm giờ ngày thường được tính thế nào?",
    "Bảo hiểm xã hội bắt buộc gồm những chế độ nào?",
    "Người lao động nước ngoài làm việc tại VN cần giấy phép gì?",
    "Công ty có được sa thải nhân viên tự ý nghỉ việc không?",
    "Hợp đồng lao động có bắt buộc bằng văn bản không?",

    # --- HÔN NHÂN & GIA ĐÌNH ---
    "Độ tuổi kết hôn hợp pháp là bao nhiêu?",
    "Thủ tục đăng ký kết hôn được quy định thế nào?",
    "Các trường hợp cấm kết hôn?",
    "Ly hôn đơn phương khi nào được chấp nhận?",
    "Tài sản chung của vợ chồng được chia khi ly hôn thế nào?",
    "Con dưới 36 tháng tuổi khi ly hôn thuộc quyền nuôi ai?",
    "Cấp dưỡng nuôi con sau ly hôn được quy định thế nào?",
    "Kết hôn giả có bị xử lý hình sự không?",
    "Tảo hôn bị xử phạt thế nào?",
    "Vợ chồng có thể thỏa thuận chia tài sản trong thời kỳ hôn nhân không?",
    "Mang thai hộ vì mục đích nhân đạo được quy định thế nào?",
    "Xác định cha, mẹ cho con ngoài giá thú thế nào?",
    "Ly hôn có yếu tố nước ngoài giải quyết ở đâu?",
    "Hôn nhân đồng giới có được công nhận tại VN?",
    "Con nuôi có được hưởng thừa kế không?",

    # --- GIAO THÔNG ---
    "Nồng độ cồn tối đa cho phép khi lái xe là bao nhiêu?",
    "Vượt đèn đỏ bị phạt bao nhiêu tiền?",
    "Không đội mũ bảo hiểm bị phạt thế nào?",
    "Bằng lái xe hạng B1 lái được xe gì?",
    "Tốc độ tối đa trong khu dân cư là bao nhiêu?",
    "Xe máy điện có cần bằng lái không?",
    "Gây tai nạn giao thông chết người bị xử lý thế nào?",
    "Uống rượu bia lái xe bị phạt bao nhiêu?",
    "Đi ngược chiều bị xử phạt thế nào?",
    "Tuổi được cấp bằng lái xe máy A1?",
    "Xe không chính chủ có bị phạt không?",
    "Sử dụng điện thoại khi lái xe bị phạt bao nhiêu?",
    "Không xi nhan khi chuyển làn bị phạt thế nào?",
    "Xe quá hạn đăng kiểm bị xử lý thế nào?",
    "Tai nạn giao thông có được bảo hiểm chi trả không?",

    # --- HÌNH SỰ ---
    "Tội trộm cắp tài sản bị phạt tù bao nhiêu năm?",
    "Giết người bị xử lý thế nào?",
    "Tội lừa đảo chiếm đoạt tài sản bị phạt bao nhiêu?",
    "Cướp tài sản bị xử phạt thế nào?",
    "Tàng trữ trái phép chất ma túy bị phạt bao nhiêu năm tù?",
    "Tội cố ý gây thương tích bị phạt thế nào?",
    "Hiếp dâm bị xử lý thế nào?",
    "Đánh bạc bị phạt bao nhiêu?",
    "Tội tham ô tài sản bị phạt thế nào?",
    "Buôn lậu bị xử lý ra sao?",
    "Tội làm giả con dấu bị phạt thế nào?",
    "Người chưa thành niên phạm tội bị xử lý thế nào?",
    "Tội rửa tiền bị phạt bao nhiêu năm?",
    "Không tố giác tội phạm có bị phạt không?",
    "Phòng vệ chính đáng được quy định thế nào?",

    # --- DÂN SỰ ---
    "Thời hiệu khởi kiện tranh chấp hợp đồng là bao lâu?",
    "Hợp đồng dân sự vô hiệu khi nào?",
    "Đặt cọc trong giao dịch dân sự được quy định thế nào?",
    "Bồi thường thiệt hại ngoài hợp đồng thế nào?",
    "Năng lực hành vi dân sự của người từ 6 đến 18 tuổi?",
    "Thừa kế theo pháp luật được chia thế nào?",
    "Di chúc hợp pháp cần điều kiện gì?",
    "Người mất năng lực hành vi dân sự được xác định thế nào?",
    "Quyền nhân thân gồm những quyền gì?",
    "Giao dịch dân sự bằng lời nói có hiệu lực không?",

    # --- ĐẤT ĐAI ---
    "Đất ở có thời hạn sử dụng bao lâu?",
    "Chuyển nhượng quyền sử dụng đất cần điều kiện gì?",
    "Thu hồi đất có được bồi thường không?",
    "Sổ đỏ và sổ hồng khác nhau thế nào?",
    "Hạn mức giao đất ở tại nông thôn?",
    "Đất nông nghiệp có được xây nhà không?",
    "Tranh chấp đất đai giải quyết ở đâu?",
    "Thừa kế quyền sử dụng đất thế nào?",
    "Chuyển mục đích sử dụng đất cần thủ tục gì?",
    "Đất lấn chiếm bị xử lý thế nào?",

    # --- DOANH NGHIỆP / THUẾ / HÀNH CHÍNH ---
    "Thành lập công ty TNHH cần điều kiện gì?",
    "Vốn điều lệ tối thiểu để thành lập công ty?",
    "Thuế thu nhập cá nhân được tính thế nào?",
    "Thuế GTGT bao nhiêu phần trăm?",
    "Hóa đơn điện tử có bắt buộc không?",
    "Thủ tục phá sản doanh nghiệp thế nào?",
    "Cổ đông sáng lập công ty cổ phần có nghĩa vụ gì?",
    "Hộ kinh doanh cá thể có phải đóng thuế không?",
    "Người đại diện theo pháp luật của công ty là ai?",
    "Xử phạt vi phạm hành chính về thuế thế nào?",
    "Khiếu nại quyết định hành chính thế nào?",
    "Người nước ngoài có được mua nhà tại VN không?",
    "Bảo hiểm y tế bắt buộc gồm những đối tượng nào?",
    "Thời hiệu xử phạt vi phạm hành chính là bao lâu?",
    "Đăng ký tạm trú, tạm vắng thế nào?",
]

assert len(QUESTIONS) == 100, f'Cần đúng 100 câu, hiện có {len(QUESTIONS)}'


# ============================================================
# 5. BENCHMARK — PHÂN LOẠI + XẾP HẠNG KHÓ → DỄ
# ============================================================
REFUSAL_MARKERS = [
    'không tìm thấy', 'không có quy định', 'không đề cập',
    'không có thông tin', 'không thể trả lời', 'không rõ',
    'chưa có quy định', 'không nằm trong', 'không được cung cấp',
    'không có trong trích dẫn', 'không đủ thông tin',
    'trích dẫn không', 'không xác định được',
]


def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)


def classify(answer, top_score, threshold):
    """Trả về 'answerable' / 'unanswerable'."""
    if not answer or len(answer.strip()) < 25:
        return 'unanswerable'
    if is_refusal(answer):
        return 'unanswerable'
    if top_score < threshold:
        return 'unanswerable'
    return 'answerable'


def run_benchmark():
    print('\n' + '=' * 60)
    print('--- 🧪 BENCHMARK 100 CÂU HỎI — XẾP HẠNG KHÓ → DỄ ---')
    print('=' * 60)

    # Pass 1: lấy điểm BM25 cho tất cả câu hỏi (nhanh)
    print('⏳ Bước 1/2: tính điểm BM25 cho 100 câu hỏi...')
    t0 = time.time()
    bm25_scores = []
    for q in QUESTIONS:
        tok = q.lower().split()
        scores = bm25.get_scores(tok) if tok else [0.0]
        bm25_scores.append(float(max(scores)) if len(scores) else 0.0)
    print(f'✅ Xong trong {time.time() - t0:.1f}s.')

    # Ngưỡng "có trích dẫn đủ liên quan" — dùng percentile 25
    sorted_scores = sorted(bm25_scores)
    threshold = sorted_scores[max(0, int(len(sorted_scores) * 0.25))]
    print(f'📏 Ngưỡng BM25 (P25) = {threshold:.3f}')

    # Pass 2: gọi model cho từng câu
    print('\n⏳ Bước 2/2: gọi model cho từng câu (chậm hơn)...')
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f'  [{i:3d}/100] {q[:70]}...', end=' ', flush=True)
        t_q = time.time()
        answer, top_score, _ = ask_model(q, top_k=5, max_tokens=400)
        dt = time.time() - t_q
        label = classify(answer, top_score, threshold)
        results.append({
            'id': i,
            'question': q,
            'answer': answer,
            'bm25_top_score': top_score,
            'label': label,
            'time_s': dt,
        })
        icon = '✅' if label == 'answerable' else '❌'
        print(f'{icon} BM25={top_score:.2f} ({dt:.1f}s)')

    # ----- Tổng hợp -----
    answerable = [r for r in results if r['label'] == 'answerable']
    unanswerable = [r for r in results if r['label'] == 'unanswerable']

    print('\n' + '=' * 60)
    print('📊 KẾT QUẢ TỔNG HỢP')
    print('=' * 60)
    print(f'Tổng số câu hỏi     : {len(results)}')
    print(f'✅ Trả lời được     : {len(answerable)}')
    print(f'❌ Không trả lời được: {len(unanswerable)}')

    # ----- Xếp hạng KHÓ → DỄ -----
    # Điểm khó: BM25 thấp = khó; bị từ chối = cộng thêm 1.0
    max_score = max(bm25_scores) or 1.0
    def difficulty(r):
        d = 1.0 - (r['bm25_top_score'] / max_score)  # 0..1, cao = khó
        if r['label'] == 'unanswerable':
            d += 1.0
        return d

    ranked = sorted(results, key=difficulty, reverse=True)

    print('\n' + '=' * 60)
    print('🎯 BẢNG XẾP HẠNG KHÓ → DỄ (top 30 khó nhất)')
    print('=' * 60)
    for rank, r in enumerate(ranked[:30], 1):
        tag = 'UNANS' if r['label'] == 'unanswerable' else 'ANS  '
        print(f'#{rank:02d} [{tag}] BM25={r["bm25_top_score"]:.3f} | {r["question"]}')
        short = r['answer'].replace('\n', ' ')[:160]
        print(f'      → {short}{"..." if len(r["answer"]) > 160 else ""}')

    # ----- Danh sách câu TRẢ LỜI ĐƯỢC -----
    print('\n' + '=' * 60)
    print(f'✅ DANH SÁCH {len(answerable)} CÂU TRẢ LỜI ĐƯỢC')
    print('=' * 60)
    for i, r in enumerate(sorted(answerable, key=lambda x: x['bm25_top_score'], reverse=True), 1):
        print(f'{i:3d}. [BM25={r["bm25_top_score"]:.2f}] {r["question"]}')

    # ----- 20 câu KHÔNG trả lời được -----
    print('\n' + '=' * 60)
    print(f'❌ 20 CÂU KHÔNG TRẢ LỜI ĐƯỢC (trích từ {len(unanswerable)})')
    print('=' * 60)
    for i, r in enumerate(unanswerable[:20], 1):
        print(f'{i:2d}. [BM25={r["bm25_top_score"]:.3f}] {r["question"]}')
        short = r['answer'].replace('\n', ' ')[:200]
        print(f'    → {short}{"..." if len(r["answer"]) > 200 else ""}')

    if len(unanswerable) < 20:
        print(f'\n⚠️  Chỉ có {len(unanswerable)} câu không trả lời được '
              f'(< 20). Kho luật hiện tại khá đầy đủ với bộ câu hỏi này.')

    # ----- Lưu kết quả -----
    out_path = get_resource_path('benchmark_results.json')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold_p25': threshold,
                'total': len(results),
                'answerable_count': len(answerable),
                'unanswerable_count': len(unanswerable),
                'ranked_hard_to_easy': ranked,
                'answerable_list': answerable,
                'unanswerable_list': unanswerable,
            }, f, ensure_ascii=False, indent=2)
        print(f'\n💾 Đã lưu kết quả chi tiết: {out_path}')
    except Exception as e:
        print(f'⚠️ Không lưu được file kết quả: {e}')


# ============================================================
# 6. CHẾ ĐỘ CHAT (giữ nguyên hành vi cũ)
# ============================================================
def chat_loop():
    print('\n' + '=' * 60)
    print('--- 🤖 CHATBOT LUẬT VIỆT NAM — OFFLINE — SẴN SÀNG ---')
    print(f'Kho luật: {len(documents):,} phân đoạn')
    print('Gõ "exit"/"quit" để thoát, "clear" để xóa màn hình,')
    print('"benchmark" để chạy demo 100 câu hỏi.')
    print('=' * 60)
    while True:
        try:
            user_input = input('\n👤 Bạn hỏi: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n👋 Tạm biệt!')
            break
        if not user_input:
            continue
        low = user_input.lower()
        if low in ['exit', 'quit', 'thoát']:
            print('👋 Tạm biệt!')
            break
        if low == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        if low == 'benchmark':
            run_benchmark()
            continue

        print('\n🔍 [Đang tra cứu và suy luận...]\n')
        t0 = time.time()
        answer, top_score, _ = ask_model(user_input, top_k=5, max_tokens=1024)
        print('🤖 Chatbot: ', end='', flush=True)
        # In dần cho có cảm giác stream
        for chunk in [answer[i:i+80] for i in range(0, len(answer), 80)]:
            print(chunk, end='', flush=True)
        print(f'\n\n⏱️  ({time.time() - t0:.1f}s | BM25={top_score:.3f})\n')


# ============================================================
# 7. MENU KHỞI ĐỘNG
# ============================================================
def main():
    print('\n' + '=' * 60)
    print('CHỌN CHẾ ĐỘ:')
    print('  [1] Chat tự do')
    print('  [2] Benchmark 100 câu hỏi (xếp hạng khó → dễ)')
    print('=' * 60)
    try:
        choice = input('Nhập lựa chọn (1/2, mặc định 1): ').strip() or '1'
    except (EOFError, KeyboardInterrupt):
        choice = '1'
    if choice == '2':
        run_benchmark()
    else:
        chat_loop()


if __name__ == '__main__':
    main()
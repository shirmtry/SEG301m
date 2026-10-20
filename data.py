# -*- coding: utf-8 -*-
"""
Đọc benchmark_results.json → phân tích:
  1. Câu hỏi thuộc chủ đề A nhưng chứa ý của chủ đề B (cross-topic)
  2. Câu hỏi A trùng nội dung với câu hỏi B (cross-question overlap)
Xuất: cross_analysis.json + cross_analysis.pdf
"""

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from itertools import combinations

# ============================================================
# TAXONOMY CHỦ ĐỀ PHÁP LUẬT + KEYWORD ĐẶC TRƯNG
# ============================================================
TOPICS = {
    'Lao động': [
        'lao động', 'hợp đồng lao động', 'hđlđ', 'thử việc', 'sa thải',
        'nghỉ phép', 'thai sản', 'ốm đau', 'quấy rối', 'nghỉ hưu',
        'giờ làm việc', 'làm thêm', 'tiền lương', 'lương tối thiểu',
        'bảo hiểm xã hội', 'bhxh', 'người sử dụng lao động', 'nsdlđ',
        'người lao động', 'kỷ luật lao động', 'chấm dứt hợp đồng',
    ],
    'Hôn nhân & Gia đình': [
        'kết hôn', 'ly hôn', 'vợ chồng', 'tảo hôn', 'hôn nhân',
        'cấp dưỡng', 'con nuôi', 'mang thai hộ', 'cha mẹ', 'con cái',
        'con ngoài giá thú', 'kết hôn giả', 'hôn nhân đồng giới',
        'đăng ký kết hôn', 'quyền nuôi con', 'tài sản chung',
    ],
    'Giao thông': [
        'giao thông', 'lái xe', 'nồng độ cồn', 'đèn đỏ', 'mũ bảo hiểm',
        'tốc độ', 'bằng lái', 'đăng kiểm', 'tai nạn', 'ngược chiều',
        'xi nhan', 'chuyển làn', 'rượu bia', 'điện thoại khi lái xe',
        'xe máy điện', 'xe không chính chủ',
    ],
    'Hình sự': [
        'tội', 'giết người', 'trộm cắp', 'cướp', 'lừa đảo', 'ma túy',
        'hiếp dâm', 'tham ô', 'buôn lậu', 'đánh bạc', 'hình sự', 'tù',
        'phạt tù', 'cố ý gây thương tích', 'rửa tiền', 'làm giả con dấu',
        'tố giác tội phạm', 'phòng vệ chính đáng', 'chưa thành niên',
    ],
    'Dân sự': [
        'dân sự', 'hợp đồng dân sự', 'đặt cọc', 'bồi thường',
        'di chúc', 'thừa kế', 'giao dịch dân sự', 'năng lực hành vi',
        'vô hiệu', 'thời hiệu khởi kiện', 'quyền nhân thân',
    ],
    'Đất đai': [
        'đất', 'sổ đỏ', 'sổ hồng', 'chuyển nhượng', 'thu hồi',
        'lấn chiếm', 'địa chính', 'quyền sử dụng đất', 'đất ở',
        'đất nông nghiệp', 'hạn mức', 'chuyển mục đích',
        'tranh chấp đất', 'thừa kế quyền sử dụng đất',
    ],
    'Doanh nghiệp & Thuế': [
        'công ty', 'doanh nghiệp', 'cổ đông', 'vốn điều lệ', 'thuế',
        'hóa đơn', 'kế toán', 'tnhh', 'cổ phần', 'phá sản',
        'người đại diện', 'hộ kinh doanh', 'thành lập công ty',
        'thuế gtgt', 'thuế tncn', 'hóa đơn điện tử',
    ],
    'Hành chính': [
        'khiếu nại', 'xử phạt hành chính', 'tạm trú', 'tạm vắng',
        'bảo hiểm y tế', 'bhyt', 'thời hiệu xử phạt', 'thủ tục hành chính',
        'cư trú', 'hộ khẩu',
    ],
}

# Map câu hỏi → chủ đề (dựa trên ID trong bộ 100 câu của bạn)
ID_TOPIC_HINT = {
    range(1, 21):  'Lao động',
    range(21, 36): 'Hôn nhân & Gia đình',
    range(36, 51): 'Giao thông',
    range(51, 66): 'Hình sự',
    range(66, 76): 'Dân sự',
    range(76, 86): 'Đất đai',
    range(86, 96): 'Doanh nghiệp & Thuế',
    range(96, 101): 'Hành chính',
}


def hint_topic(qid):
    for r, t in ID_TOPIC_HINT.items():
        if qid in r:
            return t
    return 'Khác'


# ============================================================
# CHUẨN HOÁ TIẾNG VIỆT
# ============================================================
STOPWORDS = set("""
là bao nhiêu gì nào khi của và có được không bị phạt thế các một
trong ngoài trên dưới về theo tại cho với để mà thì là ở ra vào
người này đó kia ấy nọ ai đâu sao vậy rằng nếu nhưng hoặc hay
đã sẽ đang cũng vẫn còn phải cần nên có thể bởi vì do nên
""".split())


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalize(s):
    s = s.lower()
    s = re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]',
               ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def tokenize(s):
    toks = normalize(s).split()
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


# ============================================================
# 1. PHÂN TÍCH CROSS-TOPIC
# ============================================================
def score_topics(question):
    """Trả về dict {topic: số keyword khớp}."""
    q = normalize(question)
    scores = {}
    for topic, kws in TOPICS.items():
        hits = [kw for kw in kws if normalize(kw) in q]
        if hits:
            scores[topic] = {'count': len(hits), 'matched': hits}
    return scores


def analyze_cross_topic(records):
    """
    Với mỗi câu hỏi:
      - primary_topic   = chủ đề có nhiều keyword khớp nhất
      - secondary       = các chủ đề khác cũng khớp
      - is_cross        = True nếu có >=1 secondary
      - hint_topic      = chủ đề "kỳ vọng" theo ID
      - topic_mismatch  = primary != hint
    """
    out = []
    for r in records:
        qid = r['id']
        q = r['question']
        sc = score_topics(q)
        hint = hint_topic(qid)

        if sc:
            ranked = sorted(sc.items(), key=lambda kv: kv[1]['count'], reverse=True)
            primary = ranked[0][0]
            secondary = [t for t, _ in ranked[1:]]
        else:
            primary = hint
            secondary = []

        out.append({
            'id': qid,
            'question': q,
            'label': r.get('label'),
            'bm25': r.get('bm25_top_score'),
            'hint_topic': hint,
            'primary_topic': primary,
            'secondary_topics': secondary,
            'matched_keywords': {t: v['matched'] for t, v in sc.items()},
            'is_cross': len(secondary) > 0,
            'topic_mismatch': primary != hint,
        })
    return out


# ============================================================
# 2. PHÂN TÍCH CROSS-QUESTION (câu A trùng ý câu B)
# ============================================================
def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def analyze_cross_question(records, threshold=0.30):
    """
    Với mỗi cặp (i, j), tính Jaccard token.
    Cặp >= threshold → ghi nhận overlap.
    """
    toks = {r['id']: tokenize(r['question']) for r in records}
    pairs = []
    for a, b in combinations(records, 2):
        ta, tb = toks[a['id']], toks[b['id']]
        j = jaccard(ta, tb)
        if j >= threshold:
            shared = sorted(set(ta) & set(tb))
            pairs.append({
                'id_a': a['id'],
                'q_a': a['question'],
                'topic_a': hint_topic(a['id']),
                'id_b': b['id'],
                'q_b': b['question'],
                'topic_b': hint_topic(b['id']),
                'jaccard': round(j, 3),
                'shared_tokens': shared,
                'cross_topic': hint_topic(a['id']) != hint_topic(b['id']),
            })
    pairs.sort(key=lambda p: p['jaccard'], reverse=True)
    return pairs


# ============================================================
# 3. NHÓM CLUSTER (union-find nhỏ)
# ============================================================
def cluster_pairs(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pairs:
        union(p['id_a'], p['id_b'])

    groups = defaultdict(list)
    for node in list(parent.keys()):
        groups[find(node)].append(node)
    return [sorted(v) for v in groups.values() if len(v) > 1]


# ============================================================
# 4. MAIN
# ============================================================
def main():
    here = os.path.abspath(os.path.dirname(__file__))
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, 'benchmark_results.json')

    if not os.path.exists(json_path):
        print(f'❌ Không tìm thấy: {json_path}')
        sys.exit(1)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = data.get('ranked_hard_to_easy', [])
    print(f'📖 Đã đọc {len(records)} bản ghi.')

    # --- 1. Cross-topic ---
    cross_topic = analyze_cross_topic(records)
    cross = [r for r in cross_topic if r['is_cross']]
    mismatch = [r for r in cross_topic if r['topic_mismatch']]

    print(f'\n🎯 CROSS-TOPIC:')
    print(f'   Câu có >=2 chủ đề: {len(cross)}')
    print(f'   Câu lệch chủ đề kỳ vọng: {len(mismatch)}')

    # --- 2. Cross-question ---
    pairs = analyze_cross_question(records, threshold=0.30)
    clusters = cluster_pairs(pairs)
    print(f'\n🔗 CROSS-QUESTION:')
    print(f'   Cặp trùng >=30% token: {len(pairs)}')
    print(f'   Nhóm cluster: {len(clusters)}')

    # --- 3. In mẫu ---
    print('\n--- TOP 15 CẶP TRÙNG NỘI DUNG NHẤT ---')
    for p in pairs[:15]:
        flag = '⚡' if p['cross_topic'] else '  '
        print(f'{flag} J={p["jaccard"]:.2f} '
              f'[{p["topic_a"]} #{p["id_a"]}] ↔ [{p["topic_b"]} #{p["id_b"]}]')
        print(f'      A: {p["q_a"]}')
        print(f'      B: {p["q_b"]}')
        print(f'      Từ chung: {", ".join(p["shared_tokens"][:10])}')

    # --- 4. Lưu JSON ---
    out_json = os.path.join(os.path.dirname(json_path), 'cross_analysis.json')
    payload = {
        'created': datetime.now().isoformat(),
        'total': len(records),
        'cross_topic_count': len(cross),
        'mismatch_count': len(mismatch),
        'overlap_pair_count': len(pairs),
        'cluster_count': len(clusters),
        'cross_topic': cross,
        'mismatch': mismatch,
        'overlap_pairs': pairs,
        'clusters': [
            {
                'ids': cl,
                'questions': [next(r['question'] for r in records if r['id'] == i)
                              for i in cl],
            } for cl in clusters
        ],
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'\n💾 Đã lưu: {out_json}')

    return payload


if __name__ == '__main__':
    main()
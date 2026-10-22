import json
import re
import os

FILES = [
    r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\legal_merged.jsonl",
]

# Dấu hiệu CHẮC CHẮN của luật cũ (không thể có trong Luật 2024)
OLD_SIGNS = [
    (r"Điều\s+2\.1\.1", "Cấu trúc 'Điều 2.1.1' (cách ghi cũ)"),
    (r"3\.500\s*kg", "Ngưỡng '3.500 kg' (đã bỏ trong luật mới)"),
    (r"điểm\s+[a-zđ]\s+và\s+[a-zđ]\s+của\s+Điều", "Cách viện dẫn 'điểm x và y của Điều' (cũ)"),
    (r"hạng\s+B2\s+cấp\s+cho", "Câu định nghĩa 'B2 cấp cho' (cũ)"),
]

PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in OLD_SIGNS]

for filepath in FILES:
    print(f"\n{'='*75}")
    print(f"📁 {os.path.basename(filepath)}")
    print('='*75)
    
    if not os.path.exists(filepath):
        print("  ⚠️  File không tồn tại, bỏ qua")
        continue
    
    suspects = []
    total = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            data = json.loads(line)
            title = data.get("metadata", {}).get("title", "")
            text = data.get("text", "")
            
            for pat, label in PATTERNS:
                match = pat.search(text)
                if match:
                    suspects.append({
                        "line": i + 1,
                        "title": title[:130],
                        "label": label,
                        "match": match.group(0)[:80],
                        "text_preview": text[:250].replace("\n", " | ")
                    })
                    break
    
    print(f"📊 Tổng chunks: {total:,}")
    print(f"⚠️  Nghi vấn: {len(suspects)} chunks\n")
    
    for s in suspects[:30]:
        print(f"--- Line {s['line']} | {s['label']} ---")
        print(f"  Title: {s['title']}")
        print(f"  Match: ...{s['match']}...")
        print(f"  Text : {s['text_preview']}...")
        print()
    
    if len(suspects) > 30:
        print(f"... (còn {len(suspects)-30} chunk nữa không in)")
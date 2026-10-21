import json
from collections import Counter
import re

INPUT = r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\legal_merged_v2_FINAL.jsonl"

# Regex bắt số hiệu văn bản (VD: 52/1999/NĐ-CP, 15/1998/NĐ-CP)
PATTERN = re.compile(
    r"(?:Nghị định|Thông tư|Quyết định|Luật)\s*(?:số\s*)?"
    r"([0-9]{1,4}[-/][0-9]{4}[-/][A-ZĐ0-9/-]+|[0-9]{1,4}[-/][A-ZĐ0-9/-]+)",
    re.IGNORECASE
)

# Đếm theo năm ban hành
year_counter = Counter()
items_by_year = {}

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        title = data.get("metadata", {}).get("title", "")
        match = PATTERN.search(title)
        if not match:
            continue
        num = match.group(1)
        # Tìm năm (4 chữ số)
        year_match = re.search(r"(\d{4})", num)
        if not year_match:
            continue
        year = int(year_match.group(1))
        year_counter[year] += 1
        items_by_year.setdefault(year, []).append(title[:150])

# In báo cáo
print("=" * 80)
print("📊 THỐNG KÊ VĂN BẢN THEO NĂM BAN HÀNH")
print("=" * 80)

# Sắp xếp năm tăng dần
for year in sorted(year_counter.keys()):
    count = year_counter[year]
    if year <= 2010:  # Chỉ quan tâm văn bản cũ
        print(f"\n🔥 Năm {year}: {count:,} chunks")
        # In tối đa 5 văn bản mỗi năm để check
        unique = list(set(items_by_year[year]))[:5]
        for title in unique:
            print(f"   - {title}")

# Tổng kết
total_old = sum(c for y, c in year_counter.items() if y <= 2010)
total_all = sum(year_counter.values())
print("\n" + "=" * 80)
print(f"📈 TỔNG: {total_all:,} chunks có số hiệu")
print(f"⚠️  Văn bản trước 2010: {total_old:,} chunks ({total_old/total_all*100:.1f}%)")
print("=" * 80)
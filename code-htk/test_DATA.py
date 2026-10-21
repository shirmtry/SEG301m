import json
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm

FILE_PATH = r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\legal_merged.jsonl"

count_laws = Counter()
total_articles = 0

print("⏳ Đang phân tích file legal_merged.jsonl...")

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Đang quét dữ liệu"):
        try:
            data = json.loads(line)
            total_articles += 1
            
            # Lấy tiêu đề (VD: "Bộ luật Dân sự 2015 - Điều 1. Phạm vi điều chỉnh")
            title = data.get("metadata", {}).get("title", "Không xác định")
            
            # Tách lấy tên Luật (cắt bỏ phần "- Điều...")
            if " - Điều" in title:
                law_name = title.split(" - Điều")[0].strip()
            else:
                law_name = title.split(" - ")[0].strip()
                
            count_laws[law_name] += 1
                
        except Exception as e:
            pass

print(f"\n✅ Đã phân tích xong {total_articles:,} Điều luật!\n")

print("="*60)
print("TOP 20 BỘ LUẬT / LUẬT CÓ NHIỀU ĐIỀU NHẤT TRONG DATA")
print("="*60)
df_laws = pd.DataFrame(count_laws.most_common(20), columns=["Tên Bộ Luật / Luật", "Số lượng Điều"])
print(df_laws.to_string(index=False))

print(f"\n📊 Tổng cộng có {len(count_laws)} Bộ luật / Luật khác nhau trong cơ sở dữ liệu.")
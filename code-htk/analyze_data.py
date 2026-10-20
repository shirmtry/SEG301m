import json
from collections import Counter
from pathlib import Path

META_FILE = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\processed\legal_meta_v2.json")

print("Đang phân tích dữ liệu (có thể mất 1-2 phút vì file rất lớn)...")
meta = json.load(open(META_FILE, encoding="utf-8"))

# Đếm số lượng theo Lĩnh vực (linh_vuc)
linh_vuc_counter = Counter(item.get("linh_vuc", "Không xác định") for item in meta)
print("\n=== PHÂN BỔ THEO LĨNH VỰC (Top 15) ===")
for linh_vuc, count in linh_vuc_counter.most_common(15):
    if linh_vuc: # Bỏ qua nếu rỗng
        print(f"{linh_vuc:<40}: {count:,} chunks")

# Đếm số lượng theo Loại văn bản (loai)
loai_counter = Counter(item.get("loai", "Không xác định") for item in meta)
print("\n=== PHÂN BỔ THEO LOẠI VĂN BẢN ===")
for loai, count in loai_counter.most_common(10):
    if loai:
        print(f"{loai:<30}: {count:,} chunks")
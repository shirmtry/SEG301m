import json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cấu hình font tiếng Việt cho biểu đồ
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma'] # Thay bằng font máy bạn nếu cần
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
META_FILE = BASE_DIR / "processed" / "legal_meta_v2.json"
OUT_DIR = BASE_DIR / "report_assets"
OUT_DIR.mkdir(exist_ok=True)

print("⏳ Đang đọc dữ liệu (có thể mất 1-2 phút)...")
meta = json.load(open(META_FILE, encoding="utf-8"))

# 1. Phân tích Lĩnh vực (Linh_vuc)
linh_vuc_counts = Counter(item.get("linh_vuc", "Chưa phân loại") for item in meta)
# Lấy top 10 lĩnh vực nhiều nhất, bỏ qua các nhãn rỗng
top_linh_vuc = {k: v for k, v in linh_vuc_counts.most_common(15) if k and k != "Chưa phân loại"}

# 2. Phân tích Loại văn bản (Loai)
# Dựa vào số hiệu (VD: QH là Luật, NĐ là Nghị định, TT là Thông tư)
def get_doc_type(so_ky_hieu):
    if not so_ky_hieu: return "Khác"
    if "QH" in so_ky_hieu: return "Luật / Bộ luật"
    if "NĐ" in so_ky_hieu: return "Nghị định"
    if "TT" in so_ky_hieu: return "Thông tư"
    if "QĐ" in so_ky_hieu: return "Quyết định"
    if "CP" in so_ky_hieu: return "Nghị quyết CP"
    return "Khác"

loai_counts = Counter(get_doc_type(item.get("so_ky_hieu", "")) for item in meta)
# Loại bỏ "Khác" hoặc rỗng nếu muốn biểu đồ đẹp hơn
loai_counts.pop("Khác", None)

# 3. Phân tích độ dài chunk (số ký tự)
chunk_lengths = [len(item.get("content", "")) for item in meta]

# ===== VẼ BIỂU ĐỒ =====
sns.set_theme(style="whitegrid")

# Biểu đồ 1: Top Lĩnh vực
plt.figure(figsize=(10, 6))
sns.barplot(x=list(top_linh_vuc.values()), y=list(top_linh_vuc.keys()), palette="viridis")
plt.title("Top 15 Lĩnh vực Pháp luật phổ biến nhất trong Cơ sở dữ liệu", fontsize=14)
plt.xlabel("Số lượng Chunks", fontsize=12)
plt.ylabel("Lĩnh vực", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "top_linh_vuc.png", dpi=300)
print(f"✅ Đã lưu biểu đồ: {OUT_DIR / 'top_linh_vuc.png'}")

# Biểu đồ 2: Loại văn bản
plt.figure(figsize=(8, 8))
plt.pie(loai_counts.values(), labels=loai_counts.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("Phân bổ các Loại văn bản Pháp luật", fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "loai_van_ban.png", dpi=300)
print(f"✅ Đã lưu biểu đồ: {OUT_DIR / 'loai_van_ban.png'}")

# Biểu đồ 3: Phân bố độ dài chunk
plt.figure(figsize=(10, 5))
sns.histplot(chunk_lengths, bins=50, kde=True, color="blue")
plt.title("Phân bố độ dài các Chunks (Ký tự)", fontsize=14)
plt.xlabel("Số ký tự", fontsize=12)
plt.ylabel("Tần suất", fontsize=12)
plt.xlim(0, 4000) # Cắt bớt các outlier quá dài
plt.tight_layout()
plt.savefig(OUT_DIR / "chunk_length.png", dpi=300)
print(f"✅ Đã lưu biểu đồ: {OUT_DIR / 'chunk_length.png'}")

# ===== IN SỐ LIỆU ĐỂ ĐƯA VÀO BÁO CÁO =====
print("\n" + "="*50)
print("SỐ LIỆU TÓM TẮT CHO BÁO CÁO")
print("="*50)
print(f"Tổng số chunks: {len(meta):,}")
print(f"Tổng số văn bản (ước tính theo doc_id): {len(set(item['doc_id'] for item in meta)):,}")
print(f"Độ dài trung bình của 1 chunk: {sum(chunk_lengths)/len(chunk_lengths):.0f} ký tự")
print(f"Lĩnh vực đa dạng nhất: {list(top_linh_vuc.keys())[0]} ({list(top_linh_vuc.values())[0]:,} chunks)")
print("="*50)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from collections import Counter

# ===== CẤU HÌNH =====
BASE_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
CSV_FILE = BASE_DIR / "BAO_CAO_TEST.csv"
OUT_DIR = BASE_DIR / "report_assets"
OUT_DIR.mkdir(exist_ok=True)

# Cấu hình font tiếng Việt cho biểu đồ
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("⏳ Đang đọc và phân tích file CSV...")
try:
    df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"❌ Không tìm thấy file: {CSV_FILE}")
    exit()

# Xử lý dữ liệu
df['Thời gian (giây)'] = pd.to_numeric(df['Thời gian (giây)'], errors='coerce')

# Xác định câu bị từ chối
refusal_keywords = "không tìm thấy|không có thông tin|không thể"
df['is_refusal'] = df['Câu trả lời của Bot'].str.lower().str.contains(refusal_keywords, na=False)

# Lấy Top 1 Document (ví dụ: "83/2014/TT-BTC (0.73) | ..." -> "83/2014/TT-BTC")
def extract_top_doc(doc_str):
    if pd.isna(doc_str) or not doc_str: return None
    top1 = doc_str.split('|')[0].strip()
    doc_id = re.sub(r'\(.*?\)', '', top1).strip()
    return doc_id

df['Top1_Doc'] = df['Nguồn tham khảo (Top 2)'].apply(extract_top_doc)

# ==========================================
# BIỂU ĐỒ 1: TỶ LỆ TRẢ LỜI THÀNH CÔNG VS TỪ CHỐI
# ==========================================
success_count = len(df) - df['is_refusal'].sum()
refusal_count = df['is_refusal'].sum()

plt.figure(figsize=(7, 7))
labels = ['Trả lời thành công', 'Từ chối (Không tìm thấy)']
sizes = [success_count, refusal_count]
colors = ['#4CAF50', '#FF5252']
explode = (0.1, 0) 

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12})
plt.title('Tỷ lệ Trả lời của Chatbot (100 câu test)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / "ty_le_tra_loi.png", dpi=300)
print(f"✅ Đã lưu: {OUT_DIR / 'ty_le_tra_loi.png'}")

# ==========================================
# BIỂU ĐỒ 2: PHÂN BỐ THỜI GIAN PHẢN HỒI
# ==========================================
plt.figure(figsize=(10, 5))
sns.histplot(df['Thời gian (giây)'], bins=15, kde=True, color="teal")
plt.axvline(df['Thời gian (giây)'].mean(), color='red', linestyle='--', label=f"Trung bình: {df['Thời gian (giây)'].mean():.1f}s")
plt.title('Phân bố Thời gian Phản hồi của Chatbot (giây)', fontsize=14, fontweight='bold')
plt.xlabel('Thời gian (giây)', fontsize=12)
plt.ylabel('Số lượng câu hỏi', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "phan_bo_thoi_gian.png", dpi=300)
print(f"✅ Đã lưu: {OUT_DIR / 'phan_bo_thoi_gian.png'}")

# ==========================================
# BIỂU ĐỒ 3: TOP 10 VĂN BẢN ĐƯỢC TRÍCH DẪN NHIỀU NHẤT
# ==========================================
top_docs = df['Top1_Doc'].dropna()
doc_counts = Counter(top_docs).most_common(10)

if doc_counts:
    docs, counts = zip(*doc_counts)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(docs), hue=list(docs), palette="viridis", legend=False)
    plt.title('Top 10 Văn bản Pháp luật được trích dẫn nhiều nhất', fontsize=14, fontweight='bold')
    plt.xlabel('Số lần được trích dẫn', fontsize=12)
    plt.ylabel('Số hiệu văn bản', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_van_ban.png", dpi=300)
    print(f"✅ Đã lưu: {OUT_DIR / 'top_van_ban.png'}")

# ==========================================
# BIỂU ĐỒ 4: PHÂN BỐ ĐIỂM THỦ CÔNG (NẾU CÓ)
# ==========================================
if 'Điểm thủ công (1-5)' in df.columns:
    df['Điểm thủ công (1-5)'] = pd.to_numeric(df['Điểm thủ công (1-5)'], errors='coerce')
    valid_scores = df['Điểm thủ công (1-5)'].dropna()
    
    if not valid_scores.empty:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=valid_scores, palette="Blues_d", hue=valid_scores, legend=False)
        plt.title('Phân bố Điểm đánh giá Thủ công (Thang 1-5)', fontsize=14, fontweight='bold')
        plt.xlabel('Điểm số', fontsize=12)
        plt.ylabel('Số lượng câu hỏi', fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "phan_bo_diem.png", dpi=300)
        print(f"✅ Đã lưu: {OUT_DIR / 'phan_bo_diem.png'}")
        print(f"   👉 Điểm trung bình: {valid_scores.mean():.2f}/5.0")
    else:
        print("⚠️ Chưa có điểm thủ công nào được điền trong file CSV.")

print("\n🎉 HOÀN TẤT! Hãy vào thư mục 'report_assets' để lấy ảnh chèn vào báo cáo.")
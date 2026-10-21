import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import Counter

# Cấu hình style
sns.set_theme(style="whitegrid")
os.makedirs("report_assets", exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv("demo_results.csv")

# ==========================================
# 1. Biểu đồ tròn: Tỷ lệ Trả lời
# ==========================================
plt.figure(figsize=(8, 6))
status_counts = df["Trạng thái"].value_counts()
colors = ['#4CAF50', '#F44336'] # Xanh và Đỏ
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.05, 0), shadow=True)
plt.title("Tỷ lệ Phản hồi của Chatbot (100 câu test)", fontsize=14, fontweight='bold')
plt.savefig("report_assets/ty_le_tra_loi.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. Biểu đồ Histogram: Phân bố Thời gian
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(df["Thời gian (giây)"], bins=15, kde=True, color='teal')
plt.axvline(df["Thời gian (giây)"].mean(), color='red', linestyle='--', label=f'Trung bình: {df["Thời gian (giây)"].mean():.1f}s')
plt.title("Phân bố Thời gian Phản hồi của Chatbot (giây)", fontsize=14, fontweight='bold')
plt.xlabel("Thời gian (giây)")
plt.ylabel("Số lượng câu hỏi")
plt.legend()
plt.savefig("report_assets/phan_bo_thoi_gian.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. Biểu đồ Cột: Top 10 Văn bản được trích dẫn nhiều nhất
# ==========================================
all_sources = []
for src in df["Nguồn tham khảo"].dropna():
    sources = src.split(" | ")
    for s in sources:
        if s and s != "Không rõ":
            # Cắt lấy tên luật (bỏ phần Điều)
            law_name = s.split(" - Điều")[0].strip()
            all_sources.append(law_name)

source_counts = Counter(all_sources)
top_10 = source_counts.most_common(10)

labels = [item[0] for item in top_10]
values = [item[1] for item in top_10]

plt.figure(figsize=(12, 6))
sns.barplot(x=values, y=labels, palette="viridis")
plt.title("Top 10 Văn bản Pháp luật được trích dẫn nhiều nhất", fontsize=14, fontweight='bold')
plt.xlabel("Số lần được trích dẫn")
plt.ylabel("Tên Văn bản")
plt.tight_layout()
plt.savefig("report_assets/top_van_ban.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 4. Biểu đồ Scatter: Tương quan Thời gian vs Độ dài câu trả lời
# ==========================================
df["Độ dài câu trả lời (ký tự)"] = df["Câu trả lời"].apply(len)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Thời gian (giây)", y="Độ dài câu trả lời (ký tự)", hue="Trạng thái", palette={'Trả lời thành công': 'green', 'Không trả lời được': 'red'}, s=100)
plt.title("Tương quan giữa Thời gian và Độ dài câu trả lời", fontsize=14, fontweight='bold')
plt.xlabel("Thời gian phản hồi (giây)")
plt.ylabel("Độ dài câu trả lời (ký tự)")
plt.savefig("report_assets/tuong_quan_thoi_gian.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Đã tạo xong 4 biểu đồ trong thư mục 'report_assets'!")
print("👉 Bạn có thể dùng các ảnh này để chèn vào file Báo cáo Markdown/Word mới.")
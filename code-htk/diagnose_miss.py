"""diagnose_miss.py — Kiểm tra 143 data miss. CHỈ ĐỌC, không ghi."""
import json
import re
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
HUYTKING = BASE / "HUYTKING.jsonl"
CSV = BASE / "test_results" / "20261022_235732_914new.csv"


def main():
    # 1. Load data
    print("📂 Load HUYTKING.jsonl...")
    titles = []
    with open(HUYTKING, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line).get("metadata", {}).get("title", "")
            if t:
                titles.append(t)
    print(f"   → {len(titles):,} titles\n")

    # 2. Load miss cases
    df = pd.read_csv(CSV, encoding="utf-8-sig", on_bad_lines="skip")
    miss = df[(df["expected"] == "answer") & (df["verdict"] == "refused")]
    print(f"🔍 Phân tích {len(miss)} câu data miss\n")

    stats = {"CÓ trong data": 0, "KHÔNG có": 0, "Không parse được": 0}
    parsed_misses = []

    for _, r in miss.iterrows():
        q = str(r["question"]).strip()
        # Parse: "Điều N của Luật X", "Điều N Luật X quy định", "Luật X - Điều N"
        m1 = re.search(r"[Đđ]iều\s+(\d+).*?của\s+(.+?)[\.\?]?$", q)
        m2 = re.search(r"[Đđ]iều\s+(\d+)\s+(.+?)\s+(?:quy định|-)", q)
        m3 = re.search(r"^(.+?)\s+-\s+[Đđ]iều\s+(\d+)", q)
        m4 = re.search(r"[Đđ]iều\s+(\d+)\s+(.+?)[\.\?]?$", q)

        dieu, law = None, None
        if m1:
            dieu, law = m1.group(1), m1.group(2).strip()
        elif m3:
            law, dieu = m3.group(1).strip(), m3.group(2)
        elif m2:
            dieu, law = m2.group(1), m2.group(2).strip()
        elif m4:
            dieu, law = m4.group(1), m4.group(2).strip()

        if not dieu or not law:
            stats["Không parse được"] += 1
            parsed_misses.append((q, "?", "?"))
            continue

        # Tìm trong titles
        law_low = law.lower()[:30]  # cắt để match lỏng
        matched = [t for t in titles
                   if f"Điều {dieu}." in t or f"Điều {dieu} " in t or f"Điều {dieu}:" in t]
        matched = [t for t in matched if law_low in t.lower()]

        if matched:
            stats["CÓ trong data"] += 1
            parsed_misses.append((q, dieu, f"✅ {matched[0][:60]}"))
        else:
            stats["KHÔNG có"] += 1
            parsed_misses.append((q, dieu, "❌"))



    # 3. In báo cáo
    print("=" * 78)
    print("📊 KẾT QUẢ PHÂN TÍCH 143 DATA MISS")
    print("=" * 78)
    for k, v in stats.items():
        pct = v / len(miss) * 100
        print(f"  {k:20s}: {v:4,} ({pct:5.1f}%)")

    print("\n" + "=" * 78)
    print("📋 CHI TIẾT 30 CÂU ĐẦU")
    print("=" * 78)
    for q, dieu, result in parsed_misses[:30]:
        print(f"  Q: {q[:75]}")
        print(f"     → Điều {dieu} → {result}\n")

    # 4. Kết luận
    print("=" * 78)
    print("🧭 KẾT LUẬN")
    print("=" * 78)
    if stats["CÓ trong data"] > stats["KHÔNG có"]:
        print(f"  ⚠️  {stats['CÓ trong data']} câu có Điều trong data mà vẫn refuse")
        print(f"  → Đây là lỗi RETRIEVAL, cần fix retrieve_advanced()")
        print(f"  → Không phải lỗi data")
    else:
        print(f"  ℹ️  {stats['KHÔNG có']} câu không có Điều trong data")
        print(f"  → Đây là DATA MISS thật (data VBHN đổi số điều)")
        print(f"  → Cần crawl bổ sung HOẶC chấp nhận (an toàn hơn)")


if __name__ == "__main__":
    main()
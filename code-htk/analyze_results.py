"""analyze_results.py — Phân tích CSV kết quả test. CHỈ ĐỌC, không ghi file.

   Đọc:
     - test_results/*_914new.csv  (914 câu mới)
     - test_results/*_results.csv (500 câu cũ, nếu có)

   Không sửa file nào. Chỉ in báo cáo.
"""
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
OUTPUT_DIR = BASE_DIR / "test_results"

# ═══════════════════════════════════════════════════════════
# LOAD CSV ROBUST
# ═══════════════════════════════════════════════════════════
def load_csv_robust(path):
    """Đọc CSV, fallback nhiều cách. Trả về DataFrame hoặc None."""
    if not path.exists():
        return None
    # Cách 1: pandas strict (CSV mới dùng QUOTE_ALL — OK)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        if len(df) > 0 and "verdict" in df.columns:
            return df
    except Exception:
        pass
    # Cách 2: fallback đọc bằng csv module
    try:
        rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) < len(header):
                    continue
                rows.append(dict(zip(header, r)))
        return pd.DataFrame(rows) if rows else None
    except Exception:
        return None


def find_files():
    """Tìm file mới nhất mỗi loại."""
    new_csv = sorted(OUTPUT_DIR.glob("*_914new.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    old_csv = sorted(OUTPUT_DIR.glob("*_results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (new_csv[0] if new_csv else None), (old_csv[0] if old_csv else None)


# ═══════════════════════════════════════════════════════════
# PHÂN TÍCH
# ═══════════════════════════════════════════════════════════
def section(title):
    print(f"\n{'═'*78}")
    print(f"  {title}")
    print(f"{'═'*78}")


def subsection(title):
    print(f"\n  ── {title}")
    print(f"  {'─'*74}")


def bar(value, total, width=30, char="█"):
    if total == 0:
        return ""
    n = round(value / total * width)
    return char * n + "░" * (width - n)


def analyze_one(df, label):
    """Phân tích 1 DataFrame."""
    if df is None or len(df) == 0:
        print(f"  ⚠️  {label}: không có dữ liệu")
        return None

    n = len(df)
    section(f"📄 {label} — {n:,} câu")

    # ─── TỔNG QUAN VERDICT ───
    subsection("VERDICT TỔNG QUAN")
    verdicts = Counter(df["verdict"].astype(str))
    for v, c in verdicts.most_common():
        pct = c / n * 100
        print(f"  {v:10s} {c:5,} ({pct:5.1f}%) {bar(c, n)}")

    # ─── THEO CATEGORY ───
    subsection("THEO CATEGORY")
    cats = Counter(df["category"].astype(str))
    for cat, c in cats.most_common():
        print(f"\n  [{cat}] — {c} câu")
        sub = df[df["category"] == cat]
        sv = Counter(sub["verdict"].astype(str))
        for v, vc in sv.most_common():
            pct = vc / len(sub) * 100
            print(f"    {v:10s} {vc:5,} ({pct:5.1f}%) {bar(vc, len(sub), width=25)}")

    # ─── THEO EXPECTED ───
    if "expected" in df.columns:
        subsection("VERDICT vs EXPECTED")
        print(f"  {'Expected':10s} {'Verdict':10s} {'Count':>6s}   {'Kết quả'}")
        print(f"  {'─'*10} {'─'*10} {'─'*6}   {'─'*40}")
        ct = pd.crosstab(df["expected"].astype(str), df["verdict"].astype(str))
        for expected in ct.index:
            for verdict in ct.columns:
                count = int(ct.loc[expected, verdict])
                if count == 0:
                    continue
                # Đánh giá đúng/sai
                if expected == "refuse" and verdict == "refused":
                    mark = "✅ PASS"
                elif expected == "answer" and verdict == "answered":
                    mark = "✅ PASS"
                elif expected == "any":
                    mark = "ℹ️  any"
                elif expected == "refuse" and verdict in ("answered", "warning"):
                    mark = "❌ FAIL (lẽ ra refuse)"
                elif expected == "answer" and verdict in ("refused", "warning"):
                    mark = "⚠️  MISS (lẽ ra answer)"
                elif verdict == "blocked":
                    mark = "🛡️  BLOCKED (an toàn)"
                else:
                    mark = f"…{verdict}"
                print(f"  {expected:10s} {verdict:10s} {count:>6,}   {mark}")

    # ─── REASON ───
    subsection("REASON DISTRIBUTION")
    reasons = Counter(df["reason"].astype(str))
    for r, c in reasons.most_common(15):
        print(f"  {r:35s} {c:5,} ({c/n*100:5.1f}%)")

    # ─── TIME ───
    subsection("THỜI GIAN PHẢN HỒI")
    try:
        ts = pd.to_numeric(df["time_sec"], errors="coerce").dropna()
        if len(ts):
            print(f"  Count    : {len(ts):,}")
            print(f"  Mean     : {ts.mean():.1f}s")
            print(f"  Median   : {ts.median():.1f}s")
            print(f"  Min/Max  : {ts.min():.1f}s / {ts.max():.1f}s")
            print(f"  Total    : {ts.sum()/3600:.1f}h")
    except Exception:
        pass

    # ─── CITATION ───
    subsection("CITATION")
    try:
        cites = pd.to_numeric(df["n_citations"], errors="coerce").fillna(0).astype(int)
        print(f"  Có cite [N]  : {(cites > 0).sum():,} câu ({(cites > 0).mean()*100:.1f}%)")
        print(f"  Không cite   : {(cites == 0).sum():,} câu ({(cites == 0).mean()*100:.1f}%)")
        print(f"  Trung bình   : {cites.mean():.2f} cites/câu")
    except Exception:
        pass

    return df


def analyze_failures(df, label, limit=30):
    """Liệt kê các câu FAIL cụ thể."""
    if df is None or len(df) == 0:
        return

    section(f"❌ TOP FAIL CASES — {label}")

    # Trap mà không refuse
    trap_fail = df[(df["expected"] == "refuse") & (~df["verdict"].isin(["refused", "blocked"]))]
    subsection(f"TRAP FAIL (lẽ ra refuse, không refuse) — {len(trap_fail)} câu")
    for _, r in trap_fail.head(limit).iterrows():
        q = str(r.get("question", ""))[:70]
        v = r.get("verdict", "?")
        a = str(r.get("answer", ""))[:90].replace("\n", " ")
        print(f"  [{v}] Q: {q}")
        print(f"        A: {a}")
        print()

    # From_data mà bị refuse
    data_fail = df[(df["expected"] == "answer") & (df["verdict"] == "refused")]
    subsection(f"DATA MISS (lẽ ra answer, bị refuse) — {len(data_fail)} câu")
    for _, r in data_fail.head(limit).iterrows():
        q = str(r.get("question", ""))[:70]
        print(f"  Q: {q}")
        print()

    # Citation misuse
    if "reason" in df.columns:
        misuse = df[df["reason"] == "citation_misuse"]
        subsection(f"CITATION MISUSE — {len(misuse)} câu")
        for _, r in misuse.head(limit).iterrows():
            q = str(r.get("question", ""))[:70]
            a = str(r.get("answer", ""))[:120].replace("\n", " ")
            print(f"  Q: {q}")
            print(f"     A: {a}")
            print()

        mid = df[df["reason"] == "mid_refusal"]
        subsection(f"MID REFUSAL — {len(mid)} câu")
        for _, r in mid.head(limit).iterrows():
            q = str(r.get("question", ""))[:70]
            a = str(r.get("answer", ""))[:120].replace("\n", " ")
            print(f"  Q: {q}")
            print(f"     A: {a}")
            print()


def calc_score(df):
    """Tính điểm pass/fail tổng thể."""
    if df is None or len(df) == 0:
        return None

    n = len(df)
    stats = {
        "trap_pass": 0, "trap_total": 0,
        "data_pass": 0, "data_total": 0,
        "edge_any": 0, "edge_total": 0,
        "blocked": 0,
    }

    for _, r in df.iterrows():
        expected = str(r.get("expected", ""))
        verdict = str(r.get("verdict", ""))
        if expected == "refuse":
            stats["trap_total"] += 1
            if verdict in ("refused", "blocked"):
                stats["trap_pass"] += 1
        elif expected == "answer":
            stats["data_total"] += 1
            if verdict == "answered":
                stats["data_pass"] += 1
        elif expected == "any":
            stats["edge_total"] += 1
            stats["edge_any"] += 1
        if verdict == "blocked":
            stats["blocked"] += 1

    return stats


def main():
    print("=" * 78)
    print("📊 PHÂN TÍCH KẾT QUẢ TEST 914 CÂU")
    print("   (chỉ đọc — không ghi file nào)")
    print("=" * 78)

    new_csv, old_csv = find_files()
    print(f"\n📁 CSV mới : {new_csv.name if new_csv else 'KHÔNG CÓ'}")
    print(f"📁 CSV cũ  : {old_csv.name if old_csv else 'KHÔNG CÓ'}")

    df_new = load_csv_robust(new_csv) if new_csv else None
    df_old = load_csv_robust(old_csv) if old_csv else None

    # Phân tích từng file
    analyze_one(df_new, "914 CÂU MỚI")
    if df_old is not None:
        analyze_one(df_old, "500 CÂU CŨ")

    # Fail cases
    analyze_failures(df_new, "914 CÂU MỚI")

    # ─── ĐIỂM PASS/FAIL ───
    section("🎯 ĐIỂM PASS/FAIL TỔNG THỂ")
    for label, df in [("914 câu mới", df_new), ("500 câu cũ", df_old)]:
        if df is None:
            continue
        stats = calc_score(df)
        if not stats:
            continue
        print(f"\n  📄 {label}:")
        if stats["trap_total"]:
            rate = stats["trap_pass"] / stats["trap_total"] * 100
            mark = "✅" if rate >= 90 else ("⚠️" if rate >= 75 else "❌")
            print(f"    {mark} Trap refuse đúng : {stats['trap_pass']}/{stats['trap_total']} ({rate:.1f}%)")
        if stats["data_total"]:
            rate = stats["data_pass"] / stats["data_total"] * 100
            mark = "✅" if rate >= 80 else ("⚠️" if rate >= 60 else "❌")
            print(f"    {mark} Data answered đúng: {stats['data_pass']}/{stats['data_total']} ({rate:.1f}%)")
        if stats["edge_total"]:
            print(f"    ℹ️  Edge any          : {stats['edge_total']} câu")
        print(f"    🛡️  Blocked an toàn   : {stats['blocked']} câu")

    # ─── SO SÁNH ───
    if df_new is not None and df_old is not None:
        section("📈 SO SÁNH 500 CŨ vs 914 MỚI")
        vc_old = Counter(df_old["verdict"].astype(str))
        vc_new = Counter(df_new["verdict"].astype(str))
        total_old, total_new = len(df_old), len(df_new)
        print(f"\n  {'Verdict':12s} {'Cũ':>15s} {'Mới':>15s} {'Δ':>10s}")
        print(f"  {'─'*12} {'─'*15} {'─'*15} {'─'*10}")
        all_v = set(vc_old) | set(vc_new)
        for v in sorted(all_v):
            o = vc_old.get(v, 0)
            n = vc_new.get(v, 0)
            op = o / total_old * 100
            np_ = n / total_new * 100
            delta = np_ - op
            arrow = "↑" if delta > 1 else ("↓" if delta < -1 else "=")
            print(f"  {v:12s} {o:5,} ({op:5.1f}%) {n:5,} ({np_:5.1f}%) {arrow} {delta:+.1f}%")

    print("\n" + "=" * 78)
    print("✅ PHÂN TÍCH XONG. Không có file nào bị sửa.")
    print("=" * 78)


if __name__ == "__main__":
    main()
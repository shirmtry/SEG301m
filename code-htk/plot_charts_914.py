"""plot_charts_914.py — Sinh ảnh phân tích từ 914new.csv + 500results.csv.
   Đọc: test_results/*_914new.csv, test_results/*_results.csv
   Ghi: report_assets_914/*.png + report_914.txt
"""
import csv
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
OUTPUT_DIR = BASE / "test_results"
ASSET_DIR = BASE / "report_assets_914"
ASSET_DIR.mkdir(exist_ok=True)


def load_csv_robust(path):
    """Đọc CSV, fallback nhiều cách."""
    if not path.exists():
        print(f"⚠️  Không thấy: {path.name}")
        return None
    # Cách 1: pandas strict (CSV mới có QUOTE_ALL)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
        if len(df) > 0 and "verdict" in df.columns:
            return df
    except Exception:
        pass
    # Cách 2: csv module fallback
    try:
        rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("verdict"):
                    rows.append(r)
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        print(f"❌ Lỗi đọc {path.name}: {e}")
        return None


def find_latest(pattern):
    files = sorted(OUTPUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def plot_pie_verdict(df, label, out_path):
    vc = df["verdict"].value_counts()
    colors = {"answered": "#2ecc71", "refused": "#3498db",
              "warning": "#f39c12", "blocked": "#e74c3c", "error": "#95a5a6"}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
           colors=[colors.get(k, "#999") for k in vc.index],
           startangle=90, explode=[0.02] * len(vc))
    ax.set_title(f"Tỷ lệ phản hồi — {label} ({len(df)} câu)", fontsize=13)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_path.name}")


def plot_category_verdict(df, label, out_path):
    ct = pd.crosstab(df["category"], df["verdict"])
    colors = {"answered": "#2ecc71", "refused": "#3498db",
              "warning": "#f39c12", "blocked": "#e74c3c", "error": "#95a5a6"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", ax=ax,
            color=[colors.get(c, "#999") for c in ct.columns])
    ax.set_title(f"Kết quả theo loại câu — {label} ({len(df)} câu)", fontsize=13)
    ax.set_ylabel("Số câu")
    plt.xticks(rotation=0)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_path.name}")


def plot_time_hist(df, label, out_path):
    ts = pd.to_numeric(df["time_sec"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ts, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    mean = ts.mean()
    ax.axvline(mean, color="red", linestyle="--", label=f"Mean {mean:.1f}s")
    ax.set_title(f"Phân bố thời gian phản hồi — {label}", fontsize=13)
    ax.set_xlabel("Giây"); ax.set_ylabel("Số câu")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_path.name}")


def plot_reasons(df, label, out_path):
    rc = df["reason"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    rc.plot(kind="barh", ax=ax, color="#9b59b6")
    ax.set_title(f"Phân loại lý do — {label}", fontsize=13)
    ax.set_xlabel("Số câu")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_path.name}")


def plot_refusal_rate(df, label, out_path):
    df = df.copy()
    df["_refused_blocked"] = df["verdict"].isin(["refused", "blocked"])
    rate = df.groupby("category")["_refused_blocked"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    rate.plot(kind="bar", ax=ax, color="#e67e22")
    ax.set_title(f"Tỷ lệ từ chối/block (%) — {label}", fontsize=13)
    ax.set_ylabel("%")
    plt.xticks(rotation=0)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_path.name}")


def plot_expected_vs_verdict(df, label, out_path):
    ct = pd.crosstab(df["expected"], df["verdict"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", ax=ax)
    ax.set_title(f"Expected vs Verdict — {label}", fontsize=13)
    ax.set_ylabel("Số câu")
    plt.xticks(rotation=0)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_path.name}")


def plot_compare(df_new, df_old, out_path):
    """So sánh 500 cũ vs 914 mới."""
    vc_new = df_new["verdict"].value_counts(normalize=True) * 100
    vc_old = df_old["verdict"].value_counts(normalize=True) * 100
    all_v = sorted(set(vc_new.index) | set(vc_old.index))
    new_pct = [vc_new.get(v, 0) for v in all_v]
    old_pct = [vc_old.get(v, 0) for v in all_v]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(all_v))
    w = 0.35
    ax.bar([i - w/2 for i in x], old_pct, w, label="500 cũ", color="#95a5a6")
    ax.bar([i + w/2 for i in x], new_pct, w, label="914 mới", color="#2ecc71")
    ax.set_xticks(list(x)); ax.set_xticklabels(all_v)
    ax.set_ylabel("%")
    ax.set_title("So sánh 500 cũ vs 914 mới (theo % verdict)")
    for i, (o, n) in enumerate(zip(old_pct, new_pct)):
        ax.text(i - w/2, o + 0.5, f"{o:.1f}", ha="center", fontsize=9)
        ax.text(i + w/2, n + 0.5, f"{n:.1f}", ha="center", fontsize=9)
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ {out_path.name}")


def write_report(df_new, df_old, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"📊 BÁO CÁO PHÂN TÍCH 914 CÂU MỚI\n")
        f.write("=" * 70 + "\n\n")

        # Tổng quan 914
        f.write(f"📄 914 câu mới — {len(df_new)} câu\n")
        f.write("-" * 70 + "\n")
        for v, c in df_new["verdict"].value_counts().items():
            f.write(f"  {v:10s}: {c:5,} ({c/len(df_new)*100:5.1f}%)\n")
        f.write(f"\n  Thời gian TB: {pd.to_numeric(df_new['time_sec'], errors='coerce').mean():.1f}s\n")
        f.write(f"  Tổng thời gian: {pd.to_numeric(df_new['time_sec'], errors='coerce').sum()/3600:.1f}h\n\n")

        # Theo category
        f.write("Theo category:\n")
        for cat in df_new["category"].unique():
            sub = df_new[df_new["category"] == cat]
            f.write(f"\n  [{cat}] n={len(sub)}\n")
            for v, c in sub["verdict"].value_counts().items():
                f.write(f"    {v:10s}: {c:5,} ({c/len(sub)*100:5.1f}%)\n")

        # Reasons
        f.write("\nReasons:\n")
        for r, c in df_new["reason"].value_counts().items():
            f.write(f"  {r:25s}: {c:5,}\n")

        # So sánh
        if df_old is not None:
            f.write("\n" + "=" * 70 + "\n")
            f.write("📈 SO SÁNH 500 CŨ vs 914 MỚI\n")
            f.write("=" * 70 + "\n")
            vc_old = df_old["verdict"].value_counts(normalize=True) * 100
            vc_new = df_new["verdict"].value_counts(normalize=True) * 100
            all_v = sorted(set(vc_old.index) | set(vc_new.index))
            f.write(f"  {'Verdict':12s} {'Cũ':>10s} {'Mới':>10s} {'Δ':>10s}\n")
            for v in all_v:
                o = vc_old.get(v, 0)
                n = vc_new.get(v, 0)
                delta = n - o
                arrow = "↑" if delta > 1 else ("↓" if delta < -1 else "=")
                f.write(f"  {v:12s} {o:9.1f}% {n:9.1f}% {arrow}{delta:+6.1f}%\n")

        # Traps fail
        f.write("\n" + "=" * 70 + "\n")
        fail_traps = df_new[(df_new["expected"] == "refuse") &
                            (~df_new["verdict"].isin(["refused", "blocked"]))]
        f.write(f"⚠️ TRAPS FAIL: {len(fail_traps)} câu\n")
        f.write("=" * 70 + "\n")
        for i, (_, r) in enumerate(fail_traps.head(30).iterrows(), 1):
            q = str(r["question"])[:80]
            a = str(r["answer"])[:140].replace("\n", " ")
            v = r["verdict"]
            f.write(f"\n{i}. [{v}] Q: {q}\n")
            f.write(f"     A: {a}\n")

        # Data miss
        f.write("\n" + "=" * 70 + "\n")
        miss = df_new[(df_new["expected"] == "answer") &
                      (df_new["verdict"] == "refused")]
        f.write(f"⚠️ DATA MISS (lẽ ra answer, bị refused): {len(miss)} câu\n")
        f.write("=" * 70 + "\n")
        for i, (_, r) in enumerate(miss.head(30).iterrows(), 1):
            q = str(r["question"])[:80]
            f.write(f"{i}. {q}\n")

    print(f"✅ {out_path.name}")


def main():
    print("=" * 70)
    print("📊 SINH ẢNH PHÂN TÍCH")
    print("=" * 70)

    new_csv = find_latest("*_914new.csv")
    old_csv = find_latest("*_results.csv")

    print(f"\n📁 914 mới: {new_csv.name if new_csv else 'KHÔNG CÓ'}")
    print(f"📁 500 cũ : {old_csv.name if old_csv else 'KHÔNG CÓ'}\n")

    df_new = load_csv_robust(new_csv) if new_csv else None
    df_old = load_csv_robust(old_csv) if old_csv else None

    if df_new is None:
        print("❌ Không có dữ liệu 914 mới. Dừng.")
        return

    print(f"\n✅ Đọc {len(df_new)} câu từ {new_csv.name}")
    if df_old is not None:
        print(f"✅ Đọc {len(df_old)} câu từ {old_csv.name}")

    print("\n🎨 Đang sinh biểu đồ...\n")

    plot_pie_verdict(df_new, "914 câu mới", ASSET_DIR / "01_pie_verdict_914.png")
    plot_category_verdict(df_new, "914 câu mới", ASSET_DIR / "02_category_914.png")
    plot_time_hist(df_new, "914 câu mới", ASSET_DIR / "03_time_hist_914.png")
    plot_reasons(df_new, "914 câu mới", ASSET_DIR / "04_reasons_914.png")
    plot_refusal_rate(df_new, "914 câu mới", ASSET_DIR / "05_refusal_rate_914.png")
    plot_expected_vs_verdict(df_new, "914 câu mới", ASSET_DIR / "06_expected_vs_verdict_914.png")

    if df_old is not None:
        plot_compare(df_new, df_old, ASSET_DIR / "07_compare_500_vs_914.png")

    write_report(df_new, df_old, BASE / "report_914.txt")

    print(f"\n{'='*70}")
    print(f"✅ HOÀN THÀNH. Ảnh + report trong: {ASSET_DIR}")
    print(f"📁 Report: {BASE / 'report_914.txt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
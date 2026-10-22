"""check_laws.py — Kiểm tra luật hiện có và đối chiếu luật cần thêm."""
import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat")
MERGED_FILE = BASE_DIR / "legal_merged.jsonl"

# Danh sách luật KỲ VỌNG (theo LAW_KEYWORDS v6 + bổ sung đầy đủ)
EXPECTED = {
    # Hình sự
    "Bộ luật Hình sự":                       [r"bộ luật hình sự", r"blhs"],
    "Bộ luật Tố tụng hình sự":               [r"tố tụng hình sự", r"bltths"],
    "Luật Thi hành án hình sự":              [r"thi hành án hình sự"],
    # Dân sự
    "Bộ luật Dân sự":                        [r"bộ luật dân sự", r"blds"],
    "Bộ luật Tố tụng dân sự":                [r"tố tụng dân sự", r"blttds"],
    "Luật Hôn nhân và Gia đình":             [r"hôn nhân.*gia đình"],
    "Luật Thi hành án dân sự":               [r"thi hành án dân sự"],
    # Lao động - BH
    "Bộ luật Lao động":                      [r"bộ luật lao động", r"bllđ"],
    "Luật Bảo hiểm xã hội":                  [r"bảo hiểm xã hội"],
    "Luật Bảo hiểm y tế":                    [r"bảo hiểm y tế"],
    "Luật An toàn, vệ sinh lao động":        [r"an toàn.*vệ sinh lao động"],
    # Đất đai - Nhà ở - Xây dựng
    "Luật Đất đai":                          [r"luật đất đai"],
    "Luật Nhà ở":                            [r"luật nhà ở"],
    "Luật Kinh doanh bất động sản":          [r"kinh doanh bất động sản"],
    "Luật Xây dựng":                         [r"luật xây dựng"],
    # Hành chính - Cư trú
    "Luật Xử lý vi phạm hành chính":         [r"xử lý vi phạm hành chính", r"xlvphc"],
    "Luật Tố tụng hành chính":               [r"tố tụng hành chính"],
    "Luật Cán bộ, công chức":                [r"cán bộ.*công chức"],
    "Luật Phòng, chống tham nhũng":          [r"tham nhũng"],
    "Luật Khiếu nại":                        [r"luật khiếu nại"],
    "Luật Tố cáo":                           [r"luật tố cáo"],
    "Luật Cư trú":                           [r"luật cư trú"],
    "Luật Căn cước":                         [r"luật căn cước"],
    "Luật Hộ tịch":                          [r"luật hộ tịch"],
    "Luật Quốc tịch":                        [r"luật quốc tịch"],
    "Luật Xuất cảnh, nhập cảnh":             [r"xuất cảnh.*nhập cảnh"],
    # Kinh doanh - Tài chính
    "Luật Doanh nghiệp":                     [r"luật doanh nghiệp"],
    "Luật Đầu tư":                           [r"luật đầu tư"],
    "Luật Đấu thầu":                         [r"luật đấu thầu"],
    "Luật Cạnh tranh":                       [r"luật cạnh tranh"],
    "Luật Phá sản":                          [r"luật phá sản"],
    "Luật Thương mại":                       [r"luật thương mại"],
    "Luật Trọng tài thương mại":             [r"trọng tài thương mại"],
    "Luật Bảo vệ quyền lợi người tiêu dùng": [r"người tiêu dùng"],
    "Luật Quản lý thuế":                     [r"quản lý thuế"],
    "Luật Thuế TNCN":                        [r"thu nhập cá nhân"],
    "Luật Thuế GTGT":                        [r"giá trị gia tăng"],
    "Luật Thuế TNDN":                        [r"thu nhập doanh nghiệp"],
    "Luật Hải quan":                         [r"luật hải quan"],
    "Luật Chứng khoán":                      [r"luật chứng khoán"],
    "Luật Các tổ chức tín dụng":             [r"tổ chức tín dụng"],
    # SHTT - CNTT
    "Luật Sở hữu trí tuệ":                   [r"sở hữu trí tuệ"],
    "Luật Công nghệ thông tin":              [r"công nghệ thông tin"],
    "Luật An ninh mạng":                     [r"an ninh mạng"],
    "Luật An toàn thông tin mạng":           [r"an toàn thông tin"],
    "Luật Giao dịch điện tử":                [r"giao dịch điện tử"],
    # Giao thông
    "Luật Trật tự, ATGT đường bộ":           [r"trật tự.*an toàn giao thông"],
    "Luật Đường bộ":                         [r"luật đường bộ"],
    "Luật Đường sắt":                        [r"luật đường sắt"],
    "Luật Hàng không dân dụng":              [r"hàng không"],
    "Luật Hàng hải":                         [r"luật hàng hải"],
    # Xã hội
    "Luật Giáo dục":                         [r"luật giáo dục"],
    "Luật Giáo dục đại học":                 [r"giáo dục đại học"],
    "Luật Trẻ em":                           [r"luật trẻ em"],
    "Luật Người khuyết tật":                 [r"người khuyết tật"],
    "Luật Bình đẳng giới":                   [r"bình đẳng giới"],
    "Luật Phòng, chống bạo lực gia đình":    [r"bạo lực gia đình"],
    "Luật Nghĩa vụ quân sự":                 [r"nghĩa vụ quân sự"],
    "Luật Khám bệnh, chữa bệnh":             [r"khám bệnh.*chữa bệnh"],
    "Luật Dược":                             [r"luật dược"],
    "Luật An toàn thực phẩm":                [r"an toàn thực phẩm"],
    "Luật Bảo vệ môi trường":                [r"bảo vệ môi trường"],
    # Tư pháp
    "Luật Công chứng":                       [r"luật công chứng"],
    "Luật Luật sư":                          [r"luật luật sư"],
    "Luật Trợ giúp pháp lý":                 [r"trợ giúp pháp lý"],
    "Luật Đấu giá tài sản":                  [r"đấu giá tài sản"],
    "Luật Nuôi con nuôi":                    [r"nuôi con nuôi"],
    # Nền tảng
    "Hiến pháp":                             [r"hiến pháp"],
    "Luật Ban hành VBQPPL":                  [r"ban hành văn bản"],
}

def main():
    laws = Counter()
    total = 0

    if not MERGED_FILE.exists():
        print(f"❌ Không tìm thấy: {MERGED_FILE}")
        return

    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                total += 1
                title = data.get("metadata", {}).get("title", "")
                if " - Điều" in title:
                    law = title.split(" - Điều")[0].strip()
                    laws[law] += 1
            except json.JSONDecodeError:
                continue

    print("=" * 70)
    print(f"📊 Tổng records     : {total}")
    print(f"📚 Số luật hiện có   : {len(laws)}")
    print(f"📝 Tổng số điều     : {sum(laws.values())}")
    print("=" * 70)
    print("\n📖 DANH SÁCH LUẬT HIỆN CÓ (sắp xếp theo số điều):\n")
    for law, count in sorted(laws.items(), key=lambda x: -x[1]):
        print(f"  {count:5d} điều | {law}")

    # Đối chiếu
    print("\n" + "=" * 70)
    print("🔍 ĐỐI CHIẾU VỚI DANH SÁCH KỲ VỌNG")
    print("=" * 70)
    missing = []
    for exp, pats in EXPECTED.items():
        found = any(re.search(p, law, re.I) for law in laws for p in pats)
        if not found:
            missing.append(exp)

    if missing:
        print(f"\n⚠️  THIẾU {len(missing)} LUẬT:\n")
        for m in missing:
            print(f"  ✗ {m}")
    else:
        print("\n✅ Đã có đủ toàn bộ luật trong danh sách kỳ vọng.")

    cov = (len(EXPECTED) - len(missing)) / len(EXPECTED) * 100
    print(f"\n📈 Độ phủ: {cov:.1f}%  ({len(EXPECTED)-len(missing)}/{len(EXPECTED)})")

    # Cảnh báo luật có ít điều (có thể chưa đủ dữ liệu)
    print("\n" + "=" * 70)
    print("⚠️  LUẬT CÓ < 30 ĐIỀU (nên crawl bổ sung):")
    print("=" * 70)
    for law, count in sorted(laws.items(), key=lambda x: x[1]):
        if count < 30:
            print(f"  {count:5d} điều | {law}")

if __name__ == "__main__":
    main()
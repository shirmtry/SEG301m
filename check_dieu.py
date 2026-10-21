import json

FILE_PATH = r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\legal_merged.jsonl"

print("🔍 Đang tìm kiếm Điều 123 và Điều 134 trong Bộ luật Hình sự...")

found_123 = False
found_134 = False

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        title = data.get("metadata", {}).get("title", "")
        
        # Kiểm tra xem có chữ "Bộ luật Hình sự" và số điều tương ứng không
        if "Bộ luật Hình sự" in title:
            if "Điều 123." in title or "Điều 123 " in title:
                print(f"✅ ĐÃ TÌM THẤY: {title}")
                found_123 = True
            if "Điều 134." in title or "Điều 134 " in title:
                print(f"✅ ĐÃ TÌM THẤY: {title}")
                found_134 = True

if not found_123:
    print("❌ KHÔNG TÌM THẤY Điều 123 của Bộ luật Hình sự trong file!")
if not found_134:
    print("❌ KHÔNG TÌM THẤY Điều 134 của Bộ luật Hình sự trong file!")
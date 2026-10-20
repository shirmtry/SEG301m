# -*- coding: utf-8 -*-
"""
XỬ LÝ PDF LUẬT VN -> JSONL (TỰ ĐỘNG OCR)
- PDF có text layer  -> trích text trực tiếp
- PDF scan (ảnh)     -> tự động OCR bằng Tesseract

Schema output: {"text", "law_type", "source", "page"}

Chạy: py cook_all.py
"""
import os
import re
import json
import time
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pymupdf  # PyMuPDF
from tqdm import tqdm


# =========================================================
# CẤU HÌNH  (ĐÃ ĐIỀN SẴN ĐƯỜNG DẪN CỦA BẠN)
# =========================================================
POPPLER_PATH = r"C:\Users\dinht\Downloads\Release-26.09.0-0\poppler-26.09.0\Library\bin"
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_LANG = "vie"
OCR_DPI = 300
MIN_TEXT_CHARS_PER_PAGE = 50

# Thư mục + file output mặc định (có thể đổi qua CLI)
DEFAULT_INPUT = r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\data"
DEFAULT_OUTPUT = r"C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat\data_luat.jsonl"

# Map tên file -> loại luật (sửa cho khớp file của bạn)
FILE_TO_LAW = {
    "hinhsu.pdf": "Hình sự",
    "dansu.pdf": "Dân sự",
    "hanhchinh.pdf": "Hành chính",
    "laodong.pdf": "Lao động",
    "honnhangiadinh.pdf": "Hôn nhân gia đình",
    "totungdansu.pdf": "Tố tụng dân sự",
    "totungdansu1.pdf": "Tố tụng dân sự",
    "totungdansu2.pdf": "Tố tụng dân sự",
    "kinhte.pdf": "Kinh tế",
    "taichinh.pdf": "Tài chính",
}


def _setup_tesseract():
    try:
        import pytesseract
        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        return pytesseract
    except ImportError:
        return None


# =========================================================
# 1. TRÍCH XUẤT TEXT (TỰ ĐỘNG TEXT LAYER / OCR)
# =========================================================
def extract_text(pdf_path: str):
    """
    Trả về (text, method)
    method = "text_layer" hoặc "ocr"
    """
    doc = pymupdf.open(pdf_path)
    n_pages = len(doc)

    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()

    total_chars = sum(len(t.strip()) for t in pages_text)
    avg_chars = total_chars / max(n_pages, 1)

    if avg_chars >= MIN_TEXT_CHARS_PER_PAGE:
        return "\n".join(pages_text), "text_layer"

    ocr_text = _ocr_pdf(pdf_path)
    return ocr_text, "ocr"


def _ocr_pdf(pdf_path: str) -> str:
    from pdf2image import convert_from_path

    pytesseract = _setup_tesseract()
    if pytesseract is None:
        raise RuntimeError("Chưa cài pytesseract. Chạy: pip install pytesseract pdf2image")

    if not os.path.exists(POPPLER_PATH):
        raise RuntimeError(f"Không tìm thấy Poppler tại: {POPPLER_PATH}")

    images = convert_from_path(pdf_path, dpi=OCR_DPI, poppler_path=POPPLER_PATH)

    parts = []
    for img in images:
        try:
            txt = pytesseract.image_to_string(img, lang=TESS_LANG)
        except Exception:
            txt = ""
        parts.append(txt)
    return "\n".join(parts)


# =========================================================
# 2. LÀM SẠCH TEXT (giữ nguyên logic file gốc của bạn)
# =========================================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(
        r'[^\w\s\.\,\;\:\-\(\)\/\'\"àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị'
        r'òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]',
        ' ', text
    )
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================================================
# 3. CHIA CHUNK (giữ nguyên logic file gốc)
# =========================================================
def split_into_chunks(text, min_length=100, max_length=1500):
    if not text:
        return []

    dieu_pattern = re.compile(r'(?=Điều\s+\d+[\.\:])', re.IGNORECASE)
    parts = dieu_pattern.split(text)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_length:
            sentences = re.split(r'(?<=[\.\;\:])\s+', part)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) < max_length:
                    current += " " + sent
                else:
                    if len(current.strip()) >= min_length:
                        chunks.append(current.strip())
                    current = sent
            if len(current.strip()) >= min_length:
                chunks.append(current.strip())
        else:
            if len(part) >= min_length:
                chunks.append(part)

    return chunks


# =========================================================
# 4. XỬ LÝ 1 FILE PDF
# =========================================================
def process_one_pdf(args):
    """
    Trả về (pdf_path, chunks_list, method, error)
    Mỗi chunk: {"text", "law_type", "source", "page"}
    """
    pdf_path, law_type = args
    try:
        raw, method = extract_text(pdf_path)
        if not raw.strip():
            return pdf_path, [], method, "PDF rỗng sau xử lý (kể cả OCR)"

        # Chunk toàn bộ văn bản (không tracking page riêng vì OCR/text layer
        # thường gộp trang; page sẽ để 0 hoặc có thể chỉnh nếu cần)
        cleaned = clean_text(raw)
        chunks = split_into_chunks(cleaned)

        rows = [
            {
                "text": ch,
                "law_type": law_type,
                "source": os.path.basename(pdf_path),
                "page": 0,
            }
            for ch in chunks
        ]
        return pdf_path, rows, method, None
    except Exception as e:
        return pdf_path, [], "error", str(e)


# =========================================================
# 5. MAIN
# =========================================================
def main(input_dir, output_file, workers=4):
    pdf_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True))
    if not pdf_files:
        print(f"❌ Không tìm thấy PDF trong: {input_dir}")
        return

    print(f"📚 Tìm thấy {len(pdf_files)} file PDF trong: {input_dir}")
    print(f"⚙️  Workers: {workers} | OCR: BẬT")
    print(f"🔧 Poppler  : {POPPLER_PATH}")
    print(f"🔧 Tesseract: {TESSERACT_CMD} (lang={TESS_LANG})")
    print("─" * 60)

    tasks = [(p, FILE_TO_LAW.get(os.path.basename(p), Path(p).stem)) for p in pdf_files]

    total_chunks = 0
    stats = {"text_layer": 0, "ocr": 0, "error": 0}
    failed = []
    t0 = time.time()

    with open(output_file, "w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one_pdf, t): t for t in tasks}
            pbar = tqdm(total=len(futures), desc="📄 Cook", unit="file")

            for fut in as_completed(futures):
                pdf_path, rows, method, err = fut.result()
                name = os.path.basename(pdf_path)

                if err:
                    stats["error"] += 1
                    failed.append((name, err))
                    pbar.set_postfix_str(f"❌ {name[:30]}")
                else:
                    stats[method] = stats.get(method, 0) + 1
                    for row in rows:
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_chunks += len(rows)
                    icon = "🔍" if method == "ocr" else "✅"
                    pbar.set_postfix_str(f"{icon} {name[:30]} ({len(rows)})")

                pbar.update(1)
            pbar.close()

    dt = time.time() - t0

    print("\n" + "=" * 60)
    print(f"🎉 HOÀN TẤT trong {dt:.1f}s")
    print(f"📄 Tổng file      : {len(pdf_files)}")
    print(f"✅ Text layer     : {stats.get('text_layer', 0)}")
    print(f"🔍 Phải OCR       : {stats.get('ocr', 0)}")
    print(f"❌ Lỗi            : {stats.get('error', 0)}")
    print(f"📦 Tổng chunks    : {total_chunks}")
    print(f"💾 File JSONL     : {os.path.abspath(output_file)}")
    print("=" * 60)

    if failed:
        print("\n⚠️  FILE LỖI:")
        for name, err in failed[:20]:
            print(f"   - {name}: {err[:100]}")


# =========================================================
# 6. CLI
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Folder PDF")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="File JSONL")
    ap.add_argument("-w", "--workers", type=int, default=4)
    args = ap.parse_args()

    main(args.input, args.output, workers=args.workers)
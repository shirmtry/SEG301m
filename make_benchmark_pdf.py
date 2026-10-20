# -*- coding: utf-8 -*-
"""
Đọc benchmark_results.json → xuất PDF báo cáo:
  - Trang tổng quan + biểu đồ ASCII
  - Danh sách CÂU TRẢ LỜI ĐƯỢC (có trích đoạn)
  - Danh sách CÂU KHÔNG TRẢ LỜI ĐƯỢC (kèm trích đoạn)
  - Phụ lục: bảng xếp hạng khó → dễ
"""

import json
import os
import sys
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)


# ============================================================
# 1. ĐĂNG KÝ FONT UNICODE (để hiển thị tiếng Việt)
# ============================================================
FONT_CANDIDATES = [
    # Windows
    (r'C:\Windows\Fonts\arial.ttf',       r'C:\Windows\Fonts\arialbd.ttf',   'Arial'),
    (r'C:\Windows\Fonts\segoeui.ttf',     r'C:\Windows\Fonts\segoeuib.ttf',  'SegoeUI'),
    (r'C:\Windows\Fonts\times.ttf',       r'C:\Windows\Fonts\timesbd.ttf',   'Times'),
    # macOS
    ('/Library/Fonts/Arial.ttf',          '/Library/Fonts/Arial Bold.ttf',   'Arial'),
    ('/System/Library/Fonts/Supplemental/Arial.ttf',
     '/System/Library/Fonts/Supplemental/Arial Bold.ttf',                   'Arial'),
    # Linux
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
     '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',                'DejaVu'),
    ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
     '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',        'Liberation'),
]

FONT_NAME = 'Helvetica'      # fallback
FONT_BOLD = 'Helvetica-Bold'


def register_font():
    global FONT_NAME, FONT_BOLD
    for regular, bold, name in FONT_CANDIDATES:
        if os.path.exists(regular):
            try:
                pdfmetrics.registerFont(TTFont(name, regular))
                pdfmetrics.registerFont(TTFont(
                    name + '-Bold',
                    bold if os.path.exists(bold) else regular,
                ))
                FONT_NAME = name
                FONT_BOLD = name + '-Bold'
                print(f'✅ Dùng font: {name} ({regular})')
                return
            except Exception as e:
                print(f'⚠️ Không đăng ký được {name}: {e}')
    print('⚠️ Không tìm thấy font Unicode — PDF có thể hiển thị lỗi tiếng Việt.')


register_font()


# ============================================================
# 2. STYLE
# ============================================================
styles = getSampleStyleSheet()


def make_style(name, **kwargs):
    """
    FIX: cho phép caller override fontName mà không bị trùng keyword.
    Trước đây hard-code fontName=FONT_NAME nên khi gọi kèm fontName=... sẽ lỗi.
    """
    base = kwargs.pop('parent', styles['Normal'])
    kwargs.setdefault('fontName', FONT_NAME)   # chỉ set nếu caller chưa truyền
    return ParagraphStyle(name, parent=base, **kwargs)


S_TITLE    = make_style('Title2',   fontSize=22, leading=28, fontName=FONT_BOLD,
                        textColor=colors.HexColor('#0B3D91'), alignment=1,
                        spaceAfter=6)
S_SUBTITLE = make_style('Subtitle', fontSize=12, leading=16,
                        textColor=colors.HexColor('#555555'), alignment=1,
                        spaceAfter=20)
S_H1       = make_style('H1',       fontSize=16, leading=22, fontName=FONT_BOLD,
                        textColor=colors.HexColor('#0B3D91'),
                        spaceBefore=14, spaceAfter=8)
S_H2       = make_style('H2',       fontSize=13, leading=18, fontName=FONT_BOLD,
                        textColor=colors.HexColor('#1F4E79'),
                        spaceBefore=10, spaceAfter=6)
S_BODY     = make_style('Body',     fontSize=10, leading=14)
S_SMALL    = make_style('Small',    fontSize=9,  leading=12,
                        textColor=colors.HexColor('#666666'))
S_Q        = make_style('Q',        fontSize=11, leading=15, fontName=FONT_BOLD,
                        textColor=colors.HexColor('#111111'),
                        spaceBefore=6, spaceAfter=2)
S_A        = make_style('A',        fontSize=9.5, leading=13,
                        textColor=colors.HexColor('#333333'),
                        leftIndent=12)
S_REF      = make_style('Ref',      fontSize=8.5, leading=11,
                        textColor=colors.HexColor('#888888'),
                        leftIndent=12)


def esc(t):
    """Escape XML cho reportlab Paragraph."""
    if t is None:
        return ''
    return (str(t).replace('&', '&amp;')
                  .replace('<', '&lt;')
                  .replace('>', '&gt;'))


def shorten(text, n=500):
    text = (text or '').strip().replace('\n', ' ')
    return text if len(text) <= n else text[:n].rstrip() + '…'


# ============================================================
# 3. ĐỌC JSON
# ============================================================
def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 4. TẠO PDF
# ============================================================
def _add_page_number(canvas, doc):
    """Vẽ số trang + footer ở mỗi trang."""
    canvas.saveState()
    canvas.setFont(FONT_NAME, 8)
    canvas.setFillColor(colors.HexColor('#888888'))
    canvas.drawString(1.6 * cm, 1.0 * cm,
                      'Báo cáo Benchmark Chatbot Luật Việt Nam')
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm,
                           f'Trang {doc.page}')
    canvas.setStrokeColor(colors.HexColor('#DDDDDD'))
    canvas.setLineWidth(0.3)
    canvas.line(1.6 * cm, 1.3 * cm, A4[0] - 1.6 * cm, 1.3 * cm)
    canvas.restoreState()


def build_pdf(data, out_path):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm,
        topMargin=1.6 * cm,  bottomMargin=1.8 * cm,
        title='Báo cáo Benchmark Chatbot Luật Việt Nam',
        author='Benchmark tool',
    )

    story = []

    # ----- Trang bìa -----
    story.append(Spacer(1, 2 * cm))
    story.append(Paragraph('BÁO CÁO BENCHMARK', S_TITLE))
    story.append(Paragraph('HỆ THỐNG HỎI ĐÁP PHÁP LUẬT VIỆT NAM', S_TITLE))
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width='80%', thickness=2,
                            color=colors.HexColor('#0B3D91'),
                            hAlign='CENTER'))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        f'Ngày tạo: {datetime.now().strftime("%d/%m/%Y %H:%M")}', S_SUBTITLE))

    total = data.get('total', 0) or 0
    ans = data.get('answerable_count', 0) or 0
    unans = data.get('unanswerable_count', 0) or 0
    thr = data.get('threshold_p25', 0) or 0

    summary_table = Table([
        ['Chỉ số', 'Giá trị'],
        ['Tổng số câu hỏi', f'{total}'],
        ['✅ Trả lời được',
         f'{ans}  ({ans * 100 / total:.1f}%)' if total else '0'],
        ['❌ Không trả lời được',
         f'{unans}  ({unans * 100 / total:.1f}%)' if total else '0'],
        ['Ngưỡng BM25 (P25)', f'{thr:.2f}'],
    ], colWidths=[7 * cm, 7 * cm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0B3D91')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), FONT_BOLD),
        ('FONTNAME',   (0, 1), (-1, -1), FONT_NAME),
        ('FONTSIZE',   (0, 0), (-1, -1), 11),
        ('ALIGN',      (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
            [colors.white, colors.HexColor('#F0F6FF')]),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        'Tài liệu liệt kê toàn bộ câu hỏi trong bộ benchmark, phân loại rõ '
        'câu trả lời được và câu không trả lời được, kèm trích đoạn câu trả lời '
        'của hệ thống để tiện đối chiếu và đánh giá thủ công.', S_SMALL))

    story.append(PageBreak())

    # ----- Mục 1: Tóm tắt -----
    story.append(Paragraph('1. TÓM TẮT ĐIỀU HÀNH', S_H1))
    if total:
        summary_para = (
            f'Bộ benchmark gồm <b>{total}</b> câu hỏi pháp luật tiếng Việt, '
            f'trong đó hệ thống trả lời được <b>{ans}</b> câu '
            f'(chiếm {ans * 100 / total:.1f}%) và không trả lời được '
            f'<b>{unans}</b> câu (chiếm {unans * 100 / total:.1f}%). '
            f'Ngưỡng BM25 top-1 dùng để phân loại là <b>{thr:.2f}</b> '
            f'(phân vị 25).'
        )
    else:
        summary_para = 'Không có dữ liệu.'
    story.append(Paragraph(summary_para, S_BODY))
    story.append(Spacer(1, 0.3 * cm))

    # Biểu đồ ASCII đơn giản
    bar_len = 30                                   # đổi 40 → 30 cho an toàn cột
    if total:
        ans_bar   = '█' * max(1, int(round(bar_len * ans / total)))
        unans_bar = '█' * max(1, int(round(bar_len * unans / total)))
    else:
        ans_bar = unans_bar = ''

    bar_table = Table([
        ['Nhóm', 'Số câu', 'Tỷ lệ', 'Biểu đồ'],
        ['✅ Trả lời được', str(ans),
         f'{ans * 100 / total:.1f}%' if total else '—',
         Paragraph(f'<font color="#2E7D32">{ans_bar}</font>', S_SMALL)],
        ['❌ Không trả lời được', str(unans),
         f'{unans * 100 / total:.1f}%' if total else '—',
         Paragraph(f'<font color="#C62828">{unans_bar}</font>', S_SMALL)],
    ], colWidths=[4.5 * cm, 2 * cm, 2 * cm, 7.5 * cm])
    bar_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F4E79')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), FONT_BOLD),
        ('FONTNAME',   (0, 1), (-1, -1), FONT_NAME),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('ALIGN',      (1, 0), (2, -1), 'CENTER'),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(bar_table)

    story.append(PageBreak())

    # ----- Mục 2: Câu trả lời được -----
    answerable = data.get('answerable_list', []) or []
    story.append(Paragraph(
        f'2. DANH SÁCH {len(answerable)} CÂU TRẢ LỜI ĐƯỢC', S_H1))
    story.append(Paragraph(
        'Sắp xếp theo điểm BM25 giảm dần (câu càng cao càng dễ).', S_SMALL))
    story.append(Spacer(1, 0.3 * cm))

    for i, r in enumerate(sorted(answerable,
                                 key=lambda x: x.get('bm25_top_score', 0),
                                 reverse=True), 1):
        _add_question_block(story, i, r, color='#2E7D32')

    story.append(PageBreak())

    # ----- Mục 3: Câu KHÔNG trả lời được -----
    unans_list = data.get('unanswerable_list', []) or []
    story.append(Paragraph(
        f'3. DANH SÁCH {len(unans_list)} CÂU KHÔNG TRẢ LỜI ĐƯỢC', S_H1))
    story.append(Paragraph(
        'Sắp xếp theo điểm BM25 tăng dần (câu càng thấp càng khó).', S_SMALL))
    story.append(Spacer(1, 0.3 * cm))

    for i, r in enumerate(sorted(unans_list,
                                 key=lambda x: x.get('bm25_top_score', 0)), 1):
        _add_question_block(story, i, r, color='#C62828')

    # ----- Mục 4: Bảng xếp hạng khó → dễ -----
    story.append(PageBreak())
    story.append(Paragraph('4. BẢNG XẾP HẠNG KHÓ → DỄ (TOÀN BỘ)', S_H1))
    story.append(Paragraph(
        'Bảng dưới đây liệt kê toàn bộ câu hỏi theo thứ tự khó nhất đến '
        'dễ nhất (tính theo điểm BM25 tăng dần).', S_SMALL))
    story.append(Spacer(1, 0.3 * cm))

    ranked = data.get('ranked_hard_to_easy', []) or []
    rows = [['#', 'ID', 'Câu hỏi', 'BM25', 'Kết quả']]
    for idx, r in enumerate(ranked, 1):
        rows.append([
            str(idx),
            str(r.get('id', '')),
            Paragraph(esc(shorten(r.get('question', ''), 90)), S_SMALL),
            f'{r.get("bm25_top_score", 0):.2f}',
            '✅' if r.get('label') == 'answerable' else '❌',
        ])

    rank_table = Table(rows,
                       colWidths=[1 * cm, 1.2 * cm, 11 * cm, 1.8 * cm, 1.5 * cm],
                       repeatRows=1)
    rank_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0B3D91')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), FONT_BOLD),
        ('FONTNAME',   (1, 1), (1, -1), FONT_NAME),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ALIGN',      (0, 0), (1, -1), 'CENTER'),
        ('ALIGN',      (3, 0), (4, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID',       (0, 0), (-1, -1), 0.25, colors.HexColor('#DDDDDD')),
    ]
    # Zebra + tô màu theo nhãn (an toàn khi ranked rỗng)
    if ranked:
        for i, r in enumerate(ranked, 1):
            if r.get('label') == 'answerable':
                rank_style.append(('TEXTCOLOR', (4, i), (4, i),
                                   colors.HexColor('#2E7D32')))
                rank_style.append(('BACKGROUND', (0, i), (-1, i),
                                   colors.HexColor('#F1F8E9')))
            else:
                rank_style.append(('TEXTCOLOR', (4, i), (4, i),
                                   colors.HexColor('#C62828')))
                rank_style.append(('BACKGROUND', (0, i), (-1, i),
                                   colors.HexColor('#FFEBEE')))

    rank_table.setStyle(TableStyle(rank_style))
    story.append(rank_table)

    # ----- Build (thêm footer số trang) -----
    doc.build(story, onFirstPage=_add_page_number,
              onLaterPages=_add_page_number)
    print(f'✅ Đã tạo PDF: {out_path}')


def _add_question_block(story, index, r, color):
    """Thêm block: câu hỏi + metadata + trích đoạn câu trả lời."""
    q = esc(r.get('question', ''))
    ans_text = esc(shorten(r.get('answer', ''), 550))

    tag_txt = ('TRẢ LỜI ĐƯỢC' if r.get('label') == 'answerable'
               else 'KHÔNG TRẢ LỜI ĐƯỢC')
    meta = (f'ID {r.get("id", "?")} &nbsp;|&nbsp; '
            f'BM25: {r.get("bm25_top_score", 0):.2f} &nbsp;|&nbsp; '
            f'Thời gian: {r.get("time_s", 0):.1f}s &nbsp;|&nbsp; '
            f'<b><font color="{color}">{tag_txt}</font></b>')

    story.append(Paragraph(f'{index}. {q}', S_Q))
    story.append(Paragraph(meta, S_SMALL))
    story.append(Paragraph(f'→ {ans_text}', S_A))
    story.append(HRFlowable(width='100%', thickness=0.3,
                            color=colors.HexColor('#EEEEEE'),
                            spaceBefore=6, spaceAfter=6))


# ============================================================
# 5. MAIN
# ============================================================
def main():
    here = os.path.abspath(os.path.dirname(__file__))
    json_path = os.path.join(here, 'benchmark_results.json')
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f'❌ Không tìm thấy: {json_path}')
        sys.exit(1)

    print(f'📖 Đọc dữ liệu: {json_path}')
    data = load_results(json_path)

    out_path = os.path.join(os.path.dirname(json_path),
                            'bao_cao_benchmark.pdf')
    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    build_pdf(data, out_path)


if __name__ == '__main__':
    main()
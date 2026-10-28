import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def export_to_pdf(data_records: list, output_filepath: str):
    """Động cơ vẽ PDF tĩnh chuẩn A4 hỗ trợ Tiếng Việt"""
    # 1. Đăng ký Font tiếng Việt: Times New Roman của Windows nếu có, nếu không dùng DejaVu Serif đi kèm repo
    font_path = r"C:\Windows\Fonts\times.ttf"
    bundled_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
    font_name = "Helvetica" # Font dự phòng cuối cùng (không có dấu tiếng Việt)
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('TimesNewRoman', font_path))
        font_name = 'TimesNewRoman'
    elif os.path.exists(os.path.join(bundled_dir, "DejaVuSerif.ttf")):
        pdfmetrics.registerFont(TTFont('DejaVuSerif', os.path.join(bundled_dir, "DejaVuSerif.ttf")))
        pdfmetrics.registerFont(TTFont('DejaVuSerif-Bold', os.path.join(bundled_dir, "DejaVuSerif-Bold.ttf")))
        pdfmetrics.registerFontFamily('DejaVuSerif', normal='DejaVuSerif', bold='DejaVuSerif-Bold', italic='DejaVuSerif', boldItalic='DejaVuSerif-Bold')
        font_name = 'DejaVuSerif'

    # 2. Tạo khung vẽ PDF (Khổ A4 nằm ngang cho bảng nhiều cột)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    pdf = SimpleDocTemplate(output_filepath, pagesize=landscape(A4))
    
    # 3. Chuẩn bị định dạng CSS cho Text
    styles = getSampleStyleSheet()
    style_header = ParagraphStyle(name='Header', fontName=font_name, fontSize=11, leading=14, alignment=1)
    style_cell = ParagraphStyle(name='Cell', fontName=font_name, fontSize=10, leading=13)
    style_cell_left = ParagraphStyle(name='CellLeft', fontName=font_name, fontSize=10, leading=13, alignment=0)
    
    # 4. Đổ dữ liệu vào mảng (Dùng Paragraph để tự động xuống dòng khi text dài)
    headers = ["Số TT", "Số, ký hiệu tài liệu", "Ngày tháng", "Tên loại hoặc trích yếu", "Tác giả tài liệu", "Tờ số"]
    table_data = [[Paragraph(f"<b>{h}</b>", style_header) for h in headers]]
    
    for idx, record in enumerate(data_records, 1):
        row = [
            Paragraph(str(idx), style_cell),
            Paragraph(str(record.get('so_ky_hieu_tai_lieu', '') or ''), style_cell),
            Paragraph(str(record.get('ngay_thang_van_ban', '') or ''), style_cell),
            Paragraph(str(record.get('ten_loai_tai_lieu', '') or ''), style_cell_left),
            Paragraph(str(record.get('tac_gia', '') or ''), style_cell),
            Paragraph(str(record.get('to_so_trang_so', '') or ''), style_cell)
        ]
        table_data.append(row)
        
    # 5. Khai báo tỷ lệ rộng các cột (Khổ A4 ngang ~ 842 điểm ảnh)
    col_widths = [45, 90, 80, 280, 180, 50]
    table = Table(table_data, colWidths=col_widths)
    
    # 6. Tô màu, kẻ khung viền cho bảng
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('BOTTOMPADDING', (0,0), (-1, -1), 8),
        ('TOPPADDING', (0,0), (-1, -1), 8),
    ]))
    
    # 7. Xuất xưởng file PDF
    pdf.build([table])
    print(f"✅ Đã xuất báo cáo PDF thành công tại: {output_filepath}")
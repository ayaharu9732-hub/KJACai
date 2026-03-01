from __future__ import annotations

from pathlib import Path

from reportlab.pdfgen import canvas

from src.kjac.report.layout import PAGE_SIZE, register_japanese_font
from src.kjac.report.sections import draw_page_1, draw_page_2, draw_page_3, draw_page_4, load_metrics


def build_report(metrics_csv: str | Path, output_pdf: str | Path) -> Path:
    csv_path = Path(metrics_csv)
    out_path = Path(output_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_rows, stats, summary = load_metrics(csv_path)
    font_name = register_japanese_font()

    c = canvas.Canvas(str(out_path), pagesize=PAGE_SIZE)
    draw_page_1(c, data_rows, stats, summary, font_name)
    draw_page_2(c, data_rows, font_name)
    draw_page_3(c, stats, font_name)
    draw_page_4(c, stats, font_name)
    c.save()
    return out_path

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Table, TableStyle

from src.kjac.runtime_paths import REPO_ROOT

PAGE_SIZE = landscape(A4)
LEFT_MARGIN = 15 * mm
TOP_MARGIN = 12 * mm


def register_japanese_font() -> str:
    candidates = [
        REPO_ROOT / "fonts" / "ipaexg.ttf",
        Path("C:/Windows/Fonts/ipaexg.ttf"),
        Path("C:/Windows/Fonts/IPAexGothic.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("IPAexGothic", str(path)))
                return "IPAexGothic"
            except Exception:
                continue
    return "Helvetica"


def draw_title(c, title: str, font_name: str) -> float:
    width, height = PAGE_SIZE
    y = height - TOP_MARGIN
    c.setFont(font_name, 17)
    c.drawString(LEFT_MARGIN, y, title)
    c.setStrokeColor(colors.grey)
    c.line(LEFT_MARGIN, y - 3 * mm, width - LEFT_MARGIN, y - 3 * mm)
    return y - 9 * mm


def draw_lines(c, lines: Iterable[str], y: float, font_name: str, size: int = 11, line_h: float = 6.2 * mm) -> float:
    c.setFont(font_name, size)
    for line in lines:
        c.drawString(LEFT_MARGIN, y, line)
        y -= line_h
    return y


def draw_table(
    c,
    data: list[list[str]],
    y: float,
    font_name: str,
    col_widths: list[float] | None = None,
    cell_bg_map: dict[tuple[int, int], Color] | None = None,
) -> float:
    if not data:
        return y
    width, _ = PAGE_SIZE
    if col_widths is None:
        per_col = (width - (LEFT_MARGIN * 2)) / max(1, len(data[0]))
        col_widths = [per_col] * len(data[0])
    table = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds: list[tuple] = [
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    if cell_bg_map:
        for (row_idx, col_idx), bg in cell_bg_map.items():
            style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), bg))
    table.setStyle(TableStyle(style_cmds))
    tw, th = table.wrapOn(c, width - (LEFT_MARGIN * 2), y)
    table.drawOn(c, LEFT_MARGIN, y - th)
    return y - th - 4 * mm

# -*- coding: utf-8 -*-
"""
patch_relay_report_v2_4_fpdf2_lnfix.py
- relay_report_generator_ai_v2_4.py の fpdf2 DeprecationWarning を消す
- cell(..., 0, 1, ...) の "ln=1相当" を new_x/new_y に置換
"""

from pathlib import Path
from datetime import datetime
import re

TARGET = Path("relay_report_generator_ai_v2_4.py")

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_lnfix_{ts}")
    bak.write_bytes(path.read_bytes())
    return bak

def main():
    s = TARGET.read_text(encoding="utf-8", errors="strict")

    # 1) header のセル（タイトル）
    # self.cell(0, 10, self.title_text, 0, 1, "C")
    s = re.sub(
        r'self\.cell\(\s*0\s*,\s*10\s*,\s*self\.title_text\s*,\s*0\s*,\s*1\s*,\s*"C"\s*\)',
        'self.cell(0, 10, self.title_text, new_x="LMARGIN", new_y="NEXT", align="C")',
        s
    )

    # 2) pdf.cell(..., 0, 1) を new_x/new_y に
    # pdf.cell(0, 8, "...", 0, 1)
    s = re.sub(
        r'pdf\.cell\(\s*0\s*,\s*([0-9]+)\s*,\s*(".*?")\s*,\s*0\s*,\s*1\s*\)',
        r'pdf.cell(0, \1, \2, new_x="LMARGIN", new_y="NEXT")',
        s
    )

    # 3) pdf.cell(0, 6, line, 0, 1) のように変数が入るパターン
    s = re.sub(
        r'pdf\.cell\(\s*0\s*,\s*([0-9]+)\s*,\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*0\s*,\s*1\s*\)',
        r'pdf.cell(0, \1, \2, new_x="LMARGIN", new_y="NEXT")',
        s
    )

    # 4) ReportPDF.header() 内の cell を align 引数の形に統一（保険）
    # もし既に new_x/new_y 形式なら触らない
    if 'new_x="LMARGIN", new_y="NEXT"' not in s:
        # 何かズレた時の保険（基本起きない）
        pass

    bak = backup(TARGET)
    TARGET.write_text(s, encoding="utf-8")
    print("[OK] patched:", TARGET)
    print("[OK] backup :", bak.name)

if __name__ == "__main__":
    main()

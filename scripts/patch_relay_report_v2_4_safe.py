#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_relay_report_v2_4_safe.py

✅ relay_report_generator_ai_v2_4.py へ安全にパッチ
- matplotlib グラフの日本語フォントを IPAexGothic に寄せる（DejaVu glyph warning 対策）
- FPDF add_font の deprecated uni(True) を除去
- header() の new_x/new_y を使わない（旧fpdf互換）
※ 文字列(トリプルクォート)を壊さないよう、行ベースで最小差分パッチ
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

TARGET = Path("relay_report_generator_ai_v2_4.py")

FONT_BLOCK = r"""
# ==============================
# matplotlib (graph) font settings (Japanese)
# ==============================
import matplotlib
from matplotlib import font_manager

# グラフ内の日本語を IPAexGothic に統一（DejaVu警告を減らす）
if os.path.exists(FONT_PATH):
    try:
        font_manager.fontManager.addfont(FONT_PATH)
        ipa_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
        matplotlib.rcParams["font.family"] = ipa_name
    except Exception:
        matplotlib.rcParams["font.family"] = "IPAexGothic"
    matplotlib.rcParams["axes.unicode_minus"] = False
""".strip("\n") + "\n"


def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_{ts}")
    bak.write_bytes(path.read_bytes())
    return bak


def patch_addfont_uni(s: str) -> str:
    # add_font("IPA", "", FONT_PATH, True) -> add_font("IPA", "", FONT_PATH)
    s = re.sub(r"(add_font\(\s*\"IPA\"\s*,\s*\"\"\s*,\s*FONT_PATH\s*),\s*True\s*\)", r"\1)", s)
    s = re.sub(r"(add_font\(\s*\"IPA\"\s*,\s*\"B\"\s*,\s*FONT_PATH\s*),\s*True\s*\)", r"\1)", s)
    return s


def inject_font_block_after_fontpath_check(s: str) -> str:
    if "matplotlib (graph) font settings (Japanese)" in s:
        return s  # already

    # FONT_PATH exists check の直後に挿入（ここはコード領域で安全）
    pat = re.compile(
        r"(FONT_PATH\s*=\s*.*\n"
        r"if\s+not\s+os\.path\.exists\(FONT_PATH\)\s*:\s*\n"
        r"(?:[ \t]+.*\n)+)",
        re.M
    )
    m = pat.search(s)
    if not m:
        raise SystemExit("[NG] FONT_PATH の存在チェックが見つからず、挿入できません。")

    insert_pos = m.end(1)
    return s[:insert_pos] + "\n" + FONT_BLOCK + "\n" + s[insert_pos:]


def patch_header_legacy(s: str) -> str:
    # header() 内の self.cell(... new_x/new_y ...) を旧式にする
    # まず header ブロックを探す
    pat = re.compile(r"(\nclass\s+ReportPDF\(FPDF\):.*?\n)(?=\n# ==============================|\Z)", re.S)
    m = pat.search(s)
    if not m:
        return s

    block = m.group(1)
    if "def header" not in block:
        return s

    # header 関数だけ差し替え（new_x/new_y があってもなくても、固定で安全版にする）
    block2 = re.sub(
        r"def\s+header\s*\(\s*self\s*\)\s*:\s*\n(?:[ \t].*\n)+",
        "def header(self):\n"
        "        self.set_font(\"IPA\", \"B\", 16)\n"
        "        # fpdf(古い)互換：new_x/new_y を使わない\n"
        "        self.cell(0, 10, self.title_text, 0, 1, \"C\")\n"
        "        self.ln(4)\n",
        block,
        flags=re.S
    )

    return s[:m.start(1)] + block2 + s[m.end(1):]


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"[NG] not found: {TARGET.resolve()}")

    original = TARGET.read_text(encoding="utf-8", errors="ignore")
    s = original

    s = patch_addfont_uni(s)
    s = inject_font_block_after_fontpath_check(s)
    s = patch_header_legacy(s)

    if s == original:
        print("[OK] no change (already patched?)")
        return

    bak = backup(TARGET)
    TARGET.write_text(s, encoding="utf-8")
    print("[OK] patched:", TARGET)
    print("[OK] backup :", bak.name)


if __name__ == "__main__":
    main()

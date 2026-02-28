# -*- coding: utf-8 -*-
"""
Page1 + Page2 + Page3+ を生成して結合するランナー
出力:
  reports/athletes/<athlete>_<start>_to_<end>_<audience>__FULL.pdf
"""

import os
import argparse

from pypdf import PdfMerger

from build_carte_page1 import build_page1_pdf
from build_carte_page2_ai import build_page2_pdf
from build_carte_pages_daily import build_daily_pages_pdf


def safe_mkdir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("athlete_id")
    ap.add_argument("--audience", choices=["coach", "parent"], default="coach")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # 1) 各パート生成
    p1 = build_page1_pdf(base_dir, args.athlete_id, args.audience, args.start, args.end)
    p2 = build_page2_pdf(base_dir, args.athlete_id, args.audience, args.start, args.end)
    p3 = build_daily_pages_pdf(base_dir, args.athlete_id, args.audience, args.start, args.end)

    # 2) 結合
    out_dir = os.path.join(base_dir, "reports", "athletes")
    safe_mkdir(out_dir)

    # build_page1側で正規化済みのファイル名から athlete_id を拾うのが確実
    # 例: A-00001_2026-01-12_to_2026-01-14_coach__p1.pdf
    base_name = os.path.basename(p1).replace("__p1.pdf", "")
    full_path = os.path.join(out_dir, f"{base_name}__FULL.pdf")

    merger = PdfMerger()
    merger.append(p1)
    merger.append(p2)
    merger.append(p3)
    merger.write(full_path)
    merger.close()

    print(f"[OK] FULL PDF: {full_path}")


if __name__ == "__main__":
    main()

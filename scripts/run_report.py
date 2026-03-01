from __future__ import annotations
import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kjac.report.pdf_builder import build_report

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", help="input metrics CSV path")
    ap.add_argument("--out", default="report.pdf", help="output PDF path")
    ns = ap.parse_args()

    csv_path = Path(ns.metrics_csv)
    if not csv_path.exists():
        print(f"[ERR] metrics csv not found: {csv_path}", file=sys.stderr)
        return 2

    out_path = build_report(csv_path, ns.out)
    print("[DONE]", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

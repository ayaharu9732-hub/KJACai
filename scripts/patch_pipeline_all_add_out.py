# scripts/patch_pipeline_all_add_out.py
# -*- coding: utf-8 -*-
"""
SAFE PATCH (py3.9):
- Add --out option to `pipeline all` subcommand in kjac_pipeline_v1_0.py
- Pass args.out through to relay report-v2-4 call inside pipeline all (if present)
- Make backup before writing

Usage:
  python .\scripts\patch_pipeline_all_add_out.py
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
TARGET = ROOT / "kjac_pipeline_v1_0.py"

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_pipeout_{ts}")
    bak.write_bytes(path.read_bytes())
    return bak

def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"not found: {TARGET}")

    src = TARGET.read_text(encoding="utf-8", errors="strict")
    bak = backup(TARGET)
    out = src

    # ------------------------------------------------------------
    # 1) Add "--out" argument to pipeline all parser
    #    Insert right after pipe_all.add_argument("--csv", ...)
    # ------------------------------------------------------------
    if re.search(r'pipe_all\.add_argument\(\s*"--out"\s*,', out) is None:
        out2, n = re.subn(
            r'(pipe_all\.add_argument\(\s*"--csv"[^\n]*\)\n)',
            r'\1    pipe_all.add_argument("--out", default=None, help="(relay) 出力PDFパス。未指定なら output/<csv名>_AI_report.pdf")\n',
            out,
            count=1
        )
        out = out2
        print(f"[INFO] add --out to pipeline all parser: {'OK' if n==1 else 'NOT FOUND'}")

    # ------------------------------------------------------------
    # 2) Pass args.out into relay_args in cmd_pipeline_all
    #    Find: relay_args = argparse.Namespace(csv=args.csv)  -> add out=...
    # ------------------------------------------------------------
    # Pattern A
    out2, n = re.subn(
        r'relay_args\s*=\s*argparse\.Namespace\(\s*csv\s*=\s*args\.csv\s*\)',
        r"relay_args = argparse.Namespace(csv=args.csv, out=getattr(args, 'out', None))",
        out
    )
    out = out2
    print(f"[INFO] patch relay_args namespace: {'OK' if n>=1 else 'NOT FOUND'}")

    # Pattern B (alternate formatting)
    if n == 0:
        out2, n2 = re.subn(
            r'relay_args\s*=\s*argparse\.Namespace\(\s*csv\s*=\s*args\.csv\s*,\s*\)',
            r"relay_args = argparse.Namespace(csv=args.csv, out=getattr(args, 'out', None))",
            out
        )
        out = out2
        print(f"[INFO] patch relay_args alt: {'OK' if n2>=1 else 'NOT FOUND'}")

    # ------------------------------------------------------------
    # 3) Write back
    # ------------------------------------------------------------
    if out == src:
        print("[WARN] no changes applied (patterns not found).")
        print("[HINT] kjac_pipeline_v1_0.py の 'pipeline all' 周りの該当行を貼ってくれたら、完全一致で当てる。")
        return

    TARGET.write_text(out, encoding="utf-8")
    print(f"[OK] patched: {TARGET.name}")
    print(f"[OK] backup : {bak.name}")

if __name__ == "__main__":
    main()

# scripts/patch_pipeline_out_default.py
# -*- coding: utf-8 -*-
"""
Patch kjac_pipeline_v1_0.py:
- Add --out option to `relay report-v2-4` and `pipeline all`
- Default relay output to <root>/output/<csv_stem>_AI_report.pdf when --out is not provided
- Ensure output directory exists
- Safe backup before writing

Usage:
  python .\scripts\patch_pipeline_out_default.py
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
TARGET = ROOT / "kjac_pipeline_v1_0.py"

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_outpatch_{ts}")
    bak.write_bytes(path.read_bytes())
    return bak

def must_find(pattern: str, text: str, flags=0) -> re.Match:
    m = re.search(pattern, text, flags)
    if not m:
        raise RuntimeError(f"Pattern not found:\n{pattern}")
    return m

def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"not found: {TARGET}")

    src = TARGET.read_text(encoding="utf-8", errors="strict")
    bak = backup(TARGET)

    out = src

    # ------------------------------------------------------------
    # 0) Ensure we can use Path in functions (already imported in your file)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 1) Patch cmd_relay_report_v2_4 to accept args.out and default to output/<stem>_AI_report.pdf
    # ------------------------------------------------------------
    # Find existing function definition block
    m = must_find(
        r"def cmd_relay_report_v2_4\(args: argparse\.Namespace, root: Path\) -> None:\n"
        r"(?:[ \t].*\n)+",
        out,
        flags=re.MULTILINE
    )
    block = m.group(0)

    # Replace the whole function body with a robust version
    new_block = (
        "def cmd_relay_report_v2_4(args: argparse.Namespace, root: Path) -> None:\n"
        "    script = root / SCRIPT_PATHS[\"relay\"]\n"
        "    ensure_exists(script)\n"
        "    if not getattr(args, \"csv\", None):\n"
        "        logger.error(\"--csv を指定してください。\")\n"
        "        sys.exit(2)\n"
        "\n"
        "    csv_path = Path(args.csv)\n"
        "    # 出力先：--out が指定されていればそれを優先。未指定なら <root>/output/<csvstem>_AI_report.pdf\n"
        "    out_pdf = getattr(args, \"out\", None)\n"
        "    if out_pdf:\n"
        "        out_pdf = Path(out_pdf)\n"
        "    else:\n"
        "        out_dir = root / \"output\"\n"
        "        out_dir.mkdir(parents=True, exist_ok=True)\n"
        "        out_pdf = out_dir / f\"{csv_path.stem}_AI_report.pdf\"\n"
        "\n"
        "    # output ディレクトリ作成（--out がフォルダ配下のときも）\n"
        "    out_pdf.parent.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "    cmdline = [sys.executable, str(script), \"--csv\", str(csv_path)]\n"
        "    # relay 側が --out を受ける前提（あなたの v2.4 は既に対応済み）\n"
        "    cmdline += [\"--out\", str(out_pdf)]\n"
        "    run(cmdline)\n"
    )
    out = out.replace(block, new_block)

    # ------------------------------------------------------------
    # 2) Patch cmd_pipeline_all: pass through --out to relay call
    # ------------------------------------------------------------
    # Find where relay_args is created
    # Original:
    # relay_args = argparse.Namespace(csv=args.csv)
    # cmd_relay_report_v2_4(relay_args, root)
    out = re.sub(
        r"relay_args\s*=\s*argparse\.Namespace\(\s*csv\s*=\s*args\.csv\s*\)\n\s*cmd_relay_report_v2_4\(relay_args,\s*root\)",
        "relay_args = argparse.Namespace(csv=args.csv, out=getattr(args, 'out', None))\n        cmd_relay_report_v2_4(relay_args, root)",
        out,
        flags=re.MULTILINE
    )

    # If substitution didn't happen (slightly different formatting), do a safer fallback edit
    if "argparse.Namespace(csv=args.csv, out=getattr(args, 'out', None))" not in out:
        # fallback: find the relay section in cmd_pipeline_all and insert out=
        m2 = must_find(r"logger\.info\(\\"\[4/4\] リレーPDF: relay v2\.4\\"\)\n(?:[ \t].*\n)+?logger\.info\(\"  → スキップ \(relay --csv 未指定\)\"\)\n", out, flags=re.MULTILINE)
        seg = m2.group(0)
        seg2 = seg.replace(
            "relay_args = argparse.Namespace(csv=args.csv)",
            "relay_args = argparse.Namespace(csv=args.csv, out=getattr(args, 'out', None))"
        )
        out = out.replace(seg, seg2)

    # ------------------------------------------------------------
    # 3) Add --out to relay report-v2-4 subcommand
    # ------------------------------------------------------------
    # Find parser block for r_v24
    # r_v24 = sp_relay.add_parser(...)
    # r_v24.add_argument("--csv", ...)
    # Add: r_v24.add_argument("--out", ...)
    if "--out" not in out:
        out = re.sub(
            r"(r_v24\s*=\s*sp_relay\.add_parser\(\"report-v2-4\".*\)\n\s*r_v24\.add_argument\(\"--csv\"[^\n]*\)\n)",
            r"\1\n    r_v24.add_argument(\"--out\", type=Path, default=None, help=\"出力PDFパス（未指定なら <root>/output/<csv名>_AI_report.pdf）\")\n",
            out,
            flags=re.MULTILINE
        )

    # ------------------------------------------------------------
    # 4) Add --out to pipeline all subcommand (relay用の出力PDF)
    # ------------------------------------------------------------
    out = re.sub(
        r"(pipe_all\.add_argument\(\"--csv\"[^\n]*\)\n)",
        r"\1    pipe_all.add_argument(\"--out\", type=Path, default=None, help=\"(relay) 出力PDFパス（未指定なら <root>/output/<csv名>_AI_report.pdf）\")\n",
        out,
        flags=re.MULTILINE
    )

    # ------------------------------------------------------------
    # 5) Ensure root/output folder exists even if user only runs relay command and no --out
    # (handled in cmd_relay_report_v2_4)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Final write
    # ------------------------------------------------------------
    TARGET.write_text(out, encoding="utf-8")
    print(f"[OK] patched: {TARGET.name}")
    print(f"[OK] backup : {bak.name}")

if __name__ == "__main__":
    main()

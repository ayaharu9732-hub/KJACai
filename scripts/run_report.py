from __future__ import annotations
import argparse
import subprocess
import sys
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPL = ROOT / "scripts" / "pose_reporter_pdf_ai_v5_5_3_gpt_all_final.py"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=str(DEFAULT_IMPL))
    ap.add_argument("--args", default="", help="extra args passed to impl (as a single string)")
    ns = ap.parse_args()

    impl = Path(ns.impl)
    if not impl.exists():
        print(f"[ERR] impl not found: {impl}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(impl)]
    if ns.args.strip():
        cmd += shlex.split(ns.args)

    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())

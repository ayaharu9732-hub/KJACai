from __future__ import annotations
"""
Canonical KJAC pipeline CLI entrypoint.

Use this wrapper as the official way to run the pipeline.
Legacy implementation scripts remain available for compatibility.
"""
import argparse
import subprocess
import sys
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPL = REPO_ROOT / "scripts" / "kjac_pipeline_v1_3.py"

def main() -> int:
    print("[KJAC PIPELINE] canonical entrypoint")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all", help="pipeline mode (e.g. all)")
    ap.add_argument("--impl", default=str(DEFAULT_IMPL), help="pipeline implementation script path")
    ap.add_argument("--args", default="", help="extra args passed to impl (as a single string)")
    ns = ap.parse_args()

    impl = Path(ns.impl)
    if not impl.exists():
        print(f"[ERR] impl not found: {impl}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(impl), "pipeline", ns.mode]
    if ns.args.strip():
        cmd += shlex.split(ns.args)

    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())

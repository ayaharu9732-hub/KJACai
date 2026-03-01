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

# ensure repo root is importable (so `from src...` works when executed as a script)
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.kjac.runtime_paths import REPO_ROOT  # noqa: E402

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
    # ---- Preflight: require at least one MP4 in videos/ ----
    videos_dir = REPO_ROOT / "videos"
    mp4s = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
    if not mp4s:
        print()
        print("⚠ No input video found.")
        print("Please place an MP4 file inside:")
        print(f"  {videos_dir}")
        print()
        print("Example:")
        print("  videos/run01.mp4")
        print()
        print("Then run again:")
        print("  python scripts/run_pipeline.py")
        print()
        return 2
    cmd = [sys.executable, str(impl), "pipeline", ns.mode]
    if ns.args.strip():
        cmd += shlex.split(ns.args)

    print("[RUN]", " ".join(cmd))

    rc = subprocess.call(cmd)

    # ---- Friendly guidance for missing videos ----
    if rc != 0:
        videos_dir = REPO_ROOT / "videos"
        mp4s = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []
        if not mp4s:
            print()
            print("⚠ No input video found.")
            print("Please place an MP4 file inside:")
            print(f"  {videos_dir}")
            print()
            print("Example:")
            print("  videos/run01.mp4")
            print()
            print("Then run again:")
            print("  python scripts/run_pipeline.py")
            print()

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
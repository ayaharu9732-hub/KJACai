# scripts/patch_fix_src_calibration_signature.py
# -*- coding: utf-8 -*-
"""
Fix src/calibration.py to match src.cli_one_stage expectations:
- load_scale(scales_dir, cam_name) -> (meter_per_pixel, raw_dict)
- save_scale(scales_dir, cam_name, meter_per_pixel, raw_dict=None) -> None
- calibrate_interactive(...) remains usable
Python 3.9 compatible
"""

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "src" / "calibration.py"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

CODE = r'''# -*- coding: utf-8 -*-
"""
Calibration utilities (compatible with src.cli_one_stage)

Expected usage in cli_one_stage:
    m_per_px, raw = load_scale(scales_dir, cam_name)

We store per-camera JSON:
    <scales_dir>/<cam_name>.json

Format (minimal):
    {
      "meter_per_pixel": 0.0123,
      "updated_at": "2026-02-20T21:00:00"
    }
"""

from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any


def _scale_path(scales_dir: str | Path, cam_name: str) -> Path:
    d = Path(scales_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(cam_name))
    return d / f"{safe}.json"


def load_scale(scales_dir: str | Path, cam_name: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Returns (meter_per_pixel, raw_dict)
    - meter_per_pixel: float | None
    - raw_dict: dict (empty if file missing/invalid)
    """
    p = _scale_path(scales_dir, cam_name)
    if not p.exists():
        return None, {}

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        v = raw.get("meter_per_pixel", None)
        if v is None:
            return None, raw if isinstance(raw, dict) else {}
        v = float(v)
        if v <= 0:
            return None, raw if isinstance(raw, dict) else {}
        return v, raw if isinstance(raw, dict) else {"meter_per_pixel": v}
    except Exception:
        return None, {}


def save_scale(
    scales_dir: str | Path,
    cam_name: str,
    meter_per_pixel: float,
    raw: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save meter_per_pixel to <scales_dir>/<cam_name>.json
    Returns saved path.
    """
    p = _scale_path(scales_dir, cam_name)

    obj: Dict[str, Any] = {}
    if isinstance(raw, dict):
        obj.update(raw)

    obj["meter_per_pixel"] = float(meter_per_pixel)
    obj["updated_at"] = datetime.now().isoformat(timespec="seconds")

    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def calibrate_interactive(
    *,
    known_distance_m: float = 1.0,
    pixel_distance: Optional[float] = None,
    default_scale: float = 0.01
) -> float:
    """
    Minimal interactive calibration:
    - If pixel_distance is provided, returns known_distance_m / pixel_distance
    - Else asks via input()
    """
    if pixel_distance is None:
        try:
            s = input(
                f"[CALIB] 既知距離 {known_distance_m}m に相当するピクセル数を入力してください (例: 120): "
            ).strip()
            pixel_distance = float(s)
        except Exception:
            pixel_distance = None

    if pixel_distance is None or pixel_distance <= 0:
        return float(default_scale)

    return float(known_distance_m) / float(pixel_distance)
'''

def main():
    if not CALIB.exists():
        raise SystemExit(f"[NG] not found: {CALIB}")

    bak = CALIB.with_name(CALIB.name + f".bak_sigfix_{STAMP}")
    bak.write_text(CALIB.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    CALIB.write_text(CODE, encoding="utf-8")
    print(f"[OK] patched : {CALIB}")
    print(f"[OK] backup  : {bak}")

if __name__ == "__main__":
    main()
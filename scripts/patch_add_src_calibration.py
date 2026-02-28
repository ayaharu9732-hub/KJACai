# scripts/patch_add_src_calibration.py
# -*- coding: utf-8 -*-
"""
Add missing module: src/calibration.py
- Creates src/__init__.py if missing
- Creates src/calibration.py if missing
- Safe: never overwrites existing calibration.py (creates .bak if overwrite is needed)
Python 3.9 compatible
"""

from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
INIT = SRC / "__init__.py"
CALIB = SRC / "calibration.py"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

CALIB_CODE = r'''# -*- coding: utf-8 -*-
"""
Calibration utilities
- Used by src.cli_one_stage
- Provides:
    load_scale(path) -> float|None
    save_scale(path, scale: float) -> None
    calibrate_interactive(...) -> float
"""

from __future__ import annotations
from pathlib import Path
import json
from typing import Optional


def load_scale(path: str | Path) -> Optional[float]:
    """
    Load scale factor (meter_per_pixel) from a JSON file.
    Expected format:
        {"meter_per_pixel": 0.0123}
    Returns None if not found/invalid.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        v = obj.get("meter_per_pixel", None)
        if v is None:
            return None
        v = float(v)
        if v <= 0:
            return None
        return v
    except Exception:
        return None


def save_scale(path: str | Path, scale: float) -> None:
    """
    Save scale factor (meter_per_pixel) to JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {"meter_per_pixel": float(scale)}
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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
    This keeps pipeline moving without OpenCV GUI dependencies.

    Example:
        known_distance_m=1.0, pixel_distance=100 -> 0.01 m/px
    """
    if pixel_distance is None:
        try:
            s = input(f"[CALIB] 既知距離 {known_distance_m}m に相当するピクセル数を入力してください (例: 120): ").strip()
            pixel_distance = float(s)
        except Exception:
            pixel_distance = None

    if pixel_distance is None or pixel_distance <= 0:
        # fallback
        return float(default_scale)

    return float(known_distance_m) / float(pixel_distance)
'''

def main():
    if not SRC.exists():
        raise SystemExit(f"[NG] src フォルダが見つかりません: {SRC}")

    # __init__.py
    if not INIT.exists():
        INIT.write_text("# src package\n", encoding="utf-8")
        print(f"[OK] created: {INIT}")
    else:
        print(f"[SKIP] exists:  {INIT}")

    # calibration.py
    if CALIB.exists():
        # 既にあるなら基本は触らない（安全第一）
        print(f"[SKIP] exists:  {CALIB}")
        return

    CALIB.write_text(CALIB_CODE, encoding="utf-8")
    print(f"[OK] created: {CALIB}")

    # sanity: show size
    print(f"[INFO] size: {CALIB.stat().st_size} bytes")


if __name__ == "__main__":
    main()

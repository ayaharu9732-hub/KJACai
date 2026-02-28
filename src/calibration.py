# -*- coding: utf-8 -*-
"""
Calibration utilities (compatible with src.cli_one_stage)

Expected usage in cli_one_stage:
    m_per_px, raw = load_scale(scales_dir, cam_name)
    mpp, px, rm = calibrate_interactive(video)

We store per-camera JSON:
    <scales_dir>/<cam_name>.json

Format (minimal):
    {
      "meter_per_pixel": 0.0123,
      "updated_at": "2026-02-20T21:00:00",
      "note": "optional"
    }
"""

from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union


def _scale_path(scales_dir: Union[str, Path], cam_name: str) -> Path:
    d = Path(scales_dir)
    d.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(cam_name))
    return d / f"{safe}.json"


def load_scale(scales_dir: Union[str, Path], cam_name: str) -> Tuple[Optional[float], Dict[str, Any]]:
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
        if not isinstance(raw, dict):
            return None, {}
        v = raw.get("meter_per_pixel", None)
        if v is None:
            return None, raw
        v = float(v)
        if v <= 0:
            return None, raw
        return v, raw
    except Exception:
        return None, {}


def save_scale(
    scales_dir: Union[str, Path],
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


def _ask_float(prompt: str) -> Optional[float]:
    try:
        s = input(prompt).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def calibrate_interactive(video=None):
    """
    Interactive calibration compatible with cli_one_stage.

    Returns:
        (mpp, px, rm)
          mpp: meters per pixel (float)
          px : pixel distance used (float)
          rm : real meters used (float)

    Notes:
    - video is accepted for API compatibility; this minimal implementation does not parse frames.
    - You can type:
        real meters (e.g. 1.00)
        pixel distance (e.g. 120)
      then mpp = rm / px
    """
    # You can just press Enter to use defaults
    rm = _ask_float("[CALIB] 既知距離（メートル）を入力してください (例: 1.00) : ")
    if rm is None or rm <= 0:
        rm = 1.0

    px = _ask_float("[CALIB] 上の距離に相当するピクセル数を入力してください (例: 120) : ")
    if px is None or px <= 0:
        # fallback: a safe default (tune later)
        px = 100.0

    mpp = float(rm) / float(px)
    return mpp, float(px), float(rm)

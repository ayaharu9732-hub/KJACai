# -*- coding: utf-8 -*-
"""
cli_one_stage_v1_7.py
Start(0–20m) 専用 — 動画名から選手名と日付を抽出し、
outputs/<選手名>/<選手名><日付>.pdf に出力
"""

import argparse
import os
import cv2
import yaml
import numpy as np
import re
from datetime import datetime
from .calibration import load_scale, save_scale, calibrate_interactive
from .analyzer import draw_guides, put_label, flow_metrics, classify_phase
from .notes import generate_ai_summary


# ========== 共通ユーティリティ ==========
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _read_cfg(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _open_video_any_backend(path):
    for be in [cv2.CAP_ANY, getattr(cv2, "CAP_MSMF", 0), getattr(cv2, "CAP_FFMPEG", 1900), getattr(cv2, "CAP_DSHOW", 700)]:
        cap = cv2.VideoCapture(path, be)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(path)


def _frame_at_time(cap, t_sec, fps):
    idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
    return cap.read()


def _iter_times(start_sec, total_sec, step_sec):
    t = max(0.0, float(start_sec))
    while t <= total_sec + 1e-9:
        yield round(t, 3)
        t += step_sec


def _ema(x, alpha=0.25):
    y = []
    for i, v in enumerate(x):
        if i == 0:
            y.append(v)
        else:
            y.append(alpha * v + (1 - alpha) * y[-1])
    return np.array(y, dtype=float)


def _kalman_like(v):
    return _ema(v, alpha=0.3)


def _to_mps(px_per_s, m_per_px):
    return float(px_per_s) * float(m_per_px)


# ========== PDF生成 ==========
def make_pdf_one_stage(pdf_path, pages, times, speeds_ms, cfg, label, ai_summary_text):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_path = cfg.get("font_path", "ipaexg.ttf")
    try:
        pdfmetrics.registerFont(TTFont("IPAexGothic", font_path))
        font_jp = "IPAexGothic"
    except Exception:
        font_jp = "Helvetica"

    c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
    W, H = landscape(A4)

    def draw_speed_plot():
        import matplotlib.pyplot as plt
        import tempfile
        if len(times) == 0 or len(speeds_ms) == 0:
            return
        fig = plt.figure(figsize=(8, 3), dpi=150)
        plt.plot(times, speeds_ms, label="m/s")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("速度プロファイル")
        plt.tight_layout()
        tmp_plot = os.path.join(os.path.dirname(pdf_path), "_tmp_plot.png")
        fig.savefig(tmp_plot)
        plt.close(fig)
        c.setFont(font_jp, 18)
        c.drawString(40, H - 60, f"{label} : 速度プロファイル")
        c.drawImage(tmp_plot, 40, 140, width=760, height=360, preserveAspectRatio=True)
        os.remove(tmp_plot)
        c.showPage()

    def draw_paragraph(x, y, text, width=760, leading=18, size=12):
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        if isinstance(text, (list, tuple)):
            text = "\n\n".join(str(t) for t in text)
        elif not isinstance(text, str):
            text = str(text)
        text = text.replace("\n", "<br/>")
        styles = getSampleStyleSheet()
        style = styles["Normal"]
        style.fontName = font_jp
        style.fontSize = size
        style.leading = leading
        p = Paragraph(text, style)
        w, h = p.wrap(width, 9999)
        p.drawOn(c, x, y - h)
        return h

    # --- Page 1: Frames ---
    c.setFont(font_jp, 20)
    c.drawString(40, H - 60, f"動画レポート：{label}")
    y = H - 90
    for (t, phase, img_path, v_ms, v_kmh) in pages:
        c.setFont(font_jp, 11)
        c.drawString(40, y, f"t={t:.1f}s | Phase: {phase}")
        c.drawString(40, y - 16, f"Speed: {v_ms:.2f} m/s ({v_kmh:.1f} km/h)")
        try:
            c.drawImage(img_path, 420, y - 120, width=360, height=110, preserveAspectRatio=True)
        except Exception:
            pass
        y -= 140
        if y < 160:
            c.showPage()
            c.setFont(font_jp, 20)
            c.drawString(40, H - 60, f"動画レポート：{label}")
            y = H - 90
    c.showPage()

    # --- Page 2: Speed plot ---
    draw_speed_plot()

    # --- Page 3: AI summary ---
    c.setFont(font_jp, 18)
    c.drawString(40, H - 60, "AI分析レポート（サマリー）")
    draw_paragraph(40, H - 90, ai_summary_text)
    c.showPage()

    c.save()


# ========== 本体 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", help="動画パス（省略時はダイアログ）")
    ap.add_argument("--pick-dialog", action="store_true")
    ap.add_argument("--force-calibrate", action="store_true")
    ap.add_argument("--label", default="Start(0–20m)")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)

    # --- 動画選択 ---
    video = args.video
    if args.pick_dialog or not video:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        base = cfg.get("pick_default_dir") or os.getcwd()
        video = filedialog.askopenfilename(initialdir=base, filetypes=[("Video", "*.mp4 *.mov *.avi")])
        root.destroy()

    if not video or not os.path.exists(video):
        raise FileNotFoundError(f"動画が見つかりません: {video}")

    # --- ファイル名から選手名と日付を抽出 ---
    base_name = os.path.splitext(os.path.basename(video))[0]
    m = re.match(r"([^\d]+)([\d.]+)", base_name)
    if m:
        athlete = m.group(1).strip()
        date_tag = m.group(2).strip()
    else:
        athlete = base_name
        date_tag = datetime.now().strftime("%m%d")

    print(f"[INFO] 選手: {athlete}, 日付タグ: {date_tag}")

    # --- キャリブレーション ---
    scales_dir = _ensure_dir("scales")
    cam_name = "one_stage_default"
    m_per_px, raw = load_scale(scales_dir, cam_name)
    if args.force_calibrate or m_per_px is None:
        mpp, px, rm = calibrate_interactive(video)
        raw = save_scale(scales_dir, cam_name, mpp, px, rm)
        m_per_px = mpp
    print(f"[Calib] 1 px = {m_per_px:.6f} m  ({raw})")

    # --- 出力ディレクトリ構成 ---
    out_root = _ensure_dir(cfg.get("output_dir", "outputs"))
    athlete_dir = _ensure_dir(os.path.join(out_root, athlete))
    frames_dir = _ensure_dir(os.path.join(athlete_dir, "frames"))

    # --- 動画解析 ---
    cap = _open_video_any_backend(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps if total_frames > 0 else 0
    step_sec = float(cfg.get("step_sec", 0.1))
    start_sec = float(cfg.get("start_sec", 0.0))

    ok0, frame0 = _frame_at_time(cap, start_sec, fps)
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_pts = None
    times, speeds_ms, pages = [], [], []

    for t in _iter_times(start_sec, total_sec, step_sec):
        ok, frame = _frame_at_time(cap, t, fps)
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v_px, prev_pts, *_ = flow_metrics(prev_gray, gray, step_sec, prev_pts)
        prev_gray = gray
        v_ms = _to_mps(v_px, m_per_px)
        v_kmh = v_ms * 3.6
        phase = classify_phase(v_px, v_px, None, None)
        draw_guides(frame)
        put_label(frame, f"{args.label} | {phase} | t={t:.1f}s")
        img_path = os.path.join(frames_dir, f"t_{t:.1f}.jpg".replace(".", "_"))
        cv2.imwrite(img_path, frame)
        times.append(t); speeds_ms.append(v_ms)
        pages.append((t, phase, img_path, v_ms, v_kmh))

    cap.release()
    speeds_ms_smooth = _kalman_like(speeds_ms)

    # --- AI summary ---
    ai_summary_text = ""
    try:
        ai_summary_text = generate_ai_summary(
            times=times,
            speeds_ms=speeds_ms_smooth,
            tilts=None,
            phases=None,
            model=cfg.get("openai_model", "gpt-4o-mini"),
            temperature=float(cfg.get("ai_temperature", 0.5)),
            max_tokens=700,
        )
    except Exception as e:
        ai_summary_text = f"AIサマリー生成に失敗しました: {e}"

    # --- PDF出力 ---
    pdf_path = os.path.join(athlete_dir, f"{athlete}{date_tag}.pdf")
    make_pdf_one_stage(pdf_path, pages, times, speeds_ms_smooth, cfg, label=args.label, ai_summary_text=ai_summary_text)
    print(f"[OK] 出力完了: {pdf_path}")


if __name__ == "__main__":
    main()

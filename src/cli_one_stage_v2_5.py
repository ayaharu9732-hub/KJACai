# -*- coding: utf-8 -*-
"""
cli_one_stage_v2_5.py
Start(0–20m) 完全安定版（画像出力・選手別フォルダ・距離入力付き）
"""

import argparse
import os
import re
import cv2
import yaml
import numpy as np
from datetime import datetime
from PIL import Image
from .calibration import calibrate_interactive, save_scale
from .analyzer import draw_guides, put_label, flow_metrics, classify_phase
from .notes import generate_ai_summary


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _read_cfg(p):
    with open(p, encoding="utf-8") as f:
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


def _iter_times(start, total, step):
    t = max(0.0, start)
    while t <= total + 1e-9:
        yield round(t, 3)
        t += step


def _kalman_like(v):
    a = 0.3
    y = []
    for i, x in enumerate(v):
        y.append(x if i == 0 else a * x + (1 - a) * y[-1])
    return np.array(y)


def _parse_filename(path):
    base = os.path.splitext(os.path.basename(path))[0].replace(" ", "").replace("_", "")
    m = re.match(r"^([^\d]+)([\d.]+)$", base)
    if m:
        return m.group(1), m.group(2).rstrip(".")
    return base, ""


def make_pdf_one_stage(pdf_path, pages, times, speeds_ms, cfg, label, ai_summary_text):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import getSampleStyleSheet

    font_path = cfg.get("font_path", "ipaexg.ttf")
    try:
        pdfmetrics.registerFont(TTFont("IPAexGothic", font_path))
        font_jp = "IPAexGothic"
    except Exception:
        font_jp = "Helvetica"

    c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
    W, H = landscape(A4)

    def _draw_image_safe(img_path, x, y, w=360, h=110):
        try:
            c.drawImage(img_path, x, y, width=w, height=h, preserveAspectRatio=True)
        except Exception as e:
            print("[WARN] 画像描画失敗:", e)

    def draw_speed_plot():
        import matplotlib.pyplot as plt
        if len(times) == 0 or len(speeds_ms) == 0:
            return
        fig = plt.figure(figsize=(8, 3), dpi=150)
        plt.plot(times, speeds_ms)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("速度プロファイル")
        plt.tight_layout()
        tmp = os.path.join(os.path.dirname(pdf_path), "_plot.png")
        fig.savefig(tmp)
        plt.close(fig)
        c.setFont(font_jp, 18)
        c.drawString(40, H - 60, f"{label} : 速度プロファイル")
        _draw_image_safe(tmp, 40, 140, 760, 360)
        c.showPage()

    def draw_paragraph(x, y, text, width=760, size=12):
        styles = getSampleStyleSheet()
        st = styles["Normal"]
        st.fontName = font_jp
        st.fontSize = size
        st.leading = 18
        p = Paragraph(str(text).replace("\n", "<br/>"), st)
        w, h = p.wrap(width, 9999)
        p.drawOn(c, x, y - h)
        return h

    c.setFont(font_jp, 20)
    c.drawString(40, H - 60, f"動画レポート：{label}")
    y = H - 90
    for (t, phase, img, v_ms, v_kmh) in pages:
        c.setFont(font_jp, 11)
        c.drawString(40, y, f"t={t:.1f}s | Phase: {phase}")
        c.drawString(40, y - 16, f"Speed: {v_ms:.2f} m/s ({v_kmh:.1f} km/h)")
        _draw_image_safe(img, 420, y - 120)
        y -= 140
        if y < 160:
            c.showPage()
            c.setFont(font_jp, 20)
            c.drawString(40, H - 60, f"動画レポート：{label}")
            y = H - 90
    c.showPage()
    draw_speed_plot()
    c.setFont(font_jp, 18)
    c.drawString(40, H - 60, "AI分析レポート（サマリー）")
    draw_paragraph(40, H - 90, ai_summary_text)
    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video")
    ap.add_argument("--pick-dialog", action="store_true")
    ap.add_argument("--label", default="Start(0–20m)")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)

    # 動画選択
    video = args.video
    if args.pick_dialog or not video:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        video = filedialog.askopenfilename(
            initialdir=cfg.get("pick_default_dir", "videos"),
            filetypes=[("Video", "*.mp4 *.mov *.avi")]
        )
        r.destroy()
    if not video:
        raise FileNotFoundError("動画が選択されていません")

    athlete, date = _parse_filename(video)
    print(f"[INFO] 選手: {athlete}, 日付タグ: {date}")

    # 校正
    mpp, px, rm = calibrate_interactive(video)
    save_scale("scales", "one_stage_default", mpp, px, rm)
    print(f"[Calib] 1 px = {mpp:.6f} m ({rm}m/{px:.1f}px)")

    # 出力フォルダ
    out_root = _ensure_dir(cfg.get("output_dir", "outputs"))
    athlete_dir = _ensure_dir(os.path.join(out_root, athlete))
    frames_dir = _ensure_dir(os.path.join(athlete_dir, "frames"))

    cap = _open_video_any_backend(video)
    if not cap.isOpened():
        raise RuntimeError("動画を開けません")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    start = cfg.get("start_sec", 0.0)
    step = cfg.get("step_sec", 0.1)
    ok0, f0 = _frame_at_time(cap, start, fps)
    prev = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    prev_pts = None

    times, speeds, pages = [], [], []
    vpeak = 0
    for t in _iter_times(start, total, step):
        ok, fr = _frame_at_time(cap, t, fps)
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        v_px, prev_pts, dx, dy = flow_metrics(prev, gray, step, prev_pts)
        prev = gray
        v_mps = float(v_px) * float(mpp)
        v_kmh = v_mps * 3.6
        vpeak = max(vpeak, v_px)
        phase = classify_phase(v_px, vpeak, None, None)
        draw_guides(fr)
        put_label(fr, f"{args.label} | {phase} | t={t:.1f}s")

        # --- ✅ 拡張子付きで保存 ---
        img_name = f"t_{str(t).replace('.', '_')}.png"
        img = os.path.join(frames_dir, img_name)
        Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).save(img)

        pages.append((t, phase, img, v_mps, v_kmh))
        times.append(t)
        speeds.append(v_mps)
    cap.release()

    speeds_s = _kalman_like(speeds)
    ai_summary = generate_ai_summary(
        times=list(times),
        speeds_ms=list(speeds_s),
        tilts=None,
        phases=None,
        model=cfg.get("openai_model", "gpt-4o-mini"),
        temperature=cfg.get("ai_temperature", 0.5),
        max_tokens=700,
    )

    pdf = os.path.join(athlete_dir, f"{athlete}{date}.pdf")
    make_pdf_one_stage(pdf, pages, times, speeds_s, cfg, args.label, ai_summary)
    print(f"[OK] PDF出力: {pdf}")


if __name__ == "__main__":
    main()

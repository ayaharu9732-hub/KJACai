# -*- coding: utf-8 -*-
"""
cli_one_stage.py v1.6
1本の動画を解析し、Start(0–20m)などのPDFレポートを生成。
動画名から選手名と日付タグを自動抽出し、outputs/{athlete}/{athlete}{date_tag}.pdf に保存。
"""

import argparse
import os
import re
import cv2
import yaml
import numpy as np
from datetime import datetime

from .calibration import load_scale, save_scale, calibrate_interactive
from .analyzer import draw_guides, put_label, flow_metrics, get_trunk_tilt_deg, classify_phase
from .notes import generate_ai_summary


# ========== Utility ==========
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _read_cfg(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _open_video_any_backend(path):
    backends = [
        cv2.CAP_ANY,
        getattr(cv2, "CAP_MSMF", 0),
        getattr(cv2, "CAP_FFMPEG", 1900),
        getattr(cv2, "CAP_DSHOW", 700),
    ]
    for be in backends:
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
    step = max(1e-6, float(step_sec))
    while t <= total_sec + 1e-9:
        yield round(t, 3)
        t += step


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


# ========== 新機能：動画名から選手名と日付を抽出 ==========
def parse_filename_for_info(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    base = base.replace(" ", "").replace("_", "")

    # パターン例: 小池季10.7, 二村綾香20251018 など
    m = re.match(r"^([^\d]+)([\d\.]+)$", base)
    if m:
        athlete = m.group(1)
        date_tag = m.group(2)
    else:
        athlete = base
        date_tag = datetime.now().strftime("%Y%m%d")
    return athlete, date_tag


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

    try:
        import matplotlib as mpl
        mpl.rcParams["font.sans-serif"] = ["IPAexGothic"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    c = canvas.Canvas(pdf_path, pagesize=landscape(A4))
    W, H = landscape(A4)

    def draw_paragraph(x, y, text, width=760, leading=18, size=12):
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet

        if isinstance(text, (list, tuple)):
            text = "\n\n".join([str(t) for t in text])
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

    def draw_speed_plot():
        if not times or not speeds_ms:
            return
        import matplotlib.pyplot as plt
        import tempfile

        fig = plt.figure(figsize=(8, 3), dpi=150)
        plt.plot(times, speeds_ms, color="blue", label="Speed (m/s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("速度プロファイル")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        tmp_plot = os.path.join(os.path.dirname(pdf_path), "_tmp_plot.png")
        fig.savefig(tmp_plot)
        plt.close(fig)

        c.setFont(font_jp, 18)
        c.drawString(40, H - 60, f"{label} : 速度プロファイル")
        c.drawImage(tmp_plot, 40, 140, width=760, height=360, preserveAspectRatio=True)
        os.remove(tmp_plot)
        c.showPage()

    # 1ページ目：フレーム
    c.setFont(font_jp, 20)
    c.drawString(40, H - 60, f"動画レポート：{label}")
    y = H - 90
    for (t, phase, img_path, v_px, v_ms, v_kmh, tilt, notes) in pages:
        c.setFont(font_jp, 11)
        c.drawString(40, y, f"t={t:.1f}s | Phase: {phase} | Speed: {v_ms:.2f} m/s")
        try:
            c.drawImage(img_path, 380, y - 110, width=360, height=100, preserveAspectRatio=True)
        except Exception:
            pass
        y -= 130
        if y < 160:
            c.showPage()
            c.setFont(font_jp, 20)
            c.drawString(40, H - 60, f"動画レポート：{label}")
            y = H - 90
    c.showPage()

    # 2ページ目：速度プロット
    draw_speed_plot()

    # 3ページ目：AIサマリー
    c.setFont(font_jp, 18)
    c.drawString(40, H - 60, "AI分析レポート（サマリー）")
    draw_paragraph(40, H - 90, ai_summary_text, width=760)
    c.showPage()
    c.save()


# ========== Main ==========
def main():
    ap = argparse.ArgumentParser(description="1本の動画を解析し、PDFに出力")
    ap.add_argument("--config", required=True, help="config.yaml")
    ap.add_argument("--video", help="動画パス（省略時はダイアログ）")
    ap.add_argument("--pick-dialog", action="store_true", help="ダイアログで動画を選ぶ")
    ap.add_argument("--force-calibrate", action="store_true", help="毎回キャリブレーション")
    ap.add_argument("--label", default="Start(0–20m)", help="レポート見出し")
    ap.add_argument("--athlete", help="選手名（省略時は動画名から自動抽出）")
    ap.add_argument("--date_tag", help="日付タグ（省略時は動画名から自動抽出）")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)

    # 動画選択
    video = args.video
    if args.pick_dialog or not video:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        base = cfg.get("pick_default_dir") or os.getcwd()
        video = filedialog.askopenfilename(initialdir=base, filetypes=[("Video", "*.mp4 *.mov *.avi")])
        root.destroy()

    if not video or not os.path.exists(video):
        raise FileNotFoundError(f"動画が見つかりません: {video}")

    # 選手名・日付を自動抽出（手動指定があればそれを優先）
    athlete, date_tag = args.athlete, args.date_tag
    if not athlete or not date_tag:
        auto_ath, auto_date = parse_filename_for_info(video)
        athlete = athlete or auto_ath
        date_tag = date_tag or auto_date
    print(f"[INFO] 選手: {athlete}, 日付タグ: {date_tag}")

    # 校正
    scales_dir = _ensure_dir("scales")
    cam_name = "one_stage_default"
    m_per_px, raw = load_scale(scales_dir, cam_name)
    if args.force_calibrate or m_per_px is None:
        mpp, px, rm = calibrate_interactive(video)
        raw = save_scale(scales_dir, cam_name, mpp, px, rm)
        m_per_px = mpp
    print(f"[Calib] 1 px = {m_per_px:.6f} m  ({raw})")

    # 解析開始
    cap = _open_video_any_backend(video)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_sec = (total_frames - 1) / fps if total_frames > 0 else 0.0

    start_sec = float(cfg.get("start_sec", 0.0))
    step_sec = float(cfg.get("step_sec", 0.1))

    out_root = _ensure_dir(cfg.get("output_dir", "outputs"))
    athlete_dir = _ensure_dir(os.path.join(out_root, athlete))
    out_dir = _ensure_dir(os.path.join(athlete_dir, f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    prev_gray = None
    prev_pts = None
    pages, times, speeds_ms, speeds_px = [], [], [], []
    v_peak_px = 0.0

    ok0, frame0 = _frame_at_time(cap, start_sec, fps)
    if not ok0:
        raise RuntimeError("開始フレーム取得に失敗")
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    for t in _iter_times(start_sec, total_sec, step_sec):
        ok, frame = _frame_at_time(cap, t, fps)
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v_px_s, prev_pts, dx_std, dy_std = flow_metrics(prev_gray, gray, step_sec, prev_pts)
        prev_gray = gray

        v_ms = _to_mps(v_px_s, m_per_px)
        v_kmh = v_ms * 3.6
        dv_px = None if len(speeds_px) == 0 else (v_px_s - speeds_px[-1]) / max(step_sec, 1e-6)
        v_peak_px = max(v_peak_px, v_px_s)
        phase = classify_phase(v_px_s, v_peak_px, dv_px, tilt=None)

        annotated = frame.copy()
        draw_guides(annotated)
        put_label(annotated, f"{args.label} | {phase} | t={t:.1f}s")
        img_path = os.path.join(out_dir, f"t_{t:.1f}.jpg")
        cv2.imwrite(img_path, annotated)

        pages.append((t, phase, img_path, v_px_s, v_ms, v_kmh, None, []))
        times.append(t)
        speeds_ms.append(v_ms)
        speeds_px.append(v_px_s)

    cap.release()

    speeds_ms_smooth = _kalman_like(speeds_ms) if speeds_ms else []
    ai_summary_text = ""
    if bool(cfg.get("use_ai", True)):
        try:
            ai_summary_text = generate_ai_summary(
                times=times,
                speeds_ms=speeds_ms_smooth,
                tilts=None,
                phases=None,
                model=cfg.get("openai_model", "gpt-4o-mini"),
                temperature=float(cfg.get("ai_temperature", 0.4)),
                max_tokens=700,
            )
        except Exception:
            ai_summary_text = "（AIサマリー生成に失敗しました）"

    pdf_path = os.path.join(athlete_dir, f"{athlete}{date_tag}.pdf")
    make_pdf_one_stage(pdf_path, pages, times, speeds_ms_smooth, cfg, label=args.label, ai_summary_text=ai_summary_text)
    print(f"[OK] 出力完了: {pdf_path}")


if __name__ == "__main__":
    main()





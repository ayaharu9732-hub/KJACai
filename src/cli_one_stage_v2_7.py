# -*- coding: utf-8 -*-
"""
cli_one_stage_v2_7.py
KJAC AI 完全安定版（2025-11）
-------------------------------------
機能一覧：
- 動画名から選手名と日付を自動抽出（例：二村遥香10.5.mp4 → athlete=二村遥香, date_tag=10.5）
- 校正時：必ず距離[m]入力あり
- 画像は縮小+JPEG圧縮保存でPDF軽量化（configで倍率・品質設定可能）
- 速度グラフ黒文字化＆日本語フォント使用
- AI分析レポートをKJAC-AI風自然文へ整形
-------------------------------------
出力：
outputs/<選手名>/<選手名><日付タグ>.pdf
"""

import argparse
import os
import re
import cv2
import yaml
import numpy as np
from datetime import datetime
from PIL import Image

# 既存モジュール
from .calibration import calibrate_interactive, save_scale
from .analyzer import draw_guides, put_label, flow_metrics, classify_phase
from .notes import generate_ai_summary


# ========= ユーティリティ =========
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _read_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _open_video_any_backend(path: str):
    for be in [cv2.CAP_ANY, getattr(cv2, "CAP_MSMF", 0), getattr(cv2, "CAP_FFMPEG", 1900), getattr(cv2, "CAP_DSHOW", 700)]:
        cap = cv2.VideoCapture(path, be)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(path)


def _frame_at_time(cap, t_sec: float, fps: float):
    idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
    return cap.read()


def _iter_times(start_sec: float, total_sec: float, step_sec: float):
    t = max(0.0, float(start_sec))
    step = max(1e-6, float(step_sec))
    while t <= total_sec + 1e-9:
        yield round(t, 3)
        t += step


def _kalman_like(v):
    a = 0.3
    y = []
    for i, x in enumerate(v):
        y.append(x if i == 0 else a * x + (1 - a) * y[-1])
    return np.array(y, dtype=float)


def _to_mps(px_per_s, m_per_px):
    return float(px_per_s) * float(m_per_px)


def _parse_filename(video_path: str):
    base = os.path.splitext(os.path.basename(video_path))[0]
    base_compact = base.replace(" ", "").replace("_", "")
    m = re.match(r"^([^\d]+)([\d.]+)$", base_compact)
    if m:
        athlete = m.group(1).strip()
        date_tag = m.group(2).strip().rstrip(".")
        return athlete, date_tag
    return base, ""


# ========= PDF生成 =========
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
            print(f"[WARN] 画像描画失敗: {img_path} ({e})")

    # --- 速度プロファイル ---
    def draw_speed_plot():
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "IPAexGothic"
        if len(times) == 0 or len(speeds_ms) == 0:
            return
        fig = plt.figure(figsize=(8, 3), dpi=110)
        plt.plot(times, speeds_ms, color="royalblue", linewidth=2.2)
        plt.xlabel("Time (s)", fontsize=10)
        plt.ylabel("Speed (m/s)", fontsize=10)
        plt.title(f"{label}：速度プロファイル", fontsize=13, color="black", pad=10)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()

        tmp_png = os.path.join(os.path.dirname(pdf_path), "_plot_tmp.png")
        fig.savefig(tmp_png, dpi=110)
        plt.close(fig)

        tmp_jpg = os.path.join(os.path.dirname(pdf_path), "_plot_tmp.jpg")
        q = int(cfg.get("jpeg_quality", 70))
        with Image.open(tmp_png) as im:
            im = im.convert("RGB")
            im.save(tmp_jpg, format="JPEG", quality=q, optimize=True, subsampling=2, progressive=True)
        try:
            os.remove(tmp_png)
        except Exception:
            pass

        c.setFont(font_jp, 18)
        c.drawString(40, H - 60, f"{label} : 速度プロファイル")
        _draw_image_safe(tmp_jpg, 40, 140, 760, 360)
        c.showPage()
        try:
            os.remove(tmp_jpg)
        except Exception:
            pass

    # --- 段落描画 ---
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

    # --- ページ1：静止フレーム ---
    c.setFont(font_jp, 20)
    c.drawString(40, H - 60, f"KJAC AI Report：{label}")
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
            c.drawString(40, H - 60, f"KJAC AI Report：{label}")
            y = H - 90
    c.showPage()

    # --- ページ2：速度プロファイル ---
    draw_speed_plot()

    # --- ページ3：AI分析 ---
    c.setFont(font_jp, 18)
    c.drawString(40, H - 60, "KJAC-AI 自動解析レポート")
    draw_paragraph(40, H - 90, ai_summary_text)
    c.showPage()
    c.save()


# ========= AIサマリー整形 =========
def kjac_style_summary(raw_text, athlete, label):
    import re
    body = re.sub(r"#+\s*", "", raw_text)
    body = body.replace("**", "").replace("##", "")
    intro = f"【KJAC-AI 自動解析レポート】\n走者名：{athlete}\n対象区間：{label}\n\n"
    outro = (
        "\n---\nKJAC AIシステムは、スプリント動作を客観的に可視化し、"
        "データに基づいた成長支援を提供します。"
    )
    return intro + body.strip() + outro


# ========= メイン =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", help="動画パス")
    ap.add_argument("--pick-dialog", action="store_true")
    ap.add_argument("--label", default="Start(0–20m)")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)
    frame_downscale = float(cfg.get("frame_downscale", 0.25))
    jpeg_quality = int(cfg.get("jpeg_quality", 70))

    # --- 動画選択 ---
    video = args.video
    if args.pick_dialog or not video:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        base = cfg.get("pick_default_dir") or os.getcwd()
        video = filedialog.askopenfilename(initialdir=base, filetypes=[("Video", "*.mp4 *.mov *.avi *.MP4 *.MOV *.AVI")])
        root.destroy()

    if not video or not os.path.exists(video):
        raise FileNotFoundError(f"動画が見つかりません: {video}")

    athlete, date_tag = _parse_filename(video)
    print(f"[INFO] 選手: {athlete}, 日付タグ: {date_tag}")

    # --- 校正 ---
    scales_dir = _ensure_dir("scales")
    cam_name = "one_stage_default"
    mpp, px, rm = calibrate_interactive(video)
    save_scale(scales_dir, cam_name, mpp, px, rm)
    print(f"[Calib] 1 px = {mpp:.6f} m ({rm}m/{px:.1f}px)")

    # --- 出力構成 ---
    out_root = _ensure_dir(cfg.get("output_dir", "outputs"))
    athlete_dir = _ensure_dir(os.path.join(out_root, athlete))
    frames_dir = _ensure_dir(os.path.join(athlete_dir, "frames"))

    cap = _open_video_any_backend(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = (total_frames - 1) / fps if total_frames > 0 else 0.0
    step_sec = float(cfg.get("step_sec", 0.1))
    start_sec = float(cfg.get("start_sec", 0.0))

    ok0, frame0 = _frame_at_time(cap, start_sec, fps)
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_pts = None

    times, speeds_ms, speeds_px, pages = [], [], [], []
    v_peak_px = 0.0

    for t in _iter_times(start_sec, total_sec, step_sec):
        ok, frame = _frame_at_time(cap, t, fps)
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        v_px_s, prev_pts, dx_std, dy_std = flow_metrics(prev_gray, gray, step_sec, prev_pts)
        prev_gray = gray

        v_ms = _to_mps(v_px_s, mpp)
        v_kmh = v_ms * 3.6
        v_peak_px = max(v_peak_px, v_px_s)
        dv_px = None if len(speeds_px) == 0 else (v_px_s - speeds_px[-1]) / max(step_sec, 1e-6)
        phase = classify_phase(v_px_s, v_peak_px, dv_px, tilt=None)

        draw_guides(frame)
        put_label(frame, f"{args.label} | {phase} | t={t:.1f}s")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (int(w * frame_downscale), int(h * frame_downscale)), interpolation=cv2.INTER_AREA)

        img_name = f"t_{str(t).replace('.', '_')}.jpg"
        img_path = os.path.join(frames_dir, img_name)
        Image.fromarray(small).save(img_path, "JPEG", quality=jpeg_quality, optimize=True, subsampling=2)
        pages.append((t, phase, img_path, v_ms, v_kmh))
        times.append(t)
        speeds_px.append(v_px_s)
        speeds_ms.append(v_ms)

    cap.release()

    speeds_ms_smooth = _kalman_like(speeds_ms)
    try:
        ai_summary_text = generate_ai_summary(
            times=list(times),
            speeds_ms=list(speeds_ms_smooth),
            tilts=None,
            phases=None,
            model=cfg.get("openai_model", "gpt-4o-mini"),
            temperature=float(cfg.get("ai_temperature", 0.5)),
            max_tokens=700,
        )
        ai_summary_text = kjac_style_summary(ai_summary_text, athlete, args.label)
    except Exception as e:
        ai_summary_text = f"AIサマリー生成に失敗しました: {e}"

    pdf_path = os.path.join(athlete_dir, f"{athlete}{date_tag}.pdf")
    make_pdf_one_stage(pdf_path, pages, times, speeds_ms_smooth, cfg, label=args.label, ai_summary_text=ai_summary_text)
    print(f"[OK] 出力完了: {pdf_path}")


if __name__ == "__main__":
    main()

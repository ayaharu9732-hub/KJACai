# -*- coding: utf-8 -*-
"""
cli_one_stage_v2_0.py
Start(0–20m) 専用 — 毎回キャリブレーション（距離入力必須）
動画名から <選手名><日付タグ> を抽出し、PDFを
outputs/<選手名>/<選手名><日付タグ>.pdf に出力。

改良点:
- NumPy配列の真偽判定バグ修正（len()/tolist()対応）
- auto_trim対応（config.yamlでON/OFF, 閾値指定）
- .jpg保存名の拡張子欠落バグ修正
- ファイル名からの選手名/日付タグ抽出を安定化
"""

import argparse
import os
import re
import cv2
import yaml
import numpy as np
from datetime import datetime

# 既存モジュール
from .calibration import save_scale, calibrate_interactive
from .analyzer import draw_guides, put_label, flow_metrics, classify_phase
from .notes import generate_ai_summary


# ========= ユーティリティ =========
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _read_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
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

def _ema(x, alpha=0.25):
    if x is None:
        return np.array([])
    if isinstance(x, list):
        arr = np.array(x, dtype=float)
    elif isinstance(x, np.ndarray):
        arr = x.astype(float)
    else:
        arr = np.array([float(v) for v in x], dtype=float)
    if arr.size == 0:
        return np.array([])
    y = [arr[0]]
    for v in arr[1:]:
        y.append(alpha * v + (1 - alpha) * y[-1])
    return np.array(y, dtype=float)

def _kalman_like(v):
    return _ema(v, alpha=0.3)

def _to_mps(px_per_s, m_per_px):
    return float(px_per_s) * float(m_per_px)

def _parse_filename(video_path: str):
    """動画名から選手名と日付タグを抽出。例: '二村遥香10.5.mp4' -> ('二村遥香', '10.5')"""
    base = os.path.splitext(os.path.basename(video_path))[0]
    # 空白/アンダースコアは取り除いて判定（表示用は元を尊重してもOK）
    base_compact = base.replace(" ", "").replace("_", "")
    m = re.match(r"^([^\d]+)([\d.]+)$", base_compact)
    if m:
        athlete = m.group(1).strip()
        date_tag = m.group(2).strip().rstrip(".")
        return athlete, date_tag
    # 数字が見つからない場合のフォールバック
    return base, datetime.now().strftime("%m%d")


# ========= PDF生成 =========
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
        try:
            c.drawImage(tmp_plot, 40, 140, width=760, height=360, preserveAspectRatio=True)
        except Exception:
            pass
        try:
            os.remove(tmp_plot)
        except Exception:
            pass
        c.showPage()

    def draw_paragraph(x, y, text, width=760, leading=18, size=12):
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        if isinstance(text, (list, tuple, np.ndarray)):
            text = "\n\n".join([str(t) for t in list(text)])
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


# ========= メイン =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video", help="動画パス（省略時はダイアログ）")
    ap.add_argument("--pick-dialog", action="store_true")
    ap.add_argument("--force-calibrate", action="store_true")  # 互換のため残すが毎回入力する仕様
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
        video = filedialog.askopenfilename(
            initialdir=base,
            filetypes=[("Video", "*.mp4 *.mov *.avi *.MP4 *.MOV *.AVI")]
        )
        root.destroy()

    if not video or not os.path.exists(video):
        raise FileNotFoundError(f"動画が見つかりません: {video}")

    # --- 選手名と日付タグを抽出 ---
    athlete, date_tag = _parse_filename(video)
    print(f"[INFO] 選手: {athlete}, 日付タグ: {date_tag}")

    # --- キャリブレーション（毎回距離入力あり） ---
    scales_dir = _ensure_dir("scales")
    cam_name = "one_stage_default"
    mpp, px, rm = calibrate_interactive(video)   # 必ず対話で距離入力
    raw = save_scale(scales_dir, cam_name, mpp, px, rm)  # 履歴保存（次回も入力は求める）
    m_per_px = mpp
    print(f"[Calib] 1 px = {m_per_px:.6f} m  ({raw})")

    # --- 出力構成 ---
    out_root = _ensure_dir(cfg.get("output_dir", "outputs"))
    athlete_dir = _ensure_dir(os.path.join(out_root, athlete))
    frames_dir = _ensure_dir(os.path.join(athlete_dir, "frames"))

    # --- 動画解析 ---
    cap = _open_video_any_backend(video)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = (total_frames - 1) / fps if total_frames > 0 else 0.0

    step_sec = float(cfg.get("step_sec", 0.1))
    start_sec = float(cfg.get("start_sec", 0.0))
    auto_trim = bool(cfg.get("auto_trim", False))
    trim_th = float(cfg.get("auto_trim_threshold_px_s", 80.0))

    ok0, frame0 = _frame_at_time(cap, start_sec, fps)
    if not ok0 or frame0 is None:
        cap.release()
        raise RuntimeError("開始フレーム取得に失敗しました。")
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_pts = None

    times, speeds_ms, speeds_px = [], [], []
    v_peak_px = 0.0
    started = not auto_trim
    consec = 0

    pages = []

    for t in _iter_times(start_sec, total_sec, step_sec):
        ok, frame = _frame_at_time(cap, t, fps)
        if not ok or frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        v_px_s, prev_pts, dx_std, dy_std = flow_metrics(prev_gray, gray, step_sec, prev_pts)
        prev_gray = gray

        # auto_trim: 動きが始まるまでスキップ
        if auto_trim and not started:
            if v_px_s > trim_th:
                consec += 1
                if consec >= 3:
                    started = True
            else:
                consec = 0
            if not started:
                continue

        v_ms = _to_mps(v_px_s, m_per_px)
        v_kmh = v_ms * 3.6

        dv_px = None if len(speeds_px) == 0 else (v_px_s - speeds_px[-1]) / max(step_sec, 1e-6)
        v_peak_px = max(v_peak_px, v_px_s)
        phase = classify_phase(v_px_s, v_peak_px, dv_px, tilt=None)

        # フレーム保存
        draw_guides(frame)
        put_label(frame, f"{args.label} | {phase} | t={t:.1f}s")
        t_str = f"{t:.1f}".replace(".", "_")
        img_path = os.path.join(frames_dir, f"t_{t_str}.jpg")
        cv2.imwrite(img_path, frame)

        pages.append((t, phase, img_path, v_ms, v_kmh))
        times.append(t)
        speeds_px.append(v_px_s)
        speeds_ms.append(v_ms)

    cap.release()

    # --- 速度スムージング & AIサマリー ---
    speeds_ms_smooth = _kalman_like(speeds_ms)
    # NumPy -> list へ変換（notes側で安全化していても、こちらでも万全を期す）
    times_list = times if isinstance(times, list) else list(times)
    speeds_list = (
        speeds_ms_smooth.tolist()
        if isinstance(speeds_ms_smooth, np.ndarray)
        else list(speeds_ms_smooth)
    )

    if len(times_list) == 0 or len(speeds_list) == 0:
        ai_summary_text = "速度データが存在しないため、AIサマリーはスキップしました。"
    else:
        try:
            ai_summary_text = generate_ai_summary(
                times=times_list,
                speeds_ms=speeds_list,
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
    make_pdf_one_stage(pdf_path, pages, times_list, speeds_list, cfg, label=args.label, ai_summary_text=ai_summary_text)
    print(f"[OK] 出力完了: {pdf_path}")


if __name__ == "__main__":
    main()


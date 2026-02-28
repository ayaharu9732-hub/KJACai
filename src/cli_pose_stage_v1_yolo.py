# -*- coding: utf-8 -*-
"""
cli_pose_stage_v1_yolo.py
KJAC-AI | YOLOv8-Pose（現場用・軽量高速）版
- 動画名から 選手名/日付タグ を自動抽出（例：二村遥香10.5.mp4 → 二村遥香, 10.5）
- 校正（クリック2点＋距離入力）で m/px を取得 → 速度[m/s]算出に使用
- YOLOv8n-pose で骨格推定（CPUでも実用速度）
- 各フレームに骨格オーバーレイ＋速度・フェーズラベル
- 画像は縮小＋JPEG保存でPDF軽量化（config: frame_downscale / jpeg_quality）
- PDFは v2_7 と同体裁（Page1: フレーム, Page2: 速度プロファイル, Page3: AIサマリー）
"""

import argparse
import os
import re
import cv2
import yaml
import numpy as np
from PIL import Image
from datetime import datetime

# 既存モジュール（このリポジトリにあるもの）
from .calibration import calibrate_interactive, save_scale
from .analyzer import draw_guides, put_label, flow_metrics, classify_phase
from .notes import generate_ai_summary

# ====== ユーティリティ ======
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

def _ema(v, a=0.3):
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

# ====== 幾何ユーティリティ（角度など） ======
def _angle_deg(a, b, c):
    """3点（a-b-c）でb角の角度を度で返す。"""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba); nc = np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6: return None
    cosang = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _line_angle_deg(p1, p2):
    """p1→p2 の線分が水平方向となす角（度）。右上がり＋、右下がり−。"""
    p1 = np.array(p1, float); p2 = np.array(p2, float)
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6: return None
    rad = np.arctan2(-(dy), dx)  # 画像座標→y下向きなのでマイナス
    return float(np.degrees(rad))

# ====== YOLOv8-Pose 推定器 ======
def _load_yolo_pose(model_path: str | None = None):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "ultralytics が見つかりません。以下でインストールしてください：\n"
            "  pip install ultralytics\n"
            "（初回は weights 自動DL あり）"
        ) from e
    mdl = YOLO(model_path or "yolov8n-pose.pt")
    return mdl

def _yolo_pose_once(model, frame_bgr):
    """1枚推論。戻り値：(annotated_bgr, keypoints_xy[25,2] or None)"""
    res = model(frame_bgr, verbose=False)
    r0 = res[0]
    annotated = r0.plot()  # 骨格オーバーレイ済み BGR
    # 最も大きい人物（ボックス面積）を採用
    kps = None
    if r0.keypoints is not None and len(r0.keypoints) > 0:
        boxes = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else None
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) if boxes is not None else None
        idx = int(np.argmax(areas)) if areas is not None and areas.size>0 else 0
        kps = r0.keypoints.xy[idx].cpu().numpy()  # (K,2)
    return annotated, kps

# ====== PDF生成 ======
def _make_pdf(pdf_path, pages, times, speeds_ms_smooth, cfg, label, ai_summary_text):
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

    def _draw_img(path, x, y, w=360, h=110):
        try:
            c.drawImage(path, x, y, width=w, height=h, preserveAspectRatio=True)
        except Exception as e:
            print("[WARN] drawImage:", e)

    # 速度プロット（JPEGで軽量化）
    def draw_speed_plot():
        import matplotlib.pyplot as plt
        import os
        plt.rcParams["font.family"] = "IPAexGothic"
        if len(times) == 0 or len(speeds_ms_smooth) == 0: return
        fig = plt.figure(figsize=(8, 3), dpi=110)
        plt.plot(times, speeds_ms_smooth, linewidth=2.2)
        plt.xlabel("Time (s)"); plt.ylabel("Speed (m/s)")
        plt.title(f"{label}：速度プロファイル", color="black", fontsize=13, pad=10)
        plt.grid(True, linestyle="--", alpha=0.3); plt.tight_layout()
        tmp_png = os.path.join(os.path.dirname(pdf_path), "_plot_tmp.png")
        fig.savefig(tmp_png, dpi=110); plt.close(fig)

        from PIL import Image
        tmp_jpg = os.path.join(os.path.dirname(pdf_path), "_plot_tmp.jpg")
        q = int(cfg.get("jpeg_quality", 70))
        with Image.open(tmp_png) as im:
            im = im.convert("RGB")
            im.save(tmp_jpg, "JPEG", quality=q, optimize=True, subsampling=2, progressive=True)
        try: os.remove(tmp_png)
        except: pass

        c.setFont(font_jp, 18); c.drawString(40, H-60, f"{label} : 速度プロファイル")
        _draw_img(tmp_jpg, 40, 140, 760, 360)
        c.showPage()
        try: os.remove(tmp_jpg)
        except: pass

    def draw_paragraph(x, y, text, width=760, size=12):
        styles = getSampleStyleSheet()
        st = styles["Normal"]; st.fontName = font_jp; st.fontSize = size; st.leading = 18
        p = Paragraph(str(text).replace("\n", "<br/>"), st)
        w, h = p.wrap(width, 9999); p.drawOn(c, x, y - h); return h

    # Page 1: フレーム
    c.setFont(font_jp, 20); c.drawString(40, H-60, f"KJAC AI Pose Report：{label}")
    y = H - 90
    for (t, phase, img_path, v_ms, v_kmh, note_lines) in pages:
        c.setFont(font_jp, 11)
        c.drawString(40, y, f"t={t:.1f}s | Phase: {phase}")
        c.drawString(40, y-16, f"Speed: {v_ms:.2f} m/s ({v_kmh:.1f} km/h)")
        yy = y - 36
        for ln in note_lines[:4]:
            c.drawString(40, yy, f"・{ln}"); yy -= 14
        _draw_img(img_path, 420, y-120)
        y -= 140
        if y < 160:
            c.showPage()
            c.setFont(font_jp, 20); c.drawString(40, H-60, f"KJAC AI Pose Report：{label}")
            y = H - 90
    c.showPage()

    # Page 2: 速度プロファイル
    draw_speed_plot()

    # Page 3: AIサマリー
    c.setFont(font_jp, 18); c.drawString(40, H-60, "KJAC-AI 姿勢＋速度 分析サマリー")
    draw_paragraph(40, H-90, ai_summary_text)
    c.showPage()
    c.save()

# ====== AIサマリー整形（KJAC風） ======
def _kjac_style_summary(speed_report_text: str, pose_digest: str, athlete: str, label: str):
    intro = f"【KJAC-AI 自動解析レポート】\n走者名：{athlete}\n対象区間：{label}\n\n"
    body = ""
    if pose_digest:
        body += "■ 姿勢の観察ポイント\n" + pose_digest.strip() + "\n\n"
    if speed_report_text:
        body += "■ 速度プロファイル所見\n" + speed_report_text.strip() + "\n"
    outro = "\n---\nKJAC AIは、現場の素早い確認（YOLOv8-Pose）と自宅での精密分析（OpenPose）の両輪で、成長につながる“気づき”を提供します。"
    return intro + body + outro

# ====== メイン ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video")
    ap.add_argument("--pick-dialog", action="store_true")
    ap.add_argument("--label", default="Start(0–20m)")
    ap.add_argument("--yolo-model", default=None, help="yolov8n-pose.pt など（未指定なら自動DL/既存キャッシュ使用）")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)
    frame_downscale = float(cfg.get("frame_downscale", 0.25))
    jpeg_quality = int(cfg.get("jpeg_quality", 70))

    # 動画選択
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

    athlete, date_tag = _parse_filename(video)
    print(f"[INFO] 選手: {athlete}, 日付タグ: {date_tag}")

    # 校正（毎回距離入力）
    scales_dir = _ensure_dir("scales")
    mpp, px, rm = calibrate_interactive(video)
    save_scale(scales_dir, "one_stage_default", mpp, px, rm)
    print(f"[Calib] 1 px = {mpp:.6f} m ({rm}m/{px:.1f}px)")

    # 出力
    out_root = _ensure_dir(cfg.get("output_dir", "outputs"))
    athlete_dir = _ensure_dir(os.path.join(out_root, athlete))
    frames_dir = _ensure_dir(os.path.join(athlete_dir, "frames"))

    # YOLOv8-Pose 準備
    model = _load_yolo_pose(args.yolo_model)

    # 動画解析
    cap = _open_video_any_backend(video)
    if not cap.isOpened():
        raise RuntimeError("動画を開けません")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = (total_frames - 1) / fps if total_frames > 0 else 0.0
    start_sec = float(cfg.get("start_sec", 0.0))
    step_sec = float(cfg.get("step_sec", 0.1))

    ok0, frame0 = _frame_at_time(cap, start_sec, fps)
    if not ok0 or frame0 is None:
        cap.release(); raise RuntimeError("開始フレーム取得に失敗")

    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_pts = None

    times, speeds_ms, speeds_px = [], [], []
    v_peak_px = 0.0
    pages = []

    # 代表姿勢ダイジェストのための集計
    trunk_angles = []
    kneeL, kneeR = [], []
    ankle_dist_m = []  # 左右足首の距離（簡易指標）

    for t in _iter_times(start_sec, total_sec, step_sec):
        ok, frame = _frame_at_time(cap, t, fps)
        if not ok or frame is None:
            break

        # 速度（光学フロー）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v_px_s, prev_pts, dx_std, dy_std = flow_metrics(prev_gray, gray, step_sec, prev_pts)
        prev_gray = gray
        v_ms  = _to_mps(v_px_s, mpp)
        v_kmh = v_ms * 3.6
        v_peak_px = max(v_peak_px, v_px_s)
        dv_px = None if len(speeds_px)==0 else (v_px_s - speeds_px[-1]) / max(step_sec,1e-6)
        phase = classify_phase(v_px_s, v_peak_px, dv_px, tilt=None)

        # YOLO Pose 推定＆オーバーレイ
        annotated, kps = _yolo_pose_once(model, frame)
        draw_guides(annotated)
        put_label(annotated, f"{args.label} | {phase} | t={t:.1f}s")

        # 代表的な姿勢指標を収集（肩/腰/膝/足首）
        note_lines = []
        if kps is not None and kps.shape[0] >= 17:
            # COCO準拠: 5=RShoulder,6=LShoulder, 11=RHip,12=LHip, 13=RKnee,14=LKnee, 15=RAnkle,16=LAnkle（yolov8は17点）
            try:
                rs, ls = kps[5], kps[6]
                rh, lh = kps[11], kps[12]
                rk, lk = kps[13], kps[14]
                ra, la = kps[15], kps[16]
                sh_mid = (rs + ls) / 2.0
                hip_mid = (rh + lh) / 2.0
                trunk_deg = _line_angle_deg(hip_mid, sh_mid)  # 右向き水平=0°, 前傾で正の値になるよう設定
                if trunk_deg is not None:
                    trunk_angles.append(trunk_deg)
                    note_lines.append(f"体幹角: {trunk_deg:.1f}°")

                ang_knee_r = _angle_deg(rh, rk, ra)
                ang_knee_l = _angle_deg(lh, lk, la)
                if ang_knee_r is not None: kneeR.append(ang_knee_r)
                if ang_knee_l is not None: kneeL.append(ang_knee_l)
                if ang_knee_r is not None and ang_knee_l is not None:
                    note_lines.append(f"膝角(R/L): {ang_knee_r:.0f}° / {ang_knee_l:.0f}°")

                # 左右足首間距離（m換算・簡易）
                d_ankle_px = float(np.linalg.norm(ra - la))
                d_ankle_m  = d_ankle_px * float(mpp)
                ankle_dist_m.append(d_ankle_m)
                note_lines.append(f"左右足首距離: {d_ankle_m:.2f} m")
            except Exception:
                pass

        # 軽量画像保存（JPEG, 縮小）
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        new_w, new_h = max(1, int(w * frame_downscale)), max(1, int(h * frame_downscale))
        small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_name = f"t_{str(t).replace('.', '_')}.jpg"
        img_path = os.path.join(frames_dir, img_name)
        Image.fromarray(small).save(img_path, "JPEG", quality=jpeg_quality, optimize=True, subsampling=2, progressive=True)

        pages.append((t, phase, img_path, v_ms, v_kmh, note_lines))
        times.append(t); speeds_px.append(v_px_s); speeds_ms.append(v_ms)

    cap.release()

    # スピードAIサマリー（notes.py利用）＋姿勢ダイジェスト
    speeds_ms_smooth = _ema(speeds_ms, a=0.3) if len(speeds_ms) else []
    speed_report = ""
    try:
        speed_report = generate_ai_summary(
            times=list(times), speeds_ms=list(speeds_ms_smooth),
            tilts=None, phases=None,
            model=cfg.get("openai_model", "gpt-4o-mini"),
            temperature=float(cfg.get("ai_temperature", 0.5)),
            max_tokens=700
        )
    except Exception as e:
        speed_report = f"（速度AI所感の生成に失敗：{e}）"

    # 姿勢ダイジェスト（平均値を簡潔に）
    pose_digest = ""
    if len(trunk_angles) > 0:
        pose_digest += f"- 体幹前傾角の平均は {np.mean(trunk_angles):.1f}°（±{np.std(trunk_angles):.1f}°）。\n"
    if len(kneeR) > 0 and len(kneeL) > 0:
        pose_digest += f"- 膝角 平均 R/L = {np.mean(kneeR):.0f}° / {np.mean(kneeL):.0f}°。\n"
    if len(ankle_dist_m) > 0:
        pose_digest += f"- 左右足首距離（簡易指標）平均 = {np.mean(ankle_dist_m):.2f} m。\n"

    ai_summary_text = _kjac_style_summary(speed_report, pose_digest, athlete, args.label)

    # PDF 出力
    pdf_path = os.path.join(athlete_dir, f"{athlete}{date_tag}_pose.pdf")
    _make_pdf(pdf_path, pages, times, speeds_ms_smooth, cfg, label=args.label, ai_summary_text=ai_summary_text)
    print(f"[OK] 出力完了: {pdf_path}")


if __name__ == "__main__":
    main()

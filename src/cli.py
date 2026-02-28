# src/cli.py
import os, glob, csv, yaml, cv2
import numpy as np
from datetime import datetime

from .calibration import load_scale, save_scale, calibrate_interactive
from .analyzer import draw_guides, put_label, flow_metrics, get_trunk_tilt_deg, classify_phase
from .notes import generate_notes, generate_ai_summary, generate_ai_notes_frame
from .reporter import setup_font, make_pdf


# ========= ユーティリティ =========
def _open_video_any_backend(path):
    """Windowsで読み込めないケースに備え、複数バックエンドを順に試す"""
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

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _read_cfg(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _iter_times(start_sec, total_sec, step_sec):
    t = max(0.0, float(start_sec))
    step = max(1e-6, float(step_sec))
    while t <= total_sec + 1e-9:
        yield round(t, 3)
        t += step

def _frame_at_time(cap, t_sec, fps):
    idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
    ok, frame = cap.read()
    return ok, frame

def _to_mps(px_per_s, m_per_px):
    return float(px_per_s) * float(m_per_px)

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _pick_video_dialog(initial_dir=None):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="動画を選択",
            initialdir=initial_dir if (initial_dir and os.path.isdir(initial_dir)) else os.getcwd(),
            filetypes=[("Video", "*.mp4;*.mov;*.m4v;*.avi;*.MP4;*.MOV;*.M4V;*.AVI")]
        )
        root.update(); root.destroy()
        return path
    except Exception:
        return None


# ========= メイン処理 =========
def process_one_video(video_path, cfg):
    print(f"\n=== 解析開始: {video_path} ===")

    # 出力先
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir  = _ensure_dir(os.path.join(cfg.get("output_dir", "outputs"), vid_name))
    img_dir  = _ensure_dir(os.path.join(out_dir, "frames"))
    pdf_path = os.path.join(out_dir, "report.pdf")

    # 解析パラメータ
    start_sec = _safe_float(cfg.get("start_sec", 1.0), 1.0)
    step_sec  = _safe_float(cfg.get("step_sec", 0.1), 0.1)

    # 校正読み込み or 実施
    scales_dir   = _ensure_dir("scales")
    camera_name  = cfg.get("calibration", {}).get("camera_name", "default")
    policy       = cfg.get("calibration", {}).get("policy", "reuse_or_ask")
    force_cal    = bool(cfg.get("force_calibrate", False))  # CLIから上書きされることもある想定

    m_per_px, scale_raw = load_scale(scales_dir, camera_name)
    if force_cal or (m_per_px is None) or (policy == "always_ask"):
        # インタラクティブ校正：最初のフレームで2点クリック → コンソールで距離[m]入力
        m_per_px, px_dist, real_m = calibrate_interactive(video_path)
        scale_raw = save_scale(scales_dir, camera_name, m_per_px, px_dist, real_m)

    print(f"[Calib] 1 px = {m_per_px:.5f} m  ({scale_raw})")

    # フォント準備（豆腐対策）
    setup_font(cfg.get("font_path", "ipaexg.ttf"))

    # 動画を開く
    cap = _open_video_any_backend(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_sec = (total_frames - 1) / fps if total_frames > 0 else 0.0

    # MediaPipe（任意）
    use_mediapipe = bool(cfg.get("use_mediapipe", False))
    mp_pose = pose = RIGHT = LEFT = None
    if use_mediapipe:
        try:
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(static_image_mode=True, model_complexity=1,
                                enable_segmentation=False, min_detection_confidence=0.5)
            RIGHT = {"hip": mp_pose.PoseLandmark.RIGHT_HIP, "knee": mp_pose.PoseLandmark.RIGHT_KNEE,
                     "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE, "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER}
            LEFT  = {"hip": mp_pose.PoseLandmark.LEFT_HIP,  "knee": mp_pose.PoseLandmark.LEFT_KNEE,
                     "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,  "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER}
        except Exception:
            use_mediapipe = False  # 無ければ無効化

    # 時系列ループ
    pages = []          # (t, phase, img_path, v_px_s, v_ms, v_kmh, tilt, notes)
    times = []          # s
    speeds_px = []      # px/s
    speeds_ms = []      # m/s
    tilts_list = []     # deg (or None)
    phases_list = []
    prev_gray = None
    prev_pts = None
    v_peak_px = 0.0

    first_ok, first_frame = _frame_at_time(cap, start_sec, fps)
    if not first_ok:
        cap.release()
        raise RuntimeError("開始時刻のフレームを取得できません")
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # AIノート設定
    use_ai = bool(cfg.get("use_ai", True))
    ai_scope = cfg.get("ai_scope", "both")  # frames / summary / both
    ai_model = cfg.get("openai_model", "gpt-4o-mini")
    ai_temp  = float(cfg.get("ai_temperature", 0.5))
    max_ai_frames = int(cfg.get("ai_notes_max_frames", 12))
    ai_frames_done = 0

    for idx, t in enumerate(_iter_times(start_sec, total_sec, step_sec)):
        ok, frame = _frame_at_time(cap, t, fps)
        if not ok:
            break

        # 速度プロキシ（光学フロー）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v_px_s, prev_pts, dx_std, dy_std = flow_metrics(prev_gray, gray, step_sec, prev_pts)
        prev_gray = gray.copy()

        # 実速度換算
        v_ms = _to_mps(v_px_s, m_per_px)
        v_kmh = v_ms * 3.6

        # 体幹角（任意）
        tilt = None
        if use_mediapipe and pose is not None:
            try:
                tilt = get_trunk_tilt_deg(frame.copy(), pose=pose, RIGHT=RIGHT, LEFT=LEFT)
            except Exception:
                tilt = None

        # dv（簡易加速度）：隣接差分
        if speeds_px:
            dv_px = (v_px_s - speeds_px[-1]) / max(step_sec, 1e-6)
        else:
            dv_px = None

        # フェーズ推定
        v_peak_px = max(v_peak_px, v_px_s)
        phase = classify_phase(v_px_s, v_peak_px, dv_px, tilt)

        # 画像に注釈
        annotated = frame.copy()
        draw_guides(annotated)
        label = f"{phase} | t={t:.1f}s"
        put_label(annotated, label)

        # 保存
        t_str = f"{t:.1f}".replace(".", "_")
        fname = f"t_{t_str}s.jpg"
        img_path = os.path.join(img_dir, fname)
        if not cv2.imwrite(img_path, annotated):
            raise RuntimeError(f"画像保存に失敗しました: {img_path}")

        # ノート生成（基本）
        notes = generate_notes(
            phase, v_ms, v_kmh,
            None if dv_px is None else dv_px * m_per_px,  # m/s^2 に換算
            tilt, dx_std, dy_std
        )

        # 追加でAIノート（上限つき）
        if use_ai and ai_scope in ("frames", "both") and ai_frames_done < max_ai_frames:
            try:
                ai_lines = generate_ai_notes_frame(
                    frame_idx=idx, t=t, phase=phase,
                    v_ms=v_ms, v_kmh=v_kmh,
                    dv_ms2=None if dv_px is None else dv_px * m_per_px,
                    tilt=tilt, dx_std=dx_std, dy_std=dy_std,
                    model=ai_model, temperature=ai_temp
                )
                # 末尾にAIノートを追加表示（見分けやすいように）
                notes = notes + [f"AI: {s[1:]}" if s.startswith("・") else f"AI: {s}" for s in ai_lines]
                ai_frames_done += 1
            except Exception as e:
                os.makedirs("logs", exist_ok=True)
                with open(os.path.join("logs","ai_frame_error.log"),"a",encoding="utf-8") as f:
                    f.write(f"[frame {idx} @ t={t:.2f}] {e}\n")

        # 収集
        pages.append((t, phase, img_path, v_px_s, v_ms, v_kmh, tilt, notes))
        times.append(t); speeds_px.append(v_px_s); speeds_ms.append(v_ms)
        tilts_list.append(tilt); phases_list.append(phase)

    cap.release()
    if pose is not None:
        try: pose.close()
        except Exception: pass

    # ---- AIサマリー：常に2ページ返す（失敗でもフォールバック）----
    ai_summary_pages = None
    if use_ai and ai_scope in ("summary","both"):
        try:
            ai_summary_pages = generate_ai_summary(
                times=times, speeds_ms=speeds_ms, tilts=tilts_list, phases=phases_list,
                model=ai_model, temperature=float(cfg.get("ai_temperature",0.4))
            )
        except Exception as e:
            os.makedirs("logs", exist_ok=True)
            with open(os.path.join("logs","ai_summary_error.log"),"a",encoding="utf-8") as f:
                f.write(f"[generate_ai_summary failed] {e}\n")
            ai_summary_pages = None

    # ---- PDF生成 ----
    make_pdf(pages, pdf_path, times, speeds_px, speeds_ms, ai_summary_text=ai_summary_pages)
    print(f"[OK] 出力: {pdf_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="短距離フォーム自動レポート生成")
    ap.add_argument("--config", required=True, help="config.yaml パス")
    ap.add_argument("--video", help="単一動画パスを指定（省略時は videos_dir をスキャン）")
    ap.add_argument("--pick_dialog", action="store_true", help="GUIで動画を選択")
    ap.add_argument("--force_calibrate", action="store_true", help="毎回校正を実施")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)
    if args.force_calibrate:
        cfg["force_calibrate"] = True

    # 入力動画の決定
    target_videos = []
    if args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"指定動画が見つかりません: {args.video}")
        target_videos = [args.video]
    elif args.pick_dialog:
        init_dir = cfg.get("pick_default_dir") or os.getcwd()
        path = _pick_video_dialog(initial_dir=init_dir)
        if not path:
            raise FileNotFoundError("動画が選択されませんでした")
        target_videos = [path]
    else:
        videos_dir = cfg.get("videos_dir", "videos")
        process_subdirs = bool(cfg.get("process_subdirs", False))
        patterns = ["*.mp4", "*.mov", "*.m4v", "*.avi", "*.MP4", "*.MOV", "*.M4V", "*.AVI"]
        if process_subdirs:
            for root, _, _ in os.walk(videos_dir):
                for pat in patterns:
                    target_videos.extend(glob.glob(os.path.join(root, pat)))
        else:
            for pat in patterns:
                target_videos.extend(glob.glob(os.path.join(videos_dir, pat)))
        if not target_videos:
            raise FileNotFoundError(f"動画が見つかりません: {videos_dir}")

    # 1本ずつ処理
    for vp in target_videos:
        try:
            process_one_video(vp, cfg)
        except Exception as e:
            print(f"[ERROR] {vp}: {e}")


if __name__ == "__main__":
    main()

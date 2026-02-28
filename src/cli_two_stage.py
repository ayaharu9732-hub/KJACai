# src/cli_two_stage.py
import argparse
import os
import cv2
import yaml
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from .calibration import load_scale, save_scale, calibrate_interactive
from .analyzer import draw_guides, put_label, flow_metrics, classify_phase
from .reporter import setup_font
from .notes import generate_ai_summary


# ========== 共通ユーティリティ ==========
def _ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def _read_cfg(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _open_video_any_backend(path):
    backends = [cv2.CAP_ANY, getattr(cv2, "CAP_MSMF", 0),
                getattr(cv2, "CAP_FFMPEG", 1900), getattr(cv2, "CAP_DSHOW", 700)]
    for be in backends:
        cap = cv2.VideoCapture(path, be)
        if cap.isOpened(): return cap
        cap.release()
    return cv2.VideoCapture(path)

def _frame_at_time(cap, t_sec, fps):
    idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
    return cap.read()

def _iter_times(start_sec, total_sec, step_sec):
    t = max(0.0, float(start_sec)); step = max(1e-6, float(step_sec))
    while t <= total_sec + 1e-9:
        yield round(t, 3)
        t += step

def _ema(x, alpha=0.3):
    y=[]
    for i,v in enumerate(x):
        if i==0: y.append(v)
        else: y.append(alpha*v+(1-alpha)*y[-1])
    return np.array(y, dtype=float)

def _to_mps(px_per_s, m_per_px): return float(px_per_s) * float(m_per_px)

def _estimate_pitch_hz(times, speeds_ms):
    if len(times) < 16: return None
    t = np.array(times); v = np.array(speeds_ms)
    v = v - np.mean(v)
    if np.allclose(v.std(), 0, atol=1e-6): return None
    dt = np.median(np.diff(t))
    if dt <= 0: return None
    V = np.fft.rfft(v)
    freqs = np.fft.rfftfreq(len(v), d=dt)
    band = (freqs >= 1.5) & (freqs <= 4.0)
    if not np.any(band): return None
    peak_idx = np.argmax(np.abs(V[band]))
    return float(freqs[band][peak_idx])


# ========== 区間解析 ==========
def analyze_segment(video_path, m_per_px, cfg, segment_tag, out_dir):
    print(f"\n--- Segment {segment_tag}: {video_path} ---")

    cap = _open_video_any_backend(video_path)
    if not cap.isOpened(): raise RuntimeError(f"動画を開けません: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_sec = (total_frames-1)/fps if total_frames>0 else 0.0

    start_sec = float(cfg.get("start_sec", 0.0))
    step_sec  = float(cfg.get("step_sec", 0.1))
    img_dir   = _ensure_dir(os.path.join(out_dir, f"frames_{segment_tag}"))

    # 先頭フレーム
    ok0, frame0 = _frame_at_time(cap, start_sec, fps)
    if not ok0:
        cap.release()
        raise RuntimeError("開始フレーム取得に失敗")
    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    prev_pts  = None

    pages = []  # (t, phase, img_path, v_px_s, v_ms, v_kmh)
    times, speeds_px, speeds_ms, phases = [], [], [], []
    v_peak_px = 0.0

    # 自動トリム（静止→連続3ステップで閾値超）
    auto_trim = bool(cfg.get("auto_trim", True))
    started = not auto_trim
    consec  = 0
    th_px_s = float(cfg.get("auto_trim_threshold_px_s", 80.0))

    for t in _iter_times(start_sec, total_sec, step_sec):
        ok, frame = _frame_at_time(cap, t, fps)
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        v_px_s, prev_pts, _, _ = flow_metrics(prev_gray, gray, step_sec, prev_pts)
        prev_gray = gray

        if auto_trim and not started:
            if v_px_s > th_px_s:
                consec += 1
                if consec >= 3: started = True
            else:
                consec = 0
            if not started:
                continue

        v_ms  = _to_mps(v_px_s, m_per_px)
        v_kmh = v_ms * 3.6
        dv_px = None if len(speeds_px)==0 else (v_px_s - speeds_px[-1])/max(step_sec,1e-6)
        v_peak_px = max(v_peak_px, v_px_s)
        phase = classify_phase(v_px_s, v_peak_px, dv_px, None)

        annotated = frame.copy()
        draw_guides(annotated)
        put_label(annotated, f"{segment_tag} | {phase} | t={t:.1f}s")

        t_str = f"{t:.1f}".replace(".","_")
        img_path = os.path.join(img_dir, f"{segment_tag}_t_{t_str}.jpg")
        cv2.imwrite(img_path, annotated)

        pages.append((t, phase, img_path, v_px_s, v_ms, v_kmh))
        times.append(t); speeds_px.append(v_px_s); speeds_ms.append(v_ms); phases.append(phase)

    cap.release()

    speeds_ms_smooth = _ema(speeds_ms, alpha=0.3) if speeds_ms else []
    v_peak = float(np.max(speeds_ms_smooth)) if len(speeds_ms_smooth) else 0.0
    v_avg  = float(np.mean(speeds_ms_smooth)) if len(speeds_ms_smooth) else 0.0
    pitch_hz = _estimate_pitch_hz(times, speeds_ms_smooth)

    return {
        "segment": segment_tag,
        "pages": pages,
        "times": times,
        "speeds_ms": speeds_ms_smooth,
        "phases": phases,
        "v_peak": v_peak,
        "v_avg": v_avg,
        "pitch_hz": pitch_hz,
        "fps": fps
    }


# ========== PDF（2区間+AI比較） ==========
def make_pdf_two_stage(out_pdf, A, B, cfg, ai_text_A=None, ai_text_B=None, ai_text_compare=None):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    # ReportLabの本文フォント
    try:
        pdfmetrics.registerFont(TTFont("IPAexGothic", cfg.get("font_path", "ipaexg.ttf")))
        font_jp = "IPAexGothic"
    except:
        font_jp = "Helvetica"

    # Matplotlibに日本語フォント（あれば）を設定（無くてもOK）
    try:
        import matplotlib
        fp = cfg.get("font_path", "ipaexg.ttf")
        if os.path.exists(fp):
            matplotlib.rcParams["font.family"] = "IPAexGothic"
            from matplotlib import font_manager
            font_manager.fontManager.addfont(fp)
    except Exception:
        pass

    c = canvas.Canvas(out_pdf, pagesize=landscape(A4))
    W, H = landscape(A4)

    def draw_paragraph(x, y, text, width=760, leading=16, size=12):
        # list/None → str
        if isinstance(text, (list, tuple)):
            text = "\n".join([str(s) for s in text])
        text = str(text or "（AIサマリー生成なし）")
        from reportlab.platypus import Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        style = styles["Normal"]; style.fontName = font_jp; style.fontSize = size; style.leading = leading
        p = Paragraph(text.replace("\n", "<br/>"), style)
        w,h = p.wrap(width, 9999)
        p.drawOn(c, x, y-h)
        return h

    def draw_frames(pages, title):
        c.setFont(font_jp, 18); c.drawString(40, H-60, title)
        y = H-90
        for i, (t, phase, img_path, v_px, v_ms, v_kmh) in enumerate(pages[:8]):
            c.setFont(font_jp, 11)
            c.drawString(40, y, f"t={t:.1f}s  |  {phase}  |  {v_ms:.2f} m/s  ({v_kmh:.1f} km/h)")
            try:
                c.drawImage(img_path, 40, y-150, width=360, height=135, preserveAspectRatio=True)
            except: pass
            y -= 170
            if y < 140:
                c.showPage(); y = H-90; c.setFont(font_jp, 18); c.drawString(40, H-60, title)

    def draw_speed_plot(A, B, title):
        fig = plt.figure(figsize=(8,3), dpi=150)
        if A["times"]:
            plt.plot(A["times"], A["speeds_ms"], label="0–20m")
        if B["times"]:
            t0 = A["times"][-1] if A["times"] else 0.0
            tb = [t0 + (t - B["times"][0]) for t in B["times"]]
            plt.plot(tb, B["speeds_ms"], label="20–40m")
        plt.xlabel("Time (s)"); plt.ylabel("m/s"); plt.title(title); plt.legend(); plt.tight_layout()
        tmp = os.path.join(os.path.dirname(out_pdf), "_tmp_twostage_plot.png")
        fig.savefig(tmp); plt.close(fig)
        c.setFont(font_jp, 18); c.drawString(40, H-60, title)
        try:
            c.drawImage(tmp, 40, 140, width=760, height=360, preserveAspectRatio=True)
        except: pass
        c.showPage()
        try: os.remove(tmp)
        except: pass

    # ページ構成
    draw_frames(A["pages"], "Segment A（0–20m）")
    draw_frames(B["pages"], "Segment B（20–40m）")
    draw_speed_plot(A, B, "速度プロファイル（0→40m 連結）")

    # 指標比較
    c.setFont(font_jp, 18); c.drawString(40, H-60, "A/B 比較（指標）")
    c.setFont(font_jp, 12)
    c.drawString(40, H-90,  f"A: v_peak={A['v_peak']:.2f} m/s, v_avg={A['v_avg']:.2f} m/s, pitch≈{(A['pitch_hz'] or 0):.2f} Hz")
    c.drawString(40, H-110, f"B: v_peak={B['v_peak']:.2f} m/s, v_avg={B['v_avg']:.2f} m/s, pitch≈{(B['pitch_hz'] or 0):.2f} Hz")
    c.drawString(40, H-140, "所見例：A終盤→B冒頭の切替、ピッチ維持、終盤の失速兆候などを確認")
    c.showPage()

    # AI（各区間/比較）— 必ず段落化
    def ai_page(title, text):
        c.setFont(font_jp, 18); c.drawString(40, H-60, title)
        draw_paragraph(40, H-90, text, width=760, leading=18, size=12)
        c.showPage()

    ai_page("AI分析レポート：Segment A（0–20m）", ai_text_A)
    ai_page("AI分析レポート：Segment B（20–40m）", ai_text_B)
    ai_page("AI比較考察（0→40m）", ai_text_compare)

    c.save()
    print(f"[OK] 出力: {out_pdf}")


# ========== メイン ==========
def main():
    ap = argparse.ArgumentParser(description="二段階（0–20m / 20–40m）スプリント分析")
    ap.add_argument("--config", required=True, help="config.yaml")
    ap.add_argument("--videoA", help="0–20m 動画パス")
    ap.add_argument("--videoB", help="20–40m 動画パス")
    ap.add_argument("--pick_dialog", action="store_true", help="ダイアログでA/B両方を選ぶ")
    ap.add_argument("--force_calibrate", action="store_true")
    args = ap.parse_args()

    cfg = _read_cfg(args.config)
    setup_font(cfg.get("font_path", "ipaexg.ttf"))

    # ファイル選択
    def pick_file(initial_dir=None):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                initialdir=initial_dir or os.getcwd(),
                filetypes=[("Video","*.mp4 *.mov *.m4v *.avi *.MP4 *.MOV *.M4V *.AVI")]
            )
            root.destroy()
            return path
        except Exception:
            return None

    videoA = args.videoA
    videoB = args.videoB
    if args.pick_dialog:
        base = cfg.get("pick_default_dir")
        if not videoA: videoA = pick_file(base)
        if not videoB: videoB = pick_file(base)

    if not videoA or not os.path.exists(videoA):
        raise FileNotFoundError(f"A動画が見つかりません: {videoA}")
    if not videoB or not os.path.exists(videoB):
        raise FileNotFoundError(f"B動画が見つかりません: {videoB}")

    # 出力先
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ensure_dir(os.path.join(cfg.get("output_dir","outputs"), f"two_stage_{tag}"))
    pdf_path = os.path.join(out_dir, "report_two_stage.pdf")

    # スケール（A/B別）
    scales_dir = _ensure_dir("scales")
    camA, camB = "two_A", "two_B"
    mppA, rawA = load_scale(scales_dir, camA)
    mppB, rawB = load_scale(scales_dir, camB)

    def _prompt_real_m():
        try:
            real_m = float(input("コーン間の実距離[m]（例：20 または 30）を入力："))
        except Exception:
            real_m = None
        return real_m

    if args.force_calibrate or mppA is None:
        # クリック2点 → コンソール距離入力
        mpp, px, rm = calibrate_interactive(videoA)
        if rm is None:
            rm = _prompt_real_m()
        rawA = save_scale(scales_dir, camA, mpp, px, rm)
        mppA = mpp

    if args.force_calibrate or mppB is None:
        mpp, px, rm = calibrate_interactive(videoB)
        if rm is None:
            rm = _prompt_real_m()
        rawB = save_scale(scales_dir, camB, mpp, px, rm)
        mppB = mpp

    print(f"[Calib A] 1px={mppA:.6f} m  ({rawA})")
    print(f"[Calib B] 1px={mppB:.6f} m  ({rawB})")

    # 解析
    A = analyze_segment(videoA, mppA, cfg, "A(0-20m)", out_dir)
    B = analyze_segment(videoB, mppB, cfg, "B(20-40m)", out_dir)

    # AI（各区間 + 比較）
    aiA = aiB = aiC = "（AIサマリー生成なし）"
    if bool(cfg.get("use_ai", True)):
        try:
            aiA = generate_ai_summary(A["times"], A["speeds_ms"], [], A["phases"],
                                      model=cfg.get("openai_model","gpt-4o-mini"),
                                      temperature=float(cfg.get("ai_temperature", 0.5)))
        except Exception as e:
            aiA = f"AIサマリー失敗: {e}"

        try:
            aiB = generate_ai_summary(B["times"], B["speeds_ms"], [], B["phases"],
                                      model=cfg.get("openai_model","gpt-4o-mini"),
                                      temperature=float(cfg.get("ai_temperature", 0.5)))
        except Exception as e:
            aiB = f"AIサマリー失敗: {e}"

        # 比較考察
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = (
                "あなたは短距離走のコーチです。以下の2区間の時系列（m/s）指標から、"
                "0→40mの連続走として比較考察を日本語で段落に分けて簡潔に示してください。\n"
                f"- A(0–20m): v_peak={A['v_peak']:.2f}, v_avg={A['v_avg']:.2f}, pitch≈{(A['pitch_hz'] or 0):.2f}Hz\n"
                f"- B(20–40m): v_peak={B['v_peak']:.2f}, v_avg={B['v_avg']:.2f}, pitch≈{(B['pitch_hz'] or 0):.2f}Hz\n\n"
                "出力:\n1) 良い点（2–3文）\n2) 改善点（2–3文）\n3) 練習メニュー（3項目、各1文）"
            )
            r = client.chat.completions.create(
                model=cfg.get("openai_model","gpt-4o-mini"),
                temperature=float(cfg.get("ai_temperature", 0.5)),
                messages=[{"role":"user","content":prompt}],
                max_tokens=700
            )
            aiC = (r.choices[0].message.content or "").strip() or "（AI比較サマリー未取得）"
        except Exception as e:
            aiC = f"AI比較サマリー失敗: {e}"

    # PDF
    make_pdf_two_stage(pdf_path, A, B, cfg, ai_text_A=aiA, ai_text_B=aiB, ai_text_compare=aiC)


if __name__ == "__main__":
    main()

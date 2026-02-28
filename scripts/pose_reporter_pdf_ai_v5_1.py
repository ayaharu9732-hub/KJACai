# ==========================================================
# 📝 pose_reporter_pdf_ai_v5_1.py
# - v5.0 の改良：グラフ確実保存、数値連動コメント、日本語フォント強化
# ==========================================================

import os
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF, XPos, YPos

BASE_DIR   = r"C:\Users\Futamura\KJACai"
OUTPUTS    = os.path.join(BASE_DIR, "outputs")
IMAGES_DIR = os.path.join(OUTPUTS, "images")
PDF_DIR    = os.path.join(OUTPUTS, "pdf")
os.makedirs(PDF_DIR, exist_ok=True)

# ---------------- Matplotlib JP (任意) ----------------
def try_enable_japanese_matplotlib():
    try:
        import japanize_matplotlib  # noqa: F401
    except Exception:
        pass

# ---------------- グラフ作成 ----------------
def plot_graphs(df: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(8, 3))
    x = df["time_s"].values
    v = df["speed_mps"].values
    tilt = df["tilt_deg"].values
    comy = df["COM_y"].values

    ax1.plot(x, v, label="速度 (m/s)")
    ax1.set_xlabel("時間 [s]")
    ax1.set_ylabel("速度 [m/s]")

    ax2 = ax1.twinx()
    ax2.plot(x, tilt, linestyle="--", label="傾き (deg)")
    ax2.plot(x, comy, linestyle=":", label="COM_y (norm)")
    ax2.set_ylabel("角度/COM")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    graph_path = os.path.join(PDF_DIR, "graphs_v5_1.png")
    plt.savefig(graph_path, dpi=200)
    plt.close('all')  # ← 確実にファイルを閉じる
    return graph_path

# ---------------- 代表フレーム抽出 ----------------
def capture_representative_frame(overlay_mp4: str) -> str:
    cap = cv2.VideoCapture(overlay_mp4)
    if not cap.isOpened():
        return ""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 3))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ""
    out = os.path.join(IMAGES_DIR, os.path.splitext(os.path.basename(overlay_mp4))[0] + "_frame.png")
    cv2.imwrite(out, frame)
    return out

def infer_overlay_from_video(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(IMAGES_DIR, f"{base}_pose_overlay.mp4")

# ---------------- 日本語フォント探索 ----------------
def _candidate_font_paths():
    win_fonts = r"C:\Windows\Fonts"
    proj_fonts = os.path.join(BASE_DIR, "fonts")
    cands = [
        os.path.join(proj_fonts, "ipaexg.ttf"),
        os.path.join(proj_fonts, "IPAexGothic.ttf"),
        os.path.join(proj_fonts, "NotoSansCJKjp-Regular.otf"),
        os.path.join(win_fonts, "ipaexg.ttf"),
        os.path.join(win_fonts, "IPAexGothic.ttf"),
        os.path.join(win_fonts, "NotoSansCJKjp-Regular.otf"),
        os.path.join(win_fonts, "YuGothR.ttc"),
        os.path.join(win_fonts, "YuGothM.ttc"),
        os.path.join(win_fonts, "meiryo.ttc"),
        os.path.join(win_fonts, "msgothic.ttc"),
    ]
    seen, out = set(), []
    for p in cands:
        if p and os.path.exists(p) and p not in seen:
            seen.add(p); out.append(p)
    return out

def set_japanese_font(pdf: FPDF) -> str:
    for path in _candidate_font_paths():
        try:
            name = os.path.splitext(os.path.basename(path))[0]
            pdf.add_font(name, "", path, uni=True)
            pdf.set_font(name, size=12)
            return name
        except Exception:
            continue
    pdf.set_font("Helvetica", size=12)
    return "Helvetica"

# ---------------- 数値連動コメント ----------------
def make_ai_comment(pitch, stride, stab):
    tips = []
    # ピッチ
    if pitch < 4.0:
        tips.append("ピッチは少し低め。腕振りの前後幅をやや小さくして回転数を上げよう。")
    elif pitch > 4.8:
        tips.append("ピッチは非常に良好。この回転数を最後まで維持できるとさらに◎。")
    else:
        tips.append("ピッチは適正。接地直後の膝スイングを速くしてロスを減らそう。")
    # ストライド
    if stride < 1.5:
        tips.append("ストライドは小さめ。地面を“後ろへ押す”意識で滞空を少し長く。")
    elif stride > 1.9:
        tips.append("ストライドは十分。大きさを保ちつつピッチ低下に注意。")
    else:
        tips.append("ストライドは適正。骨盤の前向き推進でさらに伸ばせる余地あり。")
    # 安定性
    if stab < 6.0:
        tips.append("安定性は改善余地あり。体幹一直線の押し出し（Wall drill）を週2回。")
    elif stab > 8.0:
        tips.append("安定性は高い！後半のフォーム維持に直結しています。")
    else:
        tips.append("安定性は標準。接地の真下化とリラックスで揺れをさらに減らそう。")
    tips.append("最後の5〜6歩は腕をコンパクトにしてピッチ維持を狙おう。")
    return "\n".join(f"・{t}" for t in tips)

# ---------------- PDF 生成 ----------------
def build_pdf(csv_path: str, overlay_mp4: str, athlete: str = "二村 遥香") -> str:
    df = pd.read_csv(csv_path)
    df_body = df[df["frame"] != "_summary_"].copy()
    if df_body.empty:
        raise RuntimeError("CSVに本体データがありません。")

    try_enable_japanese_matplotlib()
    graph_path = plot_graphs(df_body)

    rep_img = ""
    if overlay_mp4 and os.path.exists(overlay_mp4):
        rep_img = capture_representative_frame(overlay_mp4)

    meta = df.tail(1).to_dict(orient="records")[0]
    pitch = float(meta.get("pitch_hz", 0.0))
    stride = float(meta.get("stride_m", 0.0))
    stab   = float(meta.get("stability_score", 0.0))

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    used_font = set_japanese_font(pdf)

    # タイトル
    pdf.set_font(size=16)
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.1）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=11)
    pdf.cell(0, 7,
             f"ピッチ: {pitch:.2f} 歩/秒   ストライド: {stride:.2f} m/歩   安定性: {stab:.1f}/10",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 第1章：フォーム分析
    pdf.set_font(size=12)
    pdf.cell(0, 8, "第1章：フォーム分析（スタート〜終盤）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if rep_img and os.path.exists(rep_img):
        pdf.image(rep_img, w=120); pdf.ln(2)
    pdf.set_font(size=10)
    pdf.multi_cell(0, 6,
        "・スタート：前傾角を保ちつつ接地は体のやや下に\n"
        "・中盤：骨盤の上下動を抑え、接地直後の膝前スイングを素早く\n"
        "・終盤：上体を起こし過ぎず、腕振りコンパクトでピッチ維持")

    # 第2章：ピッチ＆ストライド（テンプレ）
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第2章：ピッチ＆ストライド分析（比較表）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=10)
    pdf.multi_cell(0, 6,
        "（単走テンプレ）予選/決勝比較がある場合に歩数・ピッチ・ストライドを並べて表示")

    # 第3章：フォーム変化グラフ
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第3章：フォーム変化グラフ（速度・傾き・COM）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if os.path.exists(graph_path):
        pdf.image(graph_path, w=180); pdf.ln(2)

    # 第4章：区間別の改善ポイント（定型）
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第4章：区間別比較・改善ポイント",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=10)
    pdf.multi_cell(0, 6,
        "・0–30m：接地が前に流れやすい→地面を後ろに“押す”意識\n"
        "・30–60m：沈み込み抑制→接地直後の膝前スイングを速く\n"
        "・60–90m：上体の起き上がり過多に注意→視線は水平やや下\n"
        "・90–100m：ピッチ維持→最後の5–6歩は腕をコンパクトに")

    # 第5章：AI所見（数値連動）
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第5章：AI所見（今日のまとめ）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=10)
    pdf.multi_cell(0, 6, make_ai_comment(pitch, stride, stab))

    out_pdf = os.path.join(PDF_DIR, f"{athlete}_form_report_v5_1.pdf")
    pdf.output(out_pdf)
    print(f"✅ PDF出力: {out_pdf}  （使用フォント: {used_font}）")
    return out_pdf

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default=os.path.join(OUTPUTS, "koike_pose_metrics_v3.csv"))
    ap.add_argument("--video",  default="")
    ap.add_argument("--overlay", default="")
    ap.add_argument("--athlete", default="二村 遥香")
    args = ap.parse_args()

    overlay = args.overlay
    if not overlay and args.video:
        overlay = infer_overlay_from_video(args.video)

    try_enable_japanese_matplotlib()
    build_pdf(args.csv, overlay, args.athlete)










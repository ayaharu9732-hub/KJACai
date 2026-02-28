# ==========================================================
# 📝 pose_reporter_pdf_ai_v5_0.py  (JP font robust)
# - CSVから速度/角度/COMを読み込み、グラフ化
# - 骨格オーバーレイ動画から代表フレームを自動キャプチャ
# - ピッチ/ストライド/安定性 指標をヘッダ表示
# - 日本語フォント(TTF/OTF/TTC)を自動探索し、なければ英字にフォールバック
# - fpdf2 v2.5.2+ の新パラメータ対応（new_x/new_y）
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

def try_enable_japanese_matplotlib():
    try:
        import japanize_matplotlib  # noqa: F401
    except Exception:
        pass

def plot_graphs(df: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(8, 3))
    x = df["time_s"].values
    v = df["speed_mps"].values
    tilt = df["tilt_deg"].values
    com_y = df["COM_y"].values

    ax1.plot(x, v, label="速度 (m/s)")
    ax1.set_xlabel("時間 [s]")
    ax1.set_ylabel("速度 [m/s]")

    ax2 = ax1.twinx()
    ax2.plot(x, tilt, linestyle="--", label="傾き (deg)")
    ax2.plot(x, com_y, linestyle=":", label="COM_y (norm)")
    ax2.set_ylabel("角度/COM")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    graph_path = os.path.join(PDF_DIR, "graphs_v5.png")
    plt.savefig(graph_path, dpi=200)
    plt.close(fig)
    return graph_path

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
    cand = os.path.join(IMAGES_DIR, f"{base}_pose_overlay.mp4")
    return cand

# -------------------- 日本語フォント探索＆登録 --------------------
def _candidate_font_paths():
    win_fonts = r"C:\Windows\Fonts"
    proj_fonts = os.path.join(BASE_DIR, "fonts")  # 置いてあれば最優先
    candidates = [
        # プロジェクト同梱があれば最優先
        os.path.join(proj_fonts, "ipaexg.ttf"),
        os.path.join(proj_fonts, "IPAexGothic.ttf"),
        os.path.join(proj_fonts, "NotoSansCJKjp-Regular.otf"),
        os.path.join(proj_fonts, "Meiryo.ttf"),  # 稀に配布されている単TTF
        # Windows 典型パス
        os.path.join(win_fonts, "ipaexg.ttf"),
        os.path.join(win_fonts, "IPAexGothic.ttf"),
        os.path.join(win_fonts, "ipag.ttf"),
        os.path.join(win_fonts, "msgothic.ttc"),
        os.path.join(win_fonts, "meiryo.ttc"),
        os.path.join(win_fonts, "YuGothM.ttc"),
        os.path.join(win_fonts, "YuGothR.ttc"),
        os.path.join(win_fonts, "NotoSansCJKjp-Regular.otf"),
    ]
    # 重複排除＆存在チェック
    seen, out = set(), []
    for p in candidates:
        if p and p not in seen and os.path.exists(p):
            seen.add(p); out.append(p)
    return out

def set_japanese_font(pdf: FPDF) -> str:
    """
    使える日本語フォント(TTF/OTF/TTC)を自動登録して set_font。
    見つからなければ英字フォントをセットして戻る（クラッシュ回避）。
    戻り値は使用フォント名（英字フォールバック時は 'Helvetica'）。
    """
    for path in _candidate_font_paths():
        try:
            # フォント名はファイル名ベースでユニークに
            fname = os.path.basename(path)
            font_name = os.path.splitext(fname)[0]
            pdf.add_font(font_name, "", path, uni=True)
            pdf.set_font(font_name, size=12)
            return font_name
        except Exception:
            continue
    # フォールバック（英字）
    pdf.set_font("Helvetica", size=12)
    return "Helvetica"

# -------------------- PDF 本体 --------------------
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
    stab = float(meta.get("stability_score", 0.0))

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()

    # 日本語フォント設定（失敗しても英字で続行）
    used_font = set_japanese_font(pdf)

    # タイトル
    pdf.set_font(size=16)
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.0）",
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

    # 第2章：ピッチ＆ストライド
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第2章：ピッチ＆ストライド分析（比較表）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=10)
    pdf.multi_cell(0, 6,
        "予選/決勝の比較がある場合に歩数・ピッチ・ストライドを並べて表示（本走は単走のためテンプレ記載）")

    # 第3章：フォーム変化グラフ
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第3章：フォーム変化グラフ（速度・傾き・COM）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if os.path.exists(graph_path):
        pdf.image(graph_path, w=180); pdf.ln(2)

    # 第4章：区間別の改善ポイント
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第4章：区間別比較・改善ポイント",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=10)
    pdf.multi_cell(0, 6,
        "・0–30m：接地が前に流れやすい→地面を後ろに“押す”意識\n"
        "・30–60m：沈み込み抑制→接地直後の膝前スイングを速く\n"
        "・60–90m：上体の起き上がり過多に注意→視線は水平やや下\n"
        "・90–100m：ピッチ維持→最後の5–6歩は腕をコンパクトに")

    # 第5章：AI所見（今日のまとめ）
    pdf.ln(2); pdf.set_font(size=12)
    pdf.cell(0, 8, "第5章：AI所見（今日のまとめ）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font(size=10)
    pdf.multi_cell(0, 6,
        "・スタート：前傾を保ち、接地は体のやや下に。腕振りでリズムを先行させよう。\n"
        "・中盤：骨盤の上下動を抑え、設置直後の膝前スイングを素早く。\n"
        "・終盤：上体を起こし過ぎず、ピッチ維持。視線は水平やや下。\n\n"
        f"【今日の指標】ピッチ: {pitch:.2f} 歩/秒 / ストライド: {stride:.2f} m/歩 / 安定性: {stab:.1f}/10\n"
        "⇒ 接地の真下化と腕のコンパクトさで、最後の5〜6歩のスピード維持を狙おう。")

    out_pdf = os.path.join(PDF_DIR, f"{athlete}_form_report_v5_0.pdf")
    pdf.output(out_pdf)
    print(f"✅ PDF出力: {out_pdf}  （使用フォント: {used_font}）")
    return out_pdf

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default=os.path.join(OUTPUTS, "koike_pose_metrics_v3.csv"))
    ap.add_argument("--video",  default="")  # 元動画（overlay 推定に使用）
    ap.add_argument("--overlay", default="")
    ap.add_argument("--athlete", default="二村 遥香")
    args = ap.parse_args()

    overlay = args.overlay
    if not overlay and args.video:
        base = os.path.splitext(os.path.basename(args.video))[0]
        overlay = os.path.join(IMAGES_DIR, f"{base}_pose_overlay.mp4")

    build_pdf(args.csv, overlay, args.athlete)









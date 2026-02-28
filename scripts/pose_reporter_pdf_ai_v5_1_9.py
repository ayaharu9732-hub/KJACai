# ==========================================================
# 🧩 pose_reporter_pdf_ai_v5_1_9.py
# ✅ 右寄り文字修正版（左揃え固定・中央タイトル・最終デザイン）
# ==========================================================
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import textwrap
import matplotlib

# --- フォント設定（IPAexGothic推奨） ---
matplotlib.rcParams["font.family"] = "IPAexGothic"

CALIB_PATH = "outputs/jsonl/calibration_result.json"
OUTPUT_DIR = "outputs/pdf"

FONT_PATHS = [
    "C:/Users/Futamura/KJACai/fonts/ipaexg.ttf",
    "C:/Windows/Fonts/ipaexg.ttf",
    "C:/Windows/Fonts/YuGothR.ttc",
    "C:/Windows/Fonts/msgothic.ttc"
]


def get_font_path():
    for p in FONT_PATHS:
        if os.path.exists(p):
            return p
    print("⚠ 日本語フォントが見つかりません。ipaexg.ttf を fonts/ に配置してください。")
    return None


def wrap_text(text, width=45):
    """コメントなどを折り返して表示"""
    lines = textwrap.wrap(text, width)
    return lines if lines else [""]


def generate_ai_comments(summary):
    pitch, stride, stability = summary["pitch"], summary["stride"], summary["stability"]
    comments = []

    if pitch < 3.5:
        comments.append("ピッチが低めです。脚の回転スピードを意識してテンポを上げましょう。")
    elif pitch < 4.5:
        comments.append("ピッチは安定しています。終盤まで一定リズムを維持できています。")
    else:
        comments.append("ピッチが非常に良好で、反応速度も優れています。")

    if stride < 1.2:
        comments.append("ストライドがやや小さいため、骨盤の前傾を意識しましょう。")
    elif stride < 1.6:
        comments.append("ストライドは標準的です。接地姿勢が安定しています。")
    else:
        comments.append("ストライドが大きく、推進力のあるフォームです。")

    if stability < 7.0:
        comments.append("体幹の安定性がやや不足しています。姿勢保持の強化を行いましょう。")
    elif stability < 9.0:
        comments.append("安定性は良好です。リズムに一貫性があります。")
    else:
        comments.append("安定性が非常に高く、重心のブレが少ない理想的な走りです。")
    return comments


def create_graphs(df, athlete):
    """速度・傾き・重心の3グラフ"""
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    plt.subplots_adjust(hspace=0.4)

    titles = ["区間速度の変化 [m/s]", "体幹傾き角度の推移 [°]", "重心高さの変化 [m]"]
    colors = ["#0072BD", "#EDB120", "#77AC30"]
    cols = ["speed_mps", "tilt_deg", "COM_y_m"]

    for ax, col, title, color in zip(axs, cols, titles, colors):
        ax.plot(df["time_s"], df[col], linewidth=1.8, color=color)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("時間 [s]", fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f"outputs/{athlete}_graphs_v5_1_9.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=12)
        self.set_margins(15, 10, 15)  # 左右の余白を固定


def build_pdf(csv_path, video_path, athlete):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)

    calib = json.load(open(CALIB_PATH)) if os.path.exists(CALIB_PATH) else {}
    m_per_px = calib.get("m_per_px", 0.0016)

    summary = {
        "pitch": float(df["pitch_hz"].dropna().mean()),
        "stride": float(df["stride_m"].dropna().mean()),
        "stability": float(df["stability_score"].dropna().mean())
    }

    ai_comments = generate_ai_comments(summary)
    graph_path = create_graphs(df, athlete)
    font_path = get_font_path()

    pdf = PDF()
    pdf.add_page()
    if font_path:
        pdf.add_font("JP", "", font_path, uni=True)
        pdf.set_font("JP", "", 13)
    else:
        pdf.set_font("Helvetica", "", 13)

    page_width = pdf.w - pdf.l_margin - pdf.r_margin  # 有効幅

    # タイトル（中央）
    pdf.set_font_size(16)
    pdf.cell(page_width, 12, f"{athlete} フォーム分析レポート（v5.1.9）", new_y="NEXT", align="C")
    pdf.ln(3)

    # 基本情報（左寄せ・明示幅指定）
    pdf.set_font_size(11)
    pdf.cell(page_width, 8, f"動画: {os.path.basename(video_path)}", new_y="NEXT", align="L")
    pdf.cell(page_width, 8, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_y="NEXT", align="L")
    pdf.cell(page_width, 8, f"校正スケール: {m_per_px:.6f} m/px", new_y="NEXT", align="L")
    pdf.ln(3)

    # メトリクス（左揃え）
    pdf.cell(page_width, 8, f"ピッチ: {summary['pitch']:.2f} 歩/s", new_y="NEXT", align="L")
    pdf.cell(page_width, 8, f"ストライド: {summary['stride']:.2f} m", new_y="NEXT", align="L")
    pdf.cell(page_width, 8, f"安定性スコア: {summary['stability']:.2f}/10", new_y="NEXT", align="L")
    pdf.ln(5)

    # グラフ挿入
    pdf.image(graph_path, x=25, w=160)
    pdf.ln(8)

    # コメントセクション
    pdf.set_font_size(12)
    pdf.cell(page_width, 10, "AIフォーム解析コメント", new_y="NEXT", align="L")
    pdf.set_font_size(11)
    for c in ai_comments:
        for line in wrap_text(f"- {c}", 45):
            pdf.cell(5)  # ← インデント
            pdf.cell(page_width - 5, 7, line, new_y="NEXT", align="L")
    pdf.ln(5)

    # 総合所見
    pdf.set_font_size(12)
    pdf.cell(page_width, 10, "総合所見", new_y="NEXT", align="L")
    pdf.set_font_size(11)
    summary_text = (
        f"{athlete}選手はピッチ{summary['pitch']:.2f}歩/s、"
        f"ストライド{summary['stride']:.2f}m、安定性{summary['stability']:.1f}/10と"
        "バランスの取れた走行フォームを示しています。"
        "接地姿勢と体幹維持の良さが速度の安定に寄与しています。"
    )
    for line in wrap_text(summary_text, 45):
        pdf.cell(page_width, 7, line, new_y="NEXT", align="L")

    # 出力
    out_path = os.path.join(OUTPUT_DIR, f"{athlete}_form_report_v5_1_9.pdf")
    pdf.output(out_path)
    print(f"✅ PDF出力完了: {out_path}")
    print("🎯 右寄り文字をすべて修正済み（全行左揃え）")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", default="outputs/koike_pose_metrics_v3.csv")
    parser.add_argument("--athlete", default="不明選手")
    args = parser.parse_args()
    build_pdf(args.csv, args.video, args.athlete)














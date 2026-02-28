# ==========================================================
# 🧩 pose_reporter_pdf_ai_v5_1_6.py
# ✅ 日本語文字完全描画版（IPAexGothicフォント使用）
# ==========================================================
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import textwrap
import matplotlib

# --- IPAexゴシックを使用して日本語対応 ---
# フォントをダウンロードして次の場所に配置してください:
# C:\Users\Futamura\KJACai\fonts\ipaexg.ttf
matplotlib.rcParams['font.family'] = 'IPAexGothic'

CALIB_PATH = "outputs/jsonl/calibration_result.json"
OUTPUT_DIR = "outputs/pdf"

FONT_PATHS = [
    "C:/Users/Futamura/KJACai/fonts/ipaexg.ttf",
    "C:/Windows/Fonts/ipaexg.ttf",
    "C:/Windows/Fonts/YuGothR.ttc",
    "C:/Windows/Fonts/msgothic.ttc"
]

def get_font_path():
    for path in FONT_PATHS:
        if os.path.exists(path):
            return path
    print("⚠ 日本語フォントが見つかりません。ipaexg.ttfを指定パスに配置してください。")
    return None

# --- テキストを安全に改行 ---
def wrap_text(text, width=45):
    lines = textwrap.wrap(text, width)
    return lines if lines else [""]

# --- AIコメント生成 ---
def generate_ai_comments(summary):
    pitch, stride, stability = summary["pitch"], summary["stride"], summary["stability"]
    comments = []
    if pitch < 3.5:
        comments.append("ピッチが低めです。脚の回転スピードを意識してリズムを高めましょう。")
    elif pitch < 4.5:
        comments.append("ピッチは安定しています。終盤もこのテンポを維持できると良いです。")
    else:
        comments.append("ピッチが非常に良好で、反応速度も高いです。")

    if stride < 1.2:
        comments.append("ストライドがやや小さいため、骨盤の前傾保持を意識しましょう。")
    elif stride < 1.6:
        comments.append("ストライドは標準的です。接地姿勢が安定しています。")
    else:
        comments.append("ストライドが大きく、伸びのあるフォームです。")

    if stability < 7.0:
        comments.append("体幹がやや不安定です。ブレを抑えるトレーニングを取り入れましょう。")
    elif stability < 9.0:
        comments.append("安定性は良好です。リズムに一貫性があります。")
    else:
        comments.append("安定性が非常に高く、理想的な重心維持ができています。")
    return comments

# --- グラフ作成 ---
def create_graphs(df, athlete):
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    plt.subplots_adjust(hspace=0.4)
    axs[0].plot(df["time_s"], df["speed_mps"], linewidth=1.5)
    axs[0].set_title("区間速度の変化 [m/s]")
    axs[1].plot(df["time_s"], df["tilt_deg"], color="orange", linewidth=1.5)
    axs[1].set_title("体幹傾き角度の推移 [°]")
    axs[2].plot(df["time_s"], df["COM_y_m"], color="green", linewidth=1.5)
    axs[2].set_title("重心高さの変化 [m]")
    for ax in axs:
        ax.set_xlabel("時間 [s]")
    plt.tight_layout()
    path = f"outputs/{athlete}_graphs_v5_1_6.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path

# --- PDF生成 ---
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

    pdf = FPDF()
    pdf.add_page()
    if font_path:
        pdf.add_font("JP", "", font_path, uni=True)
        pdf.set_font("JP", "", 13)
    else:
        pdf.set_font("Helvetica", "", 13)

    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.1.6）", new_y="NEXT")
    pdf.set_font_size(11)
    pdf.cell(0, 8, f"動画: {os.path.basename(video_path)}", new_y="NEXT")
    pdf.cell(0, 8, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_y="NEXT")
    pdf.cell(0, 8, f"校正スケール: {m_per_px:.6f} m/px", new_y="NEXT")
    pdf.ln(4)

    pdf.cell(0, 8, f"ピッチ: {summary['pitch']:.2f} 歩/s", new_y="NEXT")
    pdf.cell(0, 8, f"ストライド: {summary['stride']:.2f} m", new_y="NEXT")
    pdf.cell(0, 8, f"安定性スコア: {summary['stability']:.2f}/10", new_y="NEXT")
    pdf.ln(5)

    pdf.image(graph_path, x=25, w=160)
    pdf.ln(8)

    pdf.set_font_size(12)
    pdf.cell(0, 10, "AIフォーム解析コメント", new_y="NEXT")
    pdf.set_font_size(11)

    for c in ai_comments:
        for line in wrap_text(f"- {c}", 45):
            pdf.cell(0, 7, line, new_y="NEXT")
    pdf.ln(8)

    pdf.set_font_size(12)
    pdf.cell(0, 10, "総合所見", new_y="NEXT")
    pdf.set_font_size(11)
    summary_text = (
        f"{athlete}選手はピッチ{summary['pitch']:.2f}歩/s、"
        f"ストライド{summary['stride']:.2f}m、安定性{summary['stability']:.1f}/10と"
        "バランスの取れた走行フォームを示しています。"
        "接地姿勢と体幹維持の良さが速度安定に寄与しています。"
    )
    for line in wrap_text(summary_text, 45):
        pdf.cell(0, 7, line, new_y="NEXT")

    out_path = os.path.join(OUTPUT_DIR, f"{athlete}_form_report_v5_1_6.pdf")
    pdf.output(out_path)
    print(f"✅ PDF出力完了: {out_path}")
    print("🈶 文字が出ない場合は ipaexg.ttf のパスを再確認してください。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", default="outputs/koike_pose_metrics_v3.csv")
    parser.add_argument("--athlete", default="不明選手")
    args = parser.parse_args()
    build_pdf(args.csv, args.video, args.athlete)












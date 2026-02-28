# ==========================================================
# 🧩 pose_reporter_pdf_ai_v5_1_2.py
# 📘 単走フォーム分析 + AIコメント + 3グラフ統合PDF出力版（安定版）
# ==========================================================
# 対応CSV: pose_metrics_analyzer_v3_4.py 出力（17列対応）
# 出力構成:
#   1. メトリクス要約（ピッチ・ストライド・安定性）
#   2. グラフ3種（速度[m/s], 傾き[°], COM高さ[m]）
#   3. AIコメント（定量連動）
#   4. 総合所見
# ==========================================================

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import warnings
import matplotlib

# ==========================================================
# グラフ日本語フォント対応
# ==========================================================
matplotlib.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'IPAexGothic']
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ==========================================================
# パス設定
# ==========================================================
CALIB_PATH = "outputs/jsonl/calibration_result.json"
OUTPUT_DIR = "outputs/pdf"
FONT_PATHS = [
    "C:/Windows/Fonts/YuGothR.ttc",   # Windows
    "C:/Windows/Fonts/MSGOTHIC.TTC",
    "C:/Windows/Fonts/IPAexGothic.ttf"
]

# ==========================================================
# フォント自動選択
# ==========================================================
def get_available_font():
    for path in FONT_PATHS:
        if os.path.exists(path):
            return path
    return None

# ==========================================================
# AIコメント生成
# ==========================================================
def generate_ai_comments(summary):
    pitch, stride, stability = summary["pitch"], summary["stride"], summary["stability"]
    comments = []

    # ピッチコメント
    if pitch < 3.5:
        comments.append("ピッチが低めです。リズム強化ドリルで脚回転速度を上げましょう。")
    elif pitch < 4.5:
        comments.append("ピッチは標準的です。接地後のリカバリーを速くしてリズム向上を狙いましょう。")
    else:
        comments.append("ピッチは非常に良好です。終盤まで維持できるよう体幹安定を意識しましょう。")

    # ストライドコメント
    if stride < 1.2:
        comments.append("ストライドがやや小さいため、骨盤の前傾維持と地面の押し出し強化が効果的です。")
    elif stride < 1.6:
        comments.append("ストライドは安定しています。接地時の姿勢保持ができています。")
    else:
        comments.append("ストライドが大きく、伸びのあるフォームです。反発の使い方が非常に良いです。")

    # 安定性コメント
    if stability < 7.0:
        comments.append("体幹のブレが少し大きいです。コアトレーニングや姿勢保持意識が改善ポイントです。")
    elif stability < 9.0:
        comments.append("安定性は良好。終盤のフォーム維持に寄与しています。")
    else:
        comments.append("安定性は非常に高く、全体のバランスが優れています。")

    return comments

# ==========================================================
# グラフ作成（3グラフ縦並び）
# ==========================================================
def create_graphs(df, athlete):
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    plt.subplots_adjust(hspace=0.4)

    # 1️⃣ 速度
    axs[0].plot(df["time_s"], df["speed_mps"], linewidth=1.5)
    axs[0].set_title("区間速度の変化 [m/s]", fontsize=11)
    axs[0].set_xlabel("時間 [s]")
    axs[0].set_ylabel("速度 [m/s]")

    # 2️⃣ 傾き角
    axs[1].plot(df["time_s"], df["tilt_deg"], color="orange", linewidth=1.5)
    axs[1].set_title("体幹傾き角度の推移 [°]", fontsize=11)
    axs[1].set_xlabel("時間 [s]")
    axs[1].set_ylabel("角度 [°]")

    # 3️⃣ COM高さ
    axs[2].plot(df["time_s"], df["COM_y_m"], color="green", linewidth=1.5)
    axs[2].set_title("重心高さの変化 [m]", fontsize=11)
    axs[2].set_xlabel("時間 [s]")
    axs[2].set_ylabel("高さ [m]")

    graph_path = f"outputs/{athlete}_graphs_v5_1_2.png"
    plt.tight_layout()
    plt.savefig(graph_path, dpi=200)
    plt.close(fig)
    return graph_path

# ==========================================================
# PDF生成
# ==========================================================
def build_pdf(csv_path, video_path, athlete):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)

    # 校正値読込
    calib = json.load(open(CALIB_PATH)) if os.path.exists(CALIB_PATH) else {}
    m_per_px = calib.get("m_per_px", 0.0016)

    # 要約統計
    summary = {
        "pitch": float(df["pitch_hz"].dropna().mean()),
        "stride": float(df["stride_m"].dropna().mean()),
        "stability": float(df["stability_score"].dropna().mean())
    }

    ai_comments = generate_ai_comments(summary)
    graph_path = create_graphs(df, athlete)

    font_path = get_available_font()
    pdf = FPDF()
    pdf.add_page()
    if font_path:
        pdf.add_font("JP", "", font_path, uni=True)
        pdf.set_font("JP", "", 13)
    else:
        pdf.set_font("Helvetica", "", 13)

    # タイトル
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.1.2）", ln=1)
    pdf.set_font_size(11)
    pdf.cell(0, 8, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 8, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 8, f"校正スケール: {m_per_px:.6f} m/px", ln=1)
    pdf.ln(5)

    # 定量値
    pdf.cell(0, 8, f"ピッチ: {summary['pitch']:.2f} 歩/s", ln=1)
    pdf.cell(0, 8, f"ストライド: {summary['stride']:.2f} m", ln=1)
    pdf.cell(0, 8, f"安定性スコア: {summary['stability']:.2f} / 10", ln=1)
    pdf.ln(5)

    # グラフ挿入
    pdf.image(graph_path, x=25, w=160)
    pdf.ln(8)

    # AIコメント
    pdf.set_font_size(12)
    pdf.cell(0, 10, "AIフォーム解析コメント", ln=1)
    pdf.set_font_size(11)
    for c in ai_comments:
        safe_text = c.replace("🏃‍♀️", "").replace("📈", "").replace("💬", "").replace("🧠", "")
        pdf.multi_cell(0, 8, f"- {safe_text}")
    pdf.ln(10)

    # 総合所見
    pdf.set_font_size(12)
    pdf.cell(0, 10, "総合所見", ln=1)
    pdf.set_font_size(11)
    summary_text = (
        f"{athlete}選手は、ピッチ{summary['pitch']:.2f}歩/s・ストライド{summary['stride']:.2f}m・"
        f"安定性{summary['stability']:.1f}/10と総合的にバランスの取れたフォームを示しています。\n"
        "特に体幹の安定性が高く、上半身の無駄な動きが少ないことが特徴です。"
    )
    pdf.multi_cell(0, 8, summary_text)

    # 出力
    output_path = os.path.join(OUTPUT_DIR, f"{athlete}_form_report_v5_1_2.pdf")
    pdf.output(output_path)
    print(f"✅ PDF出力完了: {output_path}")
    return output_path

# ==========================================================
# 実行エントリ
# ==========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", default="outputs/koike_pose_metrics_v3.csv")
    parser.add_argument("--athlete", default="不明選手")
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)









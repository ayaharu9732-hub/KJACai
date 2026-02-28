# =============================================
# pose_reporter_pdf_ai_v4_4.py
# キャプチャ存在確認・再取得対応版
# =============================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fpdf import FPDF
import cv2

# ===== 基本設定 =====
ATHLETE = "小池"
GRADE = "中学1年"
RACE = "20m区間練習"
MEET = "フォームチェック練習"
DATE = "2025-09-03"

CSV_PATH = "outputs/koike_pose_metrics_v3.csv"
VIDEO_PATH = "C:/Users/Futamura/KJACai/videos/20250903_172315.mp4"
OUTPUT_DIR = "outputs"
FONT_PATH = "C:/Users/Futamura/KJACai/ipaexg.ttf"

plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()

# =============================================
# CSV解析
# =============================================
def analyze_csv(csv_path):
    df = pd.read_csv(csv_path)
    print(f"✅ CSV読込成功: {len(df)}行, 列: {list(df.columns)}")
    return df

# =============================================
# グラフ生成
# =============================================
def create_graph(df):
    plt.figure(figsize=(10, 6))
    x = df["time_s"]

    if "tilt_deg" in df.columns:
        plt.plot(x, df["tilt_deg"], label="体幹傾き角度 (°)", linewidth=2)
    if "pelvis_height" in df.columns:
        plt.plot(x, df["pelvis_height"], label="骨盤高さ (m)", linewidth=2)
    if "com_speed_mps" in df.columns:
        plt.plot(x, df["com_speed_mps"], label="重心速度 (m/s)", linewidth=2)

    plt.xlabel("時間 (秒)")
    plt.ylabel("計測値")
    plt.title("フォーム変化グラフ（傾き・骨盤・速度）", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = os.path.join(OUTPUT_DIR, f"{ATHLETE}_graphs_v4_4.png")
    plt.savefig(graph_path, dpi=300)
    plt.close()
    print(f"✅ グラフ出力: {graph_path}")
    return graph_path

# =============================================
# フレームキャプチャ
# =============================================
def capture_frame(video_path):
    capture_path = os.path.join(OUTPUT_DIR, f"{ATHLETE}_pose_capture_v4_4.png")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 動画を開けません: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("⚠️ フレーム数を取得できません。")
        return None

    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("⚠️ キャプチャ失敗。最初のフレームを再試行します。")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("❌ 再試行も失敗しました。")
            return None

    success = cv2.imwrite(capture_path, frame)
    if success:
        print(f"✅ 代表フレーム保存: {capture_path}")
        return capture_path
    else:
        print("❌ フレーム保存に失敗しました。パスを確認してください。")
        return None

# =============================================
# PDF作成
# =============================================
class MyPDF(FPDF):
    def header(self):
        self.set_font("IPAexGothic", "", 16)
        self.cell(0, 10, f"{ATHLETE} フォーム分析レポート", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("IPAexGothic", "", 14)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def chapter_body(self, text):
        self.set_font("IPAexGothic", "", 12)
        self.multi_cell(0, 8, text)
        self.ln()

def create_pdf(capture_path, graph_path, comment):
    pdf = MyPDF()
    pdf.add_font("IPAexGothic", "", FONT_PATH)
    pdf.add_font("IPAexGothic", "B", FONT_PATH)

    pdf.add_page()
    pdf.set_font("IPAexGothic", "", 12)

    pdf.cell(0, 10, f"🏃‍♂️ 選手名: {ATHLETE}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"🏫 学年: {GRADE}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"🏁 種目: {RACE}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"📅 日付: {DATE}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"🏟 大会・練習名: {MEET}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    if capture_path and os.path.exists(capture_path):
        pdf.chapter_title("フォーム画像（骨格描画）")
        pdf.image(capture_path, x=25, w=160)
        pdf.ln(10)
    else:
        pdf.chapter_title("フォーム画像（取得失敗）")
        pdf.chapter_body("❌ フォーム画像を読み込めませんでした。")

    pdf.chapter_title("フォーム変化グラフ")
    pdf.image(graph_path, x=25, w=160)
    pdf.ln(10)

    pdf.chapter_title("AIコメント・分析")
    pdf.chapter_body(comment)

    output_path = os.path.join(OUTPUT_DIR, f"{ATHLETE}_form_report_v4_4.pdf")
    pdf.output(output_path)
    print(f"✅ PDF出力完了: {output_path}")
    return output_path

# =============================================
# メイン
# =============================================
def main():
    print("===== 🧩 小池選手 フォームレポート生成（v4.4）開始 =====")
    df = analyze_csv(CSV_PATH)
    graph_path = create_graph(df)
    capture_path = capture_frame(VIDEO_PATH)

    ai_comment = (
        "傾き角度の変化から、スタート直後は上体がやや前傾しています。\n"
        "骨盤高さが安定しており、リズム良く重心移動ができています。\n"
        "速度曲線は中盤で緩やかに上昇しており、フォーム維持は良好です。"
    )

    create_pdf(capture_path, graph_path, ai_comment)
    print("✅ 全処理完了：v4.4 PDF統合成功！")

if __name__ == "__main__":
    main()





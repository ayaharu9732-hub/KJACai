import os
import cv2
import argparse
import pandas as pd
from fpdf import FPDF
from datetime import datetime


# ============================================================
# pose_overlay から3枚抽出
# ============================================================

def save_pose_images(overlay_path, video_name):
    """
    pose_overlay 動画から start/mid/finish フレームを保存
    """
    if not os.path.exists(overlay_path):
        print(f"⚠ オーバーレイ動画が見つかりません: {overlay_path}")
        return None

    cap = cv2.VideoCapture(overlay_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 3:
        print("⚠ フレーム数が少なすぎます")
        return None

    frames = {
        "start": 0,
        "mid": total // 2,
        "finish": total - 1
    }

    out_dir = f"outputs/pose_images/{video_name}"
    os.makedirs(out_dir, exist_ok=True)
    saved = {}

    for key, idx in frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = f"{out_dir}/{key}.png"
        cv2.imwrite(out_path, frame)
        saved[key] = out_path

    cap.release()
    print(f"📸 骨格画像出力: {saved}")
    return saved


# ============================================================
# PDF作成
# ============================================================

def build_pdf(csv_path, video_path, athlete):
    df = pd.read_csv(csv_path)

    summary_row = df[df['frame'] == '_summary_']
    summary = {
        "pitch": float(summary_row.iloc[0, -4]),
        "stride": float(summary_row.iloc[0, -3]),
        "stability": float(summary_row.iloc[0, -1]),
    }

    # -----------------------------
    # 修正：スペース除去した動画名で overlay を探す
    # -----------------------------
    raw_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = raw_name.replace(" ", "")  # ←これが重要
    overlay_path = f"outputs/images/{video_name}_pose_overlay.mp4"

    # 骨格画像抽出
    pose_images = save_pose_images(overlay_path, video_name)

    # -----------------------------
    # PDF
    # -----------------------------

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("JP", "", "c:/Windows/Fonts/meiryo.ttc", uni=True)
    pdf.set_font("JP", "", 14)

    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.2.1）", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_font("JP", "", 12)
    pdf.cell(0, 8, f"ピッチ: {summary['pitch']:.2f} 歩/s", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"ストライド: {summary['stride']:.2f} m", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"安定性スコア: {summary['stability']:.2f}/10", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_font("JP", "", 13)
    pdf.cell(0, 10, "骨格画像", new_x="LMARGIN", new_y="NEXT")

    if pose_images:
        for label, path in pose_images.items():
            pdf.set_font("JP", "", 11)
            pdf.cell(0, 6, f"{label} フレーム:", new_x="LMARGIN", new_y="NEXT")
            pdf.image(path, w=120)
            pdf.ln(5)
    else:
        pdf.cell(0, 10, "⚠ 骨格画像が取得できませんでした。", new_x="LMARGIN", new_y="NEXT")

    out_pdf = f"outputs/pdf/{athlete}_form_report_v5_2_1.pdf"
    pdf.output(out_pdf)

    print(f"✅ PDF出力完了: {out_pdf}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--athlete", required=True)
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)


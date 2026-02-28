import os
import json
import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF


# =========================
#  ヘルパー
# =========================

def load_calibration(json_path="outputs/jsonl/calibration_result.json"):
    """距離校正結果の読み込み（なければ 1.0 を返す）"""
    if not os.path.exists(json_path):
        print("⚠ 校正ファイルが見つかりません:", json_path)
        return 1.0
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m_per_px = data.get("m_per_px", 1.0)
    print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
    return m_per_px


def make_safe_stem(video_basename_no_ext: str) -> str:
    """
    ファイル名からレポート用の短い stem を作成。
    今回は「二村遥香10.5」→「10_5」となるように調整。
    """
    stem = video_basename_no_ext
    # 特定の名前要素は削って数値部分だけにする（暫定運用）
    for ch in ["二村", "遥香", " ", "　"]:
        stem = stem.replace(ch, "")
    stem = stem.replace(".", "_")
    if not stem:
        stem = video_basename_no_ext.replace(".", "_")
    return stem


def load_metrics_and_summary(csv_path: str):
    """
    メトリクスCSVを読み込み、データ本体と_summary_行から集計値を取り出す。
    戻り値: (df_data, summary_dict)
    """
    df_raw = pd.read_csv(csv_path)
    # frame 列に _summary_ が入っている行を探す
    mask_summary = df_raw["frame"].astype(str) == "_summary_"
    if mask_summary.any():
        row = df_raw[mask_summary].iloc[0].tolist()
        # 末尾5列: pitch, stride, stability_std_y_m, stability_jerk, stability_score
        pitch = float(row[-5])
        stride = float(row[-4])
        stability_score = float(row[-1])
        summary = {
            "pitch": pitch,
            "stride": stride,
            "stability": stability_score,
        }
        df = df_raw[~mask_summary].copy()
    else:
        # summary がない場合は適当に計算
        df = df_raw.copy()
        pitch = float(df.get("pitch_hz", pd.Series([0])).mean())
        stride = float(df.get("stride_m", pd.Series([0])).mean())
        stability_score = float(df.get("stability_score", pd.Series([0])).mean())
        summary = {
            "pitch": pitch,
            "stride": stride,
            "stability": stability_score,
        }
    return df, summary


def load_angles(angle_csv_path: str):
    """角度CSVを読み込む"""
    if not os.path.exists(angle_csv_path):
        print("⚠ 角度CSVが見つかりません:", angle_csv_path)
        return None
    df = pd.read_csv(angle_csv_path)
    return df


def extract_pose_frames(video_path: str, stem: str):
    """
    オーバーレイ動画 outputs/images/<video_name>_pose_overlay.mp4 から
    start / mid / finish の3枚を抜き出して保存。
    戻り値: {'start': path, 'mid': path, 'finish': path}
    """
    base = os.path.splitext(os.path.basename(video_path))[0]
    overlay_path = os.path.join("outputs", "images", f"{base}_pose_overlay.mp4")
    if not os.path.exists(overlay_path):
        print("⚠ オーバーレイ動画が見つかりません:", overlay_path)
        return {}

    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        print("⚠ オーバーレイ動画を開けません:", overlay_path)
        return {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("⚠ オーバーレイ動画のフレーム数が 0 です")
        cap.release()
        return {}

    indices = [
        int(frame_count * 0.05),
        int(frame_count * 0.50),
        max(frame_count - 2, 0),
    ]
    labels = ["start", "mid", "finish"]

    out_dir = os.path.join("outputs", "pose_images", stem)
    os.makedirs(out_dir, exist_ok=True)

    saved = {}
    for idx, label in zip(indices, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(out_dir, f"{label}.png")
        cv2.imwrite(out_path, frame)
        saved[label] = out_path

    cap.release()
    print("📸 骨格画像出力:", saved)
    return saved


def create_main_graph(df_metrics: pd.DataFrame, stem: str):
    """
    速度・体幹傾き・COM高さの3グラフを縦並びで1枚に出力
    """
    os.makedirs("outputs/graphs", exist_ok=True)
    out_path = os.path.join("outputs", "graphs", f"{stem}_graphs_v5_3_0.png")

    t = df_metrics.get("time_s", pd.Series(range(len(df_metrics))))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # 速度
    axes[0].plot(t, df_metrics.get("speed_mps", pd.Series([0] * len(df_metrics))))
    axes[0].set_ylabel("速度 [m/s]")
    axes[0].set_title("速度の変化")

    # 傾き
    axes[1].plot(t, df_metrics.get("tilt_deg", pd.Series([0] * len(df_metrics))))
    axes[1].set_ylabel("体幹傾斜 [deg]")
    axes[1].set_title("体幹の前後傾き")

    # COM（縦位置）
    axes[2].plot(t, df_metrics.get("COM_y", pd.Series([0] * len(df_metrics))))
    axes[2].set_xlabel("時間 [s]")
    axes[2].set_ylabel("COM_y (正規化)")
    axes[2].set_title("重心高さの変化")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("📊 メイントラッキンググラフ出力:", out_path)
    return out_path


def create_angle_graphs(df_metrics: pd.DataFrame, df_angles: pd.DataFrame, stem: str):
    """
    膝・股関節・足首・腕・体幹・左右差サマリーを
    3列×2行サブプロットにまとめる
    """
    os.makedirs("outputs/graphs", exist_ok=True)
    out_path = os.path.join("outputs", "graphs", f"{stem}_angles_v5_3_0.png")

    # time_s を metrics から持ってくる（frame で join）
    time_df = df_metrics[["frame", "time_s"]].copy()
    time_df["frame"] = time_df["frame"].astype(int)

    angles = df_angles.copy()
    angles["frame"] = angles["frame"].astype(int)
    angles = pd.merge(angles, time_df, on="frame", how="left")
    t = angles["time_s"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # 1. 膝角度
    ax1.plot(t, angles["left_knee"], label="左")
    ax1.plot(t, angles["right_knee"], label="右")
    ax1.set_title("膝角度（左・右）")
    ax1.set_ylabel("角度 [deg]")
    ax1.legend()

    # 2. 股関節角度
    ax2.plot(t, angles["left_hip"], label="左")
    ax2.plot(t, angles["right_hip"], label="右")
    ax2.set_title("股関節角度（左・右）")
    ax2.set_ylabel("角度 [deg]")
    ax2.legend()

    # 3. 足首角度
    ax3.plot(t, angles["left_ankle"], label="左")
    ax3.plot(t, angles["right_ankle"], label="右")
    ax3.set_title("足首角度（左・右）")
    ax3.set_ylabel("角度 [deg]")
    ax3.legend()

    # 4. 腕角度
    ax4.plot(t, angles["left_arm"], label="左")
    ax4.plot(t, angles["right_arm"], label="右")
    ax4.set_title("腕角度（左・右）")
    ax4.set_ylabel("角度 [deg]")
    ax4.set_xlabel("時間 [s]")
    ax4.legend()

    # 5. 体幹傾斜
    ax5.plot(t, angles["torso_tilt_left"], label="左")
    ax5.plot(t, angles["torso_tilt_right"], label="右")
    ax5.set_title("体幹傾斜（左・右）")
    ax5.set_ylabel("傾き [deg]")
    ax5.set_xlabel("時間 [s]")
    ax5.legend()

    # 6. 左右差サマリー（棒グラフ）
    joints = ["knee", "hip", "ankle", "arm", "torso"]
    diffs = []
    diffs.append(np.nanmean(np.abs(angles["left_knee"] - angles["right_knee"])))
    diffs.append(np.nanmean(np.abs(angles["left_hip"] - angles["right_hip"])))
    diffs.append(np.nanmean(np.abs(angles["left_ankle"] - angles["right_ankle"])))
    diffs.append(np.nanmean(np.abs(angles["left_arm"] - angles["right_arm"])))
    diffs.append(
        np.nanmean(np.abs(angles["torso_tilt_left"] - angles["torso_tilt_right"]))
    )

    ax6.bar(joints, diffs)
    ax6.set_title("左右差サマリー（平均角度差）")
    ax6.set_ylabel("左右差 [deg]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print("📊 角度グラフ出力:", out_path)
    return out_path


def generate_ai_comments(summary: dict, df_angles: pd.DataFrame):
    """
    速度＋角度から簡易的なAIコメントを生成（ルールベース）
    """
    comments = []

    pitch = summary.get("pitch", 0.0)
    stride = summary.get("stride", 0.0)
    stability = summary.get("stability", 0.0)

    # 左右差
    knee_diff = float(np.nanmean(np.abs(df_angles["left_knee"] - df_angles["right_knee"])))
    hip_diff = float(np.nanmean(np.abs(df_angles["left_hip"] - df_angles["right_hip"])))
    ankle_diff = float(
        np.nanmean(np.abs(df_angles["left_ankle"] - df_angles["right_ankle"]))
    )
    arm_diff = float(np.nanmean(np.abs(df_angles["left_arm"] - df_angles["right_arm"])))
    torso_diff = float(
        np.nanmean(
            np.abs(df_angles["torso_tilt_left"] - df_angles["torso_tilt_right"])
        )
    )

    # 1) 全体リズム
    comments.append(
        f"ピッチ約 {pitch:.2f} 歩/秒・ストライド約 {stride:.2f} m というバランスで走れており、全体としてリズムの良い疾走フォームになっています。"
    )

    # 2) 膝・股関節
    if knee_diff < 4 and hip_diff < 4:
        comments.append(
            f"膝と股関節の左右差（平均で膝 {knee_diff:.1f} 度・股関節 {hip_diff:.1f} 度）は小さく、左右均等に力を出せている状態です。"
        )
    else:
        comments.append(
            f"膝または股関節の左右差（膝 {knee_diff:.1f} 度・股関節 {hip_diff:.1f} 度）がやや大きめなので、片脚だけ蹴りすぎていないかを意識するとさらに安定します。"
        )

    # 3) 足首・接地
    if ankle_diff < 5:
        comments.append(
            f"足首角度の左右差は平均 {ankle_diff:.1f} 度で、接地の向きは大きく乱れていません。つま先で地面を『前に押す』意識を続けると加速区間がさらに安定します。"
        )
    else:
        comments.append(
            f"足首角度の左右差が平均 {ankle_diff:.1f} 度と少し大きめなので、接地時に両足とも同じ向きで着くイメージを持つとブレーキが減らせます。"
        )

    # 4) 体幹
    if abs(torso_diff) < 3 and stability >= 8.0:
        comments.append(
            f"体幹の左右差（平均 {torso_diff:.1f} 度）と安定性スコア {stability:.1f}/10 から、上半身のブレはかなり小さく、真っ直ぐ前に進むフォームができています。"
        )
    else:
        comments.append(
            f"体幹の左右差は平均 {torso_diff:.1f} 度で、わずかに傾く時間帯があります。上半身を『おへそから前に運ぶ』イメージを持つと、軸がさらに安定してスピードロスが減ります。"
        )

    # 5) 腕振り
    if arm_diff < 6:
        comments.append(
            f"腕振りの左右差（平均 {arm_diff:.1f} 度）は小さく、脚の動きとリズムよく連動できています。腕をリラックスさせたまま大きく振れている点は今後も続けたい良いポイントです。"
        )
    else:
        comments.append(
            f"腕振り角度の左右差が平均 {arm_diff:.1f} 度あり、どちらか一方の腕だけ大きく振る傾向が見られます。肩の力を抜いて、左右同じリズム・同じ振り幅を意識すると下半身のブレも改善します。"
        )

    return comments


# =========================
#  PDF構築
# =========================

def build_pdf(csv_path: str, video_path: str, athlete: str):
    # 校正値
    m_per_px = load_calibration()

    # メトリクスとサマリー
    df_metrics, summary = load_metrics_and_summary(csv_path)

    # 角度データ
    base = os.path.splitext(os.path.basename(video_path))[0]
    angle_csv = os.path.join("outputs", f"angles_{base}.csv")
    df_angles = load_angles(angle_csv)
    if df_angles is None:
        print("❌ 角度CSVがないため、角度グラフとAIコメントはスキップされます。")

    # stem（10_5 など）
    stem = make_safe_stem(base)

    # 骨格画像（start/mid/finish）
    pose_imgs = extract_pose_frames(video_path, stem)

    # グラフ生成
    main_graph = create_main_graph(df_metrics, stem)
    angle_graph = None
    if df_angles is not None:
        angle_graph = create_angle_graphs(df_metrics, df_angles, stem)

    # PDF出力先
    os.makedirs("outputs/pdf", exist_ok=True)
    out_pdf = os.path.join("outputs", "pdf", f"{athlete}_form_report_v5_3_0.pdf")

    # PDF作成
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)

    # 日本語フォント（Meiryo）
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    if os.path.exists(font_path):
        pdf.add_font("JP", "", font_path, uni=True)
        pdf.set_font("JP", "", 12)
    else:
        pdf.set_font("Helvetica", "", 12)

    # -------------------------
    # 1ページ目
    # -------------------------
    pdf.add_page()

    # タイトル
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.3.0）", ln=1)
    pdf.set_font_size(10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 6, f"校正スケール: {m_per_px:.6f} m/px", ln=1)

    pdf.ln(2)
    pdf.cell(0, 6, f"ピッチ: {summary.get('pitch', 0.0):.2f} 歩/s", ln=1)
    pdf.cell(0, 6, f"ストライド: {summary.get('stride', 0.0):.2f} m", ln=1)
    pdf.cell(0, 6, f"安定性スコア: {summary.get('stability', 0.0):.2f} / 10", ln=1)

    pdf.ln(2)
    pdf.cell(0, 6, "骨格オーバーレイ（start / mid / finish）", ln=1)

    # 骨格3枚を横並び
    if pose_imgs:
        y_top = pdf.get_y() + 2
        page_width = pdf.w - 2 * pdf.l_margin
        img_width = page_width / 3.0 - 4
        x_positions = [
            pdf.l_margin,
            pdf.l_margin + img_width + 2,
            pdf.l_margin + (img_width + 2) * 2,
        ]
        labels = ["start", "mid", "finish"]
        max_h = 0
        for x, label in zip(x_positions, labels):
            if label in pose_imgs:
                pdf.image(pose_imgs[label], x=x, y=y_top, w=img_width)
                # 高さはおおよそ  img_width * (9/16) くらいと仮定
                max_h = max(max_h, img_width * 9 / 16)
        pdf.set_y(y_top + max_h + 4)
    else:
        pdf.ln(4)
        pdf.cell(0, 6, "※ 骨格オーバーレイ画像が見つからなかったため、このセクションは空欄です。", ln=1)

    pdf.ln(2)
    pdf.cell(0, 6, "速度・傾き・重心（COM）の変化グラフ", ln=1)
    pdf.ln(1)
    # メイングラフ
    pdf.image(main_graph, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)

    # -------------------------
    # 2ページ目：角度グラフ＋AIコメント
    # -------------------------
    pdf.add_page()
    pdf.set_font_size(12)
    pdf.cell(0, 8, "関節角度の推移と左右差", ln=1)
    pdf.ln(2)

    if angle_graph is not None and df_angles is not None:
        # 角度グラフ（6枚サブプロット画像）
        pdf.image(angle_graph, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
        pdf.ln(2)
        pdf.set_font_size(11)
        pdf.cell(0, 8, "AIフォーム解析コメント（速度＋角度の総合所見）", ln=1)
        pdf.ln(1)

        comments = generate_ai_comments(summary, df_angles)

        usable_width = pdf.w - 2 * pdf.l_margin
        pdf.set_font_size(10)
        for c in comments:
            # 安全な multi_cell 幅指定
            pdf.multi_cell(usable_width, 6, c)
            pdf.ln(1)
    else:
        pdf.set_font_size(10)
        pdf.cell(0, 8, "※ 角度データがなかったため、角度グラフとAI角度コメントは省略されています。", ln=1)

    pdf.output(out_pdf)
    print(f"✅ PDF出力完了: {out_pdf}")


# =========================
#  CLIエントリポイント
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス（koike_pose_metrics_v3.csv）")
    parser.add_argument("--athlete", required=True, help="選手名（PDFファイル名に使用）")
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()






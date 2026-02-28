#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pose_reporter_pdf_ai_v5_3_1.py

・koike_pose_metrics_v3.csv（速度・COM・安定性など）
・angles_<動画名>.csv（膝・股関節・足首・体幹・腕の角度）
から

 ① 速度・傾き・COMグラフ画像
 ② 角度グラフ画像
 ③ ピッチ・ストライド・安定性の要約
 ④ 角度まで含めた詳細AIコメント

を1つのPDFレポートにまとめて出力するスクリプト。

使い方：
  python scripts/pose_reporter_pdf_ai_v5_3_1.py \
      --video "videos/二村遥香10.5.mp4" \
      --csv outputs/koike_pose_metrics_v3.csv \
      --athlete "二村 遥香"

※角度CSVは自動で `outputs/angles_<動画ファイル名のstem>.csv`
  （例：angles_二村遥香10.5.csv）を探します。
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

# =========================
# 共通設定
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
PDF_DIR = OUTPUT_DIR / "pdf"
GRAPH_DIR = OUTPUT_DIR / "graphs"
CALIB_JSON = OUTPUT_DIR / "jsonl" / "calibration_result.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# matplotlib の日本語フォント設定（Meiryo）
plt.rcParams["font.family"] = "Meiryo"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 補助関数
# =========================

def load_calibration() -> float:
    """
    calibration_result.json から 1px あたり[m] を読み込み。
    見つからなければデフォルト 0.01[m/px] を返す。
    """
    if CALIB_JSON.exists():
        try:
            with open(CALIB_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            m_per_px = float(data.get("m_per_px", 0.01))
            print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
            return m_per_px
        except Exception as e:
            print(f"⚠ 校正ファイル読み込みエラー: {e}")
    print("⚠ 校正値が見つからないため、暫定 1px = 0.010000 m として扱います。")
    return 0.01


def load_metrics_csv(csv_path: str) -> pd.DataFrame:
    """
    速度・COM等のメトリクスCSV（koike_pose_metrics_v3.csv）読込。
    frame列を文字列として読み込み、_summary_ 行なども保持。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"メトリクスCSVが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"frame": str})
    # 数値行だけ解析に使う（_summary_ などを除外）
    df_numeric = df[df["frame"].str.match(r"^\d+$")].copy()
    df_numeric["frame"] = df_numeric["frame"].astype(int)
    df_numeric = df_numeric.sort_values("frame").reset_index(drop=True)

    print(f"✅ メトリクスCSV読込: {csv_path}  ({len(df_numeric)}行)")
    return df_numeric


def load_angle_csv_for_video(video_path: str) -> pd.DataFrame:
    """
    動画ファイル名から angles_*.csv を推定して読み込み。
    例：動画が 二村遥香10.5.mp4 → outputs/angles_二村遥香10.5.csv
    """
    video_stem = Path(video_path).stem
    angle_csv = OUTPUT_DIR / f"angles_{video_stem}.csv"
    if not angle_csv.exists():
        raise FileNotFoundError(f"角度CSVが見つかりません: {angle_csv}")

    df_angles = pd.read_csv(angle_csv)
    print(f"✅ 角度CSV読込: {angle_csv}  ({len(df_angles)}行)")
    return df_angles


def compute_summary(df_metrics: pd.DataFrame) -> dict:
    """
    メトリクスCSVからピッチ・ストライド・安定性の平均値をまとめて返す。
    """
    pitch = float(df_metrics["pitch_hz"].mean(skipna=True)) if "pitch_hz" in df_metrics.columns else 0.0
    stride = float(df_metrics["stride_m"].mean(skipna=True)) if "stride_m" in df_metrics.columns else 0.0
    stability = (
        float(df_metrics["stability_score"].mean(skipna=True))
        if "stability_score" in df_metrics.columns
        else 0.0
    )
    return {
        "pitch": pitch,
        "stride": stride,
        "stability": stability,
    }


# =========================
# グラフ生成
# =========================

def create_main_graph(df: pd.DataFrame, out_path: Path):
    """
    速度・体幹傾き・COM_Y の3つを縦に並べたグラフ画像を生成。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    time = df["time_s"].values if "time_s" in df.columns else df.index.values

    speed = df["speed_mps"].values if "speed_mps" in df.columns else None
    tilt = df["tilt_deg"].values if "tilt_deg" in df.columns else None
    com_y = df["COM_y"].values if "COM_y" in df.columns else None

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69 * 0.45), sharex=True)  # A4縦の半分くらい高さ

    # 1段目：速度
    if speed is not None:
        axes[0].plot(time, speed)
        axes[0].set_ylabel("速度 [m/s]")
        axes[0].set_title("時間ごとの速度変化")
        axes[0].grid(True)

    # 2段目：体幹傾き
    if tilt is not None:
        axes[1].plot(time, tilt)
        axes[1].set_ylabel("体幹傾き [deg]")
        axes[1].set_title("時間ごとの体幹傾き変化")
        axes[1].grid(True)

    # 3段目：COM高さ(正規化)
    if com_y is not None:
        axes[2].plot(time, com_y)
        axes[2].set_ylabel("COM高さ(正規化)")
        axes[2].set_xlabel("時間 [s]")
        axes[2].set_title("時間ごとのCOM高さ変化")
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"📊 メイングラフ出力: {out_path}")


def create_angle_graphs(df_angles: pd.DataFrame, out_path: Path):
    """
    角度用グラフ（膝・股関節・足首・体幹・腕）を1枚の画像にまとめて出力。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = df_angles["frame"].values if "frame" in df_angles.columns else df_angles.index.values

    fig, axes = plt.subplots(3, 2, figsize=(8.27, 11.69 * 0.7))
    axes = axes.flatten()

    # 膝
    if {"left_knee", "right_knee"}.issubset(df_angles.columns):
        axes[0].plot(frames, df_angles["left_knee"], label="左膝")
        axes[0].plot(frames, df_angles["right_knee"], label="右膝")
        axes[0].set_title("膝角度の変化")
        axes[0].set_ylabel("[deg]")
        axes[0].legend()
        axes[0].grid(True)

    # 股関節
    if {"left_hip", "right_hip"}.issubset(df_angles.columns):
        axes[1].plot(frames, df_angles["left_hip"], label="左股関節")
        axes[1].plot(frames, df_angles["right_hip"], label="右股関節")
        axes[1].set_title("股関節角度の変化")
        axes[1].set_ylabel("[deg]")
        axes[1].legend()
        axes[1].grid(True)

    # 足首
    if {"left_ankle", "right_ankle"}.issubset(df_angles.columns):
        axes[2].plot(frames, df_angles["left_ankle"], label="左足首")
        axes[2].plot(frames, df_angles["right_ankle"], label="右足首")
        axes[2].set_title("足首角度の変化")
        axes[2].set_ylabel("[deg]")
        axes[2].legend()
        axes[2].grid(True)

    # 体幹傾斜
    if {"torso_tilt_left", "torso_tilt_right"}.issubset(df_angles.columns):
        axes[3].plot(frames, df_angles["torso_tilt_left"], label="左体幹")
        axes[3].plot(frames, df_angles["torso_tilt_right"], label="右体幹")
        axes[3].set_title("体幹傾斜角度の変化")
        axes[3].set_ylabel("[deg]")
        axes[3].legend()
        axes[3].grid(True)

    # 腕振り
    if {"left_arm", "right_arm"}.issubset(df_angles.columns):
        axes[4].plot(frames, df_angles["left_arm"], label="左腕")
        axes[4].plot(frames, df_angles["right_arm"], label="右腕")
        axes[4].set_title("腕振り角度の変化")
        axes[4].set_ylabel("[deg]")
        axes[4].legend()
        axes[4].grid(True)

    # 残り1つのサブプロットは空白
    axes[5].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"📊 角度グラフ出力: {out_path}")


# =========================
# AIコメント生成（詳細版）
# =========================

def stats(series: pd.Series):
    return float(series.max()), float(series.min()), float(series.mean()), float(series.std())


def generate_ai_comments(summary: dict, df_angles: pd.DataFrame):
    """
    高度化AIコメント（角度の最大/最小/左右差/変動幅/時系列傾向 を解析）
    """
    comments = []

    pitch = summary.get("pitch", 0.0)
    stride = summary.get("stride", 0.0)
    stability = summary.get("stability", 0.0)

    # ---- 角度統計 ----
    lk_max = lk_min = lk_mean = lk_std = np.nan
    rk_max = rk_min = rk_mean = rk_std = np.nan
    knee_diff_mean = np.nan

    if {"left_knee", "right_knee"}.issubset(df_angles.columns):
        lk_max, lk_min, lk_mean, lk_std = stats(df_angles["left_knee"])
        rk_max, rk_min, rk_mean, rk_std = stats(df_angles["right_knee"])
        knee_diff_mean = float(np.nanmean(np.abs(df_angles["left_knee"] - df_angles["right_knee"])))

    hip_diff_mean = np.nan
    if {"left_hip", "right_hip"}.issubset(df_angles.columns):
        hip_diff_mean = float(np.nanmean(np.abs(df_angles["left_hip"] - df_angles["right_hip"])))

    ankle_diff_mean = np.nan
    if {"left_ankle", "right_ankle"}.issubset(df_angles.columns):
        ankle_diff_mean = float(np.nanmean(np.abs(df_angles["left_ankle"] - df_angles["right_ankle"])))

    torso_left_mean = torso_right_mean = torso_center = torso_diff_mean = np.nan
    if {"torso_tilt_left", "torso_tilt_right"}.issubset(df_angles.columns):
        torso_left_mean = float(df_angles["torso_tilt_left"].mean())
        torso_right_mean = float(df_angles["torso_tilt_right"].mean())
        torso_center = (torso_left_mean + torso_right_mean) / 2
        torso_diff_mean = float(
            np.nanmean(np.abs(df_angles["torso_tilt_left"] - df_angles["torso_tilt_right"]))
        )

    arm_diff_mean = np.nan
    if {"left_arm", "right_arm"}.issubset(df_angles.columns):
        arm_diff_mean = float(np.nanmean(np.abs(df_angles["left_arm"] - df_angles["right_arm"])))

    # ---------------------------
    # ① 走りのタイプ（ピッチ型 or ストライド型）
    # ---------------------------
    if pitch >= 4.0:
        run_type = "ピッチ型（回転数重視）"
    elif stride >= 1.8:
        run_type = "ストライド型（歩幅重視）"
    else:
        run_type = "ミックス型"

    comments.append(
        f"あなたの走りの特徴は「{run_type}」です。ピッチ {pitch:.2f} 歩/s、ストライド {stride:.2f} m のバランスから分類しています。"
    )

    # ---------------------------
    # ② 膝の可動域と左右差
    # ---------------------------
    if not np.isnan(lk_max):
        comments.append(
            f"膝角度は左 {lk_min:.1f}°〜{lk_max:.1f}°、右 {rk_min:.1f}°〜{rk_max:.1f}° の可動域で、左右差の平均は {knee_diff_mean:.1f}° でした。"
        )
        if knee_diff_mean < 4:
            comments.append("左右差が小さく、両脚で均等に荷重が乗っています。")
        else:
            comments.append(
                "左右差が大きい時間帯があり、蹴り脚が偏る可能性があります。接地の向きを揃えるイメージで走ると改善が期待できます。"
            )

    # ---------------------------
    # ③ 股関節：推進力の中心
    # ---------------------------
    if not np.isnan(hip_diff_mean):
        if hip_diff_mean < 5:
            comments.append(
                f"股関節の左右差は平均 {hip_diff_mean:.1f}° と小さく、骨盤の回旋が左右均等で効率的です。"
            )
        else:
            comments.append(
                f"股関節の左右差（平均 {hip_diff_mean:.1f}°）が比較的大きいため、骨盤が左右どちらかに逃げる時間帯があります。接地から抜き脚まで腰の向きをレーンに正対させる意識を持つと良いです。"
            )

    # ---------------------------
    # ④ 体幹：中心軸の安定性
    # ---------------------------
    if not np.isnan(torso_center):
        comments.append(
            f"体幹傾斜の左右平均は左 {torso_left_mean:.1f}° / 右 {torso_right_mean:.1f}° で、中心軸は {torso_center:.1f}° 付近です。"
        )
        comments.append(f"体幹傾斜の左右差の平均は {torso_diff_mean:.1f}° でした。")

        if stability >= 8.0 and torso_diff_mean < 3:
            comments.append("体幹は非常に安定しており、無駄な横ブレが少ない走りです。スピードが上がってもフォームが崩れにくいタイプです。")
        else:
            comments.append(
                "体幹の振れがやや大きく、特に加速区間で上下・左右のブレが出やすい傾向があります。みぞおちから下をまっすぐ前に運ぶ意識を持つと、接地ロスを減らせます。"
            )

    # ---------------------------
    # ⑤ 足首：接地の向き
    # ---------------------------
    if not np.isnan(ankle_diff_mean):
        if ankle_diff_mean < 5:
            comments.append(
                f"足首角度の左右差（平均 {ankle_diff_mean:.1f}°）は小さく、接地方向はよく揃っています。"
            )
        else:
            comments.append(
                f"足首の左右差がやや大きく（平均 {ankle_diff_mean:.1f}°）、接地方向が左右でズレる時間帯があります。踵からではなく母指球で前方に押し出す意識を持つと接地が安定します。"
            )

    # ---------------------------
    # ⑥ 腕振り：脚との連動
    # ---------------------------
    if not np.isnan(arm_diff_mean):
        if arm_diff_mean < 6:
            comments.append(
                f"腕振りの左右差は平均 {arm_diff_mean:.1f}° と小さく、脚との連動も良い状態です。上半身がうまくリズムを作れています。"
            )
        else:
            comments.append(
                f"腕振りに左右差（平均 {arm_diff_mean:.1f}°）があり、片側の肩に力が入りやすい傾向があります。肘を軽く曲げて、前後にまっすぐ振る意識を持つと脚の運びもスムーズになります。"
            )

    # ---------------------------
    # ⑦ 推奨ドリル
    # ---------------------------
    comments.append("【推奨トレーニング例】")
    comments.append("・もも上げドリル（左右差の少ない脚の引き上げを身につける）")
    comments.append("・スキップA/B（股関節の伸展と接地リズムの強化）")
    comments.append("・体幹ツイストラン（骨盤と体幹の連動性向上）")
    comments.append("・ハイニー＋腕振り連動（上半身と下半身の同期を高める）")

    return comments


# =========================
# PDF ビルド
# =========================

def safe_text(s: str) -> str:
    """
    PDFフォントで表示しづらい制御文字などを排除。
    （Meiryoなので日本語は基本OK。絵文字などは使っていない前提。）
    """
    if s is None:
        return ""
    # 制御文字を除去
    return "".join(ch for ch in s if ord(ch) >= 32)


def build_pdf(
    df_metrics: pd.DataFrame,
    df_angles: pd.DataFrame,
    video_path: str,
    athlete: str,
    m_per_px: float,
):
    video_stem = Path(video_path).stem

    main_graph_path = GRAPH_DIR / f"{video_stem}_main_graph_v5_3_1.png"
    angle_graph_path = GRAPH_DIR / f"{video_stem}_angle_graph_v5_3_1.png"

    create_main_graph(df_metrics, main_graph_path)
    create_angle_graphs(df_angles, angle_graph_path)

    summary = compute_summary(df_metrics)
    comments = generate_ai_comments(summary, df_angles)

    pdf_path = PDF_DIR / f"{athlete}_form_report_v5_3_1.pdf"

    # PDF 作成
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)

    # 日本語フォント（Meiryo）
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.set_font("JP", size=12)

    # ========= 1ページ目：概要＋メイングラフ =========
    pdf.add_page()

    pdf.set_font("JP", size=16)
    pdf.cell(0, 10, safe_text(f"{athlete} フォーム分析レポート（v5.3.1）"), ln=1)

    pdf.set_font("JP", size=10)
    pdf.cell(0, 6, safe_text(f"動画: {os.path.basename(video_path)}"), ln=1)
    pdf.cell(0, 6, safe_text(f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=1)
    pdf.cell(0, 6, safe_text(f"校正スケール: {m_per_px:.6f} m/px"), ln=1)

    pdf.ln(2)
    pdf.set_font("JP", size=11)
    pdf.cell(0, 6, safe_text("速度・姿勢の要約"), ln=1)
    pdf.set_font("JP", size=10)
    pdf.cell(0, 6, safe_text(f"ピッチ: {summary['pitch']:.2f} 歩/s"), ln=1)
    pdf.cell(0, 6, safe_text(f"ストライド: {summary['stride']:.2f} m"), ln=1)
    pdf.cell(0, 6, safe_text(f"安定性スコア: {summary['stability']:.2f} / 10"), ln=1)

    # グラフ画像（1ページ目）
    pdf.ln(3)
    pdf.cell(0, 6, safe_text("速度・傾き・COMの変化グラフ"), ln=1)
    pdf.image(str(main_graph_path), x=10, y=pdf.get_y() + 2, w=190)

    # ========= 2ページ目：角度グラフ＋AIコメント =========
    pdf.add_page()
    pdf.set_font("JP", size=12)
    pdf.cell(0, 8, safe_text("関節角度の変化（膝・股関節・足首・体幹・腕）"), ln=1)
    pdf.image(str(angle_graph_path), x=10, y=pdf.get_y() + 2, w=190)

    pdf.ln(90)  # 画像分スペースを空ける（画像高さによって調整）
    pdf.set_font("JP", size=12)
    pdf.cell(0, 8, safe_text("AIフォーム解析コメント（速度＋角度の総合所見）"), ln=1)
    pdf.set_font("JP", size=10)

    for c in comments:
        pdf.multi_cell(0, 6, safe_text(f"- {c}"))
        pdf.ln(1)

    pdf.output(str(pdf_path))
    print(f"✅ PDF出力完了: {pdf_path}")


# =========================
# main
# =========================

def main():
    parser = argparse.ArgumentParser(description="KJAC フォーム解析PDF生成 v5.3.1")
    parser.add_argument("--video", type=str, required=True, help="入力動画パス")
    parser.add_argument("--csv", type=str, required=True, help="メトリクスCSVパス（koike_pose_metrics_v3.csv）")
    parser.add_argument("--athlete", type=str, required=True, help="選手名（PDFファイル名に使用）")
    args = parser.parse_args()

    video_path = args.video
    csv_path = args.csv
    athlete = args.athlete

    m_per_px = load_calibration()
    df_metrics = load_metrics_csv(csv_path)
    df_angles = load_angle_csv_for_video(video_path)

    build_pdf(df_metrics, df_angles, video_path, athlete, m_per_px)


if __name__ == "__main__":
    main()







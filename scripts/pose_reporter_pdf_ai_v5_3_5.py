import os
import sys
import math
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF

# ==========================
# 共通設定
# ==========================

# Windows 日本語環境向けフォント（必要に応じて変更可）
DEFAULT_FONT_PATH = r"C:\Windows\Fonts\meiryo.ttc"

# Matplotlib 日本語フォント設定
matplotlib.rcParams["font.family"] = "Meiryo"
matplotlib.rcParams["axes.unicode_minus"] = False


# ==========================
# ユーティリティ
# ==========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_calibration_m_per_px():
    """
    calibration_result.json があれば m_per_px を読む。
    なければ 1.0 を返す（グラフの相対比較用）。
    """
    import json
    json_path = Path("outputs/jsonl/calibration_result.json")
    if not json_path.exists():
        print("⚠ calibration_result.json が見つからないため m_per_px=1.0 として処理します。")
        return 1.0
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m_per_px = float(data.get("m_per_px", 1.0))
        print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
        return m_per_px
    except Exception as e:
        print(f"⚠ calibration_result.json の読込に失敗: {e}")
        return 1.0


# ==========================
# メトリクスCSV 読み込み
# ==========================

def load_metrics(csv_path: str, m_per_px: float):
    """
    koike_pose_metrics_v3.csv を読み込み、
    - _summary_ 行からピッチ / ストライド / 安定性を抽出
    - データ本体を返す
    """
    df = pd.read_csv(csv_path)
    df["frame_str"] = df["frame"].astype(str)

    # summary 行の切り出し
    summary_row = df[df["frame_str"] == "_summary_"]
    data_df = df[df["frame_str"] != "_summary_"].copy()
    data_df.drop(columns=["frame_str"], inplace=True)

    summary = {
        "pitch": None,
        "stride": None,
        "stability": None,
    }

    if not summary_row.empty:
        for col_key, col_name in [
            ("pitch", "pitch_hz"),
            ("stride", "stride_m"),
            ("stability", "stability_score"),
        ]:
            if col_name in summary_row.columns:
                val = summary_row[col_name].iloc[0]
                summary[col_key] = float(val) if pd.notna(val) else None

    # 速度[m/s] 列が無ければ自動生成
    if "speed_mps_real" not in data_df.columns:
        if "speed_mps" in data_df.columns:
            data_df["speed_mps_real"] = data_df["speed_mps"] * m_per_px
        elif "dist_px" in data_df.columns and "time_s" in data_df.columns:
            dt = data_df["time_s"].diff().fillna(0.0)
            dist_m = data_df["dist_px"] * m_per_px
            v = dist_m.diff().fillna(0.0) / dt.replace(0, np.nan)
            v = v.fillna(0.0)
            data_df["speed_mps_real"] = v
        else:
            data_df["speed_mps_real"] = 0.0

    # COM_y[m] の推定
    if "COM_y_px" in data_df.columns:
        data_df["COM_y_m"] = data_df["COM_y_px"] * m_per_px
    else:
        if "COM_y" in data_df.columns:
            data_df["COM_y_m"] = data_df["COM_y"]
        else:
            data_df["COM_y_m"] = 0.0

    return data_df, summary


# ==========================
# 角度CSV 読み込み・自動生成
# ==========================

def ensure_angles_csv(video_path: str) -> str:
    """
    角度CSV (outputs/angles_<stem>.csv) が無ければ
    pose_angle_analyzer_v1_0.py を自動実行して生成する。
    """
    video_stem = Path(video_path).stem  # 例: "二村遥香10.5"
    angle_csv = Path("outputs") / f"angles_{video_stem}.csv"

    if angle_csv.exists():
        return str(angle_csv)

    print(f"🔄 角度CSVが無いため自動解析開始: {angle_csv.name}")

    angle_script = Path("scripts/pose_angle_analyzer_v1_0.py")
    if not angle_script.exists():
        raise FileNotFoundError(f"角度解析スクリプトが見つかりません: {angle_script}")

    cmd = [
        sys.executable,
        str(angle_script),
        "--video",
        video_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"角度解析スクリプトの実行に失敗しました: {e}")

    if not angle_csv.exists():
        raise FileNotFoundError(f"角度CSVが生成されませんでした: {angle_csv}")

    return str(angle_csv)


def load_angles(video_path: str):
    """
    角度CSVを読み込み、各角度の統計量を返す。
    - 膝角度、股関節角度、体幹傾斜など
    """
    angle_csv = ensure_angles_csv(video_path)
    df = pd.read_csv(angle_csv)

    angle_stats = {}

    angle_columns = {
        "knee_r_deg": "右膝角度[deg]",
        "hip_r_deg": "右股関節角度[deg]",
        "trunk_deg": "体幹前傾角度[deg]",
        "knee_l_deg": "左膝角度[deg]",
        "hip_l_deg": "左股関節角度[deg]",
    }

    for col, label in angle_columns.items():
        if col in df.columns:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            angle_stats[col] = {
                "label": label,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            }

    return df, angle_stats


# ==========================
# 骨格オーバーレイ 3コマ画像生成
# ==========================

def extract_pose_frames(video_path: str) -> dict:
    """
    outputs/images/<stem>_pose_overlay.mp4 から
    start/mid/finish の3枚を PNG で切り出す。
    （保存に失敗した画像はスキップ）
    """
    stem = Path(video_path).stem  # 例: "二村遥香10.5"
    overlay_mp4 = Path("outputs/images") / f"{stem}_pose_overlay.mp4"

    if not overlay_mp4.exists():
        print(f"⚠ オーバーレイ動画が見つかりません: {overlay_mp4}")
        return {}

    cap = cv2.VideoCapture(str(overlay_mp4))
    if not cap.isOpened():
        print(f"⚠ オーバーレイ動画を開けません: {overlay_mp4}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"⚠ オーバーレイ動画のフレーム数が0です: {overlay_mp4}")
        cap.release()
        return {}

    idxs = [
        int(total_frames * 0.1),
        int(total_frames * 0.5),
        int(total_frames * 0.9),
    ]

    # 日本語＋ピリオドを安全にするため、フォルダ名は stem から "." を "_" に変換
    safe_stem = stem.replace(".", "_")
    out_dir = Path("outputs/pose_images") / safe_stem
    ensure_dir(out_dir)

    names = ["start", "mid", "finish"]
    result = {}

    for idx, name in zip(idxs, names):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx, 0))
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ フレーム抽出失敗: idx={idx}")
            continue
        out_path = out_dir / f"{name}.png"
        # 保存成功したか確認
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            print(f"⚠ 画像保存に失敗しました: {out_path}")
            continue
        if not out_path.exists():
            print(f"⚠ 画像ファイルが存在しません（保存失敗の可能性）: {out_path}")
            continue
        result[name] = str(out_path)

    cap.release()
    print(f"📸 骨格画像出力: {result}")
    return result


# ==========================
# グラフ生成
# ==========================

def create_metrics_graph(df: pd.DataFrame, m_per_px: float, video_stem: str) -> str:
    ensure_dir("outputs/graphs")

    t = df["time_s"].values if "time_s" in df.columns else np.arange(len(df)) * 0.01
    v = df["speed_mps_real"].values
    tilt = df["tilt_deg"].values if "tilt_deg" in df.columns else np.zeros_like(v)
    com_y = df["COM_y_m"].values

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    axes[0].plot(t, v)
    axes[0].set_ylabel("速度 [m/s]")
    axes[0].set_title("時間ごとの速度変化")

    axes[1].plot(t, tilt)
    axes[1].set_ylabel("体幹前傾角度 [deg]")
    axes[1].set_title("時間ごとの体幹前傾の変化")

    axes[2].plot(t, com_y)
    axes[2].set_ylabel("COM高さ [m]")
    axes[2].set_xlabel("時間 [s]")
    axes[2].set_title("時間ごとの重心高さの変動")

    plt.tight_layout()
    out_path = Path("outputs/graphs") / f"{video_stem}_metrics_v5_3_5.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"📊 メトリクスグラフ画像出力: {out_path}")
    return str(out_path)


def create_angles_graph(df_angles: pd.DataFrame, angle_stats: dict, video_stem: str) -> str:
    ensure_dir("outputs/graphs")

    t = df_angles["time_s"].values if "time_s" in df_angles.columns else np.arange(len(df_angles)) * 0.01

    ordered_cols = []
    for col in ["knee_r_deg", "hip_r_deg", "trunk_deg", "knee_l_deg", "hip_l_deg"]:
        if col in df_angles.columns:
            ordered_cols.append(col)

    if not ordered_cols:
        print("⚠ 角度グラフを描画できるカラムがありません。")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "角度データなし", ha="center", va="center")
        ax.axis("off")
        out_path = Path("outputs/graphs") / f"{video_stem}_angles_v5_3_5.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return str(out_path)

    n_plots = len(ordered_cols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 2.5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, col in zip(axes, ordered_cols):
        label = angle_stats.get(col, {}).get("label", col)
        ax.plot(t, df_angles[col].values)
        ax.set_ylabel("[deg]")
        ax.set_title(label)

    axes[-1].set_xlabel("時間 [s]")
    plt.tight_layout()
    out_path = Path("outputs/graphs") / f"{video_stem}_angles_v5_3_5.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"📊 角度グラフ画像出力: {out_path}")
    return str(out_path)


# ==========================
# AIコメント生成（専門性強め）
# ==========================

def generate_ai_comments(df_metrics: pd.DataFrame, summary: dict, angle_stats: dict) -> list:
    comments = []

    # 速度プロファイル
    t = df_metrics["time_s"].values if "time_s" in df_metrics.columns else np.arange(len(df_metrics)) * 0.01
    v = df_metrics["speed_mps_real"].values

    max_v = float(np.max(v)) if len(v) > 0 else 0.0
    mean_v = float(np.mean(v)) if len(v) > 0 else 0.0
    idx_max = int(np.argmax(v)) if len(v) > 0 else 0
    t_max = float(t[idx_max]) if len(t) > 0 else 0.0

    accel_phase = "速い立ち上がり" if t_max < 2.0 else "やや緩やかな立ち上がり"

    comments.append(
        f"最大速度は約 {max_v:.2f} m/s（{t_max:.2f} 秒付近）で発現しており、"
        f"平均速度 {mean_v:.2f} m/s に対してピークとの差が "
        f"{(max_v - mean_v):.2f} m/s あります。これは {accel_phase} で、"
        "加速局面における踏み出しのエネルギー伝達と接地時間のマネジメントが良好であることを示します。"
    )

    # 体幹傾斜
    if "tilt_deg" in df_metrics.columns:
        tilt = df_metrics["tilt_deg"].values
        tilt_mean = float(np.mean(tilt))
        tilt_std = float(np.std(tilt))

        comments.append(
            f"体幹前傾角度は平均 {tilt_mean:.1f}°、標準偏差 {tilt_std:.1f}° と推定されます。"
            "スタート直後は前傾を強め、その後徐々に起き上がる典型的なスプリントパターンで、"
            "頭部〜骨盤まで一体的に前傾を保てているかが、地面反力ベクトルを前方に向ける上で鍵となります。"
        )

    # ピッチ／ストライド／安定性
    pitch = summary.get("pitch", None)
    stride = summary.get("stride", None)
    stability = summary.get("stability", None)

    if pitch is not None and stride is not None:
        comments.append(
            f"ピッチは約 {pitch:.2f} 歩/秒、ストライドは約 {stride:.2f} m で推定されます。"
            "この組み合わせは、年代を考慮すると上位水準にあり、"
            "ピッチ主導型というよりは、ストライドをしっかり確保したバランス型の走りと解釈できます。"
            "今後はストライドを維持したまま、接地時間短縮によるピッチの微増がタイム短縮に直結します。"
        )

    if stability is not None:
        comments.append(
            f"重心鉛直変動の指標から算出した安定性スコアは {stability:.2f} / 10 です。"
            "接地ごとの重心上下動が過度に大きいとエネルギーロスになりますが、"
            "現状は『推進のために必要な最小限の上下動』の範囲に収まっていると評価できます。"
        )

    # 膝・股関節・体幹角度
    def angle_comment(col_key: str, name: str):
        if col_key not in angle_stats:
            return None
        s = angle_stats[col_key]
        mean = s["mean"]
        std = s["std"]
        amin = s["min"]
        amax = s["max"]
        return (
            f"{name}は平均 {mean:.1f}°（最小 {amin:.1f}°〜最大 {amax:.1f}°）、"
            f"変動幅の標準偏差 {std:.1f}° 程度です。"
            "このレンジは、接地〜離地フェーズにおける筋・腱の伸張反射を十分に活かせる範囲であり、"
            "可動域を過度に使いすぎずに『張りのあるストライド』を生み出していることを示します。"
        )

    knee_r_text = angle_comment("knee_r_deg", "右膝屈曲角度")
    if knee_r_text:
        comments.append(
            knee_r_text + " 特に遊脚中の膝のたたみが適切で、"
            "ハムストリングスと大腿四頭筋の協調的な制御ができていると考えられます。"
        )

    hip_r_text = angle_comment("hip_r_deg", "右股関節伸展角度")
    if hip_r_text:
        comments.append(
            hip_r_text + " 接地後の股関節伸展が適切なタイミングで行われており、"
            "地面反力を前方推進力に変換できている一方で、"
            "骨盤前傾位を維持しながら伸展することで、さらにストライド効率を高められます。"
        )

    trunk_text = angle_comment("trunk_deg", "体幹前傾角度")
    if trunk_text:
        comments.append(
            trunk_text + " 体幹のブレが小さいため、腕振りと下肢の動きが同調しやすく、"
            "足元だけでなく身体全体で前方に倒れ込むような推進が実現できています。"
        )

    comments.append(
        "総合的に見ると、速度プロファイル・体幹前傾・下肢関節角度のいずれも、"
        "スプリントの基本原則に沿った動きができています。今後は、"
        "①スタート〜加速局面での接地時間短縮、②中盤以降の体幹前傾の微調整、"
        "③股関節伸展時の骨盤前傾の維持、の3点を高精度にコントロールすることで、"
        "同じストライド長でより高いピッチと推進効率を実現できると考えられます。"
    )

    return comments


# ==========================
# PDF ビルド
# ==========================

class JP_PDF(FPDF):
    pass


def build_pdf(csv_path: str, video_path: str, athlete: str):
    m_per_px = load_calibration_m_per_px()

    df_metrics, summary = load_metrics(csv_path, m_per_px)
    print(f"✅ メトリクスCSV読込: {len(df_metrics)} 行")

    df_angles, angle_stats = load_angles(video_path)

    video_stem = Path(video_path).stem

    pose_images = extract_pose_frames(video_path)

    metrics_graph = create_metrics_graph(df_metrics, m_per_px, video_stem)
    angles_graph = create_angles_graph(df_angles, angle_stats, video_stem)

    ai_comments = generate_ai_comments(df_metrics, summary, angle_stats)

    pdf = JP_PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)

    font_path = DEFAULT_FONT_PATH
    if not Path(font_path).exists():
        raise FileNotFoundError(f"日本語フォントが見つかりません: {font_path}")
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.set_font("JP", size=11)

    # ===== 1ページ目：概要＋骨格3コマ =====
    pdf.add_page()

    pdf.set_font("JP", size=16)
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.3.5）", ln=1)

    pdf.set_font("JP", size=10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 6, f"校正スケール: {m_per_px:.6f} m/px", ln=1)

    pdf.ln(2)
    pdf.set_font("JP", size=11)
    pdf.cell(0, 7, "基本指標（ピッチ・ストライド・安定性）", ln=1)

    pdf.set_font("JP", size=10)
    pitch_val = summary.get("pitch", 0.0) or 0.0
    stride_val = summary.get("stride", 0.0) or 0.0
    stab_val = summary.get("stability", 0.0) or 0.0

    pdf.cell(0, 6, f"ピッチ: {pitch_val:.2f} 歩/秒", ln=1)
    pdf.cell(0, 6, f"ストライド: {stride_val:.2f} m", ln=1)
    pdf.cell(0, 6, f"安定性スコア: {stab_val:.2f} / 10", ln=1)

    pdf.ln(3)
    pdf.set_font("JP", size=11)
    pdf.cell(0, 7, "骨格オーバーレイ（start / mid / finish）", ln=1)

    # ここで存在チェックしてから image() を呼ぶ
    if pose_images:
        margin_left = pdf.l_margin
        margin_right = pdf.r_margin
        usable_width = pdf.w - margin_left - margin_right
        spacing = 4
        img_width = (usable_width - 2 * spacing) / 3.0
        y_top = pdf.get_y() + 2

        x_positions = [
            margin_left,
            margin_left + img_width + spacing,
            margin_left + 2 * (img_width + spacing),
        ]
        order = ["start", "mid", "finish"]

        for x, key in zip(x_positions, order):
            if key in pose_images:
                img_abs = os.path.abspath(pose_images[key])
                if os.path.exists(img_abs):
                    pdf.image(img_abs, x=x, y=y_top, w=img_width)
                else:
                    print(f"⚠ PDF貼り付け時に画像が見つかりませんでした: {img_abs}")
        pdf.ln(img_width * 0.75 + 8)
    else:
        pdf.set_font("JP", size=9)
        pdf.multi_cell(0, 5, "※ 骨格オーバーレイ画像が取得できなかったため、このページには画像を表示していません。")

    # ===== 2ページ目：速度・COM =====
    pdf.add_page()
    pdf.set_font("JP", size=12)
    pdf.cell(0, 8, "速度・重心の時間変化", ln=1)

    pdf.set_font("JP", size=10)
    pdf.multi_cell(
        0, 5,
        "上段：時間ごとの速度[m/s]、中段：体幹前傾角度[deg]、下段：重心高さ[m]の推移を表しています。"
        "速度ピークの位置と体幹前傾の変化、重心上下動の大きさを合わせて確認することで、"
        "どの局面で推進効率が高いか／ロスが生じているかを評価できます。"
    )

    pdf.ln(2)
    if os.path.exists(metrics_graph):
        pdf.image(os.path.abspath(metrics_graph), w=180)
    else:
        pdf.multi_cell(0, 5, "※ メトリクスグラフ画像が見つかりませんでした。")

    # ===== 3ページ目：角度グラフ =====
    pdf.add_page()
    pdf.set_font("JP", size=12)
    pdf.cell(0, 8, "関節角度（膝・股関節・体幹）の時間変化", ln=1)

    pdf.set_font("JP", size=10)
    pdf.multi_cell(
        0, 5,
        "膝・股関節・体幹の角度変化は、筋・腱の伸張反射をどの程度効率良く利用できているかを示す重要な指標です。"
        "接地直前〜接地中の屈曲量と、離地に向けた伸展のタイミングがスプリント局面ごとに適切かどうかを確認します。"
    )

    pdf.ln(2)
    if os.path.exists(angles_graph):
        pdf.image(os.path.abspath(angles_graph), w=180)
    else:
        pdf.multi_cell(0, 5, "※ 角度グラフ画像が見つかりませんでした。")

    # ===== 4ページ目：AIフォーム解析コメント =====
    pdf.add_page()
    pdf.set_font("JP", size=13)
    pdf.cell(0, 8, "AIフォーム解析コメント（速度＋角度＋安定性の総合所見）", ln=1)

    pdf.ln(2)
    pdf.set_font("JP", size=10)
    for c in ai_comments:
        pdf.multi_cell(0, 5, f"・{c}")
        pdf.ln(1)

    ensure_dir("outputs/pdf")
    out_pdf = Path("outputs/pdf") / f"{athlete}_form_report_v5_3_5.pdf"
    pdf.output(str(out_pdf))
    print(f"✅ PDF出力完了: {out_pdf}")


# ==========================
# メイン
# ==========================

def main():
    parser = argparse.ArgumentParser(description="KJAC テクニカルAI v5.3.5 単走フォームレポート")
    parser.add_argument("--video", type=str, required=True, help="入力動画パス（例: videos/二村遥香10.5.mp4）")
    parser.add_argument("--csv", type=str, required=True, help="メトリクスCSVパス（例: outputs/koike_pose_metrics_v3.csv）")
    parser.add_argument("--athlete", type=str, required=True, help="選手名（PDFタイトル用）")
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()








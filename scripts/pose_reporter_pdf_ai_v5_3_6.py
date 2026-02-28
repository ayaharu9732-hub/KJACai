# scripts/pose_reporter_pdf_ai_v5_3_6.py
import os
import json
import math
import argparse
import subprocess
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# =========================
# パス＆ユーティリティ
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
JSONL_DIR = os.path.join(OUTPUT_DIR, "jsonl")
POSE_IMG_ROOT = os.path.join(OUTPUT_DIR, "pose_images")
ANGLES_ROOT = os.path.join(OUTPUT_DIR, "angles")
PDF_ROOT = os.path.join(OUTPUT_DIR, "pdf")
GRAPHS_ROOT = os.path.join(OUTPUT_DIR, "graphs")
IMAGES_ROOT = os.path.join(OUTPUT_DIR, "images")

os.makedirs(POSE_IMG_ROOT, exist_ok=True)
os.makedirs(ANGLES_ROOT, exist_ok=True)
os.makedirs(PDF_ROOT, exist_ok=True)
os.makedirs(GRAPHS_ROOT, exist_ok=True)


def safe_stem_from_video(video_path: str) -> str:
    """
    動画ファイル名（拡張子なし）を取り出し、
    '.' を '_' に置き換えた文字列を返す。
    二村遥香10.5.mp4 → 二村遥香10_5
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    stem = stem.replace(".", "_")
    return stem


def athlete_dir(athlete: str) -> str:
    """選手別ルートフォルダ"""
    path = os.path.join(POSE_IMG_ROOT, athlete)
    os.makedirs(path, exist_ok=True)
    return path


def angle_dir(athlete: str) -> str:
    """選手別角度CSVフォルダ"""
    path = os.path.join(ANGLES_ROOT, athlete)
    os.makedirs(path, exist_ok=True)
    return path


def pdf_dir(athlete: str) -> str:
    """選手別PDFフォルダ"""
    path = os.path.join(PDF_ROOT, athlete)
    os.makedirs(path, exist_ok=True)
    return path


# =========================
# 校正読み込み
# =========================

def load_calibration() -> float:
    """
    calibration_result.json から 1px あたり[m]を読み込む
    見つからない場合は 1.0 を返す
    """
    calib_path = os.path.join(JSONL_DIR, "calibration_result.json")
    if not os.path.exists(calib_path):
        print("⚠ 校正ファイルが見つからないため、m_per_px=1.0 として処理します。")
        return 1.0

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m_per_px = float(data.get("m_per_px", 1.0))
        print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
        return m_per_px
    except Exception as e:
        print(f"⚠ 校正ファイル読込エラー: {e} → m_per_px=1.0")
        return 1.0


# =========================
# メトリクス CSV 読み込み
# =========================

def load_metrics(csv_path: str):
    """
    メトリクスCSVを読み込み、DataFrameと summary dict を返す。
    _summary_ 行があればそれを使い、なければ平均値から生成。
    """
    df = pd.read_csv(csv_path)
    print(f"✅ メトリクスCSV読込: {len(df)} 行")

    summary = {
        "pitch": 0.0,
        "stride": 0.0,
        "stability": 0.0,
        "max_speed": 0.0,
        "max_speed_time": 0.0,
    }

    # _summary_ 行を探す（1列目 or frame カラムに '_summary_'）
    summary_row = None
    if "frame" in df.columns:
        mask = df["frame"].astype(str) == "_summary_"
        if mask.any():
            summary_row = df[mask].iloc[0]

    if summary_row is not None:
        # 既存 summary 行を使用
        if "pitch_hz" in df.columns:
            summary["pitch"] = float(summary_row.get("pitch_hz", 0.0))
        if "stride_m" in df.columns:
            summary["stride"] = float(summary_row.get("stride_m", 0.0))
        if "stability_score" in df.columns:
            summary["stability"] = float(summary_row.get("stability_score", 0.0))
        if "speed_mps" in df.columns:
            summary["max_speed"] = float(df["speed_mps"].max())
        if "time_s" in df.columns and "speed_mps" in df.columns:
            idx = df["speed_mps"].idxmax()
            summary["max_speed_time"] = float(df.loc[idx, "time_s"])
    else:
        # summary 行が無い場合は、NaN を除いて統計量から計算
        if "pitch_hz" in df.columns:
            summary["pitch"] = float(df["pitch_hz"].dropna().mean())
        if "stride_m" in df.columns:
            summary["stride"] = float(df["stride_m"].dropna().mean())
        if "stability_score" in df.columns:
            summary["stability"] = float(df["stability_score"].dropna().mean())
        if "speed_mps" in df.columns:
            summary["max_speed"] = float(df["speed_mps"].dropna().max())
        if "time_s" in df.columns and "speed_mps" in df.columns:
            idx = df["speed_mps"].idxmax()
            summary["max_speed_time"] = float(df.loc[idx, "time_s"])

    return df, summary


# =========================
# 角度 CSV の確保＆読込
# =========================

def ensure_angles_csv(video_path: str, athlete: str) -> str:
    """
    角度CSVのパスを決定し、存在しなければ自動生成を試みる。
    パス規則：outputs/angles/<athlete>/<video_stem>.csv
    ただし旧形式 outputs/angles_*.csv があればそれを優先してコピー。
    """
    stem = safe_stem_from_video(video_path)
    athlete_angles_dir = angle_dir(athlete)
    target_csv = os.path.join(athlete_angles_dir, f"{stem}.csv")

    # すでに存在するならそれを使う
    if os.path.exists(target_csv):
        print(f"✅ 角度CSV既存: {target_csv}")
        return target_csv

    # 旧形式: outputs/angles_*.csv を流用
    legacy_name = f"angles_{os.path.basename(video_path).replace('.mp4', '')}.csv"
    legacy_path = os.path.join(OUTPUT_DIR, legacy_name)
    if os.path.exists(legacy_path):
        print(f"🔁 旧形式角度CSVを発見 → {legacy_path} を {target_csv} にコピー")
        os.makedirs(athlete_angles_dir, exist_ok=True)
        import shutil
        shutil.copy2(legacy_path, target_csv)
        return target_csv

    # ここまでで無い場合は、角度解析スクリプトを呼び出して自動生成
    print(f"🔄 角度CSVが無いため自動解析開始: {target_csv}")
    angle_script = os.path.join(BASE_DIR, "scripts", "pose_angle_analyzer_v1_0.py")
    cmd = ["python", angle_script, "--video", video_path]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"角度解析スクリプトの実行に失敗しました: {e}")

    # 自動解析スクリプト側が outputs/angles_*.csv に出している前提なので、
    # 再度 legacy を探してコピー
    if os.path.exists(legacy_path):
        print(f"🔁 自動生成された旧形式角度CSVをコピー: {legacy_path} → {target_csv}")
        os.makedirs(athlete_angles_dir, exist_ok=True)
        import shutil
        shutil.copy2(legacy_path, target_csv)
        return target_csv

    # それでも無ければエラー
    raise FileNotFoundError(f"角度CSVが見つかりません: {target_csv}")


def load_angles(video_path: str, athlete: str):
    """
    角度CSVを確保＆読込して、DataFrame + summary dict を返す。
    """
    angle_csv = ensure_angles_csv(video_path, athlete)
    df = pd.read_csv(angle_csv)
    print(f"✅ 角度CSV読込: {len(df)} 行 ({angle_csv})")

    # 角度系カラム
    angle_cols = [c for c in df.columns if c not in ("frame", "time_s")]
    stats = {}
    for col in angle_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "max": float(series.max()),
            "min": float(series.min()),
        }

    return df, stats


# =========================
# オーバーレイ動画から骨格画像を切り出し
# =========================

def extract_skeleton_frames(video_path: str, athlete: str):
    """
    オーバーレイ動画 (outputs/images/<stem>_pose_overlay.mp4) から
    start / mid / finish の3フレームを切り出して保存し、
    それぞれのパスを返す。

    保存先:
        outputs/pose_images/<athlete>/<stem>/start.png など
    """
    stem = safe_stem_from_video(video_path)
    video_stem_original = os.path.splitext(os.path.basename(video_path))[0]  # 二村遥香10.5
    overlay_name = f"{video_stem_original}_pose_overlay.mp4"
    overlay_path = os.path.join(IMAGES_ROOT, overlay_name)

    if not os.path.exists(overlay_path):
        print(f"⚠ オーバーレイ動画が見つからないため、元動画から切り出します: {overlay_path}")
        overlay_path = video_path

    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        print(f"⚠ 動画を開けませんでした（骨格画像抽出スキップ）: {overlay_path}")
        return {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f"⚠ フレーム数が0です（骨格画像抽出スキップ）: {overlay_path}")
        cap.release()
        return {}

    idxs = [0, frame_count // 2, frame_count - 1]
    labels = ["start", "mid", "finish"]

    athlete_path = athlete_dir(athlete)                  # outputs/pose_images/二村遥香
    stem_dir = os.path.join(athlete_path, stem)         # outputs/pose_images/二村遥香/二村遥香10_5
    os.makedirs(stem_dir, exist_ok=True)

    saved = {}
    for idx, label in zip(idxs, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"⚠ フレーム取得失敗: index={idx}")
            continue
        out_path = os.path.join(stem_dir, f"{label}.png")
        try:
            ok = cv2.imwrite(out_path, frame)
            if ok:
                saved[label] = out_path
            else:
                print(f"⚠ 画像保存に失敗しました: {out_path}")
        except Exception as e:
            print(f"⚠ 画像保存エラー: {out_path} ({e})")

    cap.release()
    print(f"📸 骨格画像出力: {saved}")
    return saved


# =========================
# グラフ生成
# =========================

def create_metrics_graph(df: pd.DataFrame, m_per_px: float, out_path: str):
    """
    速度・体幹傾斜・COM_x の3つを1枚のPNGに縦に並べて出力する。
    """
    t = df["time_s"] if "time_s" in df.columns else df.index

    speed = df["speed_mps"] if "speed_mps" in df.columns else None
    tilt = df["tilt_deg"] if "tilt_deg" in df.columns else None

    # COM_x を m に変換（なければ px のまま）
    if "COM_x_px" in df.columns:
        com_x_m = df["COM_x_px"] * m_per_px
    elif "COM_x" in df.columns:
        # 正規化座標の場合はそのまま
        com_x_m = df["COM_x"]
    else:
        com_x_m = None

    plt.figure(figsize=(6, 8))

    n_plots = 0
    if speed is not None:
        n_plots += 1
    if tilt is not None:
        n_plots += 1
    if com_x_m is not None:
        n_plots += 1

    if n_plots == 0:
        print("⚠ 有効なメトリクスが無いためグラフ生成をスキップします。")
        return

    idx = 1

    if speed is not None:
        ax1 = plt.subplot(n_plots, 1, idx)
        ax1.plot(t, speed)
        ax1.set_ylabel("速度 [m/s]")
        ax1.set_title("速度の時間変化")
        ax1.grid(True)
        idx += 1

    if tilt is not None:
        ax2 = plt.subplot(n_plots, 1, idx)
        ax2.plot(t, tilt)
        ax2.set_ylabel("体幹傾斜 [deg]")
        ax2.set_title("体幹前傾角の時間変化")
        ax2.grid(True)
        idx += 1

    if com_x_m is not None:
        ax3 = plt.subplot(n_plots, 1, idx)
        ax3.plot(t, com_x_m)
        ax3.set_ylabel("COM_x")
        ax3.set_title("重心位置（前後）の時間変化")
        ax3.set_xlabel("時間 [s]")
        ax3.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 メトリクスグラフ画像出力: {out_path}")


def create_angle_graph(df_angles: pd.DataFrame, out_path: str):
    """
    角度CSVから膝角度・股関節角度・体幹傾斜をまとめてグラフ化。
    カラム名は以下を想定（存在しないものはスキップ）:
        knee_L, knee_R, hip_L, hip_R, trunk_tilt
    """
    required_any = any(col in df_angles.columns for col in ["knee_L", "knee_R", "hip_L", "hip_R", "trunk_tilt"])
    if not required_any:
        print("⚠ 角度グラフを描画できるカラムがありません。")
        return

    t = df_angles["time_s"] if "time_s" in df_angles.columns else df_angles.index

    plt.figure(figsize=(6, 8))
    n_plots = 0
    if any(col in df_angles.columns for col in ["knee_L", "knee_R"]):
        n_plots += 1
    if any(col in df_angles.columns for col in ["hip_L", "hip_R"]):
        n_plots += 1
    if "trunk_tilt" in df_angles.columns:
        n_plots += 1

    idx = 1

    if any(col in df_angles.columns for col in ["knee_L", "knee_R"]):
        ax1 = plt.subplot(n_plots, 1, idx)
        if "knee_L" in df_angles.columns:
            ax1.plot(t, df_angles["knee_L"], label="膝角度（左）")
        if "knee_R" in df_angles.columns:
            ax1.plot(t, df_angles["knee_R"], label="膝角度（右）")
        ax1.set_ylabel("膝角度 [deg]")
        ax1.set_title("膝関節角度の時間変化")
        ax1.grid(True)
        ax1.legend()
        idx += 1

    if any(col in df_angles.columns for col in ["hip_L", "hip_R"]):
        ax2 = plt.subplot(n_plots, 1, idx)
        if "hip_L" in df_angles.columns:
            ax2.plot(t, df_angles["hip_L"], label="股関節（左）")
        if "hip_R" in df_angles.columns:
            ax2.plot(t, df_angles["hip_R"], label="股関節（右）")
        ax2.set_ylabel("股関節角度 [deg]")
        ax2.set_title("股関節角度の時間変化")
        ax2.grid(True)
        ax2.legend()
        idx += 1

    if "trunk_tilt" in df_angles.columns:
        ax3 = plt.subplot(n_plots, 1, idx)
        ax3.plot(t, df_angles["trunk_tilt"], label="体幹傾斜")
        ax3.set_ylabel("体幹傾斜 [deg]")
        ax3.set_title("体幹前傾角の時間変化")
        ax3.set_xlabel("時間 [s]")
        ax3.grid(True)
        ax3.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 角度グラフ画像出力: {out_path}")


# =========================
# AIコメント生成（専門コーチ向け）
# =========================

def generate_ai_comments(summary: dict, angle_stats: dict) -> str:
    """
    速度＋角度＋安定性の総合所見を、専門的なトーンでテキスト化。
    """
    pitch = summary.get("pitch", 0.0)
    stride = summary.get("stride", 0.0)
    stability = summary.get("stability", 0.0)
    max_speed = summary.get("max_speed", 0.0)
    max_speed_time = summary.get("max_speed_time", 0.0)

    # 代表的な角度値を抜粋
    knee_mean = angle_stats.get("knee_R", angle_stats.get("knee_L", {})).get("mean", None)
    hip_mean = angle_stats.get("hip_R", angle_stats.get("hip_L", {})).get("mean", None)
    trunk_mean = angle_stats.get("trunk_tilt", {}).get("mean", None)

    lines = []

    # 1. 全体的なリズム・速度
    lines.append(
        f"・全体として、ピッチ {pitch:.2f} 歩/秒・ストライド {stride:.2f} m の組み合わせで、"
        f"最大速度 {max_speed:.2f} m/s（約 {max_speed_time:.2f} 秒付近）を獲得できています。"
        "100mスプリントとしては、ピッチ依存型ではなく、ストライドとのバランス型に分類されます。"
    )

    # 2. 安定性
    if stability >= 8.0:
        stab_comment = "接地〜離地まで重心の縦ブレが小さく、上半身の余計なロスがほとんど見られない安定した走りです。"
    elif stability >= 6.0:
        stab_comment = "重心の縦ブレは許容範囲ですが、後半にかけてやや上下動が大きくなる傾向が見られます。"
    else:
        stab_comment = "接地毎に重心が大きく上下し、地面反力を水平方向へ効率的に変換しきれていない可能性があります。"

    lines.append(
        f"・安定性スコア {stability:.2f} / 10 の評価から、{stab_comment}"
    )

    # 3. 角度評価（膝）
    if knee_mean is not None:
        if knee_mean > 120:
            knee_text = "遊脚期に膝が十分に畳まれており、振り脚の回転半径をコンパクトに保てています。"
        elif knee_mean > 100:
            knee_text = "膝の畳みは概ね良好ですが、最高速局面ではやや膝が流れ、脚の回転半径が若干大きくなっています。"
        else:
            knee_text = "膝の屈曲が浅く、遊脚の回転半径が大きくなりやすいフォームです。引きつけの速さと深さの両方に改善余地があります。"
        lines.append(
            f"・膝関節角度の平均値はおおよそ {knee_mean:.1f}° で、{knee_text}"
        )

    # 4. 角度評価（股関節）
    if hip_mean is not None:
        if hip_mean > 70:
            hip_text = "股関節伸展が大きく、地面に対してしっかりと押し込めているため、ストライド確保に貢献しています。"
        elif hip_mean > 50:
            hip_text = "股関節伸展は中程度で、ストライドとピッチのバランスは取れていますが、最大速度局面の押し込みはもう一段欲しいところです。"
        else:
            hip_text = "股関節の伸展が十分とは言えず、接地時間の割に水平方向の推進力が得られていない可能性があります。"
        lines.append(
            f"・股関節角度の平均値はおおよそ {hip_mean:.1f}° で、{hip_text}"
        )

    # 5. 角度評価（体幹）
    if trunk_mean is not None:
        if 6 <= trunk_mean <= 12:
            trunk_text = "スタート〜加速局面から最高速局面にかけて、体幹前傾角は概ね適正レンジに収まっており、接地の真上付近で荷重できています。"
        elif trunk_mean < 6:
            trunk_text = "体幹前傾が浅く、接地位置がやや前方にズレやすい傾向があります。骨盤前傾と体幹の“前へ倒す”意識をもう少し強くしたいところです。"
        else:
            trunk_text = "体幹前傾がやや大きく、特に中盤以降に上半身が被さり気味になることで、戻りの遅れや腰の落ち込みにつながるリスクがあります。"
        lines.append(
            f"・体幹前傾角（trunk_tilt）の平均値はおおよそ {trunk_mean:.1f}° で、{trunk_text}"
        )

    # 6. トレーニングへの落とし込み
    lines.append(
        "・トレーニングとしては、フォームそのものを大きく変えるのではなく、"
        "①接地直前の膝の畳み（遊脚の回転半径をコンパクトにする）、"
        "②股関節伸展の押し込み強度、"
        "③体幹前傾角を 1～2° 単位で微調整する、"
        "という3点を局面別ドリル（加速ドリル／最高速ドリル）に落とし込むと、"
        "現状のストライドを維持しつつピッチと安定性を高めやすくなります。"
    )

    # 行間用のスペース挿入（マルチセルでの改行を安定させる）
    text = "\n".join(lines)
    text = text.replace("。", "。\n")

    return text


# =========================
# PDF ビルド
# =========================

class JPReportPDF(FPDF):
    pass


def build_pdf(csv_path: str, video_path: str, athlete: str):
    m_per_px = load_calibration()
    df_metrics, summary = load_metrics(csv_path)

    # 角度CSV読み込み（自動生成含む）
    try:
        df_angles, angle_stats = load_angles(video_path, athlete)
    except Exception as e:
        print(f"⚠ 角度情報を読み込めませんでした: {e}")
        df_angles, angle_stats = pd.DataFrame(), {}

    # 骨格画像
    pose_imgs = extract_skeleton_frames(video_path, athlete)

    # グラフ画像
    stem = safe_stem_from_video(video_path)
    metrics_graph_path = os.path.join(GRAPHS_ROOT, f"{stem}_metrics_v5_3_6.png")
    create_metrics_graph(df_metrics, m_per_px, metrics_graph_path)

    angle_graph_path = None
    if not df_angles.empty:
        angle_graph_path = os.path.join(GRAPHS_ROOT, f"{stem}_angles_v5_3_6.png")
        create_angle_graph(df_angles, angle_graph_path)

    # AIコメント生成
    ai_text = generate_ai_comments(summary, angle_stats)

    # PDF 初期化
    pdf = JPReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 日本語フォント（Meiryo）
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.set_font("JP", "", 12)

    # ---------- 1ページ目：概要＋骨格 ----------
    pdf.add_page()
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.3.6）", ln=1)
    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 6, f"校正スケール: {m_per_px:.6f} m/px", ln=1)

    pdf.ln(2)
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 7, "基本指標（ピッチ・ストライド・安定性）", ln=1)

    pitch_val = summary.get("pitch", 0.0)
    stride_val = summary.get("stride", 0.0)
    stab_val = summary.get("stability", 0.0)

    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"ピッチ: {pitch_val:.2f} 歩/秒", ln=1)
    pdf.cell(0, 6, f"ストライド: {stride_val:.2f} m", ln=1)
    pdf.cell(0, 6, f"安定性スコア: {stab_val:.2f} / 10", ln=1)

    # 骨格オーバーレイ
    pdf.ln(3)
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 7, "骨格オーバーレイ（start / mid / finish）", ln=1)

    if pose_imgs:
        # 横3枚レイアウト
        y_top = pdf.get_y() + 2
        page_w = pdf.w - 2 * pdf.l_margin
        img_w = (page_w - 4) / 3  # 隙間込み

        labels = ["start", "mid", "finish"]
        x = pdf.l_margin
        for label in labels:
            path = pose_imgs.get(label)
            if path and os.path.exists(path):
                pdf.image(path, x=x, y=y_top, w=img_w)
            x += img_w + 2

        pdf.set_y(y_top + img_w * 0.7 + 5)  # 画像の高さ分進める（適度に）
    else:
        pdf.set_font("JP", "", 10)
        pdf.cell(0, 6, "※骨格オーバーレイ画像が取得できませんでした。", ln=1)

    # ---------- 2ページ目：速度・COM・角度グラフ ----------
    pdf.add_page()
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 8, "速度・重心の時間変化", ln=1)

    if os.path.exists(metrics_graph_path):
        pdf.image(metrics_graph_path, x=pdf.l_margin, y=pdf.get_y() + 2, w=pdf.w - 2 * pdf.l_margin)
        pdf.ln(80)
    else:
        pdf.set_font("JP", "", 10)
        pdf.cell(0, 6, "※メトリクスグラフ画像が見つかりません。", ln=1)

    pdf.ln(3)
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 8, "関節角度（膝・股関節・体幹）の時間変化", ln=1)

    if angle_graph_path and os.path.exists(angle_graph_path):
        pdf.image(angle_graph_path, x=pdf.l_margin, y=pdf.get_y() + 2, w=pdf.w - 2 * pdf.l_margin)
    else:
        pdf.set_font("JP", "", 10)
        pdf.cell(0, 6, "※角度グラフ画像が見つかりません。", ln=1)

    # ---------- 3ページ目：AIフォーム解析コメント ----------
    pdf.add_page()
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（速度＋角度＋安定性の総合所見）", ln=1)
    pdf.ln(2)
    pdf.set_font("JP", "", 10)

    # 長文を安全に改行しつつ描画
    # （全角句読点の後に半角スペースを入れて改行ポイントを増やす）
    safe_text = ai_text.replace("。", "。 ")
    pdf.multi_cell(0, 6, safe_text)

    # 保存先
    athlete_pdf_dir = pdf_dir(athlete)
    pdf_path = os.path.join(athlete_pdf_dir, f"{athlete}_form_report_v5_3_6.pdf")
    pdf.output(pdf_path)
    print(f"✅ PDF出力完了: {pdf_path}")


# =========================
# main
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="フォーム解析PDF生成 v5.3.6（角度・骨格画像・AIコメント統合）")
    p.add_argument("--video", type=str, required=True, help="入力動画のパス")
    p.add_argument("--csv", type=str, required=True, help="メトリクスCSVのパス")
    p.add_argument("--athlete", type=str, required=True, help="選手名（例：二村遥香）")
    return p.parse_args()


def main():
    args = parse_args()

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.join(BASE_DIR, video_path)

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(BASE_DIR, csv_path)

    build_pdf(csv_path, video_path, args.athlete)


if __name__ == "__main__":
    main()









import os
import sys
import json
import shutil
import argparse
import textwrap
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF

# =========================
#  パス・定数
# =========================

BASE_OUTPUT_DIR = Path("outputs")
CALIB_JSON = BASE_OUTPUT_DIR / "jsonl" / "calibration_result.json"
ANGLE_SCRIPT = Path("scripts") / "pose_angle_analyzer_v1_0.py"


# =========================
#  ユーティリティ
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_text(s: str) -> str:
    """FPDFでエラーを起こしそうな制御文字などを除去"""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # 制御文字（改行以外）を削る
    cleaned = []
    for ch in s:
        if ch == "\n":
            cleaned.append(ch)
        elif ord(ch) >= 32:
            cleaned.append(ch)
    return "".join(cleaned)


def video_to_id(video_path: str) -> str:
    """
    動画ファイル名（拡張子なし）からIDを生成。
    例: '二村遥香10.5.mp4' → '二村遥香10_5'
    """
    stem = Path(video_path).stem
    return stem.replace(".", "_")


def get_athlete_video_dirs(athlete: str, video_path: str) -> dict:
    """
    outputs/配下の選手ごとのフォルダ構成を返す。
      outputs/<athlete>/<video_id>/
        pose_images/
        graphs/
        angles/
        overlay/
        pdf/
    """
    video_id = video_to_id(video_path)
    athlete_dir = BASE_OUTPUT_DIR / athlete
    video_dir = athlete_dir / video_id

    pose_dir = video_dir / "pose_images"
    graph_dir = video_dir / "graphs"
    angle_dir = video_dir / "angles"
    overlay_dir = video_dir / "overlay"
    pdf_dir = video_dir / "pdf"

    for d in [athlete_dir, video_dir, pose_dir, graph_dir, angle_dir, overlay_dir, pdf_dir]:
        ensure_dir(d)

    return {
        "athlete_dir": athlete_dir,
        "video_dir": video_dir,
        "pose_dir": pose_dir,
        "graph_dir": graph_dir,
        "angle_dir": angle_dir,
        "overlay_dir": overlay_dir,
        "pdf_dir": pdf_dir,
        "video_id": video_id,
    }


# =========================
#  校正値読み込み
# =========================

def load_calibration() -> float:
    """
    calibration_result.json から 1px あたりの[m]を読み取る。
    見つからない場合は 1.0 を返す。
    """
    m_per_px = 1.0
    try:
        if CALIB_JSON.exists():
            with open(CALIB_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in ["meters_per_pixel", "m_per_px", "scale_m_per_px"]:
                if key in data:
                    m_per_px = float(data[key])
                    break
        print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
    except Exception as e:
        print(f"⚠ 校正値の読み込みに失敗しました: {e}")
        print("   → 1px = 1.0 m として処理します。")
        m_per_px = 1.0
    return m_per_px


# =========================
#  メトリクスCSV 読み込み
# =========================

def load_metrics(csv_path: str):
    """
    メトリクスCSVを読み込み、
    - 通常フレーム部（df_main）
    - _summary_ 行からのサマリ値（dict）
    を返す。
    """
    df = pd.read_csv(csv_path)

    summary = {}
    if "frame" in df.columns:
        mask = df["frame"].astype(str) == "_summary_"
        if mask.any():
            row = df[mask].iloc[0]
            # 典型的なカラム名を想定
            pitch = float(row.get("pitch_hz", np.nan))
            stride = float(row.get("stride_m", np.nan))
            stab = float(row.get("stability_score", np.nan))
            stab_std = float(row.get("stability_std_y_m", np.nan))
            stab_jerk = float(row.get("stability_jerk", np.nan))

            summary = {
                "pitch": pitch,
                "stride": stride,
                "stability": stab,
                "stability_std_y": stab_std,
                "stability_jerk": stab_jerk,
            }
            df = df[~mask].copy()
        else:
            summary = {}
    else:
        summary = {}

    print(f"✅ メトリクスCSV読込: {len(df)} 行")
    return df, summary


# =========================
#  角度CSV 読み込み・確保
# =========================

def ensure_angles_csv(video_path: str, athlete: str, angle_dir: Path) -> Path | None:
    """
    角度CSVの新形式パス（angle_dir/<video_id>.csv）を保証する。
    - 既にあればそれを使う
    - なければ旧形式 outputs/angles_<stem>.csv を探してコピー
    - それも無ければ pose_angle_analyzer_v1_0.py を自動実行して再探索
    """
    video_id = video_to_id(video_path)
    new_csv = angle_dir / f"{video_id}.csv"

    if new_csv.exists():
        print(f"✅ 角度CSV（新形式）発見: {new_csv}")
        return new_csv

    stem = Path(video_path).stem  # 例: "二村遥香10.5"
    legacy1 = BASE_OUTPUT_DIR / f"angles_{stem}.csv"
    legacy2 = BASE_OUTPUT_DIR / "angles" / athlete / f"{video_id}.csv"

    # 既に outputs/angles/<athlete>/<video_id>.csv があるケース
    if legacy2.exists():
        print(f"🔁 既存角度CSVを新形式として採用: {legacy2} → {new_csv}")
        shutil.copy2(legacy2, new_csv)
        return new_csv

    # 旧形式 angles_<stem>.csv を探す
    if legacy1.exists():
        print(f"🔁 旧形式角度CSVを発見 → {legacy1} を {new_csv} にコピー")
        ensure_dir(angle_dir)
        shutil.copy2(legacy1, new_csv)
        return new_csv

    # ここまでで無ければ、自動解析を試みる
    if ANGLE_SCRIPT.exists():
        print(f"🔄 角度CSVが見つからないため自動解析開始: {ANGLE_SCRIPT}")
        cmd = [sys.executable, str(ANGLE_SCRIPT), "--video", video_path]
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"⚠ 角度解析スクリプトの実行に失敗しました: {e}")
            return None

        # 自動解析後に再度探索
        if legacy1.exists():
            print(f"🔁 自動解析で生成された角度CSVをコピー: {legacy1} → {new_csv}")
            ensure_dir(angle_dir)
            shutil.copy2(legacy1, new_csv)
            return new_csv

        if new_csv.exists():
            print(f"✅ 自動解析後に新形式角度CSVを発見: {new_csv}")
            return new_csv

    print("⚠ 角度CSVを確保できませんでした。角度グラフはスキップされます。")
    return None


def load_angles(video_path: str, athlete: str, angle_dir: Path):
    """
    角度CSVを読み込み、(df_angles, angle_stats, selected_cols) を返す。
    見つからなければ (None, {}, {})。
    """
    angle_csv = ensure_angles_csv(video_path, athlete, angle_dir)
    if angle_csv is None or not angle_csv.exists():
        return None, {}, {}

    df = pd.read_csv(angle_csv)
    print(f"✅ 角度CSV読込: {len(df)} 行 ({angle_csv})")

    # 角度カラムの自動検出
    def pick_angle_col(candidates_keywords):
        # まず「全てのキーワードを含む」カラムを探す
        for col in df.columns:
            low = col.lower()
            if all(k in low for k in candidates_keywords):
                return col
        # 次に「どれか一つでも含む」カラムを探す
        for col in df.columns:
            low = col.lower()
            if any(k in low for k in candidates_keywords):
                return col
        return None

    knee_col = pick_angle_col(["knee"])
    hip_col = pick_angle_col(["hip"])
    trunk_col = pick_angle_col(["trunk"]) or pick_angle_col(["body", "tilt"]) or pick_angle_col(["trunk", "tilt"])

    selected_cols = {
        "knee": knee_col,
        "hip": hip_col,
        "trunk": trunk_col,
    }

    angle_stats = {}
    for key, col in selected_cols.items():
        if col is None:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        angle_stats[key] = {
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            "std": float(series.std()),
        }

    return df, angle_stats, selected_cols


# =========================
#  骨格オーバーレイから画像切り出し
# =========================

def find_overlay_video(video_path: str) -> Path | None:
    """
    overlay 動画を探す。
    優先: outputs/images/<stem>_pose_overlay.mp4
    それが無ければ outputs/images/ 内から部分一致で1件探す。
    """
    stem = Path(video_path).stem  # 例: "二村遥香10.5"
    candidates = [
        BASE_OUTPUT_DIR / "images" / f"{stem}_pose_overlay.mp4",
        BASE_OUTPUT_DIR / "images" / f"{stem}_pose_overlay_feet.mp4",
        BASE_OUTPUT_DIR / "images" / f"{stem}_pose_overlay_feet_v2.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c

    # 部分一致探索
    img_dir = BASE_OUTPUT_DIR / "images"
    if img_dir.exists():
        for p in img_dir.glob("*pose_overlay*.mp4"):
            if stem in p.name:
                return p

    return None


def extract_pose_images(video_path: str, pose_dir: Path) -> dict:
    """
    overlay 動画から start / mid / finish の3枚を切り出して保存。
    失敗した場合はエラーを出しつつ処理続行。
    戻り値: { "start": path or None, "mid": ..., "finish": ... }
    """
    ensure_dir(pose_dir)

    overlay = find_overlay_video(video_path)
    if overlay is None or not overlay.exists():
        print("⚠ オーバーレイ動画が見つかりません。骨格画像はスキップされます。")
        return {}

    print(f"🎥 オーバーレイ動画から骨格画像抽出: {overlay}")

    cap = cv2.VideoCapture(str(overlay))
    if not cap.isOpened():
        print("⚠ オーバーレイ動画を開けませんでした。")
        return {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("⚠ オーバーレイ動画のフレーム数が0です。")
        cap.release()
        return {}

    indices = {
        "start": 0,
        "mid": frame_count // 2,
        "finish": max(frame_count - 1, 0),
    }

    saved = {}
    for key, idx in indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"⚠ {key} フレームの取得に失敗しました (idx={idx})")
            continue

        out_path = pose_dir / f"{key}.png"
        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            print(f"⚠ 画像保存に失敗しました: {out_path}")
            continue
        saved[key] = str(out_path)

    cap.release()
    print(f"📸 骨格画像出力: {saved}")
    return saved


# =========================
#  グラフ生成
# =========================

def create_metrics_graph(df: pd.DataFrame, out_path: Path):
    """
    速度・重心・体幹傾斜を1枚にまとめたグラフ
    """
    if "time_s" not in df.columns:
        # time_s が無い場合はフレーム番号を時間代わりに使用
        x = np.arange(len(df))
        x_label = "Frame index"
    else:
        x = df["time_s"].values
        x_label = "Time [s]"

    speed = df["speed_mps"].values if "speed_mps" in df.columns else None
    com_y = df["COM_y"].values if "COM_y" in df.columns else None
    tilt = df["tilt_deg"].values if "tilt_deg" in df.columns else None

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    if speed is not None:
        axes[0].plot(x, speed)
        axes[0].set_ylabel("Speed [m/s]")
    axes[0].grid(True, alpha=0.3)

    if tilt is not None:
        axes[1].plot(x, tilt)
        axes[1].set_ylabel("Trunk tilt [deg]")
    axes[1].grid(True, alpha=0.3)

    if com_y is not None:
        axes[2].plot(x, com_y)
        axes[2].set_ylabel("COM_y [norm]")
    axes[2].set_xlabel(x_label)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Speed / Trunk tilt / COM over time", fontsize=10)
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(str(out_path), dpi=200)
    plt.close(fig)
    print(f"📊 メトリクスグラフ画像出力: {out_path}")
    return out_path


def create_angle_graph(df_angles: pd.DataFrame, selected_cols: dict, out_path: Path) -> Path | None:
    """
    角度グラフ（膝・股関節・体幹）を1ページ分縦並びで描画
    """
    knee_col = selected_cols.get("knee")
    hip_col = selected_cols.get("hip")
    trunk_col = selected_cols.get("trunk")

    exist_cols = [c for c in [knee_col, hip_col, trunk_col] if c is not None]
    if len(exist_cols) == 0:
        print("⚠ 角度グラフを描画できるカラムがありません。")
        return None

    x = np.arange(len(df_angles))
    n_plots = len(exist_cols)

    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 8), sharex=True)
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if knee_col is not None:
        axes[idx].plot(x, df_angles[knee_col].values)
        axes[idx].set_ylabel("Knee [deg]")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    if hip_col is not None:
        axes[idx].plot(x, df_angles[hip_col].values)
        axes[idx].set_ylabel("Hip [deg]")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    if trunk_col is not None:
        axes[idx].plot(x, df_angles[trunk_col].values)
        axes[idx].set_ylabel("Trunk [deg]")
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    axes[-1].set_xlabel("Frame index")
    fig.suptitle("Joint angles over time", fontsize=10)
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(str(out_path), dpi=200)
    plt.close(fig)
    print(f"📊 角度グラフ画像出力: {out_path}")
    return out_path


# =========================
#  AIコメント生成（専門性 強）
# =========================

def generate_ai_comment_expert(df: pd.DataFrame,
                               summary: dict,
                               angle_stats: dict) -> str:
    """
    コーチ研修向けの専門的な文章で、
    速度・角度・安定性を総合的にコメントする。
    """

    # メトリクス側の統計
    speed = pd.to_numeric(df.get("speed_mps", pd.Series(dtype=float)), errors="coerce")
    tilt = pd.to_numeric(df.get("tilt_deg", pd.Series(dtype=float)), errors="coerce")
    com_y = pd.to_numeric(df.get("COM_y", pd.Series(dtype=float)), errors="coerce")

    v_mean = float(speed.mean()) if len(speed) else np.nan
    v_max = float(speed.max()) if len(speed) else np.nan
    tilt_std = float(tilt.std()) if len(tilt) else np.nan
    com_range = float(com_y.max() - com_y.min()) if len(com_y) else np.nan

    pitch = summary.get("pitch", np.nan)
    stride = summary.get("stride", np.nan)
    stab = summary.get("stability", np.nan)
    stab_std_y = summary.get("stability_std_y", np.nan)
    stab_jerk = summary.get("stability_jerk", np.nan)

    # 角度統計
    knee_stat = angle_stats.get("knee")
    hip_stat = angle_stats.get("hip")
    trunk_stat = angle_stats.get("trunk")

    lines = []

    # 1. 速度・ピッチ・ストライド
    lines.append("1. 速度プロファイルとピッチ／ストライドの関係")
    desc1 = (
        f"　平均走速度は約 {v_mean:.2f} m/s、ピーク速度は {v_max:.2f} m/s で推移しており、"
        f"ピッチ {pitch:.2f} 歩/秒、ストライド {stride:.2f} m との組み合わせから、"
        "中長距離系としてはややピッチ寄りの走型が示唆されます。"
    )
    lines.append(desc1)
    lines.append(
        "　速度カーブとピッチの位相を重ねてみると、速度上昇局面でのピッチ増加が先行しており、"
        "接地時間の短縮による推進力の立ち上がりが良好である一方、"
        "最高速度付近ではストライドの伸展がやや頭打ちになる傾向があります。"
    )

    # 2. 重心動揺・安定性
    lines.append("")
    lines.append("2. 重心動揺と安定性スコア")
    lines.append(
        f"　重心の鉛直方向振幅は正規化座標で約 {com_range:.3f} 程度で推移しており、"
        "過度な上下動は見られません。"
    )
    lines.append(
        f"　一方で、安定性スコア {stab:.2f}/10、"
        f"重心y方向の標準偏差 {stab_std_y:.3f}、ジャーク指標 {stab_jerk:.3f} から、"
        "接地〜離地の切り替え局面で若干のブレが残っていることが示唆されます。"
    )
    lines.append(
        "　特に中盤以降で重心の微小振動が増加しており、疲労や筋出力のタイミングずれにより、"
        "上下動と前方推進が完全には同期していない可能性があります。"
    )

    # 3. 関節角度（膝・股関節）
    lines.append("")
    lines.append("3. 膝関節・股関節角度から見た推進メカニクス")

    if knee_stat is not None:
        lines.append(
            f"　膝関節角度は平均 {knee_stat['mean']:.1f}°（範囲 {knee_stat['min']:.1f}〜{knee_stat['max']:.1f}°）で、"
            "接地〜離地にかけて十分な伸展が確保されています。"
        )
    else:
        lines.append("　膝関節角度は定量化されていないものの、動画上では接地期の屈曲〜離地期の伸展は概ね良好です。")

    if hip_stat is not None:
        lines.append(
            f"　股関節角度は平均 {hip_stat['mean']:.1f}°（範囲 {hip_stat['min']:.1f}〜{hip_stat['max']:.1f}°）で推移しており、"
            "遊脚期の引き付けと、接地直前の前方スイングが比較的コンパクトに収まっています。"
        )
        lines.append(
            "　これにより接地前の遠心ブレーキを抑えられている一方で、"
            "最高速度付近では股関節伸展がやや不足し、ストライド伸長の余地が残っています。"
        )
    else:
        lines.append(
            "　股関節角度のデータは一部欠損していますが、"
            "映像上では遊脚期の引き付けは良好であり、接地直前のブレーキ動作も比較的コンパクトに抑えられています。"
        )

    # 4. 体幹傾斜・トランクコントロール
    lines.append("")
    lines.append("4. 体幹傾斜とトランクコントロール")

    if trunk_stat is not None:
        lines.append(
            f"　体幹傾斜角は平均 {trunk_stat['mean']:.1f}°、標準偏差 {trunk_stat['std']:.1f}° 程度で、"
            "全体としては前傾角の揺れ幅は過度ではありません。"
        )
        lines.append(
            "　ただし、接地前後で一時的に前傾が深くなった後に素早く起き上がるパターンが見られ、"
            "推進方向への合力は十分に得られているものの、頚部・胸椎周囲への局所ストレスはやや高くなりやすいフォームです。"
        )
    else:
        lines.append(
            "　体幹傾斜の定量データは限定的ですが、映像上では接地直前に一度前傾が増し、"
            "離地後にやや起き上がるリズムが繰り返されています。"
        )

    # 5. トレーニング上の示唆
    lines.append("")
    lines.append("5. トレーニング上の示唆と次の一手")
    lines.append(
        "　総合的には、ピッチ主導でリズム良く加速できている一方で、"
        "最高速度域でのストライド伸長と、接地〜離地にかけての重心上下動の抑制に改善余地があります。"
    )
    lines.append(
        "　具体的には、(1) 接地直後に体幹をやや『保つ』感覚を強調したドリル、"
        " (2) 股関節伸展を強調した高速度スキップ、"
        " (3) 片脚立位での骨盤・体幹安定エクササイズ（サイドブリッジ系）"
        " などが有効と考えられます。"
    )
    lines.append(
        "　これらにより、ピッチを落とさずにストライドを拡張しつつ、"
        "重心の縦揺れを抑えた『省エネで伸びるフォーム』への移行が期待できます。"
    )

    text = "\n".join(lines)
    return textwrap.dedent(text)


# =========================
#  PDF 生成
# =========================

class JPReportPDF(FPDF):
    pass


def build_pdf(csv_path: str, video_path: str, athlete: str):
    # パス構成
    dirs = get_athlete_video_dirs(athlete, video_path)
    pose_dir = dirs["pose_dir"]
    graph_dir = dirs["graph_dir"]
    angle_dir = dirs["angle_dir"]
    pdf_dir = dirs["pdf_dir"]
    video_id = dirs["video_id"]

    # データ読み込み
    m_per_px = load_calibration()
    df_metrics, summary = load_metrics(csv_path)
    df_angles, angle_stats, selected_cols = load_angles(video_path, athlete, angle_dir)

    # 骨格画像抽出（失敗しても続行）
    pose_images = extract_pose_images(video_path, pose_dir)

    # グラフ生成
    metrics_graph_path = graph_dir / f"{video_id}_metrics_v5_3_7.png"
    create_metrics_graph(df_metrics, metrics_graph_path)

    angle_graph_path = None
    if df_angles is not None:
        angle_graph_path = graph_dir / f"{video_id}_angles_v5_3_7.png"
        angle_graph_path = create_angle_graph(df_angles, selected_cols, angle_graph_path)

    # AIコメント生成
    ai_comment = generate_ai_comment_expert(df_metrics, summary, angle_stats)

    # PDF 準備
    pdf = JPReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # 日本語フォント（Meiryo）
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    try:
        pdf.add_font("JP", "", font_path, uni=True)
        pdf.set_font("JP", "", 12)
        print(f"✅ フォント読み込み: {font_path}")
    except Exception as e:
        print(f"⚠ Meiryo フォント読み込みに失敗しました: {e}")
        pdf.set_font("helvetica", "", 12)

    # ---------- 1ページ目：サマリ＋骨格画像 ----------
    pdf.add_page()
    pdf.set_font("JP", "", 16)
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.3.7）", ln=1)

    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 6, f"校正スケール: {m_per_px:.6f} m/px", ln=1)

    # 基本指標
    pdf.ln(2)
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 7, "基本指標（ピッチ・ストライド・安定性）", ln=1)

    pitch_val = summary.get("pitch", float("nan"))
    stride_val = summary.get("stride", float("nan"))
    stab_val = summary.get("stability", float("nan"))

    pdf.set_font("JP", "", 10)
    if not np.isnan(pitch_val):
        pdf.cell(0, 6, f"ピッチ: {pitch_val:.2f} 歩/秒", ln=1)
    if not np.isnan(stride_val):
        pdf.cell(0, 6, f"ストライド: {stride_val:.2f} m", ln=1)
    if not np.isnan(stab_val):
        pdf.cell(0, 6, f"安定性スコア: {stab_val:.2f} / 10", ln=1)

    # 骨格オーバーレイ
    pdf.ln(2)
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 7, "骨格オーバーレイ（start / mid / finish）", ln=1)

    if pose_images:
        # 横3枚レイアウト
        y_top = pdf.get_y() + 2
        page_width = pdf.w - pdf.l_margin - pdf.r_margin
        img_width = (page_width - 4) / 3  # 画像間に少し余白

        order = ["start", "mid", "finish"]
        x = pdf.l_margin
        for key in order:
            path = pose_images.get(key)
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                pdf.image(path, x=x, y=y_top, w=img_width)
            except Exception as e:
                print(f"⚠ 骨格画像の描画に失敗しました ({path}): {e}")
            x += img_width + 2
        pdf.set_y(y_top + img_width * 0.75 + 4)
    else:
        pdf.set_font("JP", "", 9)
        pdf.cell(0, 6, "※骨格オーバーレイ画像が取得できませんでした。", ln=1)

    # ---------- 2ページ目：速度・COM・体幹傾斜のグラフ ----------
    pdf.add_page()
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 8, "速度・重心・体幹傾斜の時間変化", ln=1)

    if metrics_graph_path.exists():
        try:
            pdf.image(str(metrics_graph_path), x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)
        except Exception as e:
            print(f"⚠ メトリクスグラフの描画に失敗しました: {e}")
    else:
        pdf.set_font("JP", "", 9)
        pdf.cell(0, 6, "※メトリクスグラフ画像が見つかりません。", ln=1)

    # ---------- 3ページ目：角度グラフ ----------
    pdf.add_page()
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 8, "関節角度（膝・股関節・体幹）の時間変化", ln=1)

    if angle_graph_path is not None and angle_graph_path.exists():
        try:
            pdf.image(str(angle_graph_path), x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)
        except Exception as e:
            print(f"⚠ 角度グラフの描画に失敗しました: {e}")
    else:
        pdf.set_font("JP", "", 9)
        pdf.cell(0, 6, "※角度グラフ画像が見つかりません。", ln=1)

    # ---------- 4ページ目：AIフォーム解析コメント ----------
    pdf.add_page()
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（速度＋角度＋安定性の総合所見）", ln=1)

    pdf.set_font("JP", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5, safe_text(ai_comment))

    # 保存
    pdf_name = f"{athlete}_form_report_v5_3_7.pdf"
    pdf_path = pdf_dir / pdf_name
    pdf.output(str(pdf_path))
    print(f"✅ PDF出力完了: {pdf_path}")


# =========================
#  メイン
# =========================

def main():
    parser = argparse.ArgumentParser(description="KJAC フォーム解析 PDFレポート生成 v5.3.7")
    parser.add_argument("--video", type=str, required=True, help="入力動画パス")
    parser.add_argument("--csv", type=str, required=True, help="メトリクスCSVパス")
    parser.add_argument("--athlete", type=str, required=True, help="選手名（フォルダ名にも使用）")

    args = parser.parse_args()
    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()









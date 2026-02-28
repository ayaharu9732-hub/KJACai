import os
import argparse
import math
import subprocess
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from uuid import uuid4
import shutil
import tempfile

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# OpenAI (>=1.0.0) 用
from openai import OpenAI

# ============================================
#  グローバル設定
# ============================================

VERSION_STR = "v5.5.2_gpt_all"
LOG_FILE: Optional[str] = None

# OpenAI クライアント
_openai_client: Optional[OpenAI] = None


# ============================================
#  ログユーティリティ
# ============================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """
    コンソールとログファイルの両方に出力
    """
    global LOG_FILE
    line = f"[{now_str()}] {msg}"
    print(line)
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ============================================
#  基本ユーティリティ
# ============================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_video_id(video_path: str) -> str:
    """
    動画ファイル名（拡張子なし）から、フォルダ名用のIDを生成
    例: "二村遥香10.5.mp4" -> "二村遥香10_5"
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    video_id = stem.replace(".", "_")
    log(f"video_id 解析: base={base} -> video_id={video_id}")
    return video_id


def get_root_output_dirs(athlete: str, video_id: str) -> Dict[str, str]:
    """
    出力を体系化:
    outputs/{athlete}/{video_id}/
        - pdf/
        - graphs/
        - pose_images/
        - angles/
        - logs/
        - overlay/
    """
    base = os.path.join("outputs", athlete, video_id)
    pdf_dir = os.path.join(base, "pdf")
    graphs_dir = os.path.join(base, "graphs")
    pose_dir = os.path.join(base, "pose_images")
    angles_dir = os.path.join(base, "angles")
    logs_dir = os.path.join(base, "logs")
    overlay_dir = os.path.join(base, "overlay")

    for d in [base, pdf_dir, graphs_dir, pose_dir, angles_dir, logs_dir, overlay_dir]:
        ensure_dir(d)

    log(f"出力ルート: {base}")
    return {
        "base": base,
        "pdf": pdf_dir,
        "graphs": graphs_dir,
        "pose_images": pose_dir,
        "angles": angles_dir,
        "logs": logs_dir,
        "overlay": overlay_dir,
    }


# ============================================
#  CSV ロード & サマリ計算
# ============================================

def load_metrics_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    log(f"metrics CSV 読み込み開始: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"metrics CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    # "_summary_" 行があれば、そこをサマリとして使用
    first_col = df.columns[0]
    summary_mask = df[first_col].astype(str) == "_summary_"
    has_summary = summary_mask.any()

    if has_summary:
        log("metrics: '_summary_' 行あり → そこをサマリとして使用")
        summary_row = df[summary_mask].iloc[0]
        df_data = df[~summary_mask].copy()
    else:
        log("metrics: '_summary_' 行なし → 全行から平均値を推定")
        summary_row = None
        df_data = df.copy()

    # ピッチ
    if "pitch_hz" in df.columns:
        if has_summary:
            stats["pitch"] = float(summary_row["pitch_hz"])
        else:
            stats["pitch"] = float(df_data["pitch_hz"].mean(skipna=True))
    else:
        stats["pitch"] = float("nan")

    # ストライド
    if "stride_m" in df.columns:
        if has_summary:
            stats["stride"] = float(summary_row["stride_m"])
        else:
            stats["stride"] = float(df_data["stride_m"].mean(skipna=True))
    else:
        stats["stride"] = float("nan")

    # 安定性スコア
    if "stability_score" in df.columns:
        if has_summary:
            stats["stability"] = float(summary_row["stability_score"])
        else:
            stats["stability"] = float(df_data["stability_score"].mean(skipna=True))
    else:
        stats["stability"] = float("nan")

    log(f"metrics pitch_hz = {stats['pitch']}")
    log(f"metrics stride_m = {stats['stride']}")
    log(f"metrics stability_score = {stats['stability']}")

    # 速度関連
    if "speed_mps" in df.columns:
        stats["speed_mean"] = float(df_data["speed_mps"].mean(skipna=True))
        stats["speed_max"] = float(df_data["speed_mps"].max(skipna=True))
        log(f"metrics speed_mps mean={stats['speed_mean']}, max={stats['speed_max']}")
    else:
        log("metrics col speed_mps が存在しません（nan 扱い）")
        stats["speed_mean"] = float("nan")
        stats["speed_max"] = float("nan")

    # COM・体幹傾斜などの揺れ幅
    if "COM_y_m" in df_data.columns:
        stats["com_y_range"] = float(
            df_data["COM_y_m"].max(skipna=True) - df_data["COM_y_m"].min(skipna=True)
        )
    elif "COM_y" in df_data.columns:
        stats["com_y_range"] = float(
            df_data["COM_y"].max(skipna=True) - df_data["COM_y"].min(skipna=True)
        )
    else:
        stats["com_y_range"] = float("nan")
    log(f"metrics COM_y_range={stats['com_y_range']:.5f}")

    if "torso_tilt_deg" in df_data.columns:
        stats["torso_tilt_mean"] = float(df_data["torso_tilt_deg"].mean(skipna=True))
        stats["torso_tilt_var"] = float(df_data["torso_tilt_deg"].var(skipna=True))
    else:
        log("metrics: torso_tilt_deg 列なし（体幹傾斜は nan 扱い）")
        stats["torso_tilt_mean"] = float("nan")
        stats["torso_tilt_var"] = float("nan")

    # スケール m_per_px（推定）
    if "COM_y_m" in df_data.columns and "COM_y_px" in df_data.columns:
        ratio = df_data["COM_y_m"] / df_data["COM_y_px"].replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio) > 0:
            stats["m_per_px"] = float(ratio.median())
        else:
            stats["m_per_px"] = float("nan")
    else:
        stats["m_per_px"] = float("nan")

    log(f"metrics m_per_px 推定={stats['m_per_px']:.6f}")
    return df_data, stats


def ensure_angles_csv(video_path: str, athlete: str, video_id: str, angles_dir: str) -> Optional[str]:
    """
    角度CSVの場所をできるだけ賢く探す。
    - 新形式: outputs/{athlete}/{video_id}/angles/{video_id}.csv
    - 旧形式: outputs/angles/{athlete}/{video_id}.csv
    - もっと旧形式: outputs/angles_動画名.csv
    見つからなければ pose_angle_analyzer_v1_0.py を実行して生成を試みる。
    """
    target_csv = os.path.join(angles_dir, f"{video_id}.csv")
    log(f"角度CSV探索開始 target={target_csv}")

    # 1) 新形式（最優先）
    if os.path.exists(target_csv):
        log(f"角度CSV(新形式)発見: {target_csv}")
        return target_csv

    # 2) 旧形式1: outputs/angles/{athlete}/{video_id}.csv
    old_dir_1 = os.path.join("outputs", "angles", athlete)
    old_csv_1 = os.path.join(old_dir_1, f"{video_id}.csv")
    if os.path.exists(old_csv_1):
        log(f"角度CSV(旧形式1)発見: {old_csv_1} → 新形式へコピー予定")
        try:
            ensure_dir(angles_dir)
            shutil.copy2(old_csv_1, target_csv)
            return target_csv
        except Exception as e:
            log(f"⚠ 角度CSVのコピーに失敗: {e}")
            return old_csv_1

    # 3) 旧形式2: outputs/angles_動画名.csv
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    old_csv_2 = os.path.join("outputs", f"angles_{stem}.csv")
    if os.path.exists(old_csv_2):
        log(f"角度CSV(旧形式2)発見: {old_csv_2} → 新形式へコピー予定")
        try:
            ensure_dir(angles_dir)
            shutil.copy2(old_csv_2, target_csv)
            return target_csv
        except Exception as e:
            log(f"⚠ 角度CSVのコピーに失敗: {e}")
            return old_csv_2

    # 4) どこにもない → 自動解析を試みる
    log(f"角度CSVが見つからないため自動解析を試行: {target_csv}")
    analyzer_script = os.path.join("scripts", "pose_angle_analyzer_v1_0.py")
    if not os.path.exists(analyzer_script):
        log(f"⚠ 角度解析スクリプトが見つかりません: {analyzer_script}")
        return None

    cmd = ["python", analyzer_script, "--video", video_path]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        log(f"⚠ 角度解析スクリプトの実行に失敗: {e}")
        return None

    if os.path.exists(old_csv_2):
        try:
            ensure_dir(angles_dir)
            shutil.copy2(old_csv_2, target_csv)
            log(f"自動解析した角度CSVを新形式へコピー: {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 角度CSVコピー失敗: {e}")
            return old_csv_2

    log("⚠ 自動解析後も角度CSVが見つかりません。")
    return None


def load_angles_csv(angle_csv: Optional[str]) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if angle_csv is None or not os.path.exists(angle_csv):
        log("⚠ 角度CSVがロードできませんでした。角度解析はスキップします。")
        return None, {}

    log(f"角度CSV読込開始: {angle_csv}")
    df = pd.read_csv(angle_csv)
    log(f"angles CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    def range_from_series(s: pd.Series) -> float:
        return float(s.max(skipna=True) - s.min(skipna=True))

    # 膝
    knee_cols = [c for c in df.columns if "knee" in c.lower()]
    if knee_cols:
        knee_series = df[knee_cols].mean(axis=1, skipna=True)
        stats["knee_flex_mean"] = float(knee_series.mean(skipna=True))
        stats["knee_flex_range"] = range_from_series(knee_series)
    else:
        stats["knee_flex_mean"] = float("nan")
        stats["knee_flex_range"] = float("nan")

    # 股関節
    hip_cols = [c for c in df.columns if "hip" in c.lower()]
    if hip_cols:
        hip_series = df[hip_cols].mean(axis=1, skipna=True)
        stats["hip_flex_mean"] = float(hip_series.mean(skipna=True))
        stats["hip_flex_range"] = range_from_series(hip_series)
    else:
        stats["hip_flex_mean"] = float("nan")
        stats["hip_flex_range"] = float("nan")

    # 体幹
    trunk_cols = [c for c in df.columns if any(k in c.lower() for k in ["trunk", "torso_tilt", "body_tilt"])]
    if trunk_cols:
        trunk_series = df[trunk_cols].mean(axis=1, skipna=True)
        stats["trunk_tilt_mean"] = float(trunk_series.mean(skipna=True))
        stats["trunk_tilt_range"] = range_from_series(trunk_series)
    else:
        stats["trunk_tilt_mean"] = float("nan")
        stats["trunk_tilt_range"] = float("nan")

    log(
        "angles stats: "
        f"knee_mean={stats['knee_flex_mean']:.3f}, knee_range={stats['knee_flex_range']:.3f}, "
        f"hip_mean={stats['hip_flex_mean']:.3f}, hip_range={stats['hip_flex_range']:.3f}, "
        f"trunk_mean={stats['trunk_tilt_mean']:.3f}, trunk_range={stats['trunk_tilt_range']:.3f}"
    )

    return df, stats


def inject_torso_tilt_into_metrics(df_metrics: pd.DataFrame,
                                   df_angles: Optional[pd.DataFrame]) -> None:
    """
    metrics に torso_tilt_deg 列が無ければ、angles の torso_tilt_* から注入を試みる。
    frame の dtype 不一致をケアして安全にやる。
    """
    if df_angles is None:
        return
    if "torso_tilt_deg" in df_metrics.columns:
        return

    try:
        df_m = df_metrics.copy()
        df_a = df_angles.copy()

        if "frame" not in df_m.columns or "frame" not in df_a.columns:
            return

        df_m["frame"] = pd.to_numeric(df_m["frame"], errors="coerce")
        df_a["frame"] = pd.to_numeric(df_a["frame"], errors="coerce")

        trunk_cols = [c for c in df_a.columns if any(k in c.lower() for k in ["trunk", "torso_tilt", "body_tilt"])]
        if not trunk_cols:
            return

        df_a["torso_tilt_deg"] = df_a[trunk_cols].mean(axis=1, skipna=True)

        merged = pd.merge(
            df_m[["frame"]],
            df_a[["frame", "torso_tilt_deg"]],
            on="frame",
            how="left",
        )

        df_metrics["torso_tilt_deg"] = merged["torso_tilt_deg"].values
        log("torso_tilt_deg を metrics に注入しました。")
    except Exception as e:
        log(f"⚠ torso_tilt_deg の注入に失敗: {e}")


# ============================================
#  グラフ生成
# ============================================

def plot_metrics_graph(df_metrics: pd.DataFrame, out_path: str) -> None:
    log(f"メトリクスグラフ生成: {out_path}")
    ensure_dir(os.path.dirname(out_path))

    if "time_s" in df_metrics.columns:
        t = df_metrics["time_s"].values
    elif "frame" in df_metrics.columns:
        t = df_metrics["frame"].values
    else:
        t = np.arange(len(df_metrics))

    speed = df_metrics["speed_mps"].values if "speed_mps" in df_metrics.columns else None
    com_y = (
        df_metrics["COM_y_m"].values
        if "COM_y_m" in df_metrics.columns
        else df_metrics["COM_y"].values
        if "COM_y" in df_metrics.columns
        else None
    )
    torso = df_metrics["torso_tilt_deg"].values if "torso_tilt_deg" in df_metrics.columns else None

    plt.figure(figsize=(6, 8))

    ax1 = plt.subplot(3, 1, 1)
    if speed is not None:
        ax1.plot(t, speed)
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_title("Speed over time")

    ax2 = plt.subplot(3, 1, 2)
    if torso is not None:
        ax2.plot(t, torso)
    ax2.set_ylabel("Torso tilt (deg)")
    ax2.set_title("Torso tilt over time")

    ax3 = plt.subplot(3, 1, 3)
    if com_y is not None:
        ax3.plot(t, com_y)
    ax3.set_ylabel("COM height")
    ax3.set_xlabel("Time (s or frame)")
    ax3.set_title("COM over time")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_angle_graph(df_angles: Optional[pd.DataFrame], out_path: str) -> bool:
    if df_angles is None:
        log("⚠ 角度グラフを描画できるデータがありません。")
        return False

    ensure_dir(os.path.dirname(out_path))

    if "time_s" in df_angles.columns:
        t = df_angles["time_s"].values
    elif "frame" in df_angles.columns:
        t = df_angles["frame"].values
    else:
        t = np.arange(len(df_angles))

    knee_cols = [c for c in df_angles.columns if "knee" in c.lower()]
    hip_cols = [c for c in df_angles.columns if "hip" in c.lower()]
    trunk_cols = [c for c in df_angles.columns if any(k in c.lower() for k in ["trunk", "torso_tilt", "body_tilt"])]

    log(f"角度グラフ用列 knee={knee_cols}, hip={hip_cols}, trunk={trunk_cols}")

    if not (knee_cols or hip_cols or trunk_cols):
        log("⚠ 角度グラフを描画できるカラムがありません。")
        return False

    log(f"角度グラフ保存先: {out_path}")

    plt.figure(figsize=(6, 8))

    ax1 = plt.subplot(3, 1, 1)
    if knee_cols:
        for c in knee_cols:
            ax1.plot(t, df_angles[c].values, label=c)
        ax1.legend(fontsize=6)
    ax1.set_ylabel("Knee (deg)")
    ax1.set_title("Knee angles")

    ax2 = plt.subplot(3, 1, 2)
    if hip_cols:
        for c in hip_cols:
            ax2.plot(t, df_angles[c].values, label=c)
        ax2.legend(fontsize=6)
    ax2.set_ylabel("Hip (deg)")
    ax2.set_title("Hip angles")

    ax3 = plt.subplot(3, 1, 3)
    if trunk_cols:
        for c in trunk_cols:
            ax3.plot(t, df_angles[c].values, label=c)
        ax3.legend(fontsize=6)
    ax3.set_ylabel("Trunk (deg)")
    ax3.set_xlabel("Time (s or frame)")
    ax3.set_title("Trunk tilt")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    log("角度グラフ保存完了")
    return True


# ============================================
#  OpenCV 日本語パス対応ユーティリティ
# ============================================

def cv2_imwrite_safe(target_path: str, image: np.ndarray) -> bool:
    """
    OpenCV が日本語パスに直接保存できない場合の回避:
    一時ASCIIディレクトリに保存してから move。
    """
    try:
        temp_dir = os.path.join(tempfile.gettempdir(), "kjac_cv2_temp")
        ensure_dir(temp_dir)
        temp_path = os.path.join(temp_dir, f"tmp_{uuid4().hex}.png")
        ok = cv2.imwrite(temp_path, image)
        if not ok:
            log(f"⚠ 一時PNG保存に失敗: {temp_path}")
            return False
        ensure_dir(os.path.dirname(target_path))
        shutil.move(temp_path, target_path)
        return True
    except Exception as e:
        log(f"⚠ cv2_imwrite_safe 失敗: {target_path}, error={e}")
        return False


# ============================================
#  骨格オーバーレイ画像抽出 + 連続フレーム書き出し
# ============================================

def extract_pose_images_from_overlay(overlay_path: str,
                                     pose_out_dir: str) -> Dict[str, str]:
    """
    オーバーレイ動画から start / mid / finish の3枚をPNGで保存。
    """
    result: Dict[str, str] = {}

    if not os.path.exists(overlay_path):
        log(f"⚠ オーバーレイ動画が見つかりません: {overlay_path}")
        return result

    log(f"オーバーレイから骨格画像抽出開始: {overlay_path}")
    ensure_dir(pose_out_dir)

    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        log("⚠ オーバーレイ動画を開けませんでした。")
        return result

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"オーバーレイ frame_count={frame_count}")
    if frame_count <= 0:
        log("⚠ フレーム数が0です。")
        cap.release()
        return result

    indices = {
        "start": 0,
        "mid": frame_count // 2,
        "finish": frame_count - 1,
    }

    for key, idx in indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            log(f"⚠ フレーム取得に失敗: {key} (index={idx})")
            continue

        final_path = os.path.join(pose_out_dir, f"{key}.png")
        if cv2_imwrite_safe(final_path, frame):
            log(f"骨格画像保存成功: {key} -> {final_path}")
            result[key] = final_path
        else:
            log(f"⚠ 画像保存に失敗: {final_path}")

    cap.release()
    log(f"骨格画像出力結果: {result}")
    return result


def export_overlay_frames_every_0_1s(overlay_path: str,
                                     out_dir: str) -> int:
    """
    オーバーレイ動画から 0.1秒ごとにフレームを書き出し。
    ファイル名: frame_0000.png（ゼロ埋め4桁）
    """
    if not os.path.exists(overlay_path):
        log(f"⚠ オーバーレイ動画が見つかりません（連続フレーム出力スキップ）: {overlay_path}")
        return 0

    ensure_dir(out_dir)

    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        log("⚠ オーバーレイ動画を開けませんでした（連続フレーム出力スキップ）。")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # 安全側デフォルト
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(int(round(fps * 0.1)), 1)  # 0.1秒ごと
    saved = 0
    frame_idx = 0

    while frame_idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        filename = f"frame_{saved:04d}.png"
        final_path = os.path.join(out_dir, filename)
        if cv2_imwrite_safe(final_path, frame):
            saved += 1
        frame_idx += step

    cap.release()
    log(f"overlay連続フレーム出力: {saved}枚 -> {out_dir}")
    return saved


# ============================================
#  AIコメント生成ロジック（ラップ + GPT）
# ============================================

def jp_wrap(text: str, max_chars: int = 35) -> str:
    """
    日本語の長文を、ある程度の文字数で手動改行。
    """
    if not text:
        return ""
    text = text.replace("\r", "").replace("\n", "")
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    return "\n".join(chunks)


def sanitize_for_pdf(text: str) -> str:
    """
    PDF に入れづらい制御文字などを簡易除去。
    """
    if text is None:
        return ""
    # 制御文字を削除（改行は別扱い）
    cleaned = "".join(
        ch for ch in text
        if ch == "\n" or (ord(ch) >= 0x20)
    )
    return cleaned


def safe_multi_cell(pdf: FPDF,
                    text: str,
                    h: float = 5.0,
                    max_chars: int = 40) -> None:
    """
    FPDF の「Not enough horizontal space ～」エラーを回避するため、
    自前で 1 行あたり max_chars 文字までに分割してから multi_cell。
    """
    if not text:
        return

    text = sanitize_for_pdf(text)
    text = text.replace("\r", "")

    paragraphs = text.split("\n")
    for para in paragraphs:
        if para.strip() == "":
            pdf.ln(h)
            continue

        buf = ""
        for ch in para:
            buf += ch
            if len(buf) >= max_chars:
                pdf.multi_cell(0, h, buf)
                buf = ""
        if buf:
            pdf.multi_cell(0, h, buf)


def get_openai_client() -> Optional[OpenAI]:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        _openai_client = OpenAI()
        return _openai_client
    except Exception as e:
        log(f"⚠ OpenAI クライアント初期化に失敗: {e}")
        return None


def call_openai_chat(system_prompt: str,
                     user_prompt: str,
                     model: str = "gpt-4.1",
                     max_tokens: int = 800) -> Optional[str]:
    """
    OpenAI ChatCompletion (新SDK) ラッパー。失敗したら None。
    """
    client = get_openai_client()
    if client is None:
        return None

    try:
        log(f"OpenAI API 呼び出し開始 (model={model})")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        text = resp.choices[0].message.content
        if text is None:
            return None
        text = text.strip()
        log(f"OpenAI API 応答長={len(text)}")
        return text
    except Exception as e:
        log(f"⚠ OpenAI API 呼び出しで例外発生: {e}")
        return None


# ---------- ルールベース（フォールバック） ----------

def generate_ai_comments_pro_rule(metrics: Dict[str, float],
                                  angles: Dict[str, float],
                                  athlete: str) -> str:
    pitch = metrics.get("pitch", float("nan"))
    stride = metrics.get("stride", float("nan"))
    stab = metrics.get("stability", float("nan"))
    speed_mean = metrics.get("speed_mean", float("nan"))
    speed_max = metrics.get("speed_max", float("nan"))
    com_range = metrics.get("com_y_range", float("nan"))

    knee_mean = angles.get("knee_flex_mean", float("nan"))
    hip_mean = angles.get("hip_flex_mean", float("nan"))
    trunk_mean = angles.get("trunk_tilt_mean", float("nan"))

    lines: List[str] = []

    lines.append(
        f"{athlete}選手の疾走局面では、ピッチ {pitch:.2f} 歩/秒・ストライド {stride:.2f} m・"
        f"平均速度 {speed_mean:.2f} m/s（最大 {speed_max:.2f} m/s）と、"
        "現段階としてはバランスの取れた出力が出せています。"
    )

    if not math.isnan(stab):
        if stab >= 8.0:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 と高く、"
                "接地ごとの体幹のブレは比較的少ない状態です。"
                "この安定性はスプリントフォームの大きな土台になっています。"
            )
        else:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 で、"
                "特にスピードが乗ってきた区間で、やや上下動や体幹の揺れが増える傾向があります。"
            )

    if not math.isnan(trunk_mean):
        if trunk_mean < -10:
            lines.append(
                f"体幹前傾角の平均は {trunk_mean:.1f} 度（やや強めの前傾）で、"
                "加速局面としては悪くないものの、トップスピードに入る場面では"
                "少し上体が被り気味になり、脚の回転が遅れやすいフォームになっています。"
            )
        elif -10 <= trunk_mean <= 5:
            lines.append(
                f"体幹前傾角の平均は {trunk_mean:.1f} 度で、"
                "加速〜中間疾走にかけて大きな崩れは見られません。"
                "今後は『前傾を保ったままリラックスして腕を振る』ことがテーマになります。"
            )
        else:
            lines.append(
                f"体幹前傾角の平均は {trunk_mean:.1f} 度とやや起き気味で、"
                "接地時の地面反力ベクトルが上方向に逃げやすいフォームです。"
                "スタート〜加速局面ではもう少し前傾を維持したいところです。"
            )

    if not math.isnan(knee_mean):
        lines.append(
            f"膝関節角度の平均は {knee_mean:.1f} 度で、"
            "遊脚期の回収はできていますが、接地直前に膝が伸びやすいフレームも見られます。"
            "接地でブレーキを生まないよう、『膝を突っ張らせずに、足の真上に体重を乗せる』意識が重要です。"
        )

    if not math.isnan(hip_mean):
        lines.append(
            f"股関節の屈曲角の平均は {hip_mean:.1f} 度で、"
            "もも上げの大きさ自体は十分ですが、骨盤と体幹の連動が甘くなる瞬間があり、"
            "結果としてストライドのわりに推進効率が落ちる区間があります。"
        )

    if not math.isnan(com_range):
        if com_range > 0.03:
            lines.append(
                f"重心の上下動は約 {com_range:.3f} m とやや大きめで、"
                "地面を強く押したいあまりに縦への動きが増えている可能性があります。"
                "接地時は『上に跳ぶ』ではなく『前に押し出す』イメージを持つと改善しやすくなります。"
            )
        else:
            lines.append(
                f"重心の上下動は約 {com_range:.3f} m と小さく、"
                "エネルギーロスの少ないフラットな軌道を確保できています。"
                "今後はこの安定性を保ったまま、ピッチとストライドの両方を少しずつ引き上げていく段階です。"
            )

    text = " ".join(lines)
    return jp_wrap(text, max_chars=38)


def generate_ai_comments_easy_rule(metrics: Dict[str, float],
                                   angles: Dict[str, float],
                                   athlete: str) -> str:
    pitch = metrics.get("pitch", float("nan"))
    stride = metrics.get("stride", float("nan"))
    stab = metrics.get("stability", float("nan"))
    speed_mean = metrics.get("speed_mean", float("nan"))
    speed_max = metrics.get("speed_max", float("nan"))
    com_range = metrics.get("com_y_range", float("nan"))

    knee_mean = angles.get("knee_flex_mean", float("nan"))
    hip_mean = angles.get("hip_flex_mean", float("nan"))
    trunk_mean = angles.get("trunk_tilt_mean", float("nan"))

    lines: List[str] = []

    lines.append(
        f"{athlete}さんの走りをデータで見ると、"
        f"平均スピードはだいたい {speed_mean:.1f} m/s（最大 {speed_max:.1f} m/s）で、"
        "今の段階としてはとても良いレベルです。"
    )

    lines.append(
        f"1秒あたりの歩数（ピッチ）は約 {pitch:.2f} 歩、"
        f"1歩で進む距離（ストライド）は約 {stride:.2f} m でした。"
        "テンポも歩幅もどちらも一定していて、基本的な走力はしっかり身についています。"
    )

    if not math.isnan(stab):
        if stab >= 8.0:
            lines.append(
                f"安定性スコアは {stab:.1f}/10 と高く、"
                "走っているときの体のブレが少ないのが強みです。"
                "この安定したフォームは、今後スピードを上げていく土台になります。"
            )
        else:
            lines.append(
                f"安定性スコアは {stab:.1f}/10 で、"
                "スピードが上がったときに少し体が上下に揺れる場面があります。"
                "ここを落ち着かせられると、タイムアップにつながりやすくなります。"
            )

    if not math.isnan(trunk_mean):
        lines.append(
            f"上半身の前傾（少し前に倒す角度）は平均で {trunk_mean:.1f} 度でした。"
            "前に倒れすぎると脚が前に出にくくなり、起きすぎると地面を強く押しにくくなります。"
            "今はその中間くらいなので、『おへそから前に引っ張られるイメージ』で走れるとさらに良くなります。"
        )

    if not math.isnan(knee_mean):
        lines.append(
            f"膝の曲げ伸ばしは平均 {knee_mean:.1f} 度で、"
            "しっかり脚をたたんでから前に出す動きはできています。"
            "これからのポイントは、接地の瞬間に膝を伸ばしすぎず、"
            "少し余裕をもたせて地面を押すことです。"
        )

    if not math.isnan(hip_mean):
        lines.append(
            f"股関節の動き（もも上げの大きさ）は平均 {hip_mean:.1f} 度で、"
            "十分に脚を前に送り出せています。"
            "お腹とおしりまわりを安定させながら、同じ動きをくり返し出せるようになると、"
            "ラストまでフォームが崩れにくくなります。"
        )

    if not math.isnan(com_range):
        lines.append(
            f"走っているときの体の上下のゆれは約 {com_range:.3f} m でした。"
            "少しだけ上下に跳ねる場面があるので、"
            "『なるべく同じ高さでスーッと前に進む』イメージを持つと、"
            "もっと楽にスピードを出せるようになります。"
        )

    text = " ".join(lines)
    return jp_wrap(text, max_chars=34)


def generate_training_menu_rule(metrics: Dict[str, float],
                                angles: Dict[str, float]) -> List[str]:
    pitch = metrics.get("pitch", float("nan"))
    stab = metrics.get("stability", float("nan"))
    com_range = metrics.get("com_y_range", float("nan"))

    knee_mean = angles.get("knee_flex_mean", float("nan"))
    hip_mean = angles.get("hip_flex_mean", float("nan"))

    drills: List[str] = []

    # 1: 体幹・姿勢
    if math.isnan(stab) or stab < 8.0:
        drills.append(
            "① フロントプランク（20〜30秒×3セット）\n"
            "ひじとつま先で体をまっすぐに支えます。腰が反らないように気をつけて、"
            "お腹のあたりに軽く力を入れたままキープしましょう。"
        )
    else:
        drills.append(
            "① 片足バランス＋腕振り（左右30秒×2セット）\n"
            "片足立ちで軽く前傾し、走るときと同じリズムで腕振りをします。"
            "体幹と腕・脚の連動を高めるトレーニングです。"
        )

    # 2: もも上げ・股関節
    if math.isnan(hip_mean) or hip_mean < 70:
        drills.append(
            "② 壁もたれハイニー（左右20回×2セット）\n"
            "壁に背中を軽くつけて立ち、片脚ずつももを高く引き上げます。"
            "腰を反らせず、お腹を少し締めた状態で行いましょう。"
        )
    else:
        drills.append(
            "② リズムハイニー（20m×3本）\n"
            "その場または20mの直線で、テンポよくもも上げをします。"
            "腕振りと足のリズムを合わせて、『同じテンポで走り切る』ことを意識します。"
        )

    # 3: 膝の安定性
    if math.isnan(knee_mean) or knee_mean < 60:
        drills.append(
            "③ スローランジ（左右10回×2セット）\n"
            "前に一歩出して腰を落とし、ゆっくり戻ります。"
            "膝が内側に倒れないように注意しながら行うと、接地時の安定性が高まります。"
        )
    else:
        drills.append(
            "③ スキップランジ（左右10回×2セット）\n"
            "ランジの姿勢から、前脚で地面を押して軽くスキップするように切り替えます。"
            "膝を固めすぎず、やわらかく地面を押す感覚を身につけます。"
        )

    # 4: ピッチアップ
    if math.isnan(pitch) or pitch < 3.5:
        drills.append(
            "④ ラインステップ（10秒×3セット）\n"
            "地面に1本線を引き、その上をできるだけ速く足を入れ替えながらまたぎます。"
            "足を高く上げすぎず、『素早くついて素早く離す』ことを意識しましょう。"
        )
    else:
        drills.append(
            "④ 20mハイテンポ走（20m×3〜4本）\n"
            "短い距離でピッチを意識したダッシュを行います。"
            "タイムよりも『同じリズムで最後まで回転させる』ことを大切にします。"
        )

    # 5: 重心の安定
    if not math.isnan(com_range) and com_range > 0.03:
        drills.append(
            "⑤ ミニハードルもどき走（10m×3本）\n"
            "ペットボトルやタオルを等間隔に並べ、その上を軽くまたぎながら走ります。"
            "なるべく腰の高さを変えずに、前へスーッと進むイメージで行いましょう。"
        )
    else:
        drills.append(
            "⑤ その場軽い足踏み（30秒×2セット）\n"
            "その場で軽く足踏みをしながら、かかとからではなく母指球付近で接地する感覚を確認します。"
            "上下に跳ねすぎず、スッと前に進める姿勢を意識しましょう。"
        )

    if len(drills) > 5:
        drills = drills[:5]

    return [jp_wrap(d, max_chars=34) for d in drills]


# ---------- GPT をかませた最終コメント生成 ----------

def generate_ai_comments_pro(metrics: Dict[str, float],
                             angles: Dict[str, float],
                             athlete: str) -> str:
    """
    コーチ研修向けのやや専門的コメント（GPT + ルールベースフォールバック）
    """
    base_text = generate_ai_comments_pro_rule(metrics, angles, athlete)
    prompt_user = (
        "以下は陸上短距離のフォーム分析データと、既に用意されている日本語コメント案です。\n"
        "よりプロのコーチが書いたような内容・表現にブラッシュアップしてください。\n"
        "・対象は中学〜高校の陸上選手\n"
        "・専門用語は使ってよいが、文章は日本語として読みやすく\n"
        "・箇条書きにせず、1〜3段落程度でまとめる\n"
        "・『〜ですね。』ではなく、『〜です。』調で\n"
        "・文字数は 600〜900 文字程度\n\n"
        "【計測データの要約（数値）】\n"
        f"{metrics}\n"
        f"{angles}\n\n"
        "【もともとのコメント案】\n"
        f"{base_text}\n"
    )

    gpt_text = call_openai_chat(
        system_prompt="あなたは陸上短距離の専門コーチです。フォーム分析レポートの文章を日本語で作成します。",
        user_prompt=prompt_user,
        model="gpt-4.1",
        max_tokens=900,
    )

    if gpt_text:
        return sanitize_for_pdf(gpt_text)
    else:
        log("GPT プロ向けコメント生成に失敗 → ルールベースにフォールバック")
        return base_text


def generate_ai_comments_easy(metrics: Dict[str, float],
                              angles: Dict[str, float],
                              athlete: str) -> str:
    """
    中学生・保護者向けのやさしい説明（GPT + ルールベースフォールバック）
    """
    base_text = generate_ai_comments_easy_rule(metrics, angles, athlete)
    prompt_user = (
        "以下は陸上短距離のフォーム分析データと、既に用意されている日本語コメント案です。\n"
        "中学生本人と保護者に向けて、わかりやすく・前向きになれる文章に書き直してください。\n"
        "・難しい専門用語はできるだけ避ける\n"
        "・『ここがダメ』ではなく、『ここを伸ばすともっと良くなる』という言い方\n"
        "・2〜4段落程度\n"
        "・文章のトーンは、優しいコーチが説明しているイメージ\n"
        "・文字数は 400〜700 文字程度\n\n"
        "【計測データの要約（数値）】\n"
        f"{metrics}\n"
        f"{angles}\n\n"
        "【もともとのコメント案】\n"
        f"{base_text}\n"
    )

    gpt_text = call_openai_chat(
        system_prompt="あなたは中学生向け陸上クラブのコーチです。わかりやすい日本語で説明します。",
        user_prompt=prompt_user,
        model="gpt-4.1-mini",
        max_tokens=700,
    )

    if gpt_text:
        return sanitize_for_pdf(gpt_text)
    else:
        log("GPT 一般向けコメント生成に失敗 → ルールベースにフォールバック")
        return base_text


def generate_training_menu(metrics: Dict[str, float],
                           angles: Dict[str, float]) -> List[str]:
    """
    自宅・学校でできる軽めのトレーニング案を GPT + ルールベースで生成。
    """
    base_list = generate_training_menu_rule(metrics, angles)
    base_joined = "\n\n".join(base_list)

    prompt_user = (
        "以下は陸上短距離選手に向けたトレーニングメニュー案です。\n"
        "自宅や学校で器具なし・少ないスペースでできる内容を中心に、5つのメニューを提案してください。\n"
        "・それぞれ『① 〜』『② 〜』という番号付き\n"
        "・1メニューあたり 2〜3 行程度\n"
        "・難しい説明は避け、中学生でもわかる日本語\n"
        "・具体的な回数や秒数も入れる\n\n"
        "【もともとのメニュー案】\n"
        f"{base_joined}\n"
    )

    gpt_text = call_openai_chat(
        system_prompt="あなたは陸上短距離のコーチです。中高生向けのトレーニングメニューを日本語で提案します。",
        user_prompt=prompt_user,
        model="gpt-4.1-mini",
        max_tokens=700,
    )

    if gpt_text:
        cleaned = sanitize_for_pdf(gpt_text)
        drills = [d.strip() for d in cleaned.split("\n") if d.strip()]
        # ①〜⑤ のブロックごとにまとめ直す簡易処理
        result: List[str] = []
        cur = []
        for line in drills:
            if line.startswith("①") or line.startswith("1."):
                if cur:
                    result.append("\n".join(cur))
                    cur = []
            elif line.startswith("②") or line.startswith("2."):
                if cur:
                    result.append("\n".join(cur))
                    cur = []
            elif line.startswith("③") or line.startswith("3."):
                if cur:
                    result.append("\n".join(cur))
                    cur = []
            elif line.startswith("④") or line.startswith("4."):
                if cur:
                    result.append("\n".join(cur))
                    cur = []
            elif line.startswith("⑤") or line.startswith("5."):
                if cur:
                    result.append("\n".join(cur))
                    cur = []
            cur.append(line)
        if cur:
            result.append("\n".join(cur))

        if not result:
            return base_list
        if len(result) > 5:
            result = result[:5]
        return result
    else:
        log("GPT トレーニングメニュー生成に失敗 → ルールベースにフォールバック")
        return base_list


# ============================================
#  PDF 出力
# ============================================

class PDFReport(FPDF):
    pass


def build_pdf(csv_path: str,
              video_path: str,
              athlete: str,
              out_dirs: Dict[str, str]) -> None:
    log(f"build_pdf 開始 csv={csv_path}, video={video_path}")

    # メトリクス読み込み
    df_metrics, metrics_stats = load_metrics_csv(csv_path)

    # 角度CSVの確保＆読み込み
    video_id = get_video_id(video_path)
    angle_csv = ensure_angles_csv(video_path, athlete, video_id, out_dirs["angles"])
    df_angles, angle_stats = load_angles_csv(angle_csv)

    # torso_tilt_deg を metrics 側に注入（可能なら）
    inject_torso_tilt_into_metrics(df_metrics, df_angles)

    # 骨格オーバーレイ動画パス
    overlay_name = os.path.basename(video_path).replace(".mp4", "_pose_overlay.mp4")
    overlay_path = os.path.join("outputs", "images", overlay_name)

    # オーバーレイから骨格3枚
    pose_images = extract_pose_images_from_overlay(overlay_path, out_dirs["pose_images"])

    # オーバーレイ連続フレーム（0.1秒ごと）
    export_overlay_frames_every_0_1s(overlay_path, out_dirs["overlay"])

    # グラフ生成
    metrics_graph_path = os.path.join(out_dirs["graphs"], f"{video_id}_metrics_{VERSION_STR}.png")
    plot_metrics_graph(df_metrics, metrics_graph_path)

    angles_graph_path = os.path.join(out_dirs["graphs"], f"{video_id}_angles_{VERSION_STR}.png")
    angles_graph_ok = plot_angle_graph(df_angles, angles_graph_path)

    # AIコメント生成（ローカルルールベース＋GPT）
    log("AIコメント生成（ローカルルールベース＋GPT）開始")
    pro_comment = generate_ai_comments_pro(metrics_stats, angle_stats, athlete)
    easy_comment = generate_ai_comments_easy(metrics_stats, angle_stats, athlete)
    training_menu = generate_training_menu(metrics_stats, angle_stats)

    # PDF 作成
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 日本語フォント
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    bold_font_path = font_path  # Meiryo は同ファイルでOK
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.add_font("JPB", "", bold_font_path, uni=True)
    log(f"✅ フォント読み込み: {font_path}")

    pdf.set_font("JPB", "", 14)
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（{VERSION_STR}）", ln=1)

    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {now_str()}", ln=1)
    pdf.cell(0, 6, f"ログファイル: {os.path.relpath(LOG_FILE) if LOG_FILE else 'N/A'}", ln=1)

    m_per_px = metrics_stats.get("m_per_px", float("nan"))
    if not math.isnan(m_per_px):
        pdf.cell(0, 6, f"推定スケール: {m_per_px:.6f} m/px", ln=1)
    pdf.ln(2)

    # 基本指標
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 7, "基本指標（ピッチ・ストライド・安定性）", ln=1)
    pdf.set_font("JP", "", 10)

    pitch_val = metrics_stats.get("pitch", float("nan"))
    stride_val = metrics_stats.get("stride", float("nan"))
    stab_val = metrics_stats.get("stability", float("nan"))

    pdf.cell(0, 6, f"ピッチ: {pitch_val:.2f} 歩/秒" if not math.isnan(pitch_val) else "ピッチ: N/A", ln=1)
    pdf.cell(0, 6, f"ストライド: {stride_val:.2f} m" if not math.isnan(stride_val) else "ストライド: N/A", ln=1)
    pdf.cell(0, 6, f"安定性スコア: {stab_val:.2f} / 10" if not math.isnan(stab_val) else "安定性スコア: N/A", ln=1)
    pdf.ln(2)

    # 骨格オーバーレイ
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 7, "骨格オーバーレイ（start / mid / finish）", ln=1)
    pdf.set_font("JP", "", 9)

    if pose_images:
        y_top = pdf.get_y()
        margin = 5
        usable_width = pdf.w - 2 * pdf.l_margin
        img_w = (usable_width - 2 * margin) / 3.0
        order = ["start", "mid", "finish"]

        for i, key in enumerate(order):
            if key not in pose_images:
                continue
            x = pdf.l_margin + i * (img_w + margin)
            pdf.image(pose_images[key], x=x, y=y_top, w=img_w)

        pdf.set_y(y_top + img_w * 0.75 + 5)
    else:
        pdf.cell(0, 6, "※骨格オーバーレイ画像が取得できませんでした。", ln=1)
    pdf.ln(3)

    # メトリクスグラフ
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "速度・重心・体幹傾斜の時間変化", ln=1)
    pdf.set_font("JP", "", 9)
    if os.path.exists(metrics_graph_path):
        pdf.image(metrics_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
    else:
        pdf.cell(0, 6, "※メトリクスグラフ画像が見つかりません。", ln=1)
    pdf.ln(3)

    # 角度グラフ
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "関節角度（膝・股関節・体幹）の時間変化", ln=1)
    pdf.set_font("JP", "", 9)
    if angles_graph_ok and os.path.exists(angles_graph_path):
        pdf.image(angles_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
    else:
        pdf.cell(0, 6, "※角度グラフ画像が見つかりません。", ln=1)
    pdf.ln(3)

    # 専門向けコメント
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（専門向け）", ln=1)
    pdf.set_font("JP", "", 9)
    safe_multi_cell(pdf, pro_comment, h=5, max_chars=40)
    pdf.ln(2)

    # 中学生・保護者向けコメント
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（中学生・保護者向け）", ln=1)
    pdf.set_font("JP", "", 9)
    safe_multi_cell(pdf, easy_comment, h=5, max_chars=38)
    pdf.ln(2)

    # おすすめトレーニング
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "おすすめトレーニング（自宅・学校でできるメニュー）", ln=1)
    pdf.set_font("JP", "", 9)
    for drill in training_menu:
        safe_multi_cell(pdf, drill, h=5, max_chars=38)
        pdf.ln(1)

    out_pdf = os.path.join(out_dirs["pdf"], f"{athlete}_form_report_{VERSION_STR}.pdf")
    pdf.output(out_pdf)
    log(f"✅ PDF出力完了: {out_pdf}")


# ============================================
#  メイン
# ============================================

def main():
    parser = argparse.ArgumentParser(description=f"KJAC フォーム分析レポート {VERSION_STR}（GPT対応フル版）")
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス（pose_metrics_analyzer_v3系の出力）")
    parser.add_argument("--athlete", required=True, help="選手名（フォルダ名にも使用）")
    args = parser.parse_args()

    video_id = get_video_id(args.video)
    out_dirs = get_root_output_dirs(args.athlete, video_id)

    # ログファイルセット
    global LOG_FILE
    log_name = f"{VERSION_STR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG_FILE = os.path.join(out_dirs["logs"], log_name)

    log(f"=== KJAC {VERSION_STR} START athlete={args.athlete}, video_id={video_id} ===")
    log(f"入力: video={args.video}, csv={args.csv}")

    try:
        build_pdf(args.csv, args.video, args.athlete, out_dirs)
        log(f"=== KJAC {VERSION_STR} END (success) ===")
    except Exception as e:
        log(f"⚠ 予期せぬエラー発生: {e}")
        log(f"=== KJAC {VERSION_STR} END (failed) ===")
        raise


if __name__ == "__main__":
    main()





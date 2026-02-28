import os
import argparse
import math
import subprocess
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# ============================================================
#  グローバル・ロガー
# ============================================================

LOG_FILE: Optional[str] = None


def set_log_file(path: str) -> None:
    global LOG_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    LOG_FILE = path


def log(msg: str) -> None:
    """コンソール＋ログファイル両方に出す"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # ログ書き込みでさらに落ちないように握りつぶす
            pass


# ============================================================
#  基本ユーティリティ
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_video_id(video_path: str) -> str:
    """
    動画ファイル名（拡張子なし）からフォルダ名用IDを生成
    例: "二村遥香10.5.mp4" -> "二村遥香10_5"
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    vid = stem.replace(".", "_")
    log(f"video_id 解析: base={base} -> video_id={vid}")
    return vid


def get_root_output_dirs(athlete: str, video_id: str) -> Dict[str, str]:
    """
    outputs/{athlete}/{video_id}/ にまとめて格納
        - pdf/
        - graphs/
        - pose_images/
        - angles/
        - logs/
    """
    base = os.path.join("outputs", athlete, video_id)
    pdf_dir = os.path.join(base, "pdf")
    graphs_dir = os.path.join(base, "graphs")
    pose_dir = os.path.join(base, "pose_images")
    angles_dir = os.path.join(base, "angles")
    logs_dir = os.path.join(base, "logs")

    for d in [base, pdf_dir, graphs_dir, pose_dir, angles_dir, logs_dir]:
        ensure_dir(d)

    log(f"出力ルート: {base}")
    return {
        "base": base,
        "pdf": pdf_dir,
        "graphs": graphs_dir,
        "pose_images": pose_dir,
        "angles": angles_dir,
        "logs": logs_dir,
    }


# ============================================================
#  CSV ロード & サマリ計算
# ============================================================

def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_metrics_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    log(f"metrics CSV 読み込み開始: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"metrics CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    # _summary_ 行を特別扱い
    first_col = df.columns[0]
    mask_summary = df[first_col].astype(str) == "_summary_"
    df_summary = df[mask_summary].iloc[-1] if mask_summary.any() else None
    if mask_summary.any():
        log("metrics: 先頭列 '_summary_' 行をサマリとして使用")
    df_data = df[~mask_summary].copy() if mask_summary.any() else df.copy()

    def get_stat(col_name: str, use_mean: bool = True) -> float:
        # 1) summary 行に値があればそれを採用
        if df_summary is not None and col_name in df.columns:
            val = _safe_float(df_summary[col_name])
            if not math.isnan(val):
                log(f"metrics {col_name} (summary) = {val}")
                return val
        # 2) 通常行の平均
        if use_mean and col_name in df_data.columns:
            series = df_data[col_name]
            if series.notna().any():
                val = float(series.mean(skipna=True))
                log(f"metrics {col_name} (mean) = {val}")
                return val
        log(f"metrics {col_name} は有効値なし (nan)")
        return float("nan")

    # ピッチ / ストライド / 安定性
    stats["pitch"] = get_stat("pitch_hz")
    stats["stride"] = get_stat("stride_m")
    stats["stability"] = get_stat("stability_score")

    # 速度（列名のゆれに対応: speed_mps or speed_m_s）
    if "speed_mps" in df_data.columns:
        stats["speed_mean"] = float(df_data["speed_mps"].mean(skipna=True))
        stats["speed_max"] = float(df_data["speed_mps"].max(skipna=True))
        log(f"metrics speed_mps mean={stats['speed_mean']}, max={stats['speed_max']}")
    elif "speed_m_s" in df_data.columns:
        stats["speed_mean"] = float(df_data["speed_m_s"].mean(skipna=True))
        stats["speed_max"] = float(df_data["speed_m_s"].max(skipna=True))
        log(f"metrics speed_m_s mean={stats['speed_mean']}, max={stats['speed_max']}")
    else:
        stats["speed_mean"] = float("nan")
        stats["speed_max"] = float("nan")
        log("metrics: 速度列 speed_mps / speed_m_s が見つかりません（nan 扱い）")

    # COM_y の揺れ幅
    if "COM_y_m" in df_data.columns:
        com = df_data["COM_y_m"]
        stats["com_y_range"] = float(com.max(skipna=True) - com.min(skipna=True))
    elif "COM_y" in df_data.columns:
        com = df_data["COM_y"]
        stats["com_y_range"] = float(com.max(skipna=True) - com.min(skipna=True))
    else:
        stats["com_y_range"] = float("nan")
    log(f"metrics COM_y_range={stats['com_y_range']:.5f}")

    # 体幹傾斜（torso_tilt_deg があれば）
    if "torso_tilt_deg" in df_data.columns:
        stats["torso_tilt_mean"] = float(df_data["torso_tilt_deg"].mean(skipna=True))
        stats["torso_tilt_var"] = float(df_data["torso_tilt_deg"].var(skipna=True))
        log(
            f"metrics torso_tilt_mean={stats['torso_tilt_mean']:.3f}, "
            f"var={stats['torso_tilt_var']:.5f}"
        )
    else:
        stats["torso_tilt_mean"] = float("nan")
        stats["torso_tilt_var"] = float("nan")
        log("metrics: torso_tilt_deg 列なし（体幹傾斜は nan 扱い）")

    # m_per_px （COM_y_m / COM_y_px から推定）
    m_per_px = float("nan")
    try:
        if "COM_y_m" in df_data.columns and "COM_y_px" in df_data.columns:
            ratio = df_data[["COM_y_m", "COM_y_px"]].copy()
            ratio = ratio[(ratio["COM_y_px"].abs() > 1e-6) & ratio["COM_y_m"].notna()]
            if not ratio.empty:
                m_per_px = float((ratio["COM_y_m"] / ratio["COM_y_px"]).median())
                log(f"metrics m_per_px 推定={m_per_px:.6f}")
            else:
                log("metrics: COM_y_m と COM_y_px から m_per_px を推定できませんでした")
        else:
            log("metrics: COM_y_m または COM_y_px 列なし（m_per_px 推定不能）")
    except Exception as e:
        log(f"metrics: m_per_px 推定中にエラー: {e}")
    stats["m_per_px"] = m_per_px

    return df_data, stats


def ensure_angles_csv(
    video_path: str,
    athlete: str,
    video_id: str,
    angles_dir: str,
) -> Optional[str]:
    """
    角度CSVの場所を賢く探索＆自動生成
    優先順位:
      1) outputs/{athlete}/{video_id}/angles/{video_id}.csv
      2) outputs/angles/{athlete}/{video_id}.csv
      3) outputs/angles_動画名.csv
      4) なければ pose_angle_analyzer_v1_0.py を実行
    """
    # 1) 新形式
    target_csv = os.path.join(angles_dir, f"{video_id}.csv")
    log(f"角度CSV探索開始 target={target_csv}")
    if os.path.exists(target_csv):
        log(f"角度CSV(新形式)発見: {target_csv}")
        return target_csv

    # 2) 旧形式: outputs/angles/{athlete}/{video_id}.csv
    old_dir_1 = os.path.join("outputs", "angles", athlete)
    old_csv_1 = os.path.join(old_dir_1, f"{video_id}.csv")
    if os.path.exists(old_csv_1):
        log(f"角度CSV(旧形式1)発見: {old_csv_1}")
        ensure_dir(angles_dir)
        new_path = target_csv
        try:
            import shutil
            shutil.copy2(old_csv_1, new_path)
            log(f"角度CSVを新形式へコピー: {old_csv_1} → {new_path}")
            return new_path
        except Exception as e:
            log(f"角度CSVコピー失敗: {e} → 旧パスをそのまま使用")
            return old_csv_1

    # 3) 旧形式: outputs/angles_動画名.csv
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    old_csv_2 = os.path.join("outputs", f"angles_{stem}.csv")
    if os.path.exists(old_csv_2):
        log(f"角度CSV(旧形式2)発見: {old_csv_2}")
        ensure_dir(angles_dir)
        new_path = target_csv
        try:
            import shutil
            shutil.copy2(old_csv_2, new_path)
            log(f"角度CSV(旧形式2)を新形式へコピー: {old_csv_2} → {new_path}")
            return new_path
        except Exception as e:
            log(f"角度CSVコピー失敗: {e} → 旧パスをそのまま使用")
            return old_csv_2

    # 4) どこにもない → 自動解析
    log(f"角度CSVが見つかりません。自動解析を試みます: {target_csv}")
    analyzer_script = os.path.join("scripts", "pose_angle_analyzer_v1_0.py")
    if not os.path.exists(analyzer_script):
        log(f"⚠ 角度解析スクリプトが見つかりません: {analyzer_script}")
        return None

    cmd = ["python", analyzer_script, "--video", video_path]
    log(f"角度解析スクリプト実行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        log(f"⚠ 角度解析スクリプトの実行に失敗しました: {e}")
        return None

    # 自動解析後、旧形式2に出力されている想定
    if os.path.exists(old_csv_2):
        try:
            import shutil
            ensure_dir(os.path.dirname(target_csv))
            shutil.copy2(old_csv_2, target_csv)
            log(f"✅ 自動解析した角度CSVを新形式へコピー: {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 自動解析CSVコピー失敗: {e}")
            return old_csv_2

    log("⚠ 自動解析後も角度CSVが見つかりませんでした。")
    return None


def load_angles_csv(angle_csv: Optional[str]) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if angle_csv is None or not os.path.exists(angle_csv):
        log("⚠ 角度CSVがロードできませんでした。角度解析はスキップします。")
        return None, {}

    log(f"角度CSV読込開始: {angle_csv}")
    df = pd.read_csv(angle_csv)
    log(f"angles CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    def mean_of_cols(cols: List[str]) -> float:
        vals = []
        for c in cols:
            if c in df.columns:
                vals.append(df[c].astype(float))
        if not vals:
            return float("nan")
        concat = pd.concat(vals, axis=0)
        return float(concat.mean(skipna=True))

    def range_of_cols(cols: List[str]) -> float:
        vals = []
        for c in cols:
            if c in df.columns:
                vals.append(df[c].astype(float))
        if not vals:
            return float("nan")
        concat = pd.concat(vals, axis=0)
        return float(concat.max(skipna=True) - concat.min(skipna=True))

    # v1.0 のカラム仕様に合わせて代表値を取得
    # 両膝
    knee_cols = [c for c in ["left_knee", "right_knee"] if c in df.columns]
    hip_cols = [c for c in ["left_hip", "right_hip"] if c in df.columns]
    trunk_cols = [c for c in ["torso_tilt_left", "torso_tilt_right"] if c in df.columns]

    stats["knee_flex_mean"] = mean_of_cols(knee_cols)
    stats["knee_flex_range"] = range_of_cols(knee_cols)
    stats["hip_flex_mean"] = mean_of_cols(hip_cols)
    stats["hip_flex_range"] = range_of_cols(hip_cols)
    stats["trunk_tilt_mean"] = mean_of_cols(trunk_cols)
    stats["trunk_tilt_range"] = range_of_cols(trunk_cols)

    log(
        "angles stats: "
        f"knee_mean={stats['knee_flex_mean']:.3f}, knee_range={stats['knee_flex_range']:.3f}, "
        f"hip_mean={stats['hip_flex_mean']:.3f}, hip_range={stats['hip_flex_range']:.3f}, "
        f"trunk_mean={stats['trunk_tilt_mean']:.3f}, trunk_range={stats['trunk_tilt_range']:.3f}"
    )

    return df, stats


# ============================================================
#  グラフ生成
# ============================================================

def plot_metrics_graph(df_metrics: pd.DataFrame, out_path: str) -> None:
    log(f"メトリクスグラフ生成: {out_path}")
    ensure_dir(os.path.dirname(out_path))

    # X軸
    if "time_s" in df_metrics.columns:
        t = df_metrics["time_s"].values
    elif "frame" in df_metrics.columns:
        t = df_metrics["frame"].values
    else:
        t = np.arange(len(df_metrics))

    # 列名ゆれに対応
    speed = None
    if "speed_mps" in df_metrics.columns:
        speed = df_metrics["speed_mps"].values
    elif "speed_m_s" in df_metrics.columns:
        speed = df_metrics["speed_m_s"].values

    com_y = None
    if "COM_y_m" in df_metrics.columns:
        com_y = df_metrics["COM_y_m"].values
    elif "COM_y" in df_metrics.columns:
        com_y = df_metrics["COM_y"].values

    torso = df_metrics["torso_tilt_deg"].values if "torso_tilt_deg" in df_metrics.columns else None

    plt.figure(figsize=(6, 8))

    # 1段目: 速度
    ax1 = plt.subplot(3, 1, 1)
    if speed is not None:
        ax1.plot(t, speed)
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_title("Speed over time")

    # 2段目: 体幹傾斜
    ax2 = plt.subplot(3, 1, 2)
    if torso is not None:
        ax2.plot(t, torso)
    ax2.set_ylabel("Torso tilt (deg)")
    ax2.set_title("Torso tilt over time")

    # 3段目: COM_y
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
    trunk_cols = [c for c in df_angles.columns if any(k in c.lower()
                                                      for k in ["trunk", "torso", "body_tilt"])]

    log(
        f"角度グラフ用列 "
        f"knee={knee_cols}, hip={hip_cols}, trunk={trunk_cols}"
    )

    if not (knee_cols or hip_cols or trunk_cols):
        log("⚠ 角度グラフを描画できるカラムがありません。")
        return False

    log(f"角度グラフ保存先: {out_path}")

    plt.figure(figsize=(6, 8))

    # 膝
    ax1 = plt.subplot(3, 1, 1)
    if knee_cols:
        for c in knee_cols:
            ax1.plot(t, df_angles[c].values, label=c)
        ax1.legend(fontsize=6)
    ax1.set_ylabel("Knee (deg)")
    ax1.set_title("Knee angles")

    # 股関節
    ax2 = plt.subplot(3, 1, 2)
    if hip_cols:
        for c in hip_cols:
            ax2.plot(t, df_angles[c].values, label=c)
        ax2.legend(fontsize=6)
    ax2.set_ylabel("Hip (deg)")
    ax2.set_title("Hip angles")

    # 体幹
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


# ============================================================
#  骨格オーバーレイ画像抽出
# ============================================================

def extract_pose_images_from_overlay(overlay_path: str, out_dir: str) -> Dict[str, str]:
    """
    オーバーレイ動画から start / mid / finish の3枚をPNGで保存。
    失敗した場合は空dictを返す。
    """
    result: Dict[str, str] = {}
    if not os.path.exists(overlay_path):
        log(f"⚠ オーバーレイ動画が見つかりません: {overlay_path}")
        return result

    log(f"オーバーレイから骨格画像抽出開始: {overlay_path}")
    ensure_dir(out_dir)

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
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                log(f"⚠ フレーム取得に失敗: {key} (index={idx})")
                continue
            out_path = os.path.join(out_dir, f"{key}.png")
            abs_path = os.path.abspath(out_path)
            ok = cv2.imwrite(abs_path, frame)
            if ok:
                result[key] = abs_path
                log(f"骨格画像保存成功: {key} -> {abs_path}")
            else:
                log(f"⚠ 画像保存に失敗: {abs_path}")
        except Exception as e:
            log(f"⚠ 画像保存エラー: key={key}, index={idx}, error={e}")

    cap.release()
    log(f"骨格画像出力結果: {result}")
    return result


# ============================================================
#  日本語テキスト整形 & 安全 multi_cell
# ============================================================

def jp_wrap(text: str, max_chars: int = 35) -> str:
    """
    日本語の長文を、おおよその文字数で折り返す。
    （行ごとに max_chars でカット）
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = text.split("\n")

    chunks: List[str] = []
    for part in parts:
        part = part.strip()
        if part == "":
            chunks.append("")
            continue
        for i in range(0, len(part), max_chars):
            chunks.append(part[i:i + max_chars])

    return "\n".join(chunks)


def safe_multicell(pdf: FPDF, text: str, line_height: float = 5.0, max_chars: int = 35) -> None:
    """
    FPDF の multi_cell で「Not enough horizontal space」を起こさないように、
    幅を明示＋文字数で自前ラップしてから1行ずつ描画。
    """
    if text is None:
        return

    wrapped = jp_wrap(text, max_chars=max_chars)
    w = pdf.w - pdf.l_margin - pdf.r_margin
    if w <= 0:
        # 万一マージン設定がおかしくても落ちないように
        w = pdf.w

    for raw_line in wrapped.split("\n"):
        line = raw_line.rstrip("\r")
        if line.strip() == "":
            pdf.ln(line_height)
        else:
            pdf.multi_cell(w, line_height, line)


# ============================================================
#  AIコメント生成ロジック
# ============================================================

def generate_ai_comments_pro(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str
) -> str:
    """
    コーチ研修向けの専門的コメント
    """
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
        f"平均速度 {speed_mean:.2f} m/s（最大 {speed_max:.2f} m/s）という指標から、"
        "中〜高強度のスプリントとしては妥当な出力が得られています。"
    )

    if not math.isnan(stab):
        if stab >= 8.0:
            lines.append(
                f"重心変動および体幹の揺れを統合した安定性スコアは {stab:.2f}/10 と高く、"
                "接地のリズムと体幹スタビリティの両面で再現性の高い走動作が獲得されています。"
            )
        elif stab >= 6.0:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 と中程度であり、全体としては許容範囲ながら、"
                "加速後半〜トップスピード局面でわずかな重心の上下動が増える傾向がみられます。"
            )
        else:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 とやや低く、接地毎の重心上下動が大きくなることで、"
                "水平方向への力の伝達効率が低下している可能性が示唆されます。"
            )

    if not math.isnan(trunk_mean):
        if trunk_mean < 5:
            lines.append(
                "体幹前傾角は平均でごく小さく、やや直立位に近い姿勢で疾走しているため、"
                "地面反力のベクトルが鉛直方向に偏りやすく、推進効率の面では改善余地があります。"
            )
        elif 5 <= trunk_mean <= 15:
            lines.append(
                "体幹前傾角は平均で適正範囲（おおよそ5〜15度）に収まっており、"
                "接地時に骨盤〜体幹の一体性が保たれた効率的な力発揮が行われています。"
            )
        else:
            lines.append(
                "体幹前傾角は平均でやや大きめであり、上体の被り込みにより遊脚の引き出しが遅れる局面がみられます。"
                "特にトップスピード局面では、過度な前傾がピッチ低下のリスクとなる点に注意が必要です。"
            )

    if not math.isnan(knee_mean):
        lines.append(
            f"膝関節屈曲角の平均値はおおよそ {knee_mean:.1f} 度で、遊脚期のリカバリー動作としては概ね適正ですが、"
            "接地直前に膝伸展が早期に出るとブレーキ要素が増加するため、"
            "股関節主導での引き上げと「脚を下ろすのではなく、身体が前に進む」感覚の獲得が重要です。"
        )

    if not math.isnan(hip_mean):
        lines.append(
            f"股関節屈曲角の平均値 {hip_mean:.1f} 度から、腿上げ自体は一定水準を満たしていますが、"
            "骨盤前傾と体幹の安定性との連動がやや不安定な局面があり、"
            "結果としてストライドに対する有効なリーチの一部が失われている可能性があります。"
        )

    if not math.isnan(com_range) and com_range > 0:
        if com_range < 0.03:
            lines.append(
                f"重心の上下動範囲は {com_range:.3f} m と小さく、"
                "エネルギーロスの少ないフラットな重心軌道が確保できている点は大きな強みです。"
            )
        else:
            lines.append(
                f"重心の上下動範囲は {com_range:.3f} m とやや大きく、"
                "特に接地局面での膝・股関節の協調制御を高めることで、"
                "鉛直方向のロスを抑え、より推進効率の高い疾走フォームへと改善できる余地があります。"
            )

    return " ".join(lines)


def generate_ai_comments_easy(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str
) -> str:
    """
    中学生・保護者向けのやさしい説明（8〜12行イメージ）
    """
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
        f"{athlete}さんの走りは、全体としてスピードとリズムのバランスがよく、"
        f"平均速度は約 {speed_mean:.1f} m/s（最大 {speed_max:.1f} m/s）と、"
        "現時点のレベルとしてとても良い指標です。"
    )

    lines.append(
        f"1秒あたりの歩数（ピッチ）は約 {pitch:.2f} 歩、"
        f"1歩あたりの進む距離（ストライド）は約 {stride:.2f} m で、"
        "テンポもストライドもどちらもある程度そろっています。"
    )

    if not math.isnan(stab):
        if stab >= 8.0:
            lines.append(
                f"走っているときの体のブレをまとめた安定性スコアは {stab:.1f}/10 と高く、"
                "上半身も下半身もしっかりコントロールできている走りです。"
            )
        else:
            lines.append(
                f"安定性スコアは {stab:.1f}/10 で、走りとしては十分ですが、"
                "スピードが上がったときに少しだけ体が上下にゆれる場面が見られます。"
            )

    if not math.isnan(trunk_mean):
        lines.append(
            f"上半身の前傾（少し前に倒す角度）は平均で {trunk_mean:.1f} 度ほどで、"
            "大きく崩れてはいませんが、スタートから加速の場面では"
            "「少し前に倒す → 徐々に起きてくる」という変化をもう少しはっきり作れると、"
            "もっと走りがスムーズになります。"
        )

    if not math.isnan(knee_mean):
        lines.append(
            f"膝の曲げ伸ばしは平均で {knee_mean:.1f} 度くらいで、"
            "しっかり脚をたたんでから前に出す動きはできています。"
            "あとは接地の瞬間に膝を突っ張りすぎないようにすることで、"
            "ブレーキを減らして、前への進み方をもっと良くできます。"
        )

    if not math.isnan(hip_mean):
        lines.append(
            f"股関節の動き（もも上げの大きさ）は平均 {hip_mean:.1f} 度で、"
            "十分に前へ脚を出せています。骨盤やお腹まわりを安定させながら、"
            "同じ動きをくり返し出せるようになると、長い距離でもフォームが崩れにくくなります。"
        )

    if not math.isnan(com_range):
        lines.append(
            f"走っているときの重心（体の真ん中）の上下のゆれは約 {com_range:.3f} m で、"
            "少しだけ上下にゆれる場面がありますが、"
            "今のうちから体幹やおしりまわりを強くしていくと、"
            "もっと地面をうまく押せるようになります。"
        )

    return " ".join(lines)


def generate_training_menu(
    metrics: Dict[str, float],
    angles: Dict[str, float],
) -> List[str]:
    """
    自宅・学校でできる軽め〜中程度のトレーニングを5つ返す
    """
    pitch = metrics.get("pitch", float("nan"))
    stab = metrics.get("stability", float("nan"))
    com_range = metrics.get("com_y_range", float("nan"))

    knee_mean = angles.get("knee_flex_mean", float("nan"))
    hip_mean = angles.get("hip_flex_mean", float("nan"))
    trunk_mean = angles.get("trunk_tilt_mean", float("nan"))

    drills: List[str] = []

    # 1. 体幹＆姿勢
    if math.isnan(stab) or stab < 8.0:
        drills.append(
            "① フロントプランク＋姿勢キープ歩行（30秒×3セット）："
            "ひじとつま先で体を一直線に保つプランクを行い、その後、"
            "胸を張ったままゆっくり10歩ずつ前後に歩いて、"
            "走るときの体幹の安定感を高めます。"
        )
    else:
        drills.append(
            "① 片足バランス＋腕振り（左右30秒×2セット）："
            "片足立ちで軽く前傾しながら、スプリントの腕振りを行います。"
            "接地時の体幹の安定と腕・脚の連動を養います。"
        )

    # 2. 股関節可動域＆もも上げ
    if math.isnan(hip_mean) or hip_mean < 70:
        drills.append(
            "② 壁もたれハイニー（20回×2〜3セット）："
            "壁に軽く背中をつけて立ち、片脚ずつももを高く引き上げます。"
            "腰が反らないようにお腹を軽く締め、股関節から脚を引き上げる感覚を身に付けます。"
        )
    else:
        drills.append(
            "② リズムハイニー（20m×3本）："
            "20mの直線で、腕振りと合わせたテンポの良いハイニーを行います。"
            "実際の疾走に近いリズムで股関節の可動とピッチ向上を狙います。"
        )

    # 3. 膝の安定性
    if math.isnan(knee_mean) or knee_mean < 60:
        drills.append(
            "③ スローランジ（左右10回×2セット）："
            "前後に大きく一歩を出し、ゆっくり腰を落としていきます。"
            "膝が内側に入らないように注意しながら行うことで、"
            "接地時の膝の安定性を高めます。"
        )
    else:
        drills.append(
            "③ スキップランジ（左右10回×2セット）："
            "ランジの姿勢から前脚で軽く地面を押してスキップするように切り替えます。"
            "地面を押す感覚と膝の柔らかい使い方を同時にトレーニングします。"
        )

    # 4. 接地の速さ（ピッチ）
    if math.isnan(pitch) or pitch < 3.5:
        drills.append(
            "④ ラダーもしくはラインステップ（10秒×3セット）："
            "地面に線を1本引き、その上をできるだけ速く『チョキチョキ走り』でまたぎます。"
            "足を大きく上げすぎず、接地を速くする感覚を養います。"
        )
    else:
        drills.append(
            "④ 20mハイテンポ走（20m×4本）："
            "距離を短めに設定し、ピッチを意識して素早く回転させる練習です。"
            "タイムよりも『同じリズムで走りきる』ことを重視します。"
        )

    # 5. 重心の安定（上下動の抑制）
    if not math.isnan(com_range) and com_range > 0.03:
        drills.append(
            "⑤ ミニハードルもどき走（10m×3本）："
            "地面にタオルやペットボトルなどを等間隔に並べ、"
            "それを軽くまたぎながら走ります。"
            "腰の高さをあまり上下させずに進む意識を持つことで、重心のブレを減らします。"
        )
    else:
        drills.append(
            "⑤ つま先〜母指球接地確認ドリル（20歩×2セット）："
            "その場で軽く足踏みしながら、かかとからではなく"
            "母指球付近で接地してすぐに離地する感覚を確認します。"
        )

    # 5つにそろえる
    if len(drills) > 5:
        drills = drills[:5]

    return drills


# ============================================================
#  PDF 出力
# ============================================================

class PDFReport(FPDF):
    pass  # 必要ならヘッダ・フッタを追加可能


def build_pdf(
    csv_path: str,
    video_path: str,
    athlete: str,
    video_id: str,
    out_dirs: Dict[str, str],
) -> None:
    log(f"build_pdf 開始 csv={csv_path}, video={video_path}")

    # メトリクス読み込み
    try:
        df_metrics, metrics_stats = load_metrics_csv(csv_path)
    except Exception as e:
        log(f"!!! metrics CSV 読み込み・解析でエラー: {e}")
        df_metrics = pd.DataFrame()
        metrics_stats = {}

    # 角度CSVの確保＆読み込み
    try:
        angle_csv = ensure_angles_csv(video_path, athlete, video_id, out_dirs["angles"])
        df_angles, angle_stats = load_angles_csv(angle_csv)
    except Exception as e:
        log(f"!!! 角度CSV処理でエラー: {e}")
        df_angles, angle_stats = None, {}

    # 骨格オーバーレイ動画パス
    overlay_name = os.path.basename(video_path).replace(".mp4", "_pose_overlay.mp4")
    overlay_path = os.path.join("outputs", "images", overlay_name)

    try:
        pose_images = extract_pose_images_from_overlay(overlay_path, out_dirs["pose_images"])
    except Exception as e:
        log(f"!!! オーバーレイ画像抽出でエラー: {e}")
        pose_images = {}

    # グラフ生成
    metrics_graph_path = os.path.join(
        out_dirs["graphs"],
        f"{video_id}_metrics_v5_4_1_lab.png",
    )
    try:
        if not df_metrics.empty:
            plot_metrics_graph(df_metrics, metrics_graph_path)
        else:
            log("⚠ df_metrics が空のためメトリクスグラフをスキップ")
    except Exception as e:
        log(f"!!! メトリクスグラフ描画でエラー: {e}")

    angles_graph_path = os.path.join(
        out_dirs["graphs"],
        f"{video_id}_angles_v5_4_1_lab.png",
    )
    try:
        angles_graph_ok = plot_angle_graph(df_angles, angles_graph_path)
    except Exception as e:
        log(f"!!! 角度グラフ描画でエラー: {e}")
        angles_graph_ok = False

    # AIコメント生成
    try:
        pro_comment = generate_ai_comments_pro(metrics_stats, angle_stats, athlete)
    except Exception as e:
        log(f"!!! 専門向けAIコメント生成でエラー: {e}")
        pro_comment = "※AIコメント生成中にエラーが発生しました。"

    try:
        easy_comment = generate_ai_comments_easy(metrics_stats, angle_stats, athlete)
    except Exception as e:
        log(f"!!! 一般向けAIコメント生成でエラー: {e}")
        easy_comment = "※AIコメント生成中にエラーが発生しました。"

    try:
        training_menu = generate_training_menu(metrics_stats, angle_stats)
    except Exception as e:
        log(f"!!! トレーニングメニュー生成でエラー: {e}")
        training_menu = [
            "※トレーニングメニュー生成中にエラーが発生しました。"
        ]

    # PDF 作成
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 日本語フォント
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    bold_font_path = font_path  # 簡易的に同じフォントファイルを B 用として登録
    try:
        pdf.add_font("JP", "", font_path, uni=True)
        pdf.add_font("JPB", "", bold_font_path, uni=True)
        pdf.set_font("JPB", "", 12)
        log(f"✅ フォント読み込み: {font_path}")
    except Exception as e:
        log(f"⚠ 日本語フォント読み込みに失敗、Helvetica にフォールバック: {e}")
        pdf.set_font("Helvetica", "B", 12)

    # タイトル部
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.4.1_lab）", ln=1)
    try:
        pdf.set_font("JP", "", 10)
    except Exception:
        pdf.set_font("Helvetica", "", 10)

    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 6, f"ログファイル: {os.path.relpath(LOG_FILE) if LOG_FILE else 'N/A'}", ln=1)

    m_per_px = metrics_stats.get("m_per_px", float("nan"))
    if not math.isnan(m_per_px):
        pdf.cell(0, 6, f"推定スケール: {m_per_px:.6f} m/px", ln=1)
    pdf.ln(2)

    # 基本指標
    try:
        pdf.set_font("JPB", "", 11)
    except Exception:
        pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "基本指標（ピッチ・ストライド・安定性）", ln=1)

    try:
        pdf.set_font("JP", "", 10)
    except Exception:
        pdf.set_font("Helvetica", "", 10)

    pitch_val = metrics_stats.get("pitch", float("nan"))
    stride_val = metrics_stats.get("stride", float("nan"))
    stab_val = metrics_stats.get("stability", float("nan"))

    pdf.cell(0, 6, f"ピッチ: {pitch_val:.2f} 歩/秒", ln=1)
    pdf.cell(0, 6, f"ストライド: {stride_val:.2f} m", ln=1)
    pdf.cell(0, 6, f"安定性スコア: {stab_val:.2f} / 10", ln=1)
    pdf.ln(2)

    # 骨格オーバーレイ
    try:
        pdf.set_font("JPB", "", 11)
    except Exception:
        pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "骨格オーバーレイ（start / mid / finish）", ln=1)

    try:
        pdf.set_font("JP", "", 9)
    except Exception:
        pdf.set_font("Helvetica", "", 9)

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
            try:
                pdf.image(pose_images[key], x=x, y=y_top, w=img_w)
            except Exception as e:
                log(f"⚠ PDF への骨格画像貼り付け失敗 key={key}, path={pose_images[key]}, error={e}")

        pdf.set_y(y_top + img_w * 0.75 + 5)
    else:
        pdf.cell(0, 6, "※骨格オーバーレイ画像が取得できませんでした。", ln=1)
    pdf.ln(3)

    # メトリクスグラフ
    try:
        pdf.set_font("JPB", "", 11)
    except Exception:
        pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "速度・重心・体幹傾斜の時間変化", ln=1)

    try:
        pdf.set_font("JP", "", 9)
    except Exception:
        pdf.set_font("Helvetica", "", 9)

    if os.path.exists(metrics_graph_path):
        try:
            pdf.image(metrics_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
        except Exception as e:
            log(f"⚠ PDF へのメトリクスグラフ貼り付け失敗: {e}")
            pdf.cell(0, 6, "※メトリクスグラフ画像の貼り付けに失敗しました。", ln=1)
    else:
        pdf.cell(0, 6, "※メトリクスグラフ画像が見つかりません。", ln=1)
    pdf.ln(3)

    # 角度グラフ
    try:
        pdf.set_font("JPB", "", 11)
    except Exception:
        pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "関節角度（膝・股関節・体幹）の時間変化", ln=1)

    try:
        pdf.set_font("JP", "", 9)
    except Exception:
        pdf.set_font("Helvetica", "", 9)

    if angles_graph_ok and os.path.exists(angles_graph_path):
        try:
            pdf.image(angles_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
        except Exception as e:
            log(f"⚠ PDF への角度グラフ貼り付け失敗: {e}")
            pdf.cell(0, 6, "※角度グラフ画像の貼り付けに失敗しました。", ln=1)
    else:
        pdf.cell(0, 6, "※角度グラフ画像が見つかりません。", ln=1)
    pdf.ln(3)

    # AIフォーム解析コメント（専門向け）
    try:
        pdf.set_font("JPB", "", 11)
    except Exception:
        pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（専門向け）", ln=1)

    try:
        pdf.set_font("JP", "", 9)
    except Exception:
        pdf.set_font("Helvetica", "", 9)

    safe_multicell(pdf, pro_comment, line_height=5, max_chars=38)
    pdf.ln(2)

    # AIフォーム解析コメント（中学生・保護者向け）
    try:
        pdf.set_font("JPB", "", 11)
    except Exception:
        pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（中学生・保護者向け）", ln=1)

    try:
        pdf.set_font("JP", "", 9)
    except Exception:
        pdf.set_font("Helvetica", "", 9)

    safe_multicell(pdf, easy_comment, line_height=5, max_chars=34)
    pdf.ln(2)

    # おすすめトレーニング
    try:
        pdf.set_font("JPB", "", 11)
    except Exception:
        pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "おすすめトレーニング（自宅・学校でできるメニュー）", ln=1)

    try:
        pdf.set_font("JP", "", 9)
    except Exception:
        pdf.set_font("Helvetica", "", 9)

    for drill in training_menu:
        safe_multicell(pdf, drill, line_height=5, max_chars=34)
        pdf.ln(1)

    # 保存
    out_pdf = os.path.join(out_dirs["pdf"], f"{athlete}_form_report_v5_4_1_lab.pdf")
    pdf.output(out_pdf)
    log(f"✅ PDF出力完了: {out_pdf}")


# ============================================================
#  メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KJAC フォーム分析レポート v5.4.1_lab（高機能版・実験用）")
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス（pose_metrics_analyzer_v3系の出力）")
    parser.add_argument("--athlete", required=True, help="選手名（フォルダ名にも使用）")
    args = parser.parse_args()

    video_id = get_video_id(args.video)
    out_dirs = get_root_output_dirs(args.athlete, video_id)

    # ログファイル設定
    log_path = os.path.join(out_dirs["logs"], f"{video_id}_v5_4_1_lab.log")
    set_log_file(log_path)
    log(f"=== KJAC v5.4.1_lab START athlete={args.athlete}, video_id={video_id} ===")
    log(f"入力: video={args.video}, csv={args.csv}")

    try:
        build_pdf(args.csv, args.video, args.athlete, video_id, out_dirs)
        log("=== KJAC v5.4.1_lab END (success) ===")
    except Exception as e:
        import traceback
        log(f"!!! FATAL ERROR in build_pdf: {e}")
        log(traceback.format_exc())

        # ここで“クラッシュゼロ”用の簡易エラーレポートを出力
        try:
            pdf = PDFReport()
            pdf.add_page()
            font_path = "c:/Windows/Fonts/meiryo.ttc"
            try:
                pdf.add_font("JP", "", font_path, uni=True)
                pdf.set_font("JP", "", 12)
            except Exception:
                pdf.set_font("Helvetica", "", 12)

            pdf.cell(0, 10, "フォーム分析レポート生成中にエラーが発生しました。", ln=1)
            pdf.ln(4)
            pdf.set_font("JP", "", 10)
            msg = (
                "詳細な原因はログファイルを確認してください。\n"
                f"ログ: {os.path.relpath(log_path)}\n\n"
                f"エラーメッセージ: {str(e)}"
            )
            safe_multicell(pdf, msg, line_height=5, max_chars=40)

            err_pdf = os.path.join(out_dirs["pdf"], f"{args.athlete}_form_report_v5_4_1_lab_error.pdf")
            pdf.output(err_pdf)
            log(f"⚠ エラー簡易PDFを出力しました: {err_pdf}")
        except Exception as e2:
            log(f"!!! 簡易PDFの出力にも失敗しました: {e2}")

        # ここでは再raiseせず、終了コード0相当で終わる（“クラッシュゼロ”方針）
        return


if __name__ == "__main__":
    main()

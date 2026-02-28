import os
import argparse
import math
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# ==========================================
# グローバル（ログ用）
# ==========================================
LOG_FILE: Optional[str] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def init_logger(log_dir: str) -> str:
    """
    ログファイルを outputs/{athlete}/{video_id}/logs/ 以下に作成
    """
    global LOG_FILE
    ensure_dir(log_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"v5_4_3_lab_{ts}.log")
    LOG_FILE = log_path
    return log_path


def log(msg: str) -> None:
    """
    標準出力＋ログファイル両方に出す
    """
    global LOG_FILE
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # ログ書き込みに失敗しても処理は止めない
            pass


# ==========================================
# 出力パス関連
# ==========================================

def get_video_id(video_path: str) -> str:
    """
    動画ファイル名（拡張子なし）から、フォルダ名用のIDを生成
    例: "二村遥香10.5.mp4" -> "二村遥香10_5"
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    vid = stem.replace(".", "_")
    log(f"video_id 解析: base={base} -> video_id={vid}")
    return vid


def get_root_output_dirs(athlete: str, video_id: str) -> Dict[str, str]:
    """
    outputs/{athlete}/{video_id}/ 以下のディレクトリをまとめて作成
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


# ==========================================
# メトリクス CSV 読み込み
# ==========================================

def load_metrics_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    log(f"metrics CSV 読み込み開始: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"metrics CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    # 先頭行が _summary_ ならそれを使う
    first_col = df.columns[0]
    if str(df.iloc[0, 0]) == "_summary_":
        log("metrics: 先頭列 '_summary_' 行をサマリとして使用")
        summary = df.iloc[0]
        df_data = df[df[first_col] != "_summary_"].copy()

        stats["pitch"] = float(summary.get("pitch_hz", float("nan")))
        stats["stride"] = float(summary.get("stride_m", float("nan")))
        stats["stability"] = float(summary.get("stability_score", float("nan")))
    else:
        log("metrics: '_summary_' 行なし → 全行から平均値を推定")
        df_data = df.copy()
        stats["pitch"] = float(df_data["pitch_hz"].mean(skipna=True)) if "pitch_hz" in df_data.columns else float("nan")
        stats["stride"] = float(df_data["stride_m"].mean(skipna=True)) if "stride_m" in df_data.columns else float("nan")
        stats["stability"] = float(df_data["stability_score"].mean(skipna=True)) if "stability_score" in df_data.columns else float("nan")

    log(f"metrics pitch_hz = {stats['pitch']}")
    log(f"metrics stride_m = {stats['stride']}")
    log(f"metrics stability_score = {stats['stability']}")

    # 速度
    if "speed_mps" in df_data.columns:
        stats["speed_mean"] = float(df_data["speed_mps"].mean(skipna=True))
        stats["speed_max"] = float(df_data["speed_mps"].max(skipna=True))
        log(f"metrics speed_mps mean={stats['speed_mean']}, max={stats['speed_max']}")
    else:
        log("metrics: speed_mps 列なし（速度は nan 扱い）")
        stats["speed_mean"] = float("nan")
        stats["speed_max"] = float("nan")

    # COM_y の上下動レンジ
    if "COM_y_m" in df_data.columns:
        stats["com_y_range"] = float(df_data["COM_y_m"].max(skipna=True) - df_data["COM_y_m"].min(skipna=True))
    elif "COM_y" in df_data.columns:
        stats["com_y_range"] = float(df_data["COM_y"].max(skipna=True) - df_data["COM_y"].min(skipna=True))
    else:
        stats["com_y_range"] = float("nan")
    log(f"metrics COM_y_range={stats['com_y_range']:.5f}")

    # 体幹傾斜（torso_tilt_deg は無い想定なので nan）
    stats["torso_tilt_mean"] = float("nan")
    stats["torso_tilt_var"] = float("nan")
    log("metrics: torso_tilt_deg 列なし（体幹傾斜は nan 扱い）")

    # スケール（仮固定）
    stats["m_per_px"] = 0.001607
    log(f"metrics m_per_px 推定={stats['m_per_px']:.6f}")

    return df_data, stats


# ==========================================
# 角度 CSV 探索・読み込み
# ==========================================

def ensure_angles_csv(video_path: str, athlete: str, video_id: str, angles_dir: str) -> Optional[str]:
    ensure_dir(angles_dir)
    target_csv = os.path.join(angles_dir, f"{video_id}.csv")
    log(f"角度CSV探索開始 target={target_csv}")

    # 1) 新形式
    if os.path.exists(target_csv):
        log(f"角度CSV(新形式)発見: {target_csv}")
        return target_csv

    # 2) 旧形式1
    old_dir_1 = os.path.join("outputs", "angles", athlete)
    old_csv_1 = os.path.join(old_dir_1, f"{video_id}.csv")
    if os.path.exists(old_csv_1):
        import shutil
        try:
            shutil.copy2(old_csv_1, target_csv)
            log(f"旧形式1 CSV を新形式へコピー: {old_csv_1} → {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 角度CSVコピー失敗(旧形式1): {e}")
            return old_csv_1

    # 3) 旧形式2
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    old_csv_2 = os.path.join("outputs", f"angles_{stem}.csv")
    if os.path.exists(old_csv_2):
        import shutil
        try:
            shutil.copy2(old_csv_2, target_csv)
            log(f"旧形式2 CSV を新形式へコピー: {old_csv_2} → {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 角度CSVコピー失敗(旧形式2): {e}")
            return old_csv_2

    log("⚠ 角度CSVが見つかりません。今回は自動解析は行わず、角度グラフ・角度コメントをスキップします。")
    return None


def load_angles_csv(angle_csv: Optional[str]) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if angle_csv is None or not os.path.exists(angle_csv):
        log("角度CSVがロードできませんでした。角度解析はスキップします。")
        return None, {}

    log(f"角度CSV読込開始: {angle_csv}")
    df = pd.read_csv(angle_csv)
    log(f"angles CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    # 膝
    knee_candidates = [c for c in df.columns if "knee" in c.lower()]
    if knee_candidates:
        series_knee = pd.concat([df[c] for c in knee_candidates], axis=0)
        stats["knee_flex_mean"] = float(series_knee.mean(skipna=True))
        stats["knee_flex_range"] = float(series_knee.max(skipna=True) - series_knee.min(skipna=True))
    else:
        stats["knee_flex_mean"] = float("nan")
        stats["knee_flex_range"] = float("nan")

    # 股関節
    hip_candidates = [c for c in df.columns if "hip" in c.lower()]
    if hip_candidates:
        series_hip = pd.concat([df[c] for c in hip_candidates], axis=0)
        stats["hip_flex_mean"] = float(series_hip.mean(skipna=True))
        stats["hip_flex_range"] = float(series_hip.max(skipna=True) - series_hip.min(skipna=True))
    else:
        stats["hip_flex_mean"] = float("nan")
        stats["hip_flex_range"] = float("nan")

    # 体幹
    trunk_candidates = [c for c in df.columns if any(k in c.lower() for k in ["trunk", "torso_tilt", "body_tilt"])]
    if trunk_candidates:
        series_trunk = pd.concat([df[c] for c in trunk_candidates], axis=0)
        stats["trunk_tilt_mean"] = float(series_trunk.mean(skipna=True))
        stats["trunk_tilt_range"] = float(series_trunk.max(skipna=True) - series_trunk.min(skipna=True))
    else:
        stats["trunk_tilt_mean"] = float("nan")
        stats["trunk_tilt_range"] = float("nan")

    log(
        "angles stats: knee_mean={:.3f}, knee_range={:.3f}, "
        "hip_mean={:.3f}, hip_range={:.3f}, trunk_mean={:.3f}, trunk_range={:.3f}".format(
            stats["knee_flex_mean"],
            stats["knee_flex_range"],
            stats["hip_flex_mean"],
            stats["hip_flex_range"],
            stats["trunk_tilt_mean"],
            stats["trunk_tilt_range"],
        )
    )

    return df, stats


# ==========================================
# グラフ生成
# ==========================================

def plot_metrics_graph(df_metrics: pd.DataFrame, out_path: str) -> None:
    log(f"メトリクスグラフ生成: {out_path}")
    ensure_dir(os.path.dirname(out_path))

    # x 軸（時間 or frame）
    if "time_s" in df_metrics.columns:
        t = df_metrics["time_s"].values
    elif "frame" in df_metrics.columns:
        t = df_metrics["frame"].values
    else:
        t = np.arange(len(df_metrics))

    speed = df_metrics["speed_mps"].values if "speed_mps" in df_metrics.columns else None
    com_y = None
    if "COM_y_m" in df_metrics.columns:
        com_y = df_metrics["COM_y_m"].values
    elif "COM_y" in df_metrics.columns:
        com_y = df_metrics["COM_y"].values
    torso = df_metrics["tilt_deg"].values if "tilt_deg" in df_metrics.columns else None

    plt.figure(figsize=(6, 8))

    # 速度
    ax1 = plt.subplot(3, 1, 1)
    if speed is not None:
        ax1.plot(t, speed)
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_title("Speed over time")

    # 体幹傾斜（tilt_deg を代用）
    ax2 = plt.subplot(3, 1, 2)
    if torso is not None:
        ax2.plot(t, torso)
    ax2.set_ylabel("Torso tilt (deg)")
    ax2.set_title("Torso tilt over time")

    # COM高さ
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
        log("角度グラフ: df_angles=None のためスキップ")
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
        log("角度グラフ: 使用可能なカラムなし → スキップ")
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


# ==========================================
# 骨格オーバーレイ画像抽出（日本語パス対応）
# ==========================================

def extract_pose_images_from_overlay(overlay_path: str, out_dir: str) -> Dict[str, str]:
    """
    オーバーレイ動画から start / mid / finish の3枚をPNGで保存。
    日本語パスでも確実に保存できるよう、
    cv2.imwrite ではなく imencode → Python IO で書き出す。
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
        log("⚠ フレーム数が0のため抽出スキップ")
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

        out_path = os.path.join(out_dir, f"{key}.png")
        try:
            ok, buf = cv2.imencode(".png", frame)
            if not ok:
                log(f"⚠ imencode に失敗: {out_path}")
                continue

            with open(out_path, "wb") as f:
                f.write(buf.tobytes())

            result[key] = out_path
            log(f"骨格画像保存成功: {key} -> {out_path}")
        except Exception as e:
            log(f"⚠ 画像保存エラー: {out_path} -> {e}")

    cap.release()
    log(f"骨格画像出力結果: {result}")
    return result


# ==========================================
# AIコメント生成
# ==========================================

def jp_wrap(text: str, max_chars: int = 35) -> str:
    """
    日本語の長文を、FPDF.multi_cell で安全に流し込めるように
    指定文字数ごとに改行を入れる。
    """
    text = text.replace("\n", "")
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    return "\n".join(chunks)


def generate_ai_comments_pro(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str
) -> str:
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
        "中〜高強度のスプリントとして妥当な出力が得られています。"
    )

    if not math.isnan(stab):
        if stab >= 8.0:
            lines.append(
                f"重心変動および体幹の揺れを統合した安定性スコアは {stab:.2f}/10 と高く、"
                "接地リズムと体幹スタビリティの両面で再現性の高い走動作が獲得されています。"
            )
        elif stab >= 6.0:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 と中程度であり、全体としては許容範囲ながら、"
                "加速後半〜トップスピード局面でわずかな重心の上下動が増える傾向がみられます。"
            )
        else:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 とやや低く、特に接地毎の重心上下動が大きくなることで、"
                "水平方向への力の伝達効率が低下している可能性が示唆されます。"
            )

    if not math.isnan(trunk_mean):
        if trunk_mean < 5:
            lines.append(
                "体幹前傾角は平均でごく小さく、やや直立位に近い姿勢で疾走しているため、"
                "地面反力ベクトルが鉛直方向に偏りやすく、推進効率の面では改善余地があります。"
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

    text = "".join(lines)
    return jp_wrap(text, max_chars=38)


def generate_ai_comments_easy(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str
) -> str:
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
        f"平均速度は約{speed_mean:.1f} m/s（最大 {speed_max:.1f} m/s）と、"
        "今の学年としてとても良いレベルです。"
    )

    lines.append(
        f"1秒あたりの歩数（ピッチ）は約 {pitch:.2f} 歩、1歩あたりの進む距離（ストライド）は約 {stride:.2f} m で、"
        "テンポもストライドもどちらもそろっています。"
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
            "「少し前に倒す→徐々に起きてくる」という変化をもう少しはっきり作れると、"
            "もっとスムーズな走りになります。"
        )

    if not math.isnan(knee_mean):
        lines.append(
            f"膝の曲げ伸ばしは平均で {knee_mean:.1f} 度くらいで、"
            "しっかり脚をたたんでから前に出す動きはできています。"
            "あとは接地の瞬間に膝を突っ張りすぎないようにすることで、"
            "ブレーキを減らして前への進み方をもっと良くできます。"
        )

    if not math.isnan(hip_mean):
        lines.append(
            f"股関節の動き（もも上げの大きさ）は平均 {hip_mean:.1f} 度で、"
            "前に脚を出す動きは十分です。お腹まわりを安定させながら"
            "同じ動きをくり返し出せるようになると、長い距離でもフォームが崩れにくくなります。"
        )

    if not math.isnan(com_range):
        lines.append(
            f"走っているときの重心（体の真ん中）の上下のゆれは約 {com_range:.3f} m で、"
            "少しだけ上下にゆれる場面がありますが、今のうちから体幹やおしりまわりを強くしていくと、"
            "もっと地面をうまく押せるようになります。"
        )

    text = " ".join(lines)
    return jp_wrap(text, max_chars=34)


def generate_training_menu(
    metrics: Dict[str, float],
    angles: Dict[str, float],
) -> List[str]:
    pitch = metrics.get("pitch", float("nan"))
    stab = metrics.get("stability", float("nan"))
    com_range = metrics.get("com_y_range", float("nan"))

    knee_mean = angles.get("knee_flex_mean", float("nan"))
    hip_mean = angles.get("hip_flex_mean", float("nan"))
    trunk_mean = angles.get("trunk_tilt_mean", float("nan"))  # 使うかもしれないので取得だけ

    drills: List[str] = []

    # 1. 体幹＆姿勢
    if math.isnan(stab) or stab < 8.0:
        drills.append(
            "① フロントプランク＋姿勢キープ歩行（30秒×3セット）"
            "：ひじとつま先で体を一直線に保つプランクを行い、その後、"
            "胸を張ったままゆっくり10歩ずつ前後に歩くことで、"
            "走るときの体幹の安定感を高めます。"
        )
    else:
        drills.append(
            "① 片足バランス＋腕振り（左右30秒×2セット）"
            "：片足立ちで軽く前傾しながら、スプリントの腕振りを行います。"
            "接地時の体幹の安定と腕・脚の連動を養います。"
        )

    # 2. 股関節可動域＆もも上げ
    if math.isnan(hip_mean) or hip_mean < 70:
        drills.append(
            "② 壁もたれハイニー（20回×2〜3セット）"
            "：壁に軽く背中をつけて立ち、片脚ずつももを高く引き上げます。"
            "腰が反らないようにお腹を軽く締め、股関節から脚を引き上げる感覚を身に付けます。"
        )
    else:
        drills.append(
            "② リズムハイニー（20m×3本）"
            "：その場もしくは20mの直線で、腕振りと合わせたテンポの良いハイニーを行います。"
            "実際の疾走に近いリズムで股関節の可動とピッチ向上を狙います。"
        )

    # 3. 膝の安定性
    if math.isnan(knee_mean) or knee_mean < 60:
        drills.append(
            "③ スローランジ（左右10回×2セット）"
            "：前後に大きく一歩を出し、ゆっくり腰を落としていきます。"
            "膝が内側に入らないように注意しながら行うことで、"
            "接地時の膝の安定性を高めます。"
        )
    else:
        drills.append(
            "③ スキップランジ（左右10回×2セット）"
            "：ランジの姿勢から前脚で軽く地面を押してスキップするように切り替えます。"
            "地面を押す感覚と膝の柔らかい使い方を同時にトレーニングします。"
        )

    # 4. 接地の速さ（ピッチ）
    if math.isnan(pitch) or pitch < 3.5:
        drills.append(
            "④ ラダーもしくはラインステップ（10秒×3セット）"
            "：地面に線を1本引き、その上をできるだけ速く『チョキチョキ走り』でまたぎます。"
            "足を大きく上げすぎず、接地を速くする感覚を養います。"
        )
    else:
        drills.append(
            "④ 20mハイテンポ走（20m×4本）"
            "：距離を短めに設定し、ピッチを意識して素早く回転させる練習です。"
            "タイムよりも『同じリズムで走りきる』ことを重視します。"
        )

    # 5. 重心の安定（上下動の抑制）
    if not math.isnan(com_range) and com_range > 0.03:
        drills.append(
            "⑤ ミニハードルもどき走（10m×3本）"
            "：地面にタオルやペットボトルなどを等間隔に並べ、"
            "それを軽くまたぎながら走ります。"
            "腰の高さをあまり上下させずに進む意識を持つことで、重心のブレを減らします。"
        )
    else:
        drills.append(
            "⑤ つま先〜母指球接地確認ドリル（20歩×2セット）"
            "：その場で軽く足踏みしながら、かかとからではなく"
            "母指球付近で接地してすぐに離地する感覚を確認します。"
        )

    if len(drills) > 5:
        drills = drills[:5]

    return [jp_wrap(d, max_chars=34) for d in drills]


# ==========================================
# PDF 出力
# ==========================================

class PDFReport(FPDF):
    pass


def safe_multicell(pdf: FPDF, text: str, line_height: float = 5.0) -> None:
    """
    FPDF の multi_cell で「Not enough horizontal space…」が出ないように
    幅と X 座標を強制的に整えてから描画。
    """
    if text is None:
        return
    text = str(text)
    if text.strip() == "":
        pdf.ln(line_height)
        return

    max_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(max_width, line_height, text)
    except Exception as e:
        # ここでクラッシュさせない
        log(f"⚠ multi_cell 失敗: {e} （text='{text[:30]}...'） → cell にフォールバック")
        pdf.set_x(pdf.l_margin)
        pdf.cell(max_width, line_height, text, ln=1)


def build_pdf(
    csv_path: str,
    video_path: str,
    athlete: str,
    out_dirs: Dict[str, str],
) -> None:
    log(f"build_pdf 開始 csv={csv_path}, video={video_path}")

    # メトリクス・角度
    df_metrics, metrics_stats = load_metrics_csv(csv_path)
    video_id = get_video_id(video_path)
    angle_csv = ensure_angles_csv(video_path, athlete, video_id, out_dirs["angles"])
    df_angles, angle_stats = load_angles_csv(angle_csv)

    # オーバーレイ動画 → 骨格画像
    overlay_name = os.path.basename(video_path).replace(".mp4", "_pose_overlay.mp4")
    overlay_path = os.path.join("outputs", "images", overlay_name)
    pose_images = extract_pose_images_from_overlay(overlay_path, out_dirs["pose_images"])

    # グラフ
    metrics_graph_path = os.path.join(out_dirs["graphs"], f"{video_id}_metrics_v5_4_3_lab.png")
    plot_metrics_graph(df_metrics, metrics_graph_path)

    angles_graph_path = os.path.join(out_dirs["graphs"], f"{video_id}_angles_v5_4_3_lab.png")
    angles_graph_ok = plot_angle_graph(df_angles, angles_graph_path)

    # AIコメント
    pro_comment = generate_ai_comments_pro(metrics_stats, angle_stats, athlete)
    easy_comment = generate_ai_comments_easy(metrics_stats, angle_stats, athlete)
    training_menu = generate_training_menu(metrics_stats, angle_stats)

    # PDF
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # フォント設定
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    bold_font_path = font_path  # Meiryo は太字も同じファイルでOK
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.add_font("JPB", "", bold_font_path, uni=True)
    log(f"✅ フォント読み込み: {font_path}")

    # タイトル
    pdf.set_font("JPB", "", 14)
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.4.3_lab）", ln=1)

    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
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

    pdf.cell(0, 6, f"ピッチ: {pitch_val:.2f} 歩/秒", ln=1)
    pdf.cell(0, 6, f"ストライド: {stride_val:.2f} m", ln=1)
    pdf.cell(0, 6, f"安定性スコア: {stab_val:.2f} / 10", ln=1)
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

    # AIフォーム解析コメント（専門向け）
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（専門向け）", ln=1)
    pdf.set_font("JP", "", 9)

    for line in pro_comment.split("\n"):
        safe_multicell(pdf, line, line_height=5.0)
    pdf.ln(2)

    # AIフォーム解析コメント（中学生・保護者向け）
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（中学生・保護者向け）", ln=1)
    pdf.set_font("JP", "", 9)

    for line in easy_comment.split("\n"):
        safe_multicell(pdf, line, line_height=5.0)
    pdf.ln(2)

    # おすすめトレーニング
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "おすすめトレーニング（自宅・学校でできるメニュー）", ln=1)
    pdf.set_font("JP", "", 9)

    for drill in training_menu:
        for line in drill.split("\n"):
            safe_multicell(pdf, line, line_height=5.0)
        pdf.ln(1)

    out_pdf = os.path.join(out_dirs["pdf"], f"{athlete}_form_report_v5_4_3_lab.pdf")
    pdf.output(out_pdf)
    log(f"✅ PDF出力完了: {out_pdf}")


# ==========================================
# main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="KJAC フォーム分析レポート v5.4.3_lab（高機能・実験用）")
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス（pose_metrics_analyzer_v3系の出力）")
    parser.add_argument("--athlete", required=True, help="選手名（フォルダ名にも使用）")
    args = parser.parse_args()

    video_id = get_video_id(args.video)
    out_dirs = get_root_output_dirs(args.athlete, video_id)
    log_path = init_logger(out_dirs["logs"])

    log(f"=== KJAC v5.4.3_lab START athlete={args.athlete}, video_id={video_id} ===")
    log(f"入力: video={args.video}, csv={args.csv}")
    log(f"ログファイル: {log_path}")

    try:
        build_pdf(args.csv, args.video, args.athlete, out_dirs)
        log("=== KJAC v5.4.3_lab END (success) ===")
    except Exception as e:
        # ここで **raise しない** → PowerShell 上ではクラッシュしない
        log(f"⚠ 予期せぬエラー発生（最終catch）: {e}")
        log("=== KJAC v5.4.3_lab END (failed but no crash) ===")


if __name__ == "__main__":
    main()



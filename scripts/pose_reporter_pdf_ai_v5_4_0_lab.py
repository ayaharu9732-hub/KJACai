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


# ==============================
#  ログまわり
# ==============================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_video_id(video_path: str) -> str:
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    return stem.replace(".", "_")


def get_root_output_dirs(athlete: str, video_id: str) -> Dict[str, str]:
    base = os.path.join("outputs", athlete, video_id)
    pdf_dir = os.path.join(base, "pdf")
    graphs_dir = os.path.join(base, "graphs")
    pose_dir = os.path.join(base, "pose_images")
    angles_dir = os.path.join(base, "angles")
    logs_dir = os.path.join(base, "logs")
    for d in [base, pdf_dir, graphs_dir, pose_dir, angles_dir, logs_dir]:
        ensure_dir(d)
    return {
        "base": base,
        "pdf": pdf_dir,
        "graphs": graphs_dir,
        "pose_images": pose_dir,
        "angles": angles_dir,
        "logs": logs_dir,
    }


def setup_logger(athlete: str, video_id: str, dirs: Dict[str, str]):
    log_path = os.path.join(dirs["logs"], f"report_v5_4_0_lab.log")

    def log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # ログ書き込み失敗しても処理は止めない
            pass

    log(f"=== KJAC v5.4.0_lab START athlete={athlete}, video_id={video_id} ===")
    log(f"出力ルート: {dirs['base']}")
    return log, log_path


# ==============================
#  CSV ロード & サマリ
# ==============================

def load_metrics_csv(csv_path: str, log) -> Tuple[pd.DataFrame, Dict[str, float]]:
    log(f"metrics CSV 読み込み開始: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"metrics CSV shape={df.shape}, columns={list(df.columns)}")

    if "_summary_" in df.iloc[:, 0].astype(str).values:
        df_data = df[df.iloc[:, 0].astype(str) != "_summary_"].copy()
        log("先頭列 '_summary_' 行を除外して解析に使用")
    else:
        df_data = df.copy()

    stats: Dict[str, float] = {}

    def safe_mean(col: str) -> float:
        if col in df_data.columns:
            v = float(df_data[col].mean(skipna=True))
            log(f"metrics mean {col}={v:.5f}")
            return v
        log(f"metrics col {col} が存在しません（nan 扱い）")
        return float("nan")

    stats["pitch"] = safe_mean("pitch_hz")
    stats["stride"] = safe_mean("stride_m")
    stats["stability"] = safe_mean("stability_score")

    stats["speed_mean"] = safe_mean("speed_m_s")
    if "speed_m_s" in df_data.columns:
        stats["speed_max"] = float(df_data["speed_m_s"].max(skipna=True))
        log(f"metrics max speed_m_s={stats['speed_max']:.5f}")
    else:
        stats["speed_max"] = float("nan")

    if "COM_y_m" in df_data.columns:
        stats["com_y_range"] = float(df_data["COM_y_m"].max(skipna=True)
                                     - df_data["COM_y_m"].min(skipna=True))
    elif "COM_y" in df_data.columns:
        stats["com_y_range"] = float(df_data["COM_y"].max(skipna=True)
                                     - df_data["COM_y"].min(skipna=True))
    else:
        stats["com_y_range"] = float("nan")
    log(f"metrics COM_y_range={stats['com_y_range']:.5f}")

    if "torso_tilt_deg" in df_data.columns:
        stats["torso_tilt_mean"] = float(df_data["torso_tilt_deg"].mean(skipna=True))
        stats["torso_tilt_var"] = float(df_data["torso_tilt_deg"].var(skipna=True))
    else:
        stats["torso_tilt_mean"] = float("nan")
        stats["torso_tilt_var"] = float("nan")
    log(f"metrics torso_tilt_mean={stats['torso_tilt_mean']:.5f}, "
        f"torso_tilt_var={stats['torso_tilt_var']:.5f}")

    # もし m_per_px があれば拾う（なければ nan）
    if "m_per_px" in df_data.columns:
        stats["m_per_px"] = float(df_data["m_per_px"].iloc[0])
        log(f"metrics m_per_px={stats['m_per_px']:.8f}")
    else:
        stats["m_per_px"] = float("nan")
        log("metrics m_per_px 列なし（スケール情報なし）")

    return df_data, stats


def ensure_angles_csv(video_path: str, athlete: str, video_id: str,
                      angles_dir: str, log) -> Optional[str]:
    target_csv = os.path.join(angles_dir, f"{video_id}.csv")
    log(f"角度CSV探索開始 target={target_csv}")

    # 1) 新形式
    if os.path.exists(target_csv):
        log(f"角度CSV(新形式)発見: {target_csv}")
        return target_csv

    # 2) 旧形式1: outputs/angles/{athlete}/{video_id}.csv
    old_dir_1 = os.path.join("outputs", "angles", athlete)
    old_csv_1 = os.path.join(old_dir_1, f"{video_id}.csv")
    if os.path.exists(old_csv_1):
        log(f"角度CSV(旧形式1)発見: {old_csv_1}")
        try:
            ensure_dir(angles_dir)
            import shutil
            shutil.copy2(old_csv_1, target_csv)
            log(f"→ 新形式へコピー: {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 旧形式1→新形式コピー失敗: {e}")
            return old_csv_1

    # 3) 旧形式2: outputs/angles_動画名.csv
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    old_csv_2 = os.path.join("outputs", f"angles_{stem}.csv")
    if os.path.exists(old_csv_2):
        log(f"角度CSV(旧形式2)発見: {old_csv_2}")
        try:
            ensure_dir(angles_dir)
            import shutil
            shutil.copy2(old_csv_2, target_csv)
            log(f"→ 新形式へコピー: {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 旧形式2→新形式コピー失敗: {e}")
            return old_csv_2

    # 4) どこにもない → 自動解析トライ
    analyzer_script = os.path.join("scripts", "pose_angle_analyzer_v1_0.py")
    log(f"角度CSV未発見。自動解析試行 script={analyzer_script}")

    if not os.path.exists(analyzer_script):
        log(f"⚠ 角度解析スクリプトが存在しません。自動解析スキップ。")
        return None

    cmd = ["python", analyzer_script, "--video", video_path]
    log(f"実行: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        log(f"⚠ 角度解析スクリプト実行失敗: {e}")
        return None

    # 自動解析後: 旧形式2 に出力された前提でもう一度チェック
    if os.path.exists(old_csv_2):
        try:
            ensure_dir(angles_dir)
            import shutil
            shutil.copy2(old_csv_2, target_csv)
            log(f"✅ 自動解析結果を新形式へコピー: {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 自動解析CSVコピー失敗: {e}")
            return old_csv_2

    log("⚠ 自動解析後も角度CSVが見つかりませんでした。")
    return None


def load_angles_csv(angle_csv: Optional[str], log) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if angle_csv is None or not os.path.exists(angle_csv):
        log("⚠ 角度CSVがロードできません。角度解析はスキップ。")
        return None, {}

    log(f"角度CSV読込開始: {angle_csv}")
    df = pd.read_csv(angle_csv)
    log(f"angles CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    def col_mean(names: List[str]) -> float:
        for n in names:
            if n in df.columns:
                v = float(df[n].mean(skipna=True))
                log(f"angles mean {n}={v:.5f}")
                return v
        log(f"angles mean {names} → 該当列なし (nan)")
        return float("nan")

    def col_range(names: List[str]) -> float:
        for n in names:
            if n in df.columns:
                v = float(df[n].max(skipna=True) - df[n].min(skipna=True))
                log(f"angles range {n}={v:.5f}")
                return v
        log(f"angles range {names} → 該当列なし (nan)")
        return float("nan")

    stats["knee_flex_mean"] = col_mean(["knee_angle_deg", "knee_flex_deg",
                                        "knee_flexion", "knee_angle"])
    stats["knee_flex_range"] = col_range(["knee_angle_deg", "knee_flex_deg",
                                          "knee_flexion", "knee_angle"])
    stats["hip_flex_mean"] = col_mean(["hip_angle_deg", "hip_flex_deg",
                                       "hip_flexion", "hip_angle"])
    stats["hip_flex_range"] = col_range(["hip_angle_deg", "hip_flex_deg",
                                         "hip_flexion", "hip_angle"])
    stats["trunk_tilt_mean"] = col_mean(["trunk_angle_deg", "torso_tilt_deg",
                                         "body_tilt_deg"])
    stats["trunk_tilt_range"] = col_range(["trunk_angle_deg", "torso_tilt_deg",
                                           "body_tilt_deg"])

    return df, stats


# ==============================
#  グラフ生成
# ==============================

def plot_metrics_graph(df_metrics: pd.DataFrame, out_path: str, log) -> None:
    log(f"メトリクスグラフ生成: {out_path}")
    ensure_dir(os.path.dirname(out_path))

    if "time_s" in df_metrics.columns:
        t = df_metrics["time_s"].values
    elif "frame" in df_metrics.columns:
        t = df_metrics["frame"].values
    else:
        t = np.arange(len(df_metrics))

    speed = df_metrics["speed_m_s"].values if "speed_m_s" in df_metrics.columns else None
    com_y = (df_metrics["COM_y_m"].values
             if "COM_y_m" in df_metrics.columns
             else df_metrics["COM_y"].values if "COM_y" in df_metrics.columns
             else None)
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


def plot_angle_graph(df_angles: Optional[pd.DataFrame], out_path: str, log) -> bool:
    if df_angles is None:
        log("角度グラフ: df_angles is None → スキップ")
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
    trunk_cols = [c for c in df_angles.columns
                  if any(k in c.lower() for k in ["trunk", "torso", "body_tilt"])]

    log(f"角度グラフ用列 knee={knee_cols}, hip={hip_cols}, trunk={trunk_cols}")

    if not (knee_cols or hip_cols or trunk_cols):
        log("角度グラフ: 使用可能な角度列が無いためスキップ")
        return False

    plt.figure(figsize=(6, 8))

    ax1 = plt.subplot(3, 1, 1)
    for c in knee_cols:
        ax1.plot(t, df_angles[c].values, label=c)
    if knee_cols:
        ax1.legend(fontsize=6)
    ax1.set_ylabel("Knee (deg)")
    ax1.set_title("Knee angles")

    ax2 = plt.subplot(3, 1, 2)
    for c in hip_cols:
        ax2.plot(t, df_angles[c].values, label=c)
    if hip_cols:
        ax2.legend(fontsize=6)
    ax2.set_ylabel("Hip (deg)")
    ax2.set_title("Hip angles")

    ax3 = plt.subplot(3, 1, 3)
    for c in trunk_cols:
        ax3.plot(t, df_angles[c].values, label=c)
    if trunk_cols:
        ax3.legend(fontsize=6)
    ax3.set_ylabel("Trunk (deg)")
    ax3.set_xlabel("Time (s or frame)")
    ax3.set_title("Trunk tilt")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"角度グラフ保存完了: {out_path}")
    return True


# ==============================
#  骨格オーバーレイから静止画3枚
# ==============================

def extract_pose_images_from_overlay(overlay_path: str, out_dir: str, log) -> Dict[str, str]:
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
        log("⚠ オーバーレイのフレーム数が0")
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
            log(f"⚠ フレーム取得失敗 key={key}, index={idx}")
            continue
        out_path = os.path.join(out_dir, f"{key}.png")
        try:
            ok = cv2.imwrite(out_path, frame)
            if ok:
                result[key] = out_path
                log(f"骨格静止画保存: {key} -> {out_path}")
            else:
                log(f"⚠ 画像保存に失敗: {out_path}")
        except Exception as e:
            log(f"⚠ 画像保存エラー: {out_path} -> {e}")

    cap.release()
    log(f"骨格画像出力結果: {result}")
    return result


# ==============================
#  AI コメント生成（5.3.8と同等）
# ==============================

def jp_wrap(text: str, max_chars: int = 35) -> str:
    text = text.replace("\n", "")
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    return "\n".join(chunks)


def generate_ai_comments_pro(metrics: Dict[str, float],
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
                f"安定性スコアは {stab:.2f}/10 とやや低く、特に接地毎の重心上下動が大きくなることで、"
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
            "股関節主導での引き上げと『脚を下ろすのではなく、身体が前に進む』感覚の獲得が重要です。"
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

    return jp_wrap("".join(lines), max_chars=38)


def generate_ai_comments_easy(metrics: Dict[str, float],
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
        f"{athlete}さんの走りは、全体としてスピードとリズムのバランスがよく、"
        f"平均速度は約{speed_mean:.1f} m/s（最大 {speed_max:.1f} m/s）と、"
        "現時点のレベルとしてとても良い指標です。"
    )
    lines.append(
        f"1秒あたりの歩数（ピッチ）は約 {pitch:.2f} 歩、1歩あたりの進む距離（ストライド）は約 {stride:.2f} m で、"
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
            "大きく崩れてはいませんが、スタートから加速の場面では『少し前に倒す→徐々に起きてくる』"
            "という変化をもう少しはっきり作れると、もっと走りがスムーズになります。"
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
            "十分に前へ脚を出せています。骨盤やお腹まわりを安定させながら"
            "同じ動きをくり返し出せるようになると、長い距離でもフォームが崩れにくくなります。"
        )

    if not math.isnan(com_range):
        lines.append(
            f"走っているときの重心（体の真ん中）の上下のゆれは約 {com_range:.3f} m で、"
            "ほんの少しだけ上下にゆれる場面がありますが、"
            "今のうちから体幹やおしりまわりを強くしていくと、"
            "もっと地面をうまく押せるようになります。"
        )

    text = " ".join(lines)
    return jp_wrap(text, max_chars=34)


def generate_training_menu(metrics: Dict[str, float],
                           angles: Dict[str, float]) -> List[str]:
    pitch = metrics.get("pitch", float("nan"))
    stab = metrics.get("stability", float("nan"))
    com_range = metrics.get("com_y_range", float("nan"))
    hip_mean = angles.get("hip_flex_mean", float("nan"))
    knee_mean = angles.get("knee_flex_mean", float("nan"))

    drills: List[str] = []

    if math.isnan(stab) or stab < 8.0:
        drills.append(
            "① フロントプランク＋姿勢キープ歩行（30秒×3セット）："
            "ひじとつま先で体を一直線に保つプランクを行い、その後、胸を張ったままゆっくり10歩ずつ前後に歩きます。"
            "走るときの体幹の安定感を高めます。"
        )
    else:
        drills.append(
            "① 片足バランス＋腕振り（左右30秒×2セット）："
            "片足立ちで軽く前傾しながらスプリントの腕振りを行い、接地時の体幹の安定と腕・脚の連動を養います。"
        )

    if math.isnan(hip_mean) or hip_mean < 70:
        drills.append(
            "② 壁もたれハイニー（20回×2〜3セット）："
            "壁に軽く背中をつけて片脚ずつももを高く引き上げ、腰が反らないようお腹を締めます。"
            "股関節から脚を引き上げる感覚を身につけます。"
        )
    else:
        drills.append(
            "② リズムハイニー（20m×3本）："
            "腕振りと合わせたテンポの良いハイニーで、実際の疾走に近いリズムで股関節の可動とピッチ向上を狙います。"
        )

    if math.isnan(knee_mean) or knee_mean < 60:
        drills.append(
            "③ スローランジ（左右10回×2セット）："
            "前後に大きく一歩を出しゆっくり腰を落とします。膝が内側に入らないよう意識し、接地時の膝の安定性を高めます。"
        )
    else:
        drills.append(
            "③ スキップランジ（左右10回×2セット）："
            "ランジ姿勢から前脚で軽く地面を押してスキップするように切り替え、地面を押す感覚と膝の柔らかい使い方を養います。"
        )

    if math.isnan(pitch) or pitch < 3.5:
        drills.append(
            "④ ラインステップ（10秒×3セット）："
            "地面に引いた線をできるだけ速くチョキチョキ走りでまたぎ、足を大きく上げすぎず接地を速くする感覚を身につけます。"
        )
    else:
        drills.append(
            "④ 20mハイテンポ走（20m×4本）："
            "距離を短めにしてピッチを意識して素早く回転させる練習です。タイムよりも同じリズムで走りきることを重視します。"
        )

    if not math.isnan(com_range) and com_range > 0.03:
        drills.append(
            "⑤ ミニハードルもどき走（10m×3本）："
            "タオルやペットボトルを等間隔に並べ、それを軽くまたぎながら走ります。腰の高さをあまり上下させずに進む意識を持ち、重心のブレを減らします。"
        )
    else:
        drills.append(
            "⑤ 母指球接地確認ドリル（20歩×2セット）："
            "その場で軽く足踏みしながら、かかとからではなく母指球付近で接地してすぐ離地する感覚を確認します。"
        )

    return [jp_wrap(d, max_chars=34) for d in drills[:5]]


# ==============================
#  PDF レポート
# ==============================

class PDFReport(FPDF):
    pass


def build_pdf(csv_path: str, video_path: str, athlete: str) -> None:
    video_id = get_video_id(video_path)
    dirs = get_root_output_dirs(athlete, video_id)
    log, log_path = setup_logger(athlete, video_id, dirs)

    log(f"build_pdf 開始 csv={csv_path}, video={video_path}")

    # メトリクス
    df_metrics, metrics_stats = load_metrics_csv(csv_path, log)

    # 角度CSV
    angle_csv = ensure_angles_csv(video_path, athlete, video_id, dirs["angles"], log)
    df_angles, angle_stats = load_angles_csv(angle_csv, log)

    # オーバーレイ動画 → 骨格画像3枚
    overlay_name = os.path.basename(video_path).replace(".mp4", "_pose_overlay.mp4")
    overlay_path = os.path.join("outputs", "images", overlay_name)
    pose_images = extract_pose_images_from_overlay(overlay_path, dirs["pose_images"], log)

    # グラフ
    metrics_graph_path = os.path.join(dirs["graphs"], f"{video_id}_metrics_v5_4_0_lab.png")
    plot_metrics_graph(df_metrics, metrics_graph_path, log)

    angles_graph_path = os.path.join(dirs["graphs"], f"{video_id}_angles_v5_4_0_lab.png")
    angles_graph_ok = plot_angle_graph(df_angles, angles_graph_path, log)

    # コメント生成
    pro_comment = generate_ai_comments_pro(metrics_stats, angle_stats, athlete)
    easy_comment = generate_ai_comments_easy(metrics_stats, angle_stats, athlete)
    training_menu = generate_training_menu(metrics_stats, angle_stats)

    # PDF
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = "c:/Windows/Fonts/meiryo.ttc"
    bold_font_path = font_path  # Meiryoは太字も同ファイルでOKなことが多い
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.add_font("JPB", "", bold_font_path, uni=True)

    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.4.0_lab）", ln=1)

    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 6, f"ログファイル: {os.path.relpath(log_path)}", ln=1)
    m_per_px = metrics_stats.get("m_per_px", float("nan"))
    if not math.isnan(m_per_px):
        pdf.cell(0, 6, f"校正スケール: {m_per_px:.6f} m/px", ln=1)
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
            try:
                pdf.image(pose_images[key], x=x, y=y_top, w=img_w)
            except Exception as e:
                log(f"⚠ PDFへの骨格画像貼り付け失敗 {key}: {e}")
        pdf.set_y(y_top + img_w * 0.75 + 5)
    else:
        pdf.cell(0, 6, "※骨格オーバーレイ画像が取得できませんでした。", ln=1)
    pdf.ln(3)

    # メトリクスグラフ
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "速度・重心・体幹傾斜の時間変化", ln=1)
    pdf.set_font("JP", "", 9)
    if os.path.exists(metrics_graph_path):
        try:
            pdf.image(metrics_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
        except Exception as e:
            log(f"⚠ メトリクスグラフ貼り付け失敗: {e}")
            pdf.cell(0, 6, "※メトリクスグラフ画像の貼り付けに失敗しました。", ln=1)
    else:
        pdf.cell(0, 6, "※メトリクスグラフ画像が見つかりません。", ln=1)
    pdf.ln(3)

    # 角度グラフ
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "関節角度（膝・股関節・体幹）の時間変化", ln=1)
    pdf.set_font("JP", "", 9)
    if angles_graph_ok and os.path.exists(angles_graph_path):
        try:
            pdf.image(angles_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
        except Exception as e:
            log(f"⚠ 角度グラフ貼り付け失敗: {e}")
            pdf.cell(0, 6, "※角度グラフ画像の貼り付けに失敗しました。", ln=1)
    else:
        pdf.cell(0, 6, "※角度グラフ画像が見つかりません。", ln=1)
    pdf.ln(3)

    # コメント（専門）
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（専門向け）", ln=1)
    pdf.set_font("JP", "", 9)
    for line in pro_comment.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 5, line)
        else:
            pdf.ln(4)
    pdf.ln(2)

    # コメント（中学生・保護者向け）
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（中学生・保護者向け）", ln=1)
    pdf.set_font("JP", "", 9)
    for line in easy_comment.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 5, line)
        else:
            pdf.ln(4)
    pdf.ln(2)

    # トレーニング
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "おすすめトレーニング（自宅・学校でできるメニュー）", ln=1)
    pdf.set_font("JP", "", 9)
    for drill in training_menu:
        for line in drill.split("\n"):
            if line.strip():
                pdf.multi_cell(0, 5, line)
            else:
                pdf.ln(3)
        pdf.ln(1)

    out_pdf = os.path.join(dirs["pdf"], f"{athlete}_form_report_v5_4_0_lab.pdf")
    pdf.output(out_pdf)
    log(f"✅ PDF出力完了: {out_pdf}")
    log("=== KJAC v5.4.0_lab END ===")


# ==============================
#  main
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="KJAC フォーム分析レポート v5.4.0_lab（高機能・ログ大量版）"
    )
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス")
    parser.add_argument("--athlete", required=True, help="選手名（フォルダ名にも使用）")
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()

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
from fpdf import FPDF, FPDFException


# ------------------------------
#  基本ユーティリティ
# ------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_video_id(video_path: str) -> str:
    """
    動画ファイル名（拡張子なし）から、フォルダ名用のIDを生成
    例: "二村遥香10.5.mp4" -> "二村遥香10_5"
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    return stem.replace(".", "_")


def get_root_output_dirs(athlete: str, video_id: str) -> Dict[str, str]:
    """
    出力を体系化:
    outputs/{athlete}/{video_id}/
        - pdf/
        - graphs/
        - pose_images/
        - angles/
    """
    base = os.path.join("outputs", athlete, video_id)
    pdf_dir = os.path.join(base, "pdf")
    graphs_dir = os.path.join(base, "graphs")
    pose_dir = os.path.join(base, "pose_images")
    angles_dir = os.path.join(base, "angles")

    for d in [base, pdf_dir, graphs_dir, pose_dir, angles_dir]:
        ensure_dir(d)

    return {
        "base": base,
        "pdf": pdf_dir,
        "graphs": graphs_dir,
        "pose_images": pose_dir,
        "angles": angles_dir,
    }


# ------------------------------
#  CSV ロード & サマリ計算
# ------------------------------

def load_metrics_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print(f"✅ メトリクスCSV読込: {csv_path}")
    df = pd.read_csv(csv_path)

    # _summary_ 行を除外（あれば）
    if "_summary_" in df.iloc[:, 0].astype(str).values:
        df_data = df[df.iloc[:, 0].astype(str) != "_summary_"].copy()
    else:
        df_data = df.copy()

    stats: Dict[str, float] = {}

    # 基本指標
    if "pitch_hz" in df_data.columns:
        stats["pitch"] = float(df_data["pitch_hz"].mean(skipna=True))
    else:
        stats["pitch"] = float("nan")

    if "stride_m" in df_data.columns:
        stats["stride"] = float(df_data["stride_m"].mean(skipna=True))
    else:
        stats["stride"] = float("nan")

    if "stability_score" in df_data.columns:
        stats["stability"] = float(df_data["stability_score"].mean(skipna=True))
    else:
        stats["stability"] = float("nan")

    # 速度関連
    if "speed_m_s" in df_data.columns:
        stats["speed_mean"] = float(df_data["speed_m_s"].mean(skipna=True))
        stats["speed_max"] = float(df_data["speed_m_s"].max(skipna=True))
    else:
        stats["speed_mean"] = float("nan")
        stats["speed_max"] = float("nan")

    # COM・体幹傾斜などの揺れ幅
    if "COM_y_m" in df_data.columns:
        stats["com_y_range"] = float(df_data["COM_y_m"].max(skipna=True) - df_data["COM_y_m"].min(skipna=True))
    elif "COM_y" in df_data.columns:
        stats["com_y_range"] = float(df_data["COM_y"].max(skipna=True) - df_data["COM_y"].min(skipna=True))
    else:
        stats["com_y_range"] = float("nan")

    if "torso_tilt_deg" in df_data.columns:
        stats["torso_tilt_mean"] = float(df_data["torso_tilt_deg"].mean(skipna=True))
        stats["torso_tilt_var"] = float(df_data["torso_tilt_deg"].var(skipna=True))
    else:
        stats["torso_tilt_mean"] = float("nan")
        stats["torso_tilt_var"] = float("nan")

    # ※ もし m_per_px をCSVに入れるならここで拾う
    if "m_per_px" in df_data.columns:
        stats["m_per_px"] = float(df_data["m_per_px"].iloc[0])
    else:
        stats["m_per_px"] = float("nan")

    return df_data, stats


def ensure_angles_csv(video_path: str, athlete: str, video_id: str, angles_dir: str) -> Optional[str]:
    """
    角度CSVの場所をできるだけ賢く探す。
    - 新形式: outputs/{athlete}/{video_id}/angles/{video_id}.csv
    - 旧形式: outputs/angles/{athlete}/{video_id}.csv
    - もっと旧形式: outputs/angles_動画名.csv
    見つからなければ pose_angle_analyzer_v1_0.py を実行して生成を試みる。
    """
    # 1) 新形式（最優先）
    target_csv = os.path.join(angles_dir, f"{video_id}.csv")
    if os.path.exists(target_csv):
        print(f"✅ 角度CSV読込候補: {target_csv}")
        return target_csv

    # 2) 旧形式1: outputs/angles/{athlete}/{video_id}.csv
    old_dir_1 = os.path.join("outputs", "angles", athlete)
    old_csv_1 = os.path.join(old_dir_1, f"{video_id}.csv")
    if os.path.exists(old_csv_1):
        ensure_dir(angles_dir)
        new_path = target_csv
        print(f"🔁 既存角度CSVを新形式として採用: {old_csv_1} → {new_path}")
        try:
            ensure_dir(os.path.dirname(new_path))
            import shutil
            shutil.copy2(old_csv_1, new_path)
            return new_path
        except Exception as e:
            print(f"⚠ 角度CSVのコピーに失敗しました: {e}")
            return old_csv_1

    # 3) 旧形式2: outputs/angles_動画名.csv
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    old_csv_2 = os.path.join("outputs", f"angles_{stem}.csv")
    if os.path.exists(old_csv_2):
        ensure_dir(angles_dir)
        new_path = target_csv
        print(f"🔁 旧形式角度CSVを新形式として採用: {old_csv_2} → {new_path}")
        try:
            ensure_dir(os.path.dirname(new_path))
            import shutil
            shutil.copy2(old_csv_2, new_path)
            return new_path
        except Exception as e:
            print(f"⚠ 角度CSVのコピーに失敗しました: {e}")
            return old_csv_2

    # 4) どこにもない → 自動解析を試みる
    print(f"🔄 角度CSVが見つかりません。自動解析を試みます: {target_csv}")
    analyzer_script = os.path.join("scripts", "pose_angle_analyzer_v1_0.py")
    if not os.path.exists(analyzer_script):
        print(f"⚠ 角度解析スクリプトが見つかりません: {analyzer_script}")
        return None

    cmd = ["python", analyzer_script, "--video", video_path]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"⚠ 角度解析スクリプトの実行に失敗しました: {e}")
        return None

    # 自動解析後は旧形式 outputs/angles_動画名.csv に出ている想定
    if os.path.exists(old_csv_2):
        try:
            ensure_dir(os.path.dirname(target_csv))
            import shutil
            shutil.copy2(old_csv_2, target_csv)
            print(f"✅ 自動解析した角度CSVを新形式へコピー: {target_csv}")
            return target_csv
        except Exception as e:
            print(f"⚠ 角度CSVコピー失敗: {e}")
            return old_csv_2

    print("⚠ 自動解析後も角度CSVが見つかりませんでした。")
    return None


def load_angles_csv(angle_csv: Optional[str]) -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    if angle_csv is None or not os.path.exists(angle_csv):
        print("⚠ 角度CSVがロードできませんでした。角度解析はスキップします。")
        return None, {}

    print(f"✅ 角度CSV読込: {angle_csv}")
    df = pd.read_csv(angle_csv)

    stats: Dict[str, float] = {}

    def col_mean(names: List[str]) -> float:
        for n in names:
            if n in df.columns:
                return float(df[n].mean(skipna=True))
        return float("nan")

    def col_range(names: List[str]) -> float:
        for n in names:
            if n in df.columns:
                return float(df[n].max(skipna=True) - df[n].min(skipna=True))
        return float("nan")

    # 両脚膝・股関節・体幹の代表値
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


# ------------------------------
#  グラフ生成
# ------------------------------

def plot_metrics_graph(df_metrics: pd.DataFrame, out_path: str) -> None:
    print(f"📊 メトリクスグラフ画像出力: {out_path}")
    ensure_dir(os.path.dirname(out_path))

    # 時間軸っぽいXを作成
    if "time_s" in df_metrics.columns:
        t = df_metrics["time_s"].values
    else:
        # フレーム番号で代用
        if "frame" in df_metrics.columns:
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
        print("⚠ 角度グラフを描画できるデータがありません。")
        return False

    ensure_dir(os.path.dirname(out_path))

    if "time_s" in df_angles.columns:
        t = df_angles["time_s"].values
    elif "frame" in df_angles.columns:
        t = df_angles["frame"].values
    else:
        t = np.arange(len(df_angles))

    # 候補列
    knee_cols = [c for c in df_angles.columns if "knee" in c.lower()]
    hip_cols = [c for c in df_angles.columns if "hip" in c.lower()]
    trunk_cols = [c for c in df_angles.columns if any(k in c.lower()
                                                      for k in ["trunk", "torso", "body_tilt"])]

    if not (knee_cols or hip_cols or trunk_cols):
        print("⚠ 角度グラフを描画できるカラムがありません。")
        return False

    print(f"📊 角度グラフ画像出力: {out_path}")

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
    return True


# ------------------------------
#  骨格オーバーレイ画像抽出
# ------------------------------

def extract_pose_images_from_overlay(overlay_path: str, out_dir: str) -> Dict[str, str]:
    """
    オーバーレイ動画から start / mid / finish の3枚をPNGで保存。
    失敗した場合は空dictを返す。
    """
    result: Dict[str, str] = {}
    if not os.path.exists(overlay_path):
        print(f"⚠ オーバーレイ動画が見つかりません: {overlay_path}")
        return result

    print(f"🎥 オーバーレイ動画から骨格画像抽出: {overlay_path}")
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        print("⚠ オーバーレイ動画を開けませんでした。")
        return result

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print("⚠ フレーム数が0です。")
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
            print(f"⚠ フレーム取得に失敗しました: {key} (index={idx})")
            continue
        out_path = os.path.join(out_dir, f"{key}.png")
        try:
            ok = cv2.imwrite(out_path, frame)
            if ok:
                result[key] = out_path
            else:
                print(f"⚠ 画像保存に失敗しました: {out_path}")
        except Exception as e:
            print(f"⚠ 画像保存エラー: {out_path} -> {e}")

    cap.release()
    print(f"📸 骨格画像出力: {result}")
    return result


# ------------------------------
#  AIコメント生成ロジック
# ------------------------------

def jp_wrap_simple(text: str, max_chars: int = 35) -> str:
    """
    ざっくり文字数ベースで改行を入れる簡易ラッパ。
    """
    text = text.replace("\r", "").replace("\n", "")
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    return "\n".join(chunks)


def safe_multicell(pdf: FPDF, text: str, h: float = 5.0) -> None:
    """
    FPDFException( Not enough horizontal space... ) を絶対出さないための安全マルチセル。

    - いったん \n で段落分割
    - 1行ずつ「この文字を足したら幅オーバーか？」を確認してから multi_cell
    - 幅は (ページ幅 - 左右マージン) を上限とする
    """
    if text is None:
        return

    max_width = pdf.w - pdf.l_margin - pdf.r_margin
    if max_width <= 0:
        max_width = 50  # 万一の保険

    # 段落ごとに処理
    paragraphs = text.replace("\r", "").split("\n")
    for para in paragraphs:
        para = para.strip()
        if para == "":
            pdf.ln(h)
            continue

        buf = ""
        for ch in para:
            test = buf + ch
            w = pdf.get_string_width(test)
            if w <= max_width:
                buf = test
            else:
                # ここまでを出力
                try:
                    pdf.multi_cell(max_width, h, buf)
                except FPDFException:
                    # それでもダメな場合、1文字ずつ出力
                    for c2 in buf:
                        pdf.multi_cell(max_width, h, c2)
                buf = ch  # 新しい行を開始

        if buf:
            try:
                pdf.multi_cell(max_width, h, buf)
            except FPDFException:
                for c2 in buf:
                    pdf.multi_cell(max_width, h, c2)


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

    return jp_wrap_simple("".join(lines), max_chars=38)


def generate_ai_comments_easy(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str
) -> str:
    """
    中学生・保護者向けのやさしい説明（8〜12行）
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
    return jp_wrap_simple(text, max_chars=34)


def generate_training_menu(
    metrics: Dict[str, float],
    angles: Dict[str, float],
) -> List[str]:
    """
    自宅・学校でできる軽めのトレーニングを5つ返す（コメント内容とリンク）
    """
    pitch = metrics.get("pitch", float("nan"))
    stride = metrics.get("stride", float("nan"))
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
            "胸を張ったままゆっくり10歩ずつ前後に歩くことで、"
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
            "その場もしくは20mの直線で、腕振りと合わせたテンポの良いハイニーを行います。"
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

    # 必ず5つに揃える（保険）
    if len(drills) > 5:
        drills = drills[:5]
    elif len(drills) < 5:
        drills.append(
            "補足：全てのドリルの前後には、軽いジョグとストレッチを行い、"
            "無理のない範囲で実施してください。痛みが出る場合は中止し、専門家に相談しましょう。"
        )

    return [jp_wrap_simple(d, max_chars=34) for d in drills[:5]]


# ------------------------------
#  PDF 出力
# ------------------------------

class PDFReport(FPDF):
    pass  # 必要ならヘッダー・フッターをカスタム可能


def build_pdf(
    csv_path: str,
    video_path: str,
    athlete: str
) -> None:
    video_id = get_video_id(video_path)
    out_dirs = get_root_output_dirs(athlete, video_id)

    # メトリクス読み込み
    df_metrics, metrics_stats = load_metrics_csv(csv_path)

    # 角度CSVの確保＆読み込み
    angle_csv = ensure_angles_csv(video_path, athlete, video_id, out_dirs["angles"])
    df_angles, angle_stats = load_angles_csv(angle_csv)

    # 骨格オーバーレイ動画パス（既存ルールに合わせる）
    overlay_name = os.path.basename(video_path).replace(".mp4", "_pose_overlay.mp4")
    overlay_path = os.path.join("outputs", "images", overlay_name)
    pose_images = extract_pose_images_from_overlay(overlay_path, out_dirs["pose_images"])

    # グラフ生成
    metrics_graph_path = os.path.join(
        out_dirs["graphs"],
        f"{video_id}_metrics_v5_3_9.png",
    )
    plot_metrics_graph(df_metrics, metrics_graph_path)

    angles_graph_path = os.path.join(
        out_dirs["graphs"],
        f"{video_id}_angles_v5_3_9.png",
    )
    angles_graph_ok = plot_angle_graph(df_angles, angles_graph_path)

    # AIコメント生成
    pro_comment = generate_ai_comments_pro(metrics_stats, angle_stats, athlete)
    easy_comment = generate_ai_comments_easy(metrics_stats, angle_stats, athlete)
    training_menu = generate_training_menu(metrics_stats, angle_stats)

    # PDF 作成
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 日本語フォント設定（通常：JP、太字：JPB）
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    bold_font_path = "c:/Windows/Fonts/meiryob.ttc"

    pdf.add_font("JP", "", font_path, uni=True)
    pdf.add_font("JPB", "", bold_font_path, uni=True)

    pdf.set_font("JP", "", 12)

    # タイトル部
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.3.9）", ln=1)
    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
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
        # 3枚横並び
        y_top = pdf.get_y()
        margin = 10
        usable_width = pdf.w - 2 * pdf.l_margin
        img_w = (usable_width - 2 * margin) / 3.0

        order = ["start", "mid", "finish"]
        for i, key in enumerate(order):
            if key not in pose_images:
                continue
            x = pdf.l_margin + i * (img_w + margin)
            pdf.image(pose_images[key], x=x, y=y_top, w=img_w)

        # 高さは画像の縦横比に依存するが、ざっくり0.75倍で前提
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
        if line.strip() == "":
            pdf.ln(4)
        else:
            safe_multicell(pdf, line, h=5)
    pdf.ln(2)

    # AIフォーム解析コメント（中学生・保護者向け）
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（中学生・保護者向け）", ln=1)
    pdf.set_font("JP", "", 9)

    for line in easy_comment.split("\n"):
        if line.strip() == "":
            pdf.ln(4)
        else:
            safe_multicell(pdf, line, h=5)
    pdf.ln(2)

    # おすすめトレーニング
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "おすすめトレーニング（自宅・学校でできるメニュー）", ln=1)
    pdf.set_font("JP", "", 9)

    for drill in training_menu:
        for line in drill.split("\n"):
            if line.strip() == "":
                pdf.ln(3)
            else:
                safe_multicell(pdf, line, h=5)
        pdf.ln(1)

    # 保存
    out_pdf = os.path.join(out_dirs["pdf"], f"{athlete}_form_report_v5_3_9.pdf")
    pdf.output(out_pdf)
    print(f"✅ PDF出力完了: {out_pdf}")


# ------------------------------
#  メイン
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="KJAC フォーム分析レポート v5.3.9（高機能版・フォント安定）")
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス（pose_metrics_analyzer_v3系の出力）")
    parser.add_argument("--athlete", required=True, help="選手名（フォルダ名にも使用）")
    args = parser.parse_args()

    print(f"📏 読み込み: video={args.video}, csv={args.csv}, athlete={args.athlete}")
    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()






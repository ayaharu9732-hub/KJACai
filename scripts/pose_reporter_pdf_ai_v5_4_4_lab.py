import os
import sys
import math
import argparse
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# OpenAI（v1系クライアント想定）
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =========================================================
#  グローバル設定
# =========================================================

VERSION_STR = "v5.4.4_lab"
LOG_FILE: Optional[str] = None


# =========================================================
#  ログ周り
# =========================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """コンソール＋ログファイル両方に出す"""
    global LOG_FILE
    line = f"[{now_str()}] {msg}"
    print(line)
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # ログ書き込みに失敗しても処理は止めない
            pass


# =========================================================
#  基本ユーティリティ
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    出力ルート:
    outputs/{athlete}/{video_id}/
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


# =========================================================
#  CSV ロード & サマリ計算
# =========================================================

def estimate_m_per_px(df: pd.DataFrame) -> float:
    """
    COM_y_m と COM_y_px からおおよその m/px を推定。
    データが不十分な場合はデフォルト 0.001607 を返す。
    """
    if "COM_y_m" in df.columns and "COM_y_px" in df.columns:
        try:
            valid = df[["COM_y_m", "COM_y_px"]].dropna()
            valid = valid[valid["COM_y_px"] != 0]
            if len(valid) > 0:
                ratios = (valid["COM_y_m"].abs() / valid["COM_y_px"].abs())
                scale = float(ratios.median())
                if scale > 0 and scale < 0.01:
                    log(f"metrics m_per_px 推定={scale:.6f}")
                    return scale
        except Exception as e:
            log(f"metrics m_per_px 推定失敗: {e}")

    default_scale = 0.001607
    log(f"metrics m_per_px 推定失敗→既定値 {default_scale:.6f} を使用")
    return default_scale


def load_metrics_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    log(f"metrics CSV 読み込み開始: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"metrics CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    # _summary_ 行を使うか、なければ全体平均
    if "_summary_" in df.iloc[:, 0].astype(str).values:
        log("metrics: '_summary_' 行あり → そこをサマリとして使用")
        df_summary = df[df.iloc[:, 0].astype(str) == "_summary_"]
        df_data = df[df.iloc[:, 0].astype(str) != "_summary_"].copy()
    else:
        log("metrics: '_summary_' 行なし → 全行から平均値を推定")
        df_summary = None
        df_data = df.copy()

    def get_val(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        if df_summary is not None:
            v = df_summary[col].iloc[0]
            try:
                return float(v)
            except Exception:
                return float("nan")
        return float(df[col].mean(skipna=True))

    # 基本指標
    stats["pitch"] = get_val("pitch_hz")
    stats["stride"] = get_val("stride_m")
    stats["stability"] = get_val("stability_score")

    log(f"metrics pitch_hz = {stats['pitch']}")
    log(f"metrics stride_m = {stats['stride']}")
    log(f"metrics stability_score = {stats['stability']}")

    if "speed_mps" in df.columns:
        stats["speed_mean"] = float(df["speed_mps"].mean(skipna=True))
        stats["speed_max"] = float(df["speed_mps"].max(skipna=True))
    else:
        log("metrics: speed_mps 列なし（nan 扱い）")
        stats["speed_mean"] = float("nan")
        stats["speed_max"] = float("nan")

    log(f"metrics speed_mps mean={stats['speed_mean']}, max={stats['speed_max']}")

    # COM_y range
    if "COM_y_m" in df.columns:
        stats["com_y_range"] = float(df["COM_y_m"].max(skipna=True) - df["COM_y_m"].min(skipna=True))
    else:
        stats["com_y_range"] = float("nan")
    log(f"metrics COM_y_range={stats['com_y_range']:.5f}" if not math.isnan(stats["com_y_range"]) else "metrics COM_y_range=nan")

    # 体幹傾斜
    if "torso_tilt_deg" in df.columns:
        stats["torso_tilt_mean"] = float(df["torso_tilt_deg"].mean(skipna=True))
        stats["torso_tilt_var"] = float(df["torso_tilt_deg"].var(skipna=True))
    else:
        log("metrics: torso_tilt_deg 列なし（体幹傾斜は nan 扱い）")
        stats["torso_tilt_mean"] = float("nan")
        stats["torso_tilt_var"] = float("nan")

    # スケール
    stats["m_per_px"] = estimate_m_per_px(df)

    return df_data, stats


def ensure_angles_csv(video_path: str, athlete: str, video_id: str, angles_dir: str) -> Optional[str]:
    """
    角度CSVの場所をできるだけ賢く探す。
    - 新形式: outputs/{athlete}/{video_id}/angles/{video_id}.csv
    - 旧形式: outputs/angles/{athlete}/{video_id}.csv
    - もっと旧形式: outputs/angles_動画名.csv
    見つからなければ pose_angle_analyzer_v1_0.py を実行して生成を試みる。
    """
    ensure_dir(angles_dir)
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
        import shutil
        try:
            shutil.copy2(old_csv_1, target_csv)
            log(f"旧形式→新形式へコピー: {old_csv_1} -> {target_csv}")
            return target_csv
        except Exception as e:
            log(f"角度CSVコピー失敗(1): {e}")
            return old_csv_1

    # 3) 旧形式2: outputs/angles_動画名.csv
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    old_csv_2 = os.path.join("outputs", f"angles_{stem}.csv")
    if os.path.exists(old_csv_2):
        import shutil
        try:
            shutil.copy2(old_csv_2, target_csv)
            log(f"旧形式→新形式へコピー: {old_csv_2} -> {target_csv}")
            return target_csv
        except Exception as e:
            log(f"角度CSVコピー失敗(2): {e}")
            return old_csv_2

    # 4) 自動解析（最後の手段）
    analyzer_script = os.path.join("scripts", "pose_angle_analyzer_v1_0.py")
    if not os.path.exists(analyzer_script):
        log(f"⚠ 角度解析スクリプトが見つかりません: {analyzer_script}")
        return None

    log(f"角度CSVが見つからないため自動解析を実行: {analyzer_script}")
    import subprocess
    cmd = ["python", analyzer_script, "--video", video_path]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        log(f"⚠ 角度解析スクリプトの実行に失敗しました: {e}")
        return None

    # 自動解析後は old_csv_2 に出ている想定
    if os.path.exists(old_csv_2):
        import shutil
        try:
            shutil.copy2(old_csv_2, target_csv)
            log(f"✅ 自動解析した角度CSVを新形式へコピー: {target_csv}")
            return target_csv
        except Exception as e:
            log(f"⚠ 自動解析角度CSVコピー失敗: {e}")
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

    # ここでは既存列を素直に使う（left/right の平均など）
    def mean_range(cols: List[str]) -> Tuple[float, float]:
        vals = []
        for c in cols:
            if c in df.columns:
                vals.append(df[c].astype(float))
        if not vals:
            return float("nan"), float("nan")
        merged = sum(vals) / len(vals)
        return float(merged.mean(skipna=True)), float(merged.max(skipna=True) - merged.min(skipna=True))

    knee_mean, knee_range = mean_range(["left_knee", "right_knee"])
    hip_mean, hip_range = mean_range(["left_hip", "right_hip"])
    trunk_mean, trunk_range = mean_range(["torso_tilt_left", "torso_tilt_right"])

    stats["knee_flex_mean"] = knee_mean
    stats["knee_flex_range"] = knee_range
    stats["hip_flex_mean"] = hip_mean
    stats["hip_flex_range"] = hip_range
    stats["trunk_tilt_mean"] = trunk_mean
    stats["trunk_tilt_range"] = trunk_range

    log(
        f"angles stats: knee_mean={knee_mean:.3f}, knee_range={knee_range:.3f}, "
        f"hip_mean={hip_mean:.3f}, hip_range={hip_range:.3f}, "
        f"trunk_mean={trunk_mean:.3f}, trunk_range={trunk_range:.3f}"
    )

    return df, stats


# =========================================================
#  グラフ生成
# =========================================================

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
    com_y = df_metrics["COM_y_m"].values if "COM_y_m" in df_metrics.columns else None
    torso = df_metrics["tilt_deg"].values if "tilt_deg" in df_metrics.columns else None

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
    ax3.set_ylabel("COM height (m)")
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

    plt.figure(figsize=(6, 8))

    # 膝
    ax1 = plt.subplot(3, 1, 1)
    for c in knee_cols:
        ax1.plot(t, df_angles[c].values, label=c)
    if knee_cols:
        ax1.legend(fontsize=6)
    ax1.set_ylabel("Knee (deg)")
    ax1.set_title("Knee angles")

    # 股関節
    ax2 = plt.subplot(3, 1, 2)
    for c in hip_cols:
        ax2.plot(t, df_angles[c].values, label=c)
    if hip_cols:
        ax2.legend(fontsize=6)
    ax2.set_ylabel("Hip (deg)")
    ax2.set_title("Hip angles")

    # 体幹
    ax3 = plt.subplot(3, 1, 3)
    for c in trunk_cols:
        ax3.plot(t, df_angles[c].values, label=c)
    if trunk_cols:
        ax3.legend(fontsize=6)
    ax3.set_ylabel("Trunk (deg)")
    ax3.set_xlabel("Time (s or frame)")
    ax3.set_title("Trunk tilt")

    plt.tight_layout()
    log(f"角度グラフ保存先: {out_path}")
    plt.savefig(out_path, dpi=200)
    plt.close()
    log("角度グラフ保存完了")
    return True


# =========================================================
#  骨格オーバーレイ画像抽出
# =========================================================

def extract_pose_images_from_overlay(overlay_path: str, out_dir: str) -> Dict[str, str]:
    """
    オーバーレイ動画から start / mid / finish の3枚をPNGで保存。
    失敗しても処理を止めずに空dictを返す。
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
            ok = cv2.imwrite(out_path, frame)
            if ok:
                log(f"骨格画像保存成功: {key} -> {out_path}")
                result[key] = out_path
            else:
                log(f"⚠ 画像保存に失敗: {out_path}")
        except Exception as e:
            log(f"⚠ 画像保存エラー: {out_path} -> {e}")

    cap.release()
    log(f"骨格画像出力結果: {result}")
    return result


# =========================================================
#  テキスト整形（multi_cell 安全ラッパー）
# =========================================================

def force_wrap_for_pdf(text: str, max_chars: int = 34) -> str:
    """
    FPDF.multi_cell で 'Not enough horizontal space' エラーにならないよう、
    ・制御文字を除去
    ・全角/半角問わず max_chars ごとに改行を挿入
    ・空行も許容
    """
    if text is None:
        return ""

    # 改行を統一
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # タブなどはスペースに
    text = text.replace("\t", " ")

    paragraphs = text.split("\n")

    out_lines: List[str] = []
    for par in paragraphs:
        p = par.strip()
        if p == "":
            # 空行として扱う
            out_lines.append("")
            continue
        # 文字数で機械的に折る（日本語前提）
        buf = ""
        for ch in p:
            buf += ch
            if len(buf) >= max_chars:
                out_lines.append(buf)
                buf = ""
        if buf:
            out_lines.append(buf)

    return "\n".join(out_lines)


def write_wrapped_multicell(pdf: FPDF, text: str, line_height: float = 5.0, max_chars: int = 34) -> None:
    """
    safe wrapper: 幅を明示的に指定し、1行ずつ multi_cell で出力。
    """
    safe_text = force_wrap_for_pdf(text, max_chars=max_chars)
    # 利用可能幅（左右マージンを除いた分）
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    if usable_width <= 0:
        # 万一の保険
        usable_width = 100

    for line in safe_text.split("\n"):
        if line.strip() == "":
            pdf.ln(line_height)
        else:
            pdf.multi_cell(usable_width, line_height, line)


# =========================================================
#  AIコメント生成（OpenAI + ルールベース）
# =========================================================

def get_openai_client() -> Optional["OpenAI"]:
    if not OPENAI_AVAILABLE:
        log("OpenAI ライブラリがインポートされていません（ルールベースのみ使用）")
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        log("⚠ OPENAI_API_KEY が設定されていません（ルールベースのみ使用）")
        return None
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        log(f"⚠ OpenAI クライアント初期化失敗: {e}")
        return None


def call_openai_for_comment(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4.1-mini",
) -> Optional[str]:
    client = get_openai_client()
    if client is None:
        return None

    try:
        log(f"OpenAI API 呼び出し開始 ({model})")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        text = resp.choices[0].message.content
        return text
    except Exception as e:
        log(f"⚠ OpenAI API 呼び出しで例外発生: {e}")
        return None


def generate_ai_comments_pro_rule(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str,
) -> str:
    """
    技術分析＋改善点重視（コーチ口調・専門寄り）
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

    # 全体像
    lines.append(
        f"{athlete}の走りを数字で見ると、ピッチが約 {pitch:.2f} 歩/秒、"
        f"ストライドが約 {stride:.2f} m、平均速度が {speed_mean:.2f} m/s"
        f"（最大 {speed_max:.2f} m/s）というバランスになっています。"
        "現時点ではピッチ寄りの走りで、テンポよく回転できているのが強みです。"
    )

    # 安定性
    if not math.isnan(stab):
        if stab >= 9.0:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 とかなり高く、"
                "接地ごとの上下動や左右ブレが小さい、きれいな重心コントロールができています。"
                "このまま『フォームを崩さないこと』を前提に、ストライドを少しずつ伸ばしていく段階です。"
            )
        elif stab >= 7.0:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 で、実戦では十分使えるレベルですが、"
                "スピードが上がった局面で少し重心の上下動が大きくなる場面が見られます。"
                "特に接地の瞬間に腰が落ちすぎないように、股関節と体幹で支える意識を持てるとさらに良くなります。"
            )
        else:
            lines.append(
                f"安定性スコアは {stab:.2f}/10 とやや低めで、"
                "接地ごとに重心が大きく上下してしまい、前への推進力が逃げている可能性があります。"
                "ストライドよりも『同じリズムで、同じ高さで走り続ける』ことを先に整えたい段階です。"
            )

    # 体幹前傾
    if not math.isnan(trunk_mean):
        lines.append(
            f"体幹の前傾角（上半身の倒し具合）は平均で {trunk_mean:.1f} 度付近です。"
            "動画を見ると、前半はしっかり前傾を取れている一方で、"
            "中盤以降に上体が起きるタイミングがやや早く、"
            "その分だけ加速の伸びが途中で止まりやすい形になっています。"
        )

    # 膝・股関節
    if not math.isnan(knee_mean):
        lines.append(
            f"膝関節の角度は平均 {knee_mean:.1f} 度付近で、脚をたたんで前に運ぶ動きは十分に出せています。"
            "改善したいポイントは『接地の直前に膝を伸ばし過ぎないこと』で、"
            "地面を真下に軽く押すイメージで接地できると、ブレーキ成分が減り、ピッチも落ちにくくなります。"
        )

    if not math.isnan(hip_mean):
        lines.append(
            f"股関節の屈曲角は平均 {hip_mean:.1f} 度と、もも上げ自体はよくできています。"
            "ただし、骨盤が前に倒れすぎたり、上半身がかぶりすぎたりすると、"
            "せっかく上げた脚が前ではなく下方向に出てしまいやすいので、"
            "『おへそから前に進む』感覚で骨盤ごと前へ送り出すイメージが大事になります。"
        )

    # 重心上下動
    if not math.isnan(com_range):
        if com_range < 0.03:
            lines.append(
                f"重心の上下動の幅は約 {com_range:.3f} m とかなり小さく、"
                "エネルギーロスの少ないフラットな重心軌道が取れています。"
                "この強みを保ったまま、接地時間を少しずつ短くしていくと、"
                "同じフォームでスピードだけを底上げしていけます。"
            )
        else:
            lines.append(
                f"重心の上下動の幅は約 {com_range:.3f} m で、やや大きめです。"
                "特にストライドを伸ばそうとしたときに、腰が沈み込んでしまい、"
                "結果として前ではなく上下にエネルギーが逃げている可能性があります。"
                "まずは『腰の高さを一定に保ったまま速く足を回す』感覚を身につけていきましょう。"
            )

    text = " ".join(lines)
    return force_wrap_for_pdf(text, max_chars=36)


def generate_ai_comments_easy_rule(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str,
) -> str:
    """
    中学生・保護者向けのやさしい説明（コーチが直接話している口調）
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
        f"{athlete}の走りを数字で見ると、1秒間にだいたい {pitch:.1f} 歩、"
        f"1歩で {stride:.2f} m 進んでいて、平均の速さは {speed_mean:.2f} m/s"
        f"（最大 {speed_max:.2f} m/s）という結果になっています。"
        "テンポよく回転できているので、この点はとても良いところです。"
    )

    if not math.isnan(stab):
        if stab >= 9.0:
            lines.append(
                f"走っているときの体のブレをまとめた『安定性スコア』は {stab:.1f}/10 と高く、"
                "フォームの形はかなり安定しています。"
                "このままフォームを崩さずに、スピードだけ少しずつ上げていけると理想的です。"
            )
        else:
            lines.append(
                f"安定性スコアは {stab:.1f}/10 で、走りとしては十分ですが、"
                "スピードが上がるときに少し体が上下にゆれる場面があります。"
                "『頭の高さをあまり上下させない』イメージを持って走ると、無駄な力が減ってきます。"
            )

    if not math.isnan(trunk_mean):
        lines.append(
            f"上半身の前への倒れ具合は平均で {trunk_mean:.1f} 度くらいです。"
            "前半はしっかり前に倒れて加速できていますが、"
            "途中で上半身が少し早く起きてしまうので、"
            "もう少しだけ長く前傾をキープできると、スピードの伸びが良くなります。"
        )

    if not math.isnan(knee_mean):
        lines.append(
            f"膝の曲げ伸ばしは平均 {knee_mean:.1f} 度で、脚をたたんで前に出す動きはよくできています。"
            "あとは、着地のときに膝を真っすぐに伸ばし切らないようにして、"
            "『軽く曲げた状態で地面を押す』感覚が身につくと、ブレーキが少なくなります。"
        )

    if not math.isnan(hip_mean):
        lines.append(
            f"股関節の動き（もも上げ）は平均 {hip_mean:.1f} 度で、"
            "ももをしっかり前に出せています。"
            "骨盤やお腹まわりを安定させたまま、同じもも上げをくり返せるようになると、"
            "長い距離でもフォームが崩れにくくなります。"
        )

    if not math.isnan(com_range):
        lines.append(
            f"体の真ん中（重心）の上下のゆれは、だいたい {com_range:.3f} m くらいです。"
            "少しゆれはありますが、今のうちから体幹やおしりまわりを鍛えていくと、"
            "もっとスムーズに前へ進めるようになります。"
        )

    text = " ".join(lines)
    return force_wrap_for_pdf(text, max_chars=34)


def generate_training_menu_rule(
    metrics: Dict[str, float],
    angles: Dict[str, float],
) -> List[str]:
    """
    自宅・学校でできる軽め〜中程度のトレーニング 5 個
    （中学生でも実施できる想定）
    """
    pitch = metrics.get("pitch", float("nan"))
    stab = metrics.get("stability", float("nan"))
    com_range = metrics.get("com_y_range", float("nan"))
    hip_mean = angles.get("hip_flex_mean", float("nan"))
    knee_mean = angles.get("knee_flex_mean", float("nan"))

    drills: List[str] = []

    # 1. 体幹・バランス
    if math.isnan(stab) or stab < 9.0:
        drills.append(
            "① フロントプランク（20〜30秒×3セット）\n"
            "ひじとつま先で体を支え、頭からかかとまでをまっすぐにキープします。"
            "腰が落ちたりおしりが上がりすぎないように、鏡や家族にチェックしてもらうとなお良いです。"
        )
    else:
        drills.append(
            "① 片足バランス＋腕振り（左右20秒×2セット）\n"
            "片足立ちになり、軽く前傾しながらスプリントの腕振りを行います。"
            "体がぐらつかないように、お腹に力を入れてキープしましょう。"
        )

    # 2. もも上げ・股関節
    if math.isnan(hip_mean) or hip_mean < 120:
        drills.append(
            "② 壁もたれもも上げ（左右15回×2セット）\n"
            "壁に軽く背中をつけて立ち、片脚ずつももを引き上げます。"
            "腰を反らさず、おへそを少しへこませるイメージで行うと、股関節から脚を上げる感覚がつかめます。"
        )
    else:
        drills.append(
            "② リズムハイニー（20m×3本 or その場で30回×2セット）\n"
            "腕振りと合わせて、テンポよくももを上げます。"
            "膝だけでなく、股関節から脚を素早く引き上げることを意識しましょう。"
        )

    # 3. 膝の安定性
    if math.isnan(knee_mean) or knee_mean < 140:
        drills.append(
            "③ スローランジ（左右10回×2セット）\n"
            "前後に大きく一歩を出し、ゆっくり腰を落とします。"
            "前脚の膝が内側に入らないように注意しながら行うことで、膝まわりの安定性が高まります。"
        )
    else:
        drills.append(
            "③ スキップランジ（左右8回×2セット）\n"
            "ランジの姿勢から前脚で地面を押し、軽くスキップするように切り替えます。"
            "地面を『トンッ』と優しく押して前へ進む感覚を身につけましょう。"
        )

    # 4. ピッチ（足の回転）
    if math.isnan(pitch) or pitch < 4.0:
        drills.append(
            "④ ラインステップ（10秒×3セット）\n"
            "地面にテープや線を1本引き、その上をできるだけ速く足踏みします。"
            "足を高く上げすぎず、素早く接地してすぐ離れることを意識します。"
        )
    else:
        drills.append(
            "④ 20mハイテンポ走（20m×4本）\n"
            "タイムよりも『同じテンポで最後まで走り切る』ことを重視します。"
            "スタートからゴールまでピッチを落とさないように意識しましょう。"
        )

    # 5. 重心の上下動
    if not math.isnan(com_range) and com_range > 0.03:
        drills.append(
            "⑤ ミニハードルもどき走（10m×3本）\n"
            "ペットボトルやタオルを等間隔に並べ、それを軽くまたぎながら走ります。"
            "頭の高さをあまり上下させないように、腰の位置を一定に保って走るイメージを持ちましょう。"
        )
    else:
        drills.append(
            "⑤ 接地感覚ドリル（その場で30回×2セット）\n"
            "その場で軽く足踏みしながら、かかとからではなく足の前の方（母指球付近）で"
            "『トンッ』と素早く接地してすぐ離れる感覚を確認します。"
        )

    drills = drills[:5]
    return [force_wrap_for_pdf(d, max_chars=34) for d in drills]


def generate_ai_comments_pro(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str,
) -> str:
    """
    専門向けコメント：
    1. まず OpenAI（あれば）で生成
    2. 失敗したらルールベース
    """
    # OpenAI プロンプト
    system_prompt = (
        "あなたは陸上競技（短距離・走り動作）の専門コーチです。"
        "与えられた数値指標（ピッチ・ストライド・速度・重心上下動・膝/股関節/体幹角度など）から、"
        "技術的な分析と、明確な改善ポイントを示すコメントを日本語で作成してください。"
        "口調は、実際に選手やコーチに説明するときの落ち着いた指導口調にしてください。"
        "対象は専門コーチ向けなので、用語はある程度専門的で構いません。"
        "ただし1つ1つの改善ポイントが分かりやすいよう、文のまとまりごとに『何が良いのか／何を直したいのか』をはっきりさせてください。"
    )

    user_prompt = (
        f"選手名: {athlete}\n"
        f"metrics: pitch_hz={metrics.get('pitch')}, stride_m={metrics.get('stride')}, "
        f"stability_score={metrics.get('stability')}, speed_mean={metrics.get('speed_mean')}, "
        f"speed_max={metrics.get('speed_max')}, com_y_range={metrics.get('com_y_range')}\n"
        f"angles: knee_mean={angles.get('knee_flex_mean')}, hip_mean={angles.get('hip_flex_mean')}, "
        f"trunk_mean={angles.get('trunk_tilt_mean')}\n"
        "これらをもとに、技術的な分析と改善ポイントを、4〜7つのまとまりで説明してください。"
        "各まとまりは2〜3文程度で、コーチが解説しているような口調でお願いします。"
    )

    ai_text = call_openai_for_comment(system_prompt, user_prompt)
    if ai_text is None:
        log("OpenAI 専門向けコメント生成に失敗 → ルールベースへフォールバック")
        return generate_ai_comments_pro_rule(metrics, angles, athlete)
    return force_wrap_for_pdf(ai_text, max_chars=36)


def generate_ai_comments_easy(
    metrics: Dict[str, float],
    angles: Dict[str, float],
    athlete: str,
) -> str:
    """
    中学生・保護者向けコメント：
    1. まず OpenAI
    2. ダメならルールベース
    """
    system_prompt = (
        "あなたは中学生の陸上選手とその保護者にフォーム結果を説明するコーチです。"
        "専門用語を使いすぎず、かみ砕いた表現で、日本語でやさしく説明してください。"
        "口調は『コーチが選手本人に話す感じ』で、上から目線ではなくサポートするトーンにしてください。"
        "良い点を1〜2個、これから直していきたい点を2〜3個、合計で8〜12行程度にまとめてください。"
    )

    user_prompt = (
        f"選手名: {athlete}\n"
        f"metrics: pitch_hz={metrics.get('pitch')}, stride_m={metrics.get('stride')}, "
        f"stability_score={metrics.get('stability')}, speed_mean={metrics.get('speed_mean')}, "
        f"speed_max={metrics.get('speed_max')}, com_y_range={metrics.get('com_y_range')}\n"
        f"angles: knee_mean={angles.get('knee_flex_mean')}, hip_mean={angles.get('hip_flex_mean')}, "
        f"trunk_mean={angles.get('trunk_tilt_mean')}\n"
        "これをもとに、中学生と保護者にも分かる言葉で、"
        "良いところと、これから一緒に直していきたいポイントを説明してください。"
    )

    ai_text = call_openai_for_comment(system_prompt, user_prompt)
    if ai_text is None:
        log("OpenAI 一般向けコメント生成に失敗 → ルールベースへフォールバック")
        return generate_ai_comments_easy_rule(metrics, angles, athlete)
    return force_wrap_for_pdf(ai_text, max_chars=34)


def generate_training_menu(
    metrics: Dict[str, float],
    angles: Dict[str, float],
) -> List[str]:
    return generate_training_menu_rule(metrics, angles)


# =========================================================
#  PDF 出力
# =========================================================

class PDFReport(FPDF):
    pass


def build_pdf(
    csv_path: str,
    video_path: str,
    athlete: str,
    out_dirs: Dict[str, str],
) -> None:
    log(f"build_pdf 開始 csv={csv_path}, video={video_path}")

    # メトリクス & 角度
    df_metrics, metrics_stats = load_metrics_csv(csv_path)
    video_id = get_video_id(video_path)
    angle_csv = ensure_angles_csv(video_path, athlete, video_id, out_dirs["angles"])
    df_angles, angle_stats = load_angles_csv(angle_csv)

    # オーバーレイ動画
    overlay_name = os.path.basename(video_path).replace(".mp4", "_pose_overlay.mp4")
    overlay_path = os.path.join("outputs", "images", overlay_name)
    pose_images = extract_pose_images_from_overlay(overlay_path, out_dirs["pose_images"])

    # グラフ
    metrics_graph_path = os.path.join(
        out_dirs["graphs"],
        f"{video_id}_metrics_{VERSION_STR}.png",
    )
    plot_metrics_graph(df_metrics, metrics_graph_path)

    angles_graph_path = os.path.join(
        out_dirs["graphs"],
        f"{video_id}_angles_{VERSION_STR}.png",
    )
    angles_graph_ok = plot_angle_graph(df_angles, angles_graph_path)

    # AIコメント
    pro_comment = generate_ai_comments_pro(metrics_stats, angle_stats, athlete)
    easy_comment = generate_ai_comments_easy(metrics_stats, angle_stats, athlete)
    training_menu = generate_training_menu(metrics_stats, angle_stats)

    # PDF開始
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # フォント
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    bold_font_path = font_path  # 同じファイルを別ファミリ名で
    try:
        pdf.add_font("JP", "", font_path, uni=True)
        pdf.add_font("JPB", "", bold_font_path, uni=True)
        log(f"✅ フォント読み込み: {font_path}")
    except Exception as e:
        log(f"⚠ フォント読み込みに失敗: {e} -> デフォルトフォント使用")
    pdf.set_font("JP", "", 12)

    # タイトル
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
        usable_width = pdf.w - pdf.l_margin - pdf.r_margin
        img_w = (usable_width - 2 * margin) / 3.0

        order = ["start", "mid", "finish"]
        for i, key in enumerate(order):
            path = pose_images.get(key)
            if not path or not os.path.exists(path):
                continue
            x = pdf.l_margin + i * (img_w + margin)
            try:
                pdf.image(path, x=x, y=y_top, w=img_w)
            except Exception as e:
                log(f"⚠ 骨格画像貼り付け失敗({key}): {e}")
        # 適当に高さを確保
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
            pdf.image(metrics_graph_path, x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)
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
            pdf.image(angles_graph_path, x=pdf.l_margin, w=pdf.w - pdf.l_margin - pdf.r_margin)
        except Exception as e:
            log(f"⚠ 角度グラフ貼り付け失敗: {e}")
            pdf.cell(0, 6, "※角度グラフ画像の貼り付けに失敗しました。", ln=1)
    else:
        pdf.cell(0, 6, "※角度グラフ画像が見つかりません。", ln=1)
    pdf.ln(3)

    # AIフォーム解析コメント（専門向け）
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（専門向け）", ln=1)
    pdf.set_font("JP", "", 9)
    write_wrapped_multicell(pdf, pro_comment, line_height=5, max_chars=36)
    pdf.ln(2)

    # AIフォーム解析コメント（中学生・保護者向け）
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "AIフォーム解析コメント（中学生・保護者向け）", ln=1)
    pdf.set_font("JP", "", 9)
    write_wrapped_multicell(pdf, easy_comment, line_height=5, max_chars=34)
    pdf.ln(2)

    # トレーニング
    pdf.set_font("JPB", "", 11)
    pdf.cell(0, 8, "おすすめトレーニング（自宅・学校でできるメニュー）", ln=1)
    pdf.set_font("JP", "", 9)
    for drill in training_menu:
        write_wrapped_multicell(pdf, drill, line_height=5, max_chars=34)
        pdf.ln(1)

    # 保存
    out_pdf = os.path.join(out_dirs["pdf"], f"{athlete}_form_report_{VERSION_STR}.pdf")
    pdf.output(out_pdf)
    log(f"✅ PDF出力完了: {out_pdf}")


# =========================================================
#  メイン
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"KJAC フォーム分析レポート {VERSION_STR}（高機能ラボ版）"
    )
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス")
    parser.add_argument("--athlete", required=True, help="選手名（フォルダ名にも使用）")
    args = parser.parse_args()

    video_id = get_video_id(args.video)
    out_dirs = get_root_output_dirs(args.athlete, video_id)

    # ログファイル準備
    global LOG_FILE
    log_name = f"{VERSION_STR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG_FILE = os.path.join(out_dirs["logs"], log_name)
    log(f"=== KJAC {VERSION_STR} START athlete={args.athlete}, video_id={video_id} ===")
    log(f"入力: video={args.video}, csv={args.csv}")
    log(f"ログファイル: {LOG_FILE}")

    try:
        build_pdf(args.csv, args.video, args.athlete, out_dirs)
        log(f"=== KJAC {VERSION_STR} END (success) ===")
    except Exception as e:
        log(f"⚠ 予期せぬエラー発生: {e}")
        log(f"=== KJAC {VERSION_STR} END (failed) ===")
        # スタックトレースもコンソールに
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




# ============================================================
#   KJAC フォーム分析レポート（完全統合版）
#   Version: v5.5.3_gpt_all_final
#   ファイル名例: scripts/pose_reporter_pdf_ai_v5_5_3_gpt_all_final.py
# ============================================================

import os
import sys
import json
import math
import shutil
import argparse
import subprocess
import re
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fpdf import FPDF
from fpdf.errors import FPDFException
from fpdf.enums import XPos, YPos

from openai import OpenAI

# ============================================================
#  グローバル設定
# ============================================================

VERSION_STR = "v5.5.3_gpt_all_final"
LOG_FILE: Optional[str] = None

# OpenAI クライアント（環境変数 OPENAI_API_KEY を使用）
try:
    openai_client = OpenAI()
except Exception:
    openai_client = None


# ============================================================
#  ログユーティリティ
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """コンソール & ログファイル同時出力"""
    global LOG_FILE
    line = f"[{now_str()}] {msg}"
    print(line)

    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ============================================================
#  基本ユーティリティ
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_video_id(video_path: str) -> str:
    """
    ex: 二村遥香10.5.mp4 → 二村遥香10_5
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    vid = stem.replace(".", "_")
    log(f"video_id 解析: base={base} -> video_id={vid}")
    return vid


def get_root_output_dirs(athlete: str, video_id: str) -> Dict[str, str]:
    base = os.path.join("outputs", athlete, video_id)

    mapping = {
        "base": base,
        "pdf": os.path.join(base, "pdf"),
        "graphs": os.path.join(base, "graphs"),
        "pose_images": os.path.join(base, "pose_images"),
        "angles": os.path.join(base, "angles"),
        "logs": os.path.join(base, "logs"),
        "overlay": os.path.join(base, "overlay"),
    }

    for d in mapping.values():
        ensure_dir(d)

    log(f"出力ルート: {base}")
    return mapping


# ============================================================
#  metrics CSV Loader / Summary
# ============================================================

def load_metrics_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """metrics CSV を読み込み、サマリ統計量を生成する"""
    log(f"metrics CSV 読み込み開始: {csv_path}")
    df = pd.read_csv(csv_path)
    log(f"metrics CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    first_col = df.columns[0]
    summary_mask = df[first_col].astype(str) == "_summary_"
    has_summary = summary_mask.any()

    if has_summary:
        log("metrics: '_summary_' 行あり → そこをサマリとして使用")
        summary_row = df[summary_mask].iloc[0]
        df_data = df[~summary_mask].copy()
    else:
        log("metrics: '_summary_' 行なし → 平均値からサマリ推定")
        summary_row = None
        df_data = df.copy()

    def pick(col):
        if col not in df.columns:
            return float("nan")
        return float(summary_row[col] if has_summary else df_data[col].mean())

    stats["pitch"] = pick("pitch_hz")
    stats["stride"] = pick("stride_m")
    stats["stability"] = pick("stability_score")

    log(f"metrics pitch_hz = {stats['pitch']}")
    log(f"metrics stride_m = {stats['stride']}")
    log(f"metrics stability_score = {stats['stability']}")

    if "speed_mps" in df.columns:
        stats["speed_mean"] = float(df_data["speed_mps"].mean())
        stats["speed_max"] = float(df_data["speed_mps"].max())
    else:
        stats["speed_mean"] = stats["speed_max"] = float("nan")

    if "COM_y_m" in df.columns:
        stats["com_y_range"] = float(df_data["COM_y_m"].max() - df_data["COM_y_m"].min())
    else:
        stats["com_y_range"] = float("nan")

    if "torso_tilt_deg" in df.columns:
        stats["torso_tilt_mean"] = float(df_data["torso_tilt_deg"].mean())
        stats["torso_tilt_var"] = float(df_data["torso_tilt_deg"].var())
    else:
        stats["torso_tilt_mean"] = float("nan")
        stats["torso_tilt_var"] = float("nan")

    try:
        if "COM_y_m" in df_data.columns and "COM_y_px" in df_data.columns:
            ratio = df_data["COM_y_m"] / df_data["COM_y_px"].replace(0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            stats["m_per_px"] = float(ratio.median()) if len(ratio) else float("nan")
        else:
            stats["m_per_px"] = float("nan")
    except Exception:
        stats["m_per_px"] = float("nan")

    log(f"metrics m_per_px 推定={stats['m_per_px']}")
    return df_data, stats


# ============================================================
#  angles CSV Loader
# ============================================================

def ensure_angles_csv(video_path: str,
                      athlete: str,
                      video_id: str,
                      angles_dir: str) -> Optional[str]:
    """角度CSVの探索（新→旧→自動解析）"""
    target = os.path.join(angles_dir, f"{video_id}.csv")
    log(f"角度CSV探索開始 target={target}")

    if os.path.exists(target):
        log(f"角度CSV(新形式)発見: {target}")
        return target

    old1 = os.path.join("outputs", "angles", athlete, f"{video_id}.csv")
    if os.path.exists(old1):
        ensure_dir(angles_dir)
        shutil.copy2(old1, target)
        log(f"角度CSV(旧1)コピー: {old1} -> {target}")
        return target

    stem = os.path.splitext(os.path.basename(video_path))[0]
    old2 = os.path.join("outputs", f"angles_{stem}.csv")
    if os.path.exists(old2):
        ensure_dir(angles_dir)
        shutil.copy2(old2, target)
        log(f"角度CSV(旧2)コピー: {old2} -> {target}")
        return target

    script = os.path.join("scripts", "pose_angle_analyzer_v1_0.py")
    if os.path.exists(script):
        log("角度CSVなし → 自動解析開始")
        try:
            subprocess.run(
                ["python", script, "--video", video_path],
                check=True
            )
        except Exception as e:
            log(f"角度解析失敗: {e}")
            return None

        if os.path.exists(old2):
            ensure_dir(angles_dir)
            shutil.copy2(old2, target)
            log(f"自動解析結果コピー: {old2} -> {target}")
            return target

    log("角度CSV見つからず")
    return None


def load_angles_csv(angle_csv: Optional[str]):
    """角度CSV読み込み & 要約計算"""
    if angle_csv is None or not os.path.exists(angle_csv):
        log("⚠ 角度CSVがありません")
        return None, {}

    df = pd.read_csv(angle_csv)
    log(f"angles CSV shape={df.shape}, columns={list(df.columns)}")

    stats: Dict[str, float] = {}

    def comp(cols):
        if not cols:
            return (float("nan"), float("nan"))
        s = df[cols].mean(axis=1)
        return float(s.mean()), float(s.max() - s.min())

    knees = [c for c in df.columns if "knee" in c.lower()]
    hips = [c for c in df.columns if "hip" in c.lower()]
    trunks = [c for c in df.columns if "trunk" in c.lower() or "torso_tilt" in c.lower()]

    stats["knee_flex_mean"], stats["knee_flex_range"] = comp(knees)
    stats["hip_flex_mean"], stats["hip_flex_range"] = comp(hips)
    stats["trunk_tilt_mean"], stats["trunk_tilt_range"] = comp(trunks)

    log(f"angles stats: {stats}")
    return df, stats


# ============================================================
#  Graph Generators
# ============================================================

def plot_metrics_graph(df: pd.DataFrame, out_path: str):
    """速度・体幹角度・COM高さの時間推移グラフ"""
    log(f"メトリクスグラフ生成: {out_path}")
    ensure_dir(os.path.dirname(out_path))

    t = df["time_s"].values if "time_s" in df.columns else df["frame"].values

    speed = df["speed_mps"].values if "speed_mps" in df.columns else None
    torso = df["torso_tilt_deg"].values if "torso_tilt_deg" in df.columns else None
    com = df["COM_y_m"].values if "COM_y_m" in df.columns else None

    plt.figure(figsize=(6, 8))

    ax1 = plt.subplot(3, 1, 1)
    if speed is not None:
        ax1.plot(t, speed)
    ax1.set_title("Speed (m/s)")

    ax2 = plt.subplot(3, 1, 2)
    if torso is not None:
        ax2.plot(t, torso)
    ax2.set_title("Torso tilt (deg)")

    ax3 = plt.subplot(3, 1, 3)
    if com is not None:
        ax3.plot(t, com)
    ax3.set_title("COM height (m)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_angle_graph(df_angles: Optional[pd.DataFrame], out_path: str) -> bool:
    """膝・股関節・体幹角度グラフ"""
    if df_angles is None:
        return False

    ensure_dir(os.path.dirname(out_path))

    t = df_angles["time_s"].values if "time_s" in df_angles.columns else df_angles["frame"].values

    knees = [c for c in df_angles.columns if "knee" in c.lower()]
    hips = [c for c in df_angles.columns if "hip" in c.lower()]
    trunks = [c for c in df_angles.columns if "trunk" in c.lower() or "torso_tilt" in c.lower()]

    plt.figure(figsize=(6, 8))

    ax1 = plt.subplot(3, 1, 1)
    for c in knees:
        ax1.plot(t, df_angles[c], label=c)
    ax1.set_title("Knee angles")
    if knees:
        ax1.legend(fontsize=6)

    ax2 = plt.subplot(3, 1, 2)
    for c in hips:
        ax2.plot(t, df_angles[c], label=c)
    ax2.set_title("Hip angles")
    if hips:
        ax2.legend(fontsize=6)

    ax3 = plt.subplot(3, 1, 3)
    for c in trunks:
        ax3.plot(t, df_angles[c], label=c)
    ax3.set_title("Trunk tilt angles")
    if trunks:
        ax3.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return True


# ============================================================
#  Overlay 画像抽出（単発 & 連続）
# ============================================================

def extract_pose_images_from_overlay(overlay_path: str, out_dir: str) -> Dict[str, str]:
    """overlay動画から start / mid / finish を抽出して画像化"""
    result: Dict[str, str] = {}
    ensure_dir(out_dir)

    if not os.path.exists(overlay_path):
        log("⚠ オーバーレイ動画が存在しません")
        return result

    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        log("⚠ オーバーレイ動画を開けません")
        return result

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"オーバーレイ frame_count={frame_count}")

    indices = {
        "start": 0,
        "mid": frame_count // 2,
        "finish": max(frame_count - 1, 0),
    }

    for key, idx in indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(out_dir, f"{key}.png")
        cv2.imwrite(path, frame)
        result[key] = path
        log(f"骨格画像保存成功: {key} -> {path}")

    cap.release()
    log(f"骨格画像出力結果: {result}")
    return result


def export_overlay_frames(overlay_path: str,
                          out_dir: str,
                          step_sec: float = 0.1) -> int:
    """overlay動画を 0.1 秒ごとに連続画像として保存"""
    ensure_dir(out_dir)

    if not os.path.exists(overlay_path):
        log("⚠ overlay 動画見つからず")
        return 0

    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        log("⚠ overlay 動画を開けない")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_step = max(int(fps * step_sec), 1)
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            path = os.path.join(out_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(path, frame)
            saved += 1
        idx += 1

    cap.release()
    log(f"overlay連続フレーム出力: {saved}枚 -> {out_dir}")
    return saved


# ============================================================
#  文字列ユーティリティ（日本語折り返し）
# ============================================================

def jp_wrap(text: str, max_chars: int = 38) -> str:
    """
    日本語テキストを「文字数ベース」でざっくり折り返す。
    単語境界はあまり気にせず、PDF 1 行あたりの文字数を制御する目的。
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []

    for para in text.split("\n"):
        p = para.strip()
        if not p:
            lines.append("")
            continue

        start = 0
        length = len(p)
        while start < length:
            end = start + max_chars
            lines.append(p[start:end])
            start = end

    return "\n".join(lines)


# ============================================================
#  OpenAI Chat ユーティリティ
# ============================================================

def call_openai_chat(model: str,
                     system_prompt: str,
                     user_prompt: str,
                     max_tokens: int = 900,
                     temperature: float = 0.6) -> str:
    """OpenAI Chat API を叩いてテキストを返す（失敗時は空文字）"""
    if openai_client is None:
        log("⚠ OpenAI クライアント未初期化（APIキー未設定？）")
        return ""

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        msg = resp.choices[0].message
        content = msg.content
        if isinstance(content, str):
            text = content
        else:
            # tool 呼び出しなどのケースは少ない想定だが一応
            text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        log(f"OpenAI API 応答長={len(text)}")
        return text.strip()
    except Exception as e:
        log(f"⚠ OpenAI API 呼び出しで例外発生: {e}")
        return ""


# ============================================================
#  GPT 用入力整形
# ============================================================

def build_gpt_inputs(metrics: Dict[str, float],
                     angles: Dict[str, float],
                     athlete: str) -> str:
    """
    GPT プロンプト向けに、metrics/angles をわかりやすく文字列化
    """
    lines: List[str] = []
    lines.append(f"選手名: {athlete}")
    lines.append("")
    lines.append("【基本指標（metrics）】")
    lines.append(json.dumps(metrics, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("【関節角度・体幹（angles）】")
    lines.append(json.dumps(angles, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("※数値はあくまで参考値です。誤差や計測条件の影響もありえます。")
    return "\n".join(lines)


# ============================================================
#  ルールベース fallback コメント
# ============================================================

def generate_ai_comments_pro_rule(metrics: Dict[str, float],
                                  angles: Dict[str, float],
                                  athlete: str) -> str:
    """GPT 失敗時の専門向けコメント（簡易版）"""
    pitch = metrics.get("pitch", float("nan"))
    stride = metrics.get("stride", float("nan"))
    speed_mean = metrics.get("speed_mean", float("nan"))
    com_y = metrics.get("com_y_range", float("nan"))
    torso = metrics.get("torso_tilt_mean", float("nan"))

    knee = angles.get("knee_flex_mean", float("nan"))
    hip = angles.get("hip_flex_mean", float("nan"))
    trunk = angles.get("trunk_tilt_mean", float("nan"))

    lines: List[str] = []
    lines.append(f"{athlete} さんの疾走フォームを、数値データに基づいて概観します。")
    lines.append("")
    lines.append("【良い点】")
    if not math.isnan(pitch):
        lines.append(f"- ピッチはおよそ {pitch:.2f} 歩/秒で、一定のリズムを保てています。")
    if not math.isnan(stride):
        lines.append(f"- ストライドはおよそ {stride:.2f} m で、身長や年齢を考えると悪くない長さです。")
    if not math.isnan(speed_mean):
        lines.append(f"- 平均速度は {speed_mean:.2f} m/s 程度で、一定以上のスピードで走れています。")

    lines.append("")
    lines.append("【改善したい点】")
    if not math.isnan(com_y):
        lines.append(f"- 重心の上下動は約 {com_y:.3f} m 程度あり、もう少し抑えられる余地があります。")
    if not math.isnan(torso):
        lines.append(f"- 体幹前傾は平均 {torso:.1f}° 付近で、局面によってはやや深すぎる可能性があります。")
    if not math.isnan(knee):
        lines.append(f"- 膝関節角度の平均は {knee:.1f}° 付近で、接地前後のたたみ（リカバリー）に改善の余地があります。")
    if not math.isnan(hip):
        lines.append(f"- 股関節角度の平均は {hip:.1f}° 付近で、引き上げのタイミング・可動域の使い方を整理したいところです。")

    lines.append("")
    lines.append("【意識してほしいキーポイント】")
    lines.append("1. 重心を上下に弾ませすぎず、前方への推進を優先する。")
    lines.append("2. 体幹を固めすぎず、骨盤からしなやかに脚を振り出す。")
    lines.append("3. 接地のタイミングと位置（体の真下付近）をそろえ、ブレーキ接地を減らす。")

    return jp_wrap("\n".join(lines), max_chars=38)


def generate_ai_comments_easy_rule(metrics: Dict[str, float],
                                   angles: Dict[str, float],
                                   athlete: str) -> str:
    """GPT 失敗時の一般向けコメント（簡易版）"""
    pitch = metrics.get("pitch", float("nan"))
    stride = metrics.get("stride", float("nan"))

    lines: List[str] = []
    lines.append("【今の走りの良いところ】")
    lines.append(f"{athlete} さんは、全体としてリズムよく走れているのが大きな長所です。")
    if not math.isnan(pitch):
        lines.append(f"ピッチ（1秒間の歩数）はおよそ {pitch:.2f} 歩/秒で、安定したテンポで走れています。")
    if not math.isnan(stride):
        lines.append(f"ストライド（1歩の距離）はおよそ {stride:.2f} m で、脚の伸びもしっかり使えています。")

    lines.append("")
    lines.append("【これから意識するとよいポイント】")
    lines.append("・体が上下にポンポン弾みすぎないように、地面を押す方向を『前へ』意識してみましょう。")
    lines.append("・お腹まわり（体幹）を少しだけ意識して、上半身がグラグラしないように走ると、ブレが減ってスピードが出やすくなります。")
    lines.append("・足を前に投げ出すよりも、体の真下あたりで接地するイメージを持つと、ブレーキがかかりにくくなります。")

    lines.append("")
    lines.append("【前向きになれる一言】")
    lines.append("今の走りにはすでに良いところがたくさんあります。")
    lines.append("今日のポイントを少しずつ意識していけば、タイムは着実に伸びていきます。焦らず、一歩ずつ積み重ねていきましょう。")

    return jp_wrap("\n".join(lines), max_chars=34)


def generate_training_menu_rule(metrics: Dict[str, float],
                                angles: Dict[str, float]) -> List[str]:
    """GPT 失敗時のトレーニングメニュー（簡易版5個）"""
    drills: List[str] = []

    drills.append(
        jp_wrap(
            "① もも上げドリル（その場）\n"
            "・姿勢をまっすぐに保ち、その場でももを交互に高く引き上げます。腕振りも合わせて行いましょう。\n"
            "・30秒×3セット。ピッチと股関節の引き上げ感覚を養います。",
            max_chars=34
        )
    )
    drills.append(
        jp_wrap(
            "② ランジウォーク\n"
            "・一歩ずつ前に大きく踏み出し、膝がつま先より前に出ないようにして腰を落とします。\n"
            "・左右10歩×2セット。股関節の可動域と安定した接地感覚に効きます。",
            max_chars=34
        )
    )
    drills.append(
        jp_wrap(
            "③ 体幹プランク\n"
            "・肘とつま先で体を支え、一直線の姿勢を30秒キープします。\n"
            "・30秒×3セット。体幹の安定を高め、上半身のブレを減らします。",
            max_chars=34
        )
    )
    drills.append(
        jp_wrap(
            "④ かかとタッチ走\n"
            "・ややゆっくりめに走りながら、接地のたびに『体の真下で着く』イメージを意識します。\n"
            "・30〜50mを数本。ブレーキの少ない接地感覚を身につけます。",
            max_chars=34
        )
    )
    drills.append(
        jp_wrap(
            "⑤ スキップ（リズム重視）\n"
            "・軽快なリズムでスキップし、腕振りと脚の振り上げを連動させます。\n"
            "・20〜30m×3本。ピッチと反発を活かした走りに繋がります。",
            max_chars=34
        )
    )

    return drills


# ============================================================
#  GPT コメント生成（プロ / 一般 / ドリル）
# ============================================================

def generate_ai_comments_pro_gpt(metrics: Dict[str, float],
                                 angles: Dict[str, float],
                                 athlete: str) -> str:
    """
    専門コーチ向けコメント（JAAF講習会レベル＋現場コーチ口調）。
    数値を必ず根拠として使い、技術的に深く解説する。
    """
    system_prompt = (
        "あなたは日本陸連公認コーチレベルの陸上競技指導者です。"
        "短距離〜中長距離の疾走フォームを、数値データと骨格情報から専門的に分析します。"
        "現場で選手に声かけをするときのような、自然で落ち着いた口調で説明してください。"
        "ただし内容は技術的に深く、フォームの改善ポイントを明確に示します。"
        "ピッチ、ストライド、速度、重心変動、体幹の傾き、膝・股関節角度など、"
        "与えられた数値を必ず根拠として引用してください。"
        "『どの局面で何が起きているか』『その結果どんな現象が起こるか』まで踏み込んで解説します。"
        "出力は日本語で、全体でおおよそ800〜900文字に収めてください。"
    )

    user_prompt = (
        "以下の疾走フォームデータをもとに、技術分析中心のコメントを書いてください。\n"
        "口調は、現場のコーチが選手と指導者ミーティングで話すイメージです。\n\n"
        "必ず、次の3つのパートに分けてください：\n"
        "【良い点】\n"
        "・ピッチ・ストライド・体幹・接地など、現状で評価できるポイントを具体的に。\n"
        "【改善したい点】\n"
        "・特に修正したい局面（加速〜中間疾走など）と、どの数値がそれを示しているかを説明。\n"
        "【意識してほしい3つのポイント】\n"
        "・箇条書きで3つ。練習やレース中に意識してほしいキーワードにまとめてください。\n\n"
        "【重要】\n"
        "・ピッチ（歩/秒）、ストライド（m）、速度（m/s）、重心上下動、体幹角度、膝・股関節角度など、\n"
        "  データにある数値は、できるだけ文章の中で具体的に引用してください。\n"
        "  例：『ピッチは4.7歩/秒と高く、前半の加速局面では〜』『体幹前傾は平均-38°とやや深めで〜』など。\n"
        "・単に誉める／注意するだけでなく、『なぜそうなるか』『どう直すか』まで書いてください。\n\n"
        "【データ】\n" + build_gpt_inputs(metrics, angles, athlete)
    )

    text = call_openai_chat(
        model="gpt-4.1",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=900,
    )

    if not text:
        log("GPT プロ向けコメント生成に失敗 → ルールベースへフォールバック")
        return generate_ai_comments_pro_rule(metrics, angles, athlete)

    return jp_wrap(text, max_chars=38)


def generate_ai_comments_easy_gpt(metrics: Dict[str, float],
                                  angles: Dict[str, float],
                                  athlete: str) -> str:
    """
    中学生・保護者向けコメント。
    少し専門用語も使うが、かならず噛み砕いて説明する。
    """
    system_prompt = (
        "あなたは中学生の陸上部を指導する、優しくて説明が上手なコーチです。"
        "フォームのデータをもとに、中学生とその保護者にも伝わる言葉で、走りの良いところと今後の課題を説明します。"
        "難しい専門用語を使うときは、必ずその場で簡単に言い換えてください。"
        "全体のトーンは前向きで、『これからこうしていけばもっと良くなる』というメッセージにしてください。"
        "出力は日本語で、全体で600〜800文字程度にしてください。"
    )

    user_prompt = (
        "以下の疾走フォームデータをもとに、中学生・保護者向けのコメントを書いてください。\n\n"
        "構成は必ず次の3つにしてください：\n"
        "【今の走りの良いところ】\n"
        "・努力が伝わるポイントや、フォームの強みをわかりやすく説明。\n"
        "【これから練習で意識するとよいポイント】\n"
        "・2〜4個程度。『腕を◯◯する』『足のつき方を◯◯する』のように、行動レベルの言葉で。\n"
        "・ピッチやストライドなどの数値に少し触れてもOKですが、かならず噛み砕いて説明してください。\n"
        "【前向きになれる一言メッセージ】\n"
        "・最後に、次の練習や大会が楽しみになるような一言を。『〜していけば必ずタイムは上がります』など。\n\n"
        "できれば、データの数値（ピッチ・ストライド・体幹前傾など）も一部紹介しつつ、\n"
        "『だから今の走りはこう見える』『こうするともっとよくなる』というつなぎ方をしてください。\n\n"
        "【データ】\n" + build_gpt_inputs(metrics, angles, athlete)
    )

    text = call_openai_chat(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=700,
    )

    if not text:
        log("GPT 一般向けコメント生成に失敗 → ルールベースへフォールバック")
        return generate_ai_comments_easy_rule(metrics, angles, athlete)

    return jp_wrap(text, max_chars=34)


def generate_training_menu_gpt(metrics: Dict[str, float],
                               angles: Dict[str, float]) -> List[str]:
    """
    フォーム改善に効く中学生向けドリルを5個生成。
    1ドリル＝タイトル＋説明（2〜3行）＋「どこに効くか」1行。
    """
    system_prompt = (
        "あなたは中学生向けの陸上コーチです。"
        "自宅や学校でできる簡単なトレーニングメニューを考えてください。"
        "器具は基本的に使わず、あってもペットボトルやタオル程度にしてください。"
        "安全面にも配慮し、『やり過ぎない』『痛みが出たら中止』などの注意も必要に応じて入れてください。"
    )

    user_prompt = (
        "以下の疾走フォームデータをもとに、フォーム改善に役立つトレーニングメニューを5個考えてください。\n\n"
        "【出力フォーマット】\n"
        "① ドリル名\n"
        "　・やり方の説明（2〜3行程度）\n"
        "　・「どこに効くか」（例：『股関節の引き上げ』『体幹の安定』など）\n"
        "\n"
        "② 〜 ⑤ も同様の形式で書いてください。\n\n"
        "【条件】\n"
        "・難易度は中学生が無理なくできるレベル。\n"
        "・主に、データ上の弱点（ピッチ不足・ストライド不足・体幹のブレ・接地時間など）を改善する内容にしてください。\n"
        "・具体的な回数や時間の目安（例：『10回×2セット』など）も入れてください。\n\n"
        "【データ】\n" + build_gpt_inputs(metrics, angles, athlete="(anonymous)")
    )

    text = call_openai_chat(
        model="gpt-4.1-mini",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=700,
    )

    if not text:
        log("GPT トレーニングメニュー生成に失敗 → ルールベースへフォールバック")
        return generate_training_menu_rule(metrics, angles)

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    if len(blocks) < 5:
        tmp: List[str] = []
        for b in blocks:
            sub = re.split(r"\n(?=[①-⑤1-5][\.\)\s　])", b)
            for s in sub:
                s = s.strip()
                if s:
                    tmp.append(s)
        blocks = tmp

    if len(blocks) >= 5:
        blocks = blocks[:5]
    else:
        rb = generate_training_menu_rule(metrics, angles)
        for d in rb:
            if len(blocks) >= 5:
                break
            blocks.append(d)

    return [jp_wrap(b, max_chars=34) for b in blocks]


def generate_comments_with_gpt(metrics_stats: Dict[str, float],
                               angle_stats: Dict[str, float],
                               athlete: str,
                               video_id: str):
    """
    専門向けコメント / 一般向けコメント / トレーニングメニュー をまとめて生成
    """
    log("AIコメント生成（ローカルルールベース＋GPT）開始")

    # 専門向け
    pro_comment = generate_ai_comments_pro_gpt(metrics_stats, angle_stats, athlete)

    # 一般向け
    easy_comment = generate_ai_comments_easy_gpt(metrics_stats, angle_stats, athlete)

    # トレーニング
    training_menu = generate_training_menu_gpt(metrics_stats, angle_stats)

    return pro_comment, easy_comment, training_menu


# ============================================================
#  PDF 出力
# ============================================================

class PDFReport(FPDF):
    pass


def safe_multi_cell(pdf: PDFReport,
                    text: str,
                    line_height: float = 5.0,
                    max_chars: int = 38):
    """
    FPDF の multi_cell を安全に使うラッパー。
    1 行あたり max_chars 文字でテキストを分割しつつ、
    FPDFException が出たときは段階的に縮めて回避する。
    """
    if not text:
        return

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")

    cleaned: List[str] = []
    for ch in text:
        if ch == "\n":
            cleaned.append(ch)
        elif ord(ch) >= 32:
            cleaned.append(ch)
    text = "".join(cleaned)

    paragraphs = text.split("\n")

    for para in paragraphs:
        s = para.strip()
        if not s:
            pdf.ln(line_height)
            continue

        idx = 0
        length = len(s)

        while idx < length:
            segment = s[idx: idx + max_chars]
            if not segment:
                break

            attempt = 0
            written = False
            while attempt < 3 and segment:
                try:
                    pdf.multi_cell(
                        pdf.epw,
                        line_height,
                        segment,
                        new_x="LMARGIN",
                        new_y="NEXT",
                    )
                    written = True
                    break
                except Exception:
                    attempt += 1
                    segment = segment[:-1]

            if not written:
                idx += 1
            else:
                idx += len(segment)

        pdf.ln(1)


def build_pdf(csv_path: str,
              video_path: str,
              athlete: str,
              out_dirs: Dict[str, str]) -> None:

    log(f"build_pdf 開始 csv={csv_path}, video={video_path}")

    df_metrics, metrics_stats = load_metrics_csv(csv_path)

    video_id = get_video_id(video_path)
    angle_csv = ensure_angles_csv(video_path, athlete, video_id, out_dirs["angles"])
    df_angles, angle_stats = load_angles_csv(angle_csv)

    # torso_tilt_deg の注入（安全版）
    try:
        if df_angles is not None:
            df_metrics["frame"] = pd.to_numeric(df_metrics["frame"], errors="coerce").astype("Int64")
            df_angles["frame"] = pd.to_numeric(df_angles["frame"], errors="coerce").astype("Int64")

            df_m = df_metrics.dropna(subset=["frame"]).copy()
            df_a = df_angles.dropna(subset=["frame"]).copy()

            if "torso_tilt_left" in df_a.columns and "torso_tilt_right" in df_a.columns:
                df_a["torso_tilt_deg"] = df_a[["torso_tilt_left", "torso_tilt_right"]].mean(axis=1)
                df_torso = df_a[["frame", "torso_tilt_deg"]].copy()
                df_merged = pd.merge(df_m, df_torso, on="frame", how="left")

                df_metrics = df_merged

                metrics_stats["torso_tilt_mean"] = float(df_metrics["torso_tilt_deg"].mean(skipna=True))
                metrics_stats["torso_tilt_var"] = float(df_metrics["torso_tilt_deg"].var(skipna=True))

                log("torso_tilt_deg を metrics に注入しました。")
    except Exception as e:
        log(f"⚠ torso_tilt_deg の注入に失敗: {e}")

    # overlay → pose_images & 連続フレーム
    overlay_name = os.path.basename(video_path).replace(".mp4", "_pose_overlay.mp4")
    overlay_path = os.path.join("outputs", "images", overlay_name)

    pose_images = extract_pose_images_from_overlay(overlay_path, out_dirs["pose_images"])
    export_overlay_frames(overlay_path, out_dirs["overlay"], step_sec=0.1)

    # グラフ生成
    metrics_graph_path = os.path.join(out_dirs["graphs"], f"{video_id}_metrics_{VERSION_STR}.png")
    angles_graph_path = os.path.join(out_dirs["graphs"], f"{video_id}_angles_{VERSION_STR}.png")

    plot_metrics_graph(df_metrics, metrics_graph_path)
    angles_graph_ok = plot_angle_graph(df_angles, angles_graph_path)

    # GPT コメント生成
    pro_comment, easy_comment, training_menu = generate_comments_with_gpt(
        metrics_stats, angle_stats, athlete, video_id
    )

    # PDF 生成
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font_path = "c:/Windows/Fonts/meiryo.ttc"
    try:
        pdf.add_font("JP", "", font_path)
        pdf.add_font("JPB", "", font_path)
        log(f"✅ フォント読み込み: {font_path}")
    except Exception as e:
        log(f"⚠ フォント読み込み失敗: {e}")
        pdf.set_font("helvetica", "", 12)

    pdf.set_font("JPB", "", 14)
    pdf.cell(
        0, 10,
        f"{athlete} フォーム分析レポート（{VERSION_STR}）",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    pdf.set_font("JP", "", 10)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"解析日時: {now_str()}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if LOG_FILE:
        pdf.cell(0, 6, f"ログファイル: {os.path.relpath(LOG_FILE)}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    mpp = metrics_stats.get("m_per_px", float("nan"))
    if not math.isnan(mpp):
        pdf.cell(0, 6, f"推定スケール: {mpp:.6f} m/px",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # 基本指標
    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 7, "基本指標（ピッチ・ストライド・安定性）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("JP", "", 10)

    def metric_line(label: str, key: str) -> str:
        v = metrics_stats.get(key, float("nan"))
        if math.isnan(v):
            return f"{label}: N/A"
        return f"{label}: {v:.2f}"

    pdf.cell(0, 6, metric_line("ピッチ", "pitch"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, metric_line("ストライド", "stride"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, metric_line("安定性スコア", "stability"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # 骨格オーバーレイ
    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 7, "骨格オーバーレイ（start / mid / finish）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if pose_images:
        y0 = pdf.get_y()
        margin = 5
        usable_w = pdf.w - 2 * pdf.l_margin
        img_w = (usable_w - 2 * margin) / 3

        for i, key in enumerate(["start", "mid", "finish"]):
            if key in pose_images:
                x = pdf.l_margin + i * (img_w + margin)
                try:
                    pdf.image(pose_images[key], x=x, y=y0, w=img_w)
                except Exception as e:
                    log(f"⚠ 骨格画像配置失敗 {key}: {e}")

        pdf.set_y(y0 + img_w * 0.75 + 8)
    else:
        pdf.set_font("JP", "", 10)
        pdf.cell(0, 6, "骨格オーバーレイ画像なし",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # メトリクスグラフ
    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 8, "速度・重心・体幹傾斜の時間変化",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if os.path.exists(metrics_graph_path):
        try:
            pdf.image(metrics_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
        except Exception as e:
            log(f"⚠ メトリクスグラフ配置失敗: {e}")
    else:
        pdf.set_font("JP", "", 10)
        pdf.cell(0, 6, "メトリクスグラフ画像なし",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # 角度グラフ
    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 8, "関節角度（膝・股関節・体幹）の時間変化",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if angles_graph_ok and os.path.exists(angles_graph_path):
        try:
            pdf.image(angles_graph_path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)
        except Exception as e:
            log(f"⚠ 角度グラフ配置失敗: {e}")
    else:
        pdf.set_font("JP", "", 10)
        pdf.cell(0, 6, "角度グラフ画像なし",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # AIコメント（専門向け）
    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 8, "AIフォーム解析コメント（専門向け）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("JP", "", 10)
    safe_multi_cell(pdf, pro_comment, line_height=5.0, max_chars=38)
    pdf.ln(3)

    # AIコメント（一般向け）
    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 8, "AIフォーム解析コメント（中学生・保護者向け）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("JP", "", 10)
    safe_multi_cell(pdf, easy_comment, line_height=5.0, max_chars=34)
    pdf.ln(3)

    # トレーニングメニュー
    pdf.set_font("JPB", "", 12)
    pdf.cell(0, 8, "おすすめトレーニング（自宅・学校でできるメニュー）",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("JP", "", 10)
    safe_multi_cell(pdf, "\n\n".join(training_menu), line_height=5.0, max_chars=34)

    out_pdf = os.path.join(out_dirs["pdf"], f"{athlete}_form_report_{VERSION_STR}.pdf")
    ensure_dir(os.path.dirname(out_pdf))
    pdf.output(out_pdf)

    log(f"✅ PDF出力完了: {out_pdf}")


# ============================================================
#  GUI から呼ぶ場合のラッパー
# ============================================================

def run_from_gui(video_path: str,
                 csv_path: str,
                 athlete: str,
                 output_base: str,
                 progress_callback=None):
    """
    GUI から呼び出されるエントリポイント。
    例外は内部で処理して True/False 返す。
    """
    def log_gui(msg: str):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass
        log(msg)

    try:
        video_id = get_video_id(video_path)

        out_dirs = {
            "base": os.path.join(output_base, athlete, video_id),
            "pdf": os.path.join(output_base, athlete, video_id, "pdf"),
            "graphs": os.path.join(output_base, athlete, video_id, "graphs"),
            "pose_images": os.path.join(output_base, athlete, video_id, "pose_images"),
            "angles": os.path.join(output_base, athlete, video_id, "angles"),
            "logs": os.path.join(output_base, athlete, video_id, "logs"),
            "overlay": os.path.join(output_base, athlete, video_id, "overlay"),
        }

        for d in out_dirs.values():
            ensure_dir(d)

        global LOG_FILE
        LOG_FILE = os.path.join(
            out_dirs["logs"],
            f"{VERSION_STR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        log_gui(f"=== GUI RUN START: {athlete} / {video_id} ===")
        log_gui(f"動画: {video_path}")
        log_gui(f"CSV : {csv_path}")
        log_gui(f"出力: {out_dirs['base']}")

        build_pdf(csv_path, video_path, athlete, out_dirs)

        log_gui("=== GUI RUN END (success) ===")
        return True, out_dirs["pdf"]

    except Exception as e:
        err_msg = f"GUI ERROR: {e}"
        log_gui(err_msg)
        log_gui("=== GUI RUN END (failed) ===")
        return False, err_msg


# ============================================================
#  CLI メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f"KJAC フォーム分析レポート {VERSION_STR}（GPT統合版）"
    )
    parser.add_argument("--video", required=True, help="入力動画パス（mp4）")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス")
    parser.add_argument("--athlete", required=True, help="選手名（例：二村遥香）")

    args = parser.parse_args()

    video_id = get_video_id(args.video)
    out_dirs = get_root_output_dirs(args.athlete, video_id)

    global LOG_FILE
    LOG_FILE = os.path.join(
        out_dirs["logs"],
        f"{VERSION_STR}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    log(f"=== KJAC START {VERSION_STR} ===")
    log(f"video    = {args.video}")
    log(f"csv      = {args.csv}")
    log(f"athlete  = {args.athlete}")
    log(f"video_id = {video_id}")

    try:
        build_pdf(args.csv, args.video, args.athlete, out_dirs)
        log("=== KJAC END (SUCCESS) ===")
    except Exception as e:
        log(f"⚠ エラー発生: {e}")
        log("=== KJAC END (FAILED) ===")
        raise


if __name__ == "__main__":
    main()

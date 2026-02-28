import os
import json
import math
import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF


# ============================================================
# 共通ユーティリティ
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sanitize_text(s: str) -> str:
    """
    FPDF でエラーを起こしやすい制御文字・絵文字などを除去。
    （Meiryo が対応していない文字も多いので、高コードポイントは落とす）
    """
    if s is None:
        return ""
    result_chars = []
    for ch in s:
        code = ord(ch)
        # 改行は許可
        if ch == "\n":
            result_chars.append(ch)
            continue
        # 通常の可視文字（ASCII〜BMPの範囲）だけ許可
        if 32 <= code < 0x1F000:
            result_chars.append(ch)
        # それ以外（多くの絵文字など）は無視
    return "".join(result_chars)


# ============================================================
# キャリブレーション読み込み
# ============================================================

def load_calibration(calib_path: str = "outputs/jsonl/calibration_result.json") -> float:
    if not os.path.exists(calib_path):
        print(f"⚠ キャリブレーションファイルが見つかりません: {calib_path}")
        return 1.0
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m_per_px = float(data.get("m_per_px", 1.0))
    print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
    return m_per_px


# ============================================================
# メトリクスCSV（速度など）解析
# ============================================================

def load_metrics(csv_path: str):
    df = pd.read_csv(csv_path)
    summary = {
        "pitch": None,
        "stride": None,
        "stability": None,
        "duration": None,
        "max_speed": None
    }

    # _summary_ 行があれば分離
    mask_summary = pd.Series(False, index=df.index)
    if "frame" in df.columns and df["frame"].dtype == object:
        mask_summary = df["frame"] == "_summary_"

    df_data = df[~mask_summary].copy()
    df_sum = df[mask_summary].copy()

    # 時間長
    if "time_s" in df_data.columns and len(df_data) > 0:
        summary["duration"] = float(df_data["time_s"].iloc[-1] - df_data["time_s"].iloc[0])
    # 速度
    if "speed_mps" in df_data.columns:
        summary["max_speed"] = float(df_data["speed_mps"].max())

    # ピッチ
    if not df_sum.empty and "pitch_hz" in df_sum.columns:
        try:
            summary["pitch"] = float(df_sum["pitch_hz"].iloc[0])
        except Exception:
            pass
    if summary["pitch"] is None and "pitch_hz" in df_data.columns:
        summary["pitch"] = float(df_data["pitch_hz"].dropna().mean()) if df_data["pitch_hz"].notna().any() else None

    # ストライド
    if not df_sum.empty and "stride_m" in df_sum.columns:
        try:
            summary["stride"] = float(df_sum["stride_m"].iloc[0])
        except Exception:
            pass
    if summary["stride"] is None and "stride_m" in df_data.columns:
        summary["stride"] = float(df_data["stride_m"].dropna().mean()) if df_data["stride_m"].notna().any() else None

    # 安定性スコア
    if not df_sum.empty and "stability_score" in df_sum.columns:
        try:
            summary["stability"] = float(df_sum["stability_score"].iloc[0])
        except Exception:
            pass
    if summary["stability"] is None and "stability_score" in df_data.columns:
        summary["stability"] = float(df_data["stability_score"].dropna().mean()) if df_data["stability_score"].notna().any() else None

    # COM高さ[m]を追加（あれば）
    if "COM_y_px" in df_data.columns:
        # COM_y_px からメートル換算は後で m_per_px を使って行う
        pass

    print(f"✅ メトリクスCSV読込: {len(df_data)} 行")
    return df_data, summary


def create_main_graph(df_metrics: pd.DataFrame, m_per_px: float, out_path: str):
    """
    速度・体幹傾き・COM高さの3つを1枚に縦3分割で描画。
    タイトル等は ASCII のみにして、日本語フォント依存を回避。
    """
    ensure_dir(os.path.dirname(out_path))

    time = df_metrics["time_s"].values if "time_s" in df_metrics.columns else np.arange(len(df_metrics))
    speed = df_metrics["speed_mps"].values if "speed_mps" in df_metrics.columns else None
    tilt = df_metrics["tilt_deg"].values if "tilt_deg" in df_metrics.columns else None

    com_y_m = None
    if "COM_y_px" in df_metrics.columns:
        com_y_m = df_metrics["COM_y_px"].values * m_per_px

    plt.figure(figsize=(6, 8))

    # 1. Speed
    ax1 = plt.subplot(3, 1, 1)
    if speed is not None:
        ax1.plot(time, speed)
    ax1.set_ylabel("Speed [m/s]")
    ax1.set_title("Speed vs Time")

    # 2. Trunk tilt
    ax2 = plt.subplot(3, 1, 2)
    if tilt is not None:
        ax2.plot(time, tilt)
    ax2.set_ylabel("Trunk tilt [deg]")
    ax2.set_title("Trunk tilt vs Time")

    # 3. COM height
    ax3 = plt.subplot(3, 1, 3)
    if com_y_m is not None:
        ax3.plot(time, com_y_m)
        ax3.set_ylabel("COM height [m]")
    else:
        ax3.plot(time, np.zeros_like(time))
        ax3.set_ylabel("COM (normalized)")
    ax3.set_xlabel("Time [s]")
    ax3.set_title("COM height vs Time")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"📊 グラフ画像出力: {out_path}")


# ============================================================
# 角度CSV解析 ＋ グラフ作成
# ============================================================

def load_angles(video_path: str):
    """
    pose_angle_analyzer_v1_0.py で保存された角度CSVを読み込み。
    ファイル名: outputs/angles_<動画ファイル名>.csv
    """
    stem = Path(video_path).name  # 拡張子込みでOK（既に angle スクリプトと合わせているため）
    angle_csv = os.path.join("outputs", f"angles_{stem}.csv")
    if not os.path.exists(angle_csv):
        raise FileNotFoundError(f"角度CSVが見つかりません: {angle_csv}")

    df = pd.read_csv(angle_csv)
    print(f"✅ 角度CSV読込: {angle_csv} ({len(df)} 行)")

    stats = {}

    def angle_stats(col_l: str, col_r: str, key: str):
        if col_l not in df.columns or col_r not in df.columns:
            return
        l = df[col_l].dropna()
        r = df[col_r].dropna()
        if len(l) == 0 or len(r) == 0:
            return
        diff = (l - r).abs()
        stats[key] = {
            "L_mean": float(l.mean()),
            "R_mean": float(r.mean()),
            "L_max": float(l.max()),
            "R_max": float(r.max()),
            "LR_diff_mean": float(diff.mean())
        }

    # 膝・股関節・足首・体幹
    angle_stats("knee_l_deg", "knee_r_deg", "knee")
    angle_stats("hip_l_deg", "hip_r_deg", "hip")
    angle_stats("ankle_l_deg", "ankle_r_deg", "ankle")

    if "trunk_deg" in df.columns:
        t = df["trunk_deg"].dropna()
        if len(t) > 0:
            stats["trunk"] = {
                "mean": float(t.mean()),
                "max": float(t.max()),
                "min": float(t.min()),
            }

    return df, stats


def create_angle_plots(df_angles: pd.DataFrame, video_path: str, out_dir: str):
    """
    膝・股関節・足首＋体幹のグラフをそれぞれ1ページ用にPNG出力。
    タイトルは英語にしてフォント依存を減らす。
    """
    ensure_dir(out_dir)
    stem = Path(video_path).stem

    time = df_angles["time_s"].values if "time_s" in df_angles.columns else np.arange(len(df_angles))

    paths = {}

    # 1) 膝角度
    if "knee_l_deg" in df_angles.columns and "knee_r_deg" in df_angles.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(time, df_angles["knee_l_deg"], label="Left knee")
        plt.plot(time, df_angles["knee_r_deg"], label="Right knee")
        plt.title("Knee flexion angle")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [deg]")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"{stem}_knee_angles.png")
        plt.savefig(p, dpi=200)
        plt.close()
        paths["knee"] = p
        print(f"📊 膝角度グラフ出力: {p}")

    # 2) 股関節角度
    if "hip_l_deg" in df_angles.columns and "hip_r_deg" in df_angles.columns:
        plt.figure(figsize=(6, 4))
        plt.plot(time, df_angles["hip_l_deg"], label="Left hip")
        plt.plot(time, df_angles["hip_r_deg"], label="Right hip")
        plt.title("Hip flexion angle")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [deg]")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"{stem}_hip_angles.png")
        plt.savefig(p, dpi=200)
        plt.close()
        paths["hip"] = p
        print(f"📊 股関節角度グラフ出力: {p}")

    # 3) 体幹＋足首角度
    has_trunk = "trunk_deg" in df_angles.columns
    has_ankle = "ankle_l_deg" in df_angles.columns and "ankle_r_deg" in df_angles.columns
    if has_trunk or has_ankle:
        plt.figure(figsize=(6, 4))
        if has_trunk:
            plt.plot(time, df_angles["trunk_deg"], label="Trunk tilt")
        if has_ankle:
            plt.plot(time, df_angles["ankle_l_deg"], label="Left ankle")
            plt.plot(time, df_angles["ankle_r_deg"], label="Right ankle")
        plt.title("Trunk & ankle angles")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [deg]")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(out_dir, f"{stem}_trunk_ankle_angles.png")
        plt.savefig(p, dpi=200)
        plt.close()
        paths["trunk_ankle"] = p
        print(f"📊 体幹＋足首角度グラフ出力: {p}")

    return paths


# ============================================================
# 骨格オーバーレイ動画 から start/mid/finish 抜き出し
# ============================================================

def extract_pose_keyframes(video_path: str, out_dir: str):
    """
    オーバーレイ動画(*_pose_overlay.mp4) から
    開始・中間・終盤の3枚をPNGとして保存。
    """
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠ オーバーレイ動画を開けません: {video_path}")
        return {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [
        int(frame_count * 0.05),
        int(frame_count * 0.50),
        int(frame_count * 0.95),
    ]
    names = ["start", "mid", "finish"]
    results = {}

    for idx, name in zip(idxs, names):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(out_path, frame)
        results[name] = out_path

    cap.release()
    print(f"📸 骨格画像出力: {results}")
    return results


# ============================================================
# AI 専門コメント生成（速度＋角度）
# ============================================================

def build_ai_comment(metrics_summary: dict, angle_stats: dict) -> str:
    lines = []

    # 速度とピッチ・ストライド
    pitch = metrics_summary.get("pitch")
    stride = metrics_summary.get("stride")
    max_speed = metrics_summary.get("max_speed")
    duration = metrics_summary.get("duration")
    stability = metrics_summary.get("stability")

    if pitch is not None and stride is not None:
        lines.append(
            f"・今回の走りでは平均ピッチ約 {pitch:.2f} 歩/秒、ストライド約 {stride:.2f} m が確認されました。"
            f" 100m 走全体としては、ピッチとストライドのバランスが良く、スプリンターとして標準〜やや高めの回転数です。"
        )
    if max_speed is not None and duration is not None:
        lines.append(
            f"・最大速度はおよそ {max_speed:.2f} m/s で、時間経過 {duration:.2f} 秒の中盤でピークに達しています。"
            " スタート後の加速局面から中盤にかけての加速がスムーズで、終盤に大きな減速は見られません。"
        )
    if stability is not None:
        lines.append(
            f"・安定性スコアは {stability:.2f}/10 で、重心のブレは比較的小さく、再現性の高い走りができています。"
        )

    # 膝
    knee = angle_stats.get("knee")
    if knee:
        lines.append(
            f"・膝関節角度は、左膝の平均屈曲角が約 {knee['L_mean']:.1f}°、右膝が約 {knee['R_mean']:.1f}° でした。"
            f" 最大屈曲角は左 {knee['L_max']:.1f}°、右 {knee['R_max']:.1f}° で、左右差の平均は {knee['LR_diff_mean']:.1f}° 程度です。"
            " 支持期〜遊脚期の切り替えで大きなアンバランスはなく、ストライドを生み出す膝のたたみは良好です。"
        )

    # 股関節
    hip = angle_stats.get("hip")
    if hip:
        lines.append(
            f"・股関節角度は、左股関節の平均屈曲が約 {hip['L_mean']:.1f}°、右股関節が約 {hip['R_mean']:.1f}°。"
            f" 特に最大屈曲角では左 {hip['L_max']:.1f}° に対して右 {hip['R_max']:.1f}° となっており、"
            f" 左右差の平均は {hip['LR_diff_mean']:.1f}° です。"
            " 右脚よりも左脚の引き上げがわずかに優位で、スタート〜中盤での接地位置のわずかな違いにつながっています。"
        )

    # 足首
    ankle = angle_stats.get("ankle")
    if ankle:
        lines.append(
            f"・足首（足関節）の背屈・底屈角は、左が平均 {ankle['L_mean']:.1f}°、右が {ankle['R_mean']:.1f}° 程度。"
            f" 最大角度では左 {ankle['L_max']:.1f}°、右 {ankle['R_max']:.1f}° で、左右差は平均 {ankle['LR_diff_mean']:.1f}° と比較的小さいです。"
            " 接地〜離地のタイミングは左右とも近く、足首の使い方は安定しています。"
        )

    # 体幹
    trunk = angle_stats.get("trunk")
    if trunk:
        lines.append(
            f"・体幹の前傾角は平均 {trunk['mean']:.1f}° 前傾、レンジは {trunk['min']:.1f}°〜{trunk['max']:.1f}° の範囲でした。"
            " スタート直後は大きめの前傾から、加速が進むにつれて徐々に起き上がる理想的なパターンに近い挙動です。"
            " ただし終盤でわずかに上体が起き上がるタイミングが早く、ラスト20mでの推進力ロスにつながる可能性があります。"
        )

    # 総合所見
    lines.append(
        "・総合すると、重心のブレが小さく、膝〜股関節〜足首の連動は安定しており、現時点でも完成度の高いフォームです。"
        " 今後の課題としては、右股関節の引き上げの速さと、ラスト20mでの体幹前傾のキープが挙げられます。"
        " これにより、終盤のスピード維持とストライドの伸びがさらに改善する可能性があります。"
    )

    # トレーニング提案（専門寄り）
    lines.append(
        "・トレーニング提案としては、"
        "①ハイニー（高いもも上げ）で左右の股関節屈曲角を揃えるドリル、"
        "②ミニハードル走で接地位置と体幹前傾を一定に保つ練習、"
        "③補強として腸腰筋・大殿筋を意識したヒップリフト系エクササイズ、"
        "などが有効です。"
    )

    return "\n".join(lines)


# ============================================================
# PDF出力
# ============================================================

class JP_PDF(FPDF):
    pass


def build_pdf(csv_path: str, video_path: str, athlete: str):
    # データ読み込み
    m_per_px = load_calibration()
    df_metrics, metrics_summary = load_metrics(csv_path)
    df_angles, angle_stats = load_angles(video_path)

    # グラフ画像
    stem = Path(video_path).stem
    graphs_dir = "outputs/graphs"
    main_graph_path = os.path.join(graphs_dir, f"{stem}_speed_tilt_com_v5_3_2.png")
    create_main_graph(df_metrics, m_per_px, main_graph_path)

    angle_graph_dir = os.path.join(graphs_dir, f"{stem}_angles_v5_3_2")
    angle_paths = create_angle_plots(df_angles, video_path, angle_graph_dir)

    # 骨格オーバーレイ画像（start/mid/finish）
    pose_images_dir = os.path.join("outputs", "pose_images", stem)
    overlay_video = os.path.join("outputs", "images", f"{stem}_pose_overlay.mp4")
    pose_imgs = {}
    if os.path.exists(overlay_video):
        pose_imgs = extract_pose_keyframes(overlay_video, pose_images_dir)
    else:
        print(f"⚠ オーバーレイ動画が見つかりません: {overlay_video}")

    # AIコメント
    ai_comment = build_ai_comment(metrics_summary, angle_stats)
    ai_comment = sanitize_text(ai_comment)

    # PDF準備
    ensure_dir("outputs/pdf")
    pdf = JP_PDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # 日本語フォント設定（Meiryo）
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.set_font("JP", size=11)

    # --------------------------------------------------------
    # 1ページ目：概要＋骨格3枚＋速度グラフ
    # --------------------------------------------------------
    pdf.add_page()
    page_w = pdf.w
    margin = 10

    # タイトル
    title = f"{athlete} フォーム分析レポート（v5.3.2）"
    pdf.set_font("JP", size=14)
    pdf.cell(0, 10, sanitize_text(title), ln=1)

    pdf.set_font("JP", size=10)
    pdf.cell(0, 6, sanitize_text(f"動画: {os.path.basename(video_path)}"), ln=1)
    pdf.cell(0, 6, sanitize_text(f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=1)
    pdf.cell(0, 6, sanitize_text(f"校正スケール: {m_per_px:.6f} m/px"), ln=1)

    # ピッチ・ストライドなど
    pdf.ln(2)
    pdf.set_font("JP", size=10)
    pitch = metrics_summary.get("pitch") or 0.0
    stride = metrics_summary.get("stride") or 0.0
    stability = metrics_summary.get("stability") or 0.0
    max_speed = metrics_summary.get("max_speed") or 0.0
    duration = metrics_summary.get("duration") or 0.0

    pdf.cell(0, 6, sanitize_text(f"ピッチ: {pitch:.2f} 歩/s　ストライド: {stride:.2f} m　最大速度: {max_speed:.2f} m/s"), ln=1)
    pdf.cell(0, 6, sanitize_text(f"レース時間: 約 {duration:.2f} s　安定性スコア: {stability:.2f} / 10"), ln=1)

    # 骨格3枚
    if pose_imgs:
        pdf.ln(2)
        pdf.cell(0, 6, sanitize_text("骨格オーバーレイ（start / mid / finish）"), ln=1)
        y_top = pdf.get_y()
        spacing = 3
        img_width = (page_w - 2 * margin - 2 * spacing) / 3.0
        img_height = img_width * 0.75  # アスペクト比ざっくり

        order = ["start", "mid", "finish"]
        x = margin
        for name in order:
            p = pose_imgs.get(name)
            if p and os.path.exists(p):
                pdf.image(p, x=x, y=y_top + 2, w=img_width)
            x += img_width + spacing

        pdf.set_y(y_top + img_height + 8)

    # 速度・傾き・COM グラフ
    if os.path.exists(main_graph_path):
        pdf.ln(2)
        pdf.cell(0, 6, sanitize_text("速度・体幹傾き・重心高さの推移"), ln=1)
        pdf.image(main_graph_path, x=margin, w=page_w - 2 * margin)

    # --------------------------------------------------------
    # 2ページ目：膝角度
    # --------------------------------------------------------
    knee_path = angle_paths.get("knee")
    if knee_path and os.path.exists(knee_path):
        pdf.add_page()
        pdf.set_font("JP", size=12)
        pdf.cell(0, 8, sanitize_text("膝関節角度の推移（左／右）"), ln=1)
        pdf.set_font("JP", size=9)
        pdf.cell(0, 6, sanitize_text("支持期〜遊脚期にかけての膝の屈伸パターンと左右差を確認できます。"), ln=1)
        pdf.ln(2)
        pdf.image(knee_path, x=margin, w=page_w - 2 * margin)

    # --------------------------------------------------------
    # 3ページ目：股関節角度
    # --------------------------------------------------------
    hip_path = angle_paths.get("hip")
    if hip_path and os.path.exists(hip_path):
        pdf.add_page()
        pdf.set_font("JP", size=12)
        pdf.cell(0, 8, sanitize_text("股関節角度の推移（左／右）"), ln=1)
        pdf.set_font("JP", size=9)
        pdf.cell(0, 6, sanitize_text("ももの引き上げ（股関節屈曲）のタイミングと左右差を見るためのグラフです。"), ln=1)
        pdf.ln(2)
        pdf.image(hip_path, x=margin, w=page_w - 2 * margin)

    # --------------------------------------------------------
    # 4ページ目：体幹＋足首角度 ＋ AIコメント
    # --------------------------------------------------------
    trunk_path = angle_paths.get("trunk_ankle")
    pdf.add_page()
    pdf.set_font("JP", size=12)
    pdf.cell(0, 8, sanitize_text("体幹前傾と足首角度の推移"), ln=1)
    pdf.set_font("JP", size=9)
    pdf.cell(0, 6, sanitize_text("スタート〜中盤〜終盤にかけての上体の倒しこみと足首の使い方を確認できます。"), ln=1)
    pdf.ln(2)

    if trunk_path and os.path.exists(trunk_path):
        pdf.image(trunk_path, x=margin, w=page_w - 2 * margin)
        pdf.ln(2)

    # AIフォーム解析コメント（専門寄り）
    pdf.set_font("JP", size=11)
    pdf.cell(0, 8, sanitize_text("AIフォーム解析コメント（速度＋角度の総合所見）"), ln=1)
    pdf.set_font("JP", size=9)

    for line in ai_comment.split("\n"):
        txt = sanitize_text(line)
        if not txt:
            pdf.ln(2)
            continue
        pdf.multi_cell(0, 5, txt)

    # --------------------------------------------------------
    # 保存
    out_pdf = os.path.join("outputs", "pdf", f"{athlete}_form_report_v5_3_2.pdf")
    out_pdf = out_pdf.replace("\\", "/")
    pdf.output(out_pdf)
    print(f"✅ PDF出力完了: {out_pdf}")


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KJAC フォームレポート生成 v5.3.2（速度＋角度＋専門コメント）")
    parser.add_argument("--video", required=True, help="入力動画パス")
    parser.add_argument("--csv", required=True, help="メトリクスCSVパス（pose_metrics_analyzer_v3_4.py の出力）")
    parser.add_argument("--athlete", required=True, help="選手名（PDFファイル名に使用）")
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()







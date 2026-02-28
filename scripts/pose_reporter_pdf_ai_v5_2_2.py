import os
import json
import cv2
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from fpdf import FPDF


# ============================================================
# ユーティリティ
# ============================================================

def load_calibration_m_per_px():
    """
    calibration_result.json から 1px あたり[m] を読み込む。
    ない場合は 1.0 を返す（グラフ表示には支障なし）。
    """
    calib_path = os.path.join("outputs", "jsonl", "calibration_result.json")
    if not os.path.exists(calib_path):
        print("※ calibration_result.json が見つかりません。m_per_px=1.0 とします。")
        return 1.0

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m_per_px = float(data.get("m_per_px", 1.0))
        print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
        return m_per_px
    except Exception as e:
        print(f"※ 校正ファイル読込エラー: {e} -> m_per_px=1.0 とします。")
        return 1.0


def save_pose_images(overlay_path, video_name_clean):
    """
    pose_overlay 動画から start / mid / finish の3枚を PNG として保存。
    戻り値: dict(label -> 絶対パス)  /  失敗時 None
    """
    overlay_abs = os.path.abspath(overlay_path)
    if not os.path.exists(overlay_abs):
        print(f"※ オーバーレイ動画が見つかりません: {overlay_abs}")
        return None

    cap = cv2.VideoCapture(overlay_abs)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < 3:
        print("※ オーバーレイ動画のフレーム数が少なすぎます。")
        cap.release()
        return None

    frames = {
        "start": 0,
        "mid": total // 2,
        "finish": total - 1
    }

    out_dir = os.path.join("outputs", "pose_images", video_name_clean)
    os.makedirs(out_dir, exist_ok=True)

    saved = {}
    for label, idx in frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"※ フレーム取得失敗: {label} (frame={idx})")
            continue
        out_path = os.path.join(out_dir, f"{label}.png")
        cv2.imwrite(out_path, frame)
        saved[label] = os.path.abspath(out_path)

    cap.release()
    print(f"📸 骨格画像出力: {saved}")
    return saved if saved else None


def create_graphs_from_csv(df, video_name_clean):
    """
    CSVデータから 3つのグラフ（速度・傾き・COM_x）を縦並びに描画し、
    1つの PNG に保存してパスを返す。
    """
    # Matplotlib日本語フォント設定（Meiryo）
    plt.rcParams["font.family"] = "Meiryo"

    # _summary_ 行を除いたデータ本体
    df_main = df[df["frame"] != "_summary_"].copy()

    # x軸: 時間[s]があればそれを使う。なければフレーム番号を使う
    if "time_s" in df_main.columns:
        x = df_main["time_s"].values
        x_label = "時間 [s]"
    else:
        x = df_main.index.values
        x_label = "フレーム"

    # y軸候補
    y_speed = df_main["speed_mps"].values if "speed_mps" in df_main.columns else None
    y_tilt = df_main["tilt_deg"].values if "tilt_deg" in df_main.columns else None

    if "COM_x_px" in df_main.columns:
        y_com = df_main["COM_x_px"].values
        com_label = "COM前後位置 [px]"
    elif "COM_x" in df_main.columns:
        y_com = df_main["COM_x"].values
        com_label = "COM前後位置 [正規化]"
    else:
        y_com = None
        com_label = "COM前後位置"

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), sharex=True)  # A4縦イメージ

    # 速度
    if y_speed is not None:
        axes[0].plot(x, y_speed)
        axes[0].set_ylabel("速度 [m/s]")
        axes[0].set_title("速度の時間変化")
    else:
        axes[0].text(0.5, 0.5, "速度データなし", ha="center", va="center")
        axes[0].set_ylabel("速度")

    # 傾き
    if y_tilt is not None:
        axes[1].plot(x, y_tilt)
        axes[1].set_ylabel("体幹傾き [deg]")
        axes[1].set_title("体幹の前後傾きの変化")
    else:
        axes[1].text(0.5, 0.5, "傾きデータなし", ha="center", va="center")
        axes[1].set_ylabel("傾き")

    # COM
    if y_com is not None:
        axes[2].plot(x, y_com)
        axes[2].set_ylabel(com_label)
        axes[2].set_title("COMの前後位置の時間変化")
    else:
        axes[2].text(0.5, 0.5, "COM位置データなし", ha="center", va="center")
        axes[2].set_ylabel("COM")

    axes[2].set_xlabel(x_label)

    plt.tight_layout()

    out_dir = os.path.join("outputs", "graphs")
    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, f"{video_name_clean}_graphs_v5_2_2.png")
    plt.savefig(graph_path, dpi=200)
    plt.close(fig)

    graph_abs = os.path.abspath(graph_path)
    print(f"📊 グラフ画像出力: {graph_abs}")
    return graph_abs


def generate_simple_ai_comments(summary):
    """
    OpenAI API を使わない簡易AIコメント。
    あくまで雰囲気用。必要なら後で本物のAPI連携に差し替え可能。
    """
    pitch = summary.get("pitch", 0.0)
    stride = summary.get("stride", 0.0)
    stability = summary.get("stability", 0.0)

    comments = []

    # ピッチ系コメント
    if pitch > 4.5:
        comments.append(
            f"ピッチは約 {pitch:.2f} 歩/秒で、かなり回転数が高い走りができています。接地時間を短く保てている点は大きな長所です。"
        )
    else:
        comments.append(
            f"ピッチは約 {pitch:.2f} 歩/秒で、無理のない回転数です。今後はスタート～加速局面で一時的にピッチをもう少し高くできると、前半の伸びがさらに良くなります。"
        )

    # ストライド系コメント
    if stride > 1.9:
        comments.append(
            f"ストライドは平均 {stride:.2f} m と、しっかり“歩幅の大きい走り”ができています。体の前ではなく、やや前方への押し出しで伸びている点は良い傾向です。"
        )
    else:
        comments.append(
            f"ストライドは平均 {stride:.2f} m で、現時点では無理のない長さです。骨盤から脚を前に出す意識や、地面を後ろへ押す動きを強化すると、自然にストライドが伸びてきます。"
        )

    # 安定性スコア
    if stability >= 8.5:
        comments.append(
            f"安定性スコアは {stability:.2f} / 10 と非常に高く、体幹や頭のブレが少ない走りができています。この安定感をキープしたまま、スタートや後半のスピードアップを狙っていきましょう。"
        )
    elif stability >= 7.0:
        comments.append(
            f"安定性スコアは {stability:.2f} / 10 で、全体としてフォームはかなり安定しています。ラスト局面で少し上下動が大きくなる傾向があるので、最後までリラックスして走れるとさらに良くなります。"
        )
    else:
        comments.append(
            f"安定性スコアは {stability:.2f} / 10 で、まだ伸びしろが大きい状態です。特に接地ごとの上下動や体の左右ブレを減らしていくことで、スピードロスを大きく減らせます。"
        )

    return comments


# ============================================================
# PDF生成
# ============================================================

def build_pdf(csv_path, video_path, athlete):
    # CSV 読み込み
    df = pd.read_csv(csv_path)

    # サマリー行取得
    summary_row = df[df["frame"] == "_summary_"]
    summary = {}

    if not summary_row.empty:
        # 最後の5列を想定: [..., pitch, stride, stability_std, stability_jerk, stability_score]
        # 例: ...,4.73,0.30,0.008,0.47,9.24
        try:
            values = summary_row.iloc[0].values
            pitch = float(values[-5])
            stride = float(values[-4])
            stability_score = float(values[-1])
            summary = {
                "pitch": pitch,
                "stride": stride,
                "stability": stability_score,
            }
        except Exception as e:
            print(f"※ サマリー行の解析に失敗しました: {e}")
    else:
        print("※ _summary_ 行が見つかりません。簡易集計を使用します。")
        df_main = df[df["frame"] != "_summary_"].copy()
        summary = {
            "pitch": float(df_main["pitch_hz"].mean()) if "pitch_hz" in df_main.columns else 0.0,
            "stride": float(df_main["stride_m"].mean()) if "stride_m" in df_main.columns else 0.0,
            "stability": float(df_main["stability_score"].mean()) if "stability_score" in df_main.columns else 0.0,
        }

    # 校正値読み込み
    m_per_px = load_calibration_m_per_px()

    # 動画名（拡張子なし）をクリーン化（空白除去）
    raw_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name_clean = raw_name.replace(" ", "")

    # オーバーレイ動画パス
    overlay_path = os.path.join("outputs", "images", f"{video_name_clean}_pose_overlay.mp4")

    # 骨格画像生成
    pose_images = save_pose_images(overlay_path, video_name_clean)

    # グラフ画像生成
    graph_path = create_graphs_from_csv(df, video_name_clean)

    # AI風コメント生成
    ai_comments = generate_simple_ai_comments(summary)

    # ========================================================
    # PDF作成
    # ========================================================
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # 日本語フォント登録（Meiryo）
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.set_font("JP", "", 14)

    # ---------- 1ページ目：タイトル＋骨格画像 ----------
    pdf.add_page()

    # タイトル
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.2.2）", ln=1)
    pdf.set_font("JP", "", 11)
    pdf.cell(0, 6, f"動画: {os.path.basename(video_path)}", ln=1)
    pdf.cell(0, 6, f"解析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.cell(0, 6, f"校正スケール: {m_per_px:.6f} m/px", ln=1)

    pdf.ln(4)
    pdf.set_font("JP", "", 12)
    pdf.cell(0, 8, f"ピッチ: {summary.get('pitch', 0.0):.2f} 歩/s", ln=1)
    pdf.cell(0, 8, f"ストライド: {summary.get('stride', 0.0):.2f} m", ln=1)
    pdf.cell(0, 8, f"安定性スコア: {summary.get('stability', 0.0):.2f} / 10", ln=1)

    pdf.ln(6)
    pdf.set_font("JP", "", 13)
    pdf.cell(0, 8, "骨格オーバーレイ（start / mid / finish）", ln=1)

    if pose_images:
        # 横3枚で配置
        pdf.ln(3)
        page_width = pdf.w
        margin = pdf.l_margin
        usable_width = page_width - 2 * margin
        gap = 4  # 画像間の隙間
        img_width = (usable_width - 2 * gap) / 3.0

        y_top = pdf.get_y()

        order = ["start", "mid", "finish"]
        labels_jp = {
            "start": "スタート付近",
            "mid": "中間付近",
            "finish": "フィニッシュ付近"
        }

        x = margin
        pdf.set_font("JP", "", 10)

        # ラベルと画像配置
        for key in order:
            if key not in pose_images:
                continue
            img_path = pose_images[key]
            img_abs = os.path.abspath(img_path)

            # ラベル
            pdf.set_xy(x, y_top)
            pdf.cell(img_width, 5, labels_jp.get(key, key), align="C")

            # 画像
            pdf.set_xy(x, y_top + 5)
            pdf.image(img_abs, x=x, y=y_top + 5, w=img_width)

            x += img_width + gap

        pdf.ln(img_width + 12)
    else:
        pdf.ln(3)
        pdf.set_font("JP", "", 11)
        pdf.cell(0, 8, "※ 骨格オーバーレイ動画が見つからなかったため、静止画像を表示できませんでした。", ln=1)

    # ---------- 2ページ目：グラフ＋AIコメント ----------
    pdf.add_page()
    pdf.set_font("JP", "", 13)
    pdf.cell(0, 8, "速度・傾き・COMの変化グラフ", ln=1)

    if graph_path and os.path.exists(graph_path):
        graph_abs = os.path.abspath(graph_path)
        pdf.ln(3)
        # ページ幅に合わせてグラフ画像を配置
        page_width = pdf.w
        margin = pdf.l_margin
        usable_width = page_width - 2 * margin
        pdf.image(graph_abs, x=margin, w=usable_width)
        pdf.ln(usable_width * 0.75 / 3)  # 適当に下にスペース
    else:
        pdf.ln(5)
        pdf.set_font("JP", "", 11)
        pdf.cell(0, 8, "※ グラフ画像を生成できませんでした。", ln=1)

    pdf.ln(6)
    pdf.set_font("JP", "", 13)
    pdf.cell(0, 8, "AIフォーム解析コメント（自動生成）", ln=1)
    pdf.ln(2)
    pdf.set_font("JP", "", 11)

    for c in ai_comments:
        safe_text = c.replace("⚠", "※")
        pdf.multi_cell(0, 6, f"・{safe_text}")
        pdf.ln(1)

    # 出力
    out_dir = os.path.join("outputs", "pdf")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"{athlete}_form_report_v5_2_2.pdf")
    pdf.output(out_pdf)

    print(f"✅ PDF出力完了: {out_pdf}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="解析対象動画のパス")
    parser.add_argument("--csv", required=True, help="pose_metrics_analyzer_v3_x のCSVパス")
    parser.add_argument("--athlete", required=True, help="選手名（PDFファイル名などに使用）")
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()


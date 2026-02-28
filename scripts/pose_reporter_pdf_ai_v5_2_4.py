import os
import re
import json
import cv2
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF


# ============================================================
# 共通ユーティリティ
# ============================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def to_ascii_folder(name: str) -> str:
    """
    日本語や記号を含む動画名から、「ASCIIだけ」のフォルダ名を作る。
    例）"二村遥香10.5" → "_10_5" → "10_5"
    """
    ascii_name = re.sub(r'[^A-Za-z0-9]+', '_', name)
    if ascii_name == "":
        ascii_name = "video"
    return ascii_name.strip("_") or "video"


def load_calibration_m_per_px():
    """
    calibration_result.json から 1px あたり[m] を読み込む。
    """
    calib_path = os.path.join("outputs", "jsonl", "calibration_result.json")
    if not os.path.exists(calib_path):
        print("※ calibration_result.json が見つからないため、m_per_px=1.0 とします。")
        return 1.0

    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m_per_px = float(data.get("m_per_px", 1.0))
        print(f"📏 読み込み: 1px = {m_per_px:.6f} m")
        return m_per_px
    except Exception as e:
        print(f"※ 校正ファイル読込エラー: {e} → m_per_px=1.0 とします。")
        return 1.0


# ============================================================
# グラフ生成（速度・傾き・COM位置）
# ============================================================

def create_graphs_from_csv(df: pd.DataFrame, video_name_clean: str, ascii_folder: str) -> str:
    """
    CSVデータから速度・傾き・COM位置の3グラフを縦に並べた1枚PNGを生成。
    戻り値は保存先の絶対パス。
    """
    # Matplotlib 日本語フォント設定（Meiryo）
    plt.rcParams["font.family"] = "Meiryo"

    # _summary_ 行を除いたデータ本体
    df_main = df[df["frame"] != "_summary_"].copy()

    # x 軸（time_s があれば使う）
    if "time_s" in df_main.columns:
        x = df_main["time_s"].values
        x_label = "時間 [s]"
    else:
        x = df_main.index.values
        x_label = "フレーム"

    # 速度
    y_speed = None
    if "speed_mps" in df_main.columns:
        y_speed = df_main["speed_mps"].values

    # 傾き
    y_tilt = None
    if "tilt_deg" in df_main.columns:
        y_tilt = df_main["tilt_deg"].values

    # COM前後位置
    y_com = None
    com_label = "COM前後位置"
    if "COM_x_px" in df_main.columns:
        y_com = df_main["COM_x_px"].values
        com_label = "COM前後位置 [px]"
    elif "COM_x" in df_main.columns:
        y_com = df_main["COM_x"].values
        com_label = "COM前後位置 [正規化]"

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69), sharex=True)  # A4縦イメージ

    # --- 速度 ---
    if y_speed is not None:
        axes[0].plot(x, y_speed)
        axes[0].set_ylabel("速度 [m/s]")
        axes[0].set_title("速度の時間変化")
    else:
        axes[0].text(0.5, 0.5, "速度データなし", ha="center", va="center")
        axes[0].set_ylabel("速度")

    # --- 傾き ---
    if y_tilt is not None:
        axes[1].plot(x, y_tilt)
        axes[1].set_ylabel("体幹傾き [deg]")
        axes[1].set_title("体幹の前後傾きの変化")
    else:
        axes[1].text(0.5, 0.5, "傾きデータなし", ha="center", va="center")
        axes[1].set_ylabel("傾き")

    # --- COM前後 ---
    if y_com is not None:
        axes[2].plot(x, y_com)
        axes[2].set_ylabel(com_label)
        axes[2].set_title("COMの前後位置の時間変化")
    else:
        axes[2].text(0.5, 0.5, "COM位置データなし", ha="center", va="center")
        axes[2].set_ylabel("COM")

    axes[2].set_xlabel(x_label)
    plt.tight_layout()

    graphs_dir = os.path.join("outputs", "graphs")
    ensure_dir(graphs_dir)

    # グラフファイル名も ASCII フォルダ名で安全に
    graph_path = os.path.join(graphs_dir, f"{ascii_folder}_graphs_v5_2_4.png")
    plt.savefig(graph_path, dpi=200)
    plt.close(fig)

    graph_abs = os.path.abspath(graph_path)
    print(f"📊 グラフ画像出力: {graph_abs}")
    return graph_abs


# ============================================================
# 骨格画像生成（start / mid / finish）
# ============================================================

def save_pose_images(overlay_path: str, ascii_folder: str):
    """
    pose_overlay 動画から start / mid / finish の3枚を PNG で保存。
    フォルダは ASCII のみの名前を使う → FPDF / OpenCV の日本語パス問題を回避。
    戻り値: dict(label -> 絶対パス)  / 失敗時 None
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
        "finish": total - 1,
    }

    out_dir = os.path.join("outputs", "pose_images", ascii_folder)
    ensure_dir(out_dir)

    saved = {}
    for label, idx in frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"※ フレーム取得失敗: {label} (frame={idx})")
            continue
        out_path = os.path.join(out_dir, f"{label}.png")
        # ASCII パスなので OpenCV でも安全
        cv2.imwrite(out_path, frame)
        saved[label] = os.path.abspath(out_path)

    cap.release()
    print(f"📸 骨格画像出力: {saved}")
    return saved if saved else None


# ============================================================
# コマ送り用フレーム書き出し（5フレームごと）
# ============================================================

def export_pose_frames(overlay_path: str, ascii_folder: str, step: int = 5):
    """
    オーバーレイ動画から step フレームごとに PNG を保存。
    例：step=5 → frame_0000, frame_0005, frame_0010, ...
    保存先: outputs/pose_images/<ascii_folder>/frames/
    """
    overlay_abs = os.path.abspath(overlay_path)
    if not os.path.exists(overlay_abs):
        print(f"※ オーバーレイ動画が見つからないため、コマ送り出力をスキップします: {overlay_abs}")
        return

    cap = cv2.VideoCapture(overlay_abs)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print("※ オーバーレイ動画のフレーム数が 0 のため、コマ送り出力をスキップします。")
        cap.release()
        return

    frames_dir = os.path.join("outputs", "pose_images", ascii_folder, "frames")
    ensure_dir(frames_dir)

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % step == 0:
            filename = f"frame_{frame_idx:04d}.png"
            out_path = os.path.join(frames_dir, filename)
            cv2.imwrite(out_path, frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"🎞 コマ送り用フレーム保存: {saved_count}枚 → {os.path.abspath(frames_dir)}")


# ============================================================
# AIっぽいコメント生成（簡易版）
# ============================================================

def generate_simple_ai_comments(summary: dict):
    """
    OpenAI API を使わずに、「それっぽい」コメントを生成。
    """
    pitch = summary.get("pitch", 0.0)
    stride = summary.get("stride", 0.0)
    stability = summary.get("stability", 0.0)

    comments = []

    # ピッチ
    if pitch > 4.5:
        comments.append(
            f"ピッチは約 {pitch:.2f} 歩/秒で、かなり回転数の高い走りができています。接地時間が短く、テンポ良く前へ進めている点は大きな強みです。"
        )
    else:
        comments.append(
            f"ピッチは約 {pitch:.2f} 歩/秒で、落ち着いた回転数です。特にスタート〜加速局面で一時的にもう少しピッチを上げられると、前半の伸びがさらに良くなります。"
        )

    # ストライド
    if stride > 1.9:
        comments.append(
            f"ストライドは平均 {stride:.2f} m と、しっかりとした歩幅で走れています。体の真下〜やや前方で接地できていると、このストライドがスピードにしっかりつながります。"
        )
    else:
        comments.append(
            f"ストライドは平均 {stride:.2f} m で、現時点では無理のない長さです。骨盤から脚を前に運ぶ意識や、地面を後ろへ強く押す動きを高めていくと、自然にストライドも伸びていきます。"
        )

    # 安定性スコア
    if stability >= 8.5:
        comments.append(
            f"安定性スコアは {stability:.2f} / 10 と非常に高く、上半身や頭のブレが少ないフォームです。この安定感をキープしながら、スタートやラストのスピードアップを狙っていくと記録更新につながります。"
        )
    elif stability >= 7.0:
        comments.append(
            f"安定性スコアは {stability:.2f} / 10 で、全体としてかなり安定した走りができています。特にラスト局面で少し上下動が大きくなる傾向があるので、最後までリラックスして走り切れるとさらに良くなります。"
        )
    else:
        comments.append(
            f"安定性スコアは {stability:.2f} / 10 で、まだ伸びしろが大きい状態です。接地ごとの上下動や左右ブレを少しずつ減らしていくことで、スピードロスを減らし、タイム短縮につながります。"
        )

    return comments


# ============================================================
# PDF生成本体
# ============================================================

def build_pdf(csv_path: str, video_path: str, athlete: str):
    # CSV 読み込み
    df = pd.read_csv(csv_path)

    # サマリー取得
    summary_row = df[df["frame"] == "_summary_"]
    summary = {}

    if not summary_row.empty:
        try:
            values = summary_row.iloc[0].values
            # 末尾5つを [pitch, stride, stability_std, stability_jerk, stability_score] とみなす
            pitch = float(values[-5])
            stride = float(values[-4])
            stability_score = float(values[-1])
            summary = {
                "pitch": pitch,
                "stride": stride,
                "stability": stability_score,
            }
        except Exception as e:
            print(f"※ サマリー行の解析に失敗: {e}")
    else:
        print("※ _summary_ 行が見つからないため、簡易平均で代用します。")
        df_main = df[df["frame"] != "_summary_"].copy()
        summary = {
            "pitch": float(df_main["pitch_hz"].mean()) if "pitch_hz" in df_main.columns else 0.0,
            "stride": float(df_main["stride_m"].mean()) if "stride_m" in df_main.columns else 0.0,
            "stability": float(df_main["stability_score"].mean()) if "stability_score" in df_main.columns else 0.0,
        }

    # 校正値
    m_per_px = load_calibration_m_per_px()

    # 動画名と ASCII フォルダ名
    raw_name = os.path.splitext(os.path.basename(video_path))[0]  # 例: "二村遥香10.5"
    video_name_clean = raw_name.replace(" ", "")
    ascii_folder = to_ascii_folder(video_name_clean)

    # オーバーレイ動画パス（こちらは日本語名のまま）
    overlay_path = os.path.join("outputs", "images", f"{video_name_clean}_pose_overlay.mp4")

    # 骨格画像生成（ASCIIフォルダ配下）
    pose_images = save_pose_images(overlay_path, ascii_folder)

    # 5フレームごとのコマ送り画像出力（ASCIIフォルダ配下）
    export_pose_frames(overlay_path, ascii_folder, step=5)

    # グラフ画像生成（ファイル名も ASCII）
    graph_path = create_graphs_from_csv(df, video_name_clean, ascii_folder)

    # AI風コメント
    ai_comments = generate_simple_ai_comments(summary)

    # ---------------- PDF作成 ----------------
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # 日本語フォント
    font_path = "c:/Windows/Fonts/meiryo.ttc"
    pdf.add_font("JP", "", font_path, uni=True)
    pdf.set_font("JP", "", 14)

    # ---------- 1ページ目：タイトル＋骨格画像 ----------
    pdf.add_page()

    # タイトル
    pdf.cell(0, 10, f"{athlete} フォーム分析レポート（v5.2.4）", ln=1)
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
        pdf.ln(3)
        page_width = pdf.w
        margin = pdf.l_margin
        usable_width = page_width - 2 * margin
        gap = 4
        img_width = (usable_width - 2 * gap) / 3.0

        y_top = pdf.get_y()
        x = margin

        labels_jp = {
            "start": "スタート付近",
            "mid": "中間付近",
            "finish": "フィニッシュ付近",
        }

        pdf.set_font("JP", "", 10)
        order = ["start", "mid", "finish"]
        for key in order:
            if key not in pose_images:
                continue
            img_abs = pose_images[key]

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

    # ---------- 2ページ目：グラフ＋コメント ----------
    pdf.add_page()
    pdf.set_font("JP", "", 13)
    pdf.cell(0, 8, "速度・傾き・COMの変化グラフ", ln=1)

    if graph_path and os.path.exists(graph_path):
        graph_abs = os.path.abspath(graph_path)
        pdf.ln(3)
        page_width = pdf.w
        margin = pdf.l_margin
        usable_width = page_width - 2 * margin
        pdf.image(graph_abs, x=margin, w=usable_width)
        pdf.ln(5)
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
    ensure_dir(out_dir)
    out_pdf = os.path.join(out_dir, f"{athlete}_form_report_v5_2_4.pdf")
    pdf.output(out_pdf)

    print(f"✅ PDF出力完了: {out_pdf}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="解析対象動画のパス")
    parser.add_argument("--csv", required=True, help="pose_metrics_analyzer の CSV パス")
    parser.add_argument("--athlete", required=True, help="選手名（PDFファイル名などに使用）")
    args = parser.parse_args()

    build_pdf(args.csv, args.video, args.athlete)


if __name__ == "__main__":
    main()




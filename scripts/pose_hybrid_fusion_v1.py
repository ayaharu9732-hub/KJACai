# ============================================================
#  pose_hybrid_fusion_v1.py
#
#  役割:
#    - YOLO v2 CSV（全身）と foot v1 CSV（足部）を frame 単位で結合
#    - outputs/{athlete}/{video_id}/yolo_pose/{video_id}_hybrid_v1.csv を生成
#
#  入力:
#    - yolo_csv: *_yolo_keypoints_v2.csv
#    - foot_csv: *_foot_v1.csv
#
#  出力:
#    - hybrid CSV:
#      frame, track_id, x1, y1, x2, y2, img_w, img_h,
#      kp0_x,...,kpN_conf,
#      foot0_x,...,footM_vis
# ============================================================

import os
import sys
import argparse
from datetime import datetime

import pandas as pd


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now_str()}] {msg}")


def parse_athlete_and_video_id(video_path: str):
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    i = len(stem) - 1
    while i >= 0 and (stem[i].isdigit() or stem[i] == "."):
        i -= 1

    athlete = stem[: i + 1].strip() if i >= 0 else stem
    video_id = stem.replace(".", "_")

    if not athlete:
        athlete = stem

    return athlete, video_id


def get_yolo_output_dir(athlete: str, video_id: str) -> str:
    root = os.path.join("outputs", athlete, video_id, "yolo_pose")
    os.makedirs(root, exist_ok=True)
    return root


def run_fusion(video_path: str, yolo_csv: str | None = None, foot_csv: str | None = None):
    athlete, video_id = parse_athlete_and_video_id(video_path)
    yolo_dir = get_yolo_output_dir(athlete, video_id)

    if yolo_csv is None:
        yolo_csv = os.path.join(yolo_dir, f"{video_id}_yolo_keypoints_v2.csv")
    if foot_csv is None:
        foot_csv = os.path.join(yolo_dir, f"{video_id}_foot_v1.csv")

    if not os.path.isfile(yolo_csv):
        log(f"ERROR: YOLO CSV が見つかりません: {yolo_csv}")
        return
    if not os.path.isfile(foot_csv):
        log(f"ERROR: foot CSV が見つかりません: {foot_csv}")
        return

    log(f"[fusion] YOLO CSV: {yolo_csv}")
    log(f"[fusion] FOOT CSV: {foot_csv}")

    df_yolo = pd.read_csv(yolo_csv)
    df_foot = pd.read_csv(foot_csv)

    if "frame" not in df_yolo.columns or "frame" not in df_foot.columns:
        log("ERROR: CSV に frame 列がありません。")
        return

    # foot 側は frame のみ unique と仮定（1フレーム1セット）
    # yolo 側は複数 track_id があり得るので、全て foot 情報を merge
    # → frame で left join
    df_hybrid = pd.merge(
        df_yolo,
        df_foot,
        on="frame",
        how="left",
        suffixes=("", "_foot"),
    )

    out_csv = os.path.join(yolo_dir, f"{video_id}_hybrid_v1.csv")
    df_hybrid.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log(f"[fusion] ハイブリッドCSVを書き出しました: {out_csv} (rows={len(df_hybrid)})")
    log("=== pose_hybrid_fusion_v1 完了 ===")


def main():
    parser = argparse.ArgumentParser(description="YOLO + FootLandmarker ハイブリッドCSV生成 v1")
    parser.add_argument("--video", required=True, help="元の動画パス (mp4)")
    parser.add_argument("--yolo_csv", default=None, help="YOLO v2 CSV パス（省略時は自動推定）")
    parser.add_argument("--foot_csv", default=None, help="foot v1 CSV パス（省略時は自動推定）")
    args = parser.parse_args()

    run_fusion(args.video, args.yolo_csv, args.foot_csv)


if __name__ == "__main__":
    main()



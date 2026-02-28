# ============================================================
# heavy_zoom_step1_track_bbox.py  （YOLOトラッキング専用）
#
# - YOLOv8x-pose で動画全体をトラッキング
# - 各フレームのバウンディングボックスを CSV に保存
# - 後続の AI-ZOOM (Step2) で、この CSV から最適ズーム領域を計算する
#
# 実行例:
#   (mp_env) python scripts/heavy_zoom_step1_track_bbox.py ^
#       --video "C:\Users\Futamura\KJACai\videos\岩本結々スタブロ11.15.mp4"
# ============================================================

import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ------------------------------------------------------------
# ログ
# ------------------------------------------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


# ------------------------------------------------------------
# 選手名 / video_id / 出力パス
# ------------------------------------------------------------
def parse_athlete_and_video_id(video_path: str) -> Tuple[str, str]:
    """
    例:
      岩本結々スタブロ11.15.mp4 → athlete=岩本結々スタブロ, video_id=岩本結々スタブロ11_15
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    i = len(stem) - 1
    while i >= 0 and (stem[i].isdigit() or stem[i] == "."):
        i -= 1

    if i < 0:
        athlete = stem
    else:
        athlete = stem[: i + 1]

    athlete = athlete.strip() or stem
    video_id = stem.replace(".", "_")
    return athlete, video_id


def get_output_csv_path(athlete: str, video_id: str) -> str:
    """
    bbox 用 CSV:
      outputs/{athlete}/{video_id}/zoom_prep/{video_id}_yolo_tracks.csv
    """
    root = os.path.join("outputs", athlete, video_id, "zoom_prep")
    os.makedirs(root, exist_ok=True)
    csv_path = os.path.join(root, f"{video_id}_yolo_tracks.csv")
    return csv_path


# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
def run(video_path: str) -> None:
    log(f"動画: {video_path}")
    if not os.path.exists(video_path):
        log("[ERROR] 動画が存在しません")
        return

    athlete, video_id = parse_athlete_and_video_id(video_path)
    csv_path = get_output_csv_path(athlete, video_id)

    # 動画情報取得
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("[ERROR] 動画を開けませんでした")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    log(f"選手名: {athlete}, video_id: {video_id}")
    log(f"FPS = {fps}")
    log(f"解像度 = {width} x {height}, 総フレーム = {total_frames}")
    log(f"出力CSV: {csv_path}")

    # YOLO モデル
    log("YOLOv8x-pose 読み込み中…")
    model = YOLO("yolov8x-pose.pt")

    rows: List[Dict[str, float]] = []

    frame_idx = -1

    # YOLO track（ByteTrack）を使うことで track_id が付与される
    for result in model.track(
        source=video_path,
        stream=True,
        conf=0.5,
        iou=0.5,
        persist=True,
        verbose=False,
    ):
        frame_idx += 1
        if frame_idx >= total_frames:
            break

        if frame_idx % 30 == 0:
            log(f"処理中… frame={frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

        # boxes がないフレーム
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes

        # xyxy, conf, cls
        xyxy = boxes.xyxy.cpu().numpy()        # (n,4)
        confs = boxes.conf.cpu().numpy()       # (n,)
        classes = boxes.cls.cpu().numpy()      # (n,)

        # track id（ない場合は -1）
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.full((xyxy.shape[0],), -1, dtype=int)

        t_sec = frame_idx / fps

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            track_id = int(ids[i])
            conf = float(confs[i])
            cls = int(classes[i])

            row = {
                "frame": frame_idx,
                "time_s": t_sec,
                "track_id": track_id,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "conf": conf,
                "cls": cls,
            }
            rows.append(row)

    # CSV 保存
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        log(f"[OK] YOLO bbox CSV 書き出し完了: {csv_path} (rows={len(df)})")
    else:
        log("[WARN] 1件も検出がありませんでした（conf が高すぎる？動画パス？）")


def main():
    parser = argparse.ArgumentParser(
        description="Step1: YOLOv8x-pose で全フレームの bbox を CSV 出力（AI-ZOOM 前処理用）"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="入力動画パス (mp4)",
    )
    args = parser.parse_args()
    run(args.video)


if __name__ == "__main__":
    main()



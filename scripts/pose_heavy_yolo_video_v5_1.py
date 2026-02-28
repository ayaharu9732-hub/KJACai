# ============================================================
# pose_heavy_yolo_video_v5_1.py
#  - YOLOv8x-pose(トラッキング) + MediaPipe Heavy Pose(3D)
#  - 足部補強あり
#  - v5 からの改良点：
#       ▶ HeavyPose の進捗ログを30フレームごとに表示
#       ▶ 推定残り時間(ETA)表示
# ============================================================

import os
import cv2
import time
import math
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, List

from ultralytics import YOLO

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------------------------------------------------
# ログ出力
# ------------------------------------------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg):
    print(f"[{now_str()}] {msg}")

# ------------------------------------------------------------
# Heavy PoseLandmarker 初期化
# ------------------------------------------------------------
def load_heavy_pose(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)
    return landmarker

# ------------------------------------------------------------
# HeavyPose の実行
# ------------------------------------------------------------
def run_heavy_pose(landmarker, crop_bgr: np.ndarray):
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_bgr)
        result = landmarker.detect(mp_image)
        return result
    except Exception:
        return None

# ------------------------------------------------------------
# 選手名・video_id
# ------------------------------------------------------------
def parse_athlete_and_video_id(path: str):
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)

    i = len(stem) - 1
    while i >= 0 and (stem[i].isdigit() or stem[i] == "."):
        i -= 1
    athlete = stem if i < 0 else stem[: i + 1]

    return athlete.strip(), stem.replace(".", "_")

# ------------------------------------------------------------
# CSV の1行作成（簡略版）
# ------------------------------------------------------------
def make_csv_row(frame_idx, time_s, mp_result):
    row = {
        "frame": frame_idx,
        "time_s": time_s,
    }

    if (
        mp_result is None or
        mp_result.pose_landmarks is None or
        len(mp_result.pose_landmarks) == 0
    ):
        for i in range(33):
            row[f"lm{i}_x"] = math.nan
            row[f"lm{i}_y"] = math.nan
            row[f"lm{i}_z"] = math.nan
        return row

    lms = mp_result.pose_landmarks[0]
    for i, lm in enumerate(lms):
        row[f"lm{i}_x"] = lm.x
        row[f"lm{i}_y"] = lm.y
        row[f"lm{i}_z"] = lm.z

    return row

# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
def run(video_path: str):

    log(f"[INFO] 動画: {video_path}")
    athlete, video_id = parse_athlete_and_video_id(video_path)

    out_dir = os.path.join("outputs", athlete, video_id, "pose_heavy_v5")
    os.makedirs(out_dir, exist_ok=True)

    overlay_path = os.path.join(out_dir, f"{video_id}_pose_heavy_v5_overlay.mp4")
    csv_path = os.path.join(out_dir, f"{video_id}_pose_heavy_v5_landmarks.csv")

    log(f"[INFO] 出力Overlay: {overlay_path}")
    log(f"[INFO] 出力CSV     : {csv_path}")

    # ---- 動画
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 30

    log(f"[INFO] FPS = {fps}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        overlay_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    # ---- YOLO
    log("[INFO] YOLOv8x-pose 読み込み中…")
    yolo_model = YOLO("yolov8x-pose.pt")

    # ---- HeavyPose
    MODEL_PATH = "pose_landmarker_heavy.task"
    log(f"[INFO] PoseLandmarker モデル: {MODEL_PATH}")
    landmarker = load_heavy_pose(MODEL_PATH)

    # ---- YOLO track stream
    log("[INFO] YOLO トラッキング開始 (ByteTrack)")
    main_track_id = None

    rows = []
    frame_idx = -1
    t0 = time.time()

    for result in yolo_model.track(
        source=video_path,
        stream=True,
        verbose=False,
        conf=0.5,
        iou=0.5,
        persist=True,
    ):
        frame_idx += 1
        time_s = frame_idx / fps
        frame_bgr = result.orig_img

        # ---- 進捗ログ（30フレーム毎）
        if frame_idx % 30 == 0 and frame_idx > 0:
            pct = frame_idx / total_frames * 100
            elapsed = time.time() - t0
            if frame_idx > 0:
                eta = elapsed / frame_idx * (total_frames - frame_idx)
            else:
                eta = 0
            log(f"[INFO] 処理中… frame={frame_idx}/{total_frames} ({pct:.1f}%)  ETA={eta/60:.1f}分")

        # ---- YOLO検出の有無
        if result.boxes is None or result.keypoints is None:
            writer.write(frame_bgr)
            continue

        boxes = result.boxes
        if boxes.id is None:
            writer.write(frame_bgr)
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        if len(ids) == 0:
            writer.write(frame_bgr)
            continue

        # ---- 最初の main_track_id 決定
        if main_track_id is None:
            idx = int(np.argmax((xyxy[:,2]-xyxy[:,0])*(xyxy[:,3]-xyxy[:,1])))
            main_track_id = int(ids[idx])
            log(f"[INFO] main_track_id = {main_track_id}")

        # ---- main_track_id 探す
        idxs = [i for i,t in enumerate(ids) if t == main_track_id]
        if not idxs:
            writer.write(frame_bgr)
            continue
        idx = idxs[0]

        x1,y1,x2,y2 = xyxy[idx].astype(int)

        # ---- 安全クロップ
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame_w-1, x2); y2 = min(frame_h-1, y2)
        crop = frame_bgr[y1:y2, x1:x2]

        # ---- HeavyPose
        mp_result = run_heavy_pose(landmarker, crop)

        # ---- CSV
        rows.append(make_csv_row(frame_idx, time_s, mp_result))

        # ---- 簡易描画（クロップ位置）
        cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)

        # ---- HeavyPose が取得できたら鼻だけ描画（処理簡略）
        if (
            mp_result is not None and
            mp_result.pose_landmarks and
            len(mp_result.pose_landmarks) > 0
        ):
            nose = mp_result.pose_landmarks[0][0]
            nx = int(nose.x * (x2-x1) + x1)
            ny = int(nose.y * (y2-y1) + y1)
            cv2.circle(frame_bgr, (nx,ny), 6, (0,255,255), -1)

        writer.write(frame_bgr)

    # ---- 保存
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, encoding="utf-8-sig", index=False)
    log(f"[INFO] CSV 書き出し完了 → {csv_path}")

    writer.release()
    cap.release()

    log("[INFO] Overlay 完了")
    log("=== pose_heavy_yolo_video_v5.1 完了 ===")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    run(args.video)

if __name__ == "__main__":
    main()



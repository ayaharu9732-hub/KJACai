# ============================================================
#  KJAC フォーム解析 - メトリクス解析 v3.4
#  入力: 動画(mp4)
#  出力: metrics CSV (COM, 速度, ピッチ, ストライド, 安定性など)
#
#  実行例:
#    (.venv) python scripts/pose_metrics_analyzer_v3_4.py --video path/to/video.mp4
# ============================================================

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------

COM_LANDMARK_IDS = [
    mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
]

HIP_IDS = [
    mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
]
SHOULDER_IDS = [
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
]


@dataclass
class FrameMetrics:
    frame: int
    com_x: float
    com_y: float
    com_x_px: float
    com_y_px: float
    tilt_deg: float
    dx_px: float
    dy_px: float
    dist_px: float
    com_y_m: float
    speed_mps: float
    time_s: float
    pitch_hz: float
    stride_m: float
    stability_std_y_m: float
    stability_jerk: float
    stability_score: float
    COM_x: float
    COM_y: float
    COM_x_px: float
    COM_y_px: float
    COM_y_m: float


# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------

def parse_athlete_and_video_id(video_path: str) -> Tuple[str, str]:
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    i = len(stem) - 1
    while i >= 0 and (stem[i].isdigit() or stem[i] == "."):
        i -= 1

    if i < 0:
        athlete = stem
    else:
        athlete = stem[: i + 1]

    video_id = stem.replace(".", "_")

    athlete = athlete.strip()
    if not athlete:
        athlete = stem

    return athlete, video_id


def get_output_dirs(base_dir: str, athlete: str, video_id: str):
    root = os.path.join(base_dir, "outputs", athlete, video_id)
    metrics_dir = os.path.join(root, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    return {"root": root, "metrics": metrics_dir}


# ------------------------------------------------------------
# メトリクス解析本体
# ------------------------------------------------------------

def compute_com_and_tilt(landmarks, width: int, height: int):
    xs, ys = [], []
    for idx in COM_LANDMARK_IDS:
        lm = landmarks[idx]
        xs.append(lm.x)
        ys.append(lm.y)

    if not xs:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    com_x = float(np.mean(xs))
    com_y = float(np.mean(ys))
    com_x_px = com_x * width
    com_y_px = com_y * height

    hip_points = []
    shoulder_points = []
    for idx in HIP_IDS:
        lm = landmarks[idx]
        hip_points.append((lm.x * width, lm.y * height))
    for idx in SHOULDER_IDS:
        lm = landmarks[idx]
        shoulder_points.append((lm.x * width, lm.y * height))

    if not hip_points or not shoulder_points:
        tilt_deg = np.nan
    else:
        hip_center = np.mean(hip_points, axis=0)
        shoulder_center = np.mean(shoulder_points, axis=0)
        vx = shoulder_center[0] - hip_center[0]
        vy = shoulder_center[1] - hip_center[1]
        angle_rad = math.atan2(vx, -vy)  # 前傾で+になるよう調整
        tilt_deg = math.degrees(angle_rad)

    return com_x, com_y, com_x_px, com_y_px, tilt_deg


def analyze_video(video_path: str) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    metrics: List[FrameMetrics] = []

    prev_com_x_px = None
    prev_com_y_px = None

    # COM_y_m の重複列用
    M_PER_PX = 0.0016  # 仮のスケール

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)

        if not res.pose_landmarks:
            fm = FrameMetrics(
                frame=frame_idx,
                com_x=np.nan,
                com_y=np.nan,
                com_x_px=np.nan,
                com_y_px=np.nan,
                tilt_deg=np.nan,
                dx_px=np.nan,
                dy_px=np.nan,
                dist_px=np.nan,
                com_y_m=np.nan,
                speed_mps=np.nan,
                time_s=frame_idx / fps,
                pitch_hz=np.nan,
                stride_m=np.nan,
                stability_std_y_m=np.nan,
                stability_jerk=np.nan,
                stability_score=np.nan,
                COM_x=np.nan,
                COM_y=np.nan,
                COM_x_px=np.nan,
                COM_y_px=np.nan,
                COM_y_m=np.nan,
            )
            metrics.append(fm)
            continue

        lm = res.pose_landmarks.landmark
        com_x, com_y, com_x_px, com_y_px, tilt_deg = compute_com_and_tilt(lm, width, height)

        if prev_com_x_px is None:
            dx_px = np.nan
            dy_px = np.nan
            dist_px = np.nan
        else:
            dx_px = com_x_px - prev_com_x_px
            dy_px = com_y_px - prev_com_y_px
            dist_px = math.hypot(dx_px, dy_px)

        prev_com_x_px = com_x_px
        prev_com_y_px = com_y_px

        COM_y_m = com_y_px * M_PER_PX
        if np.isnan(dist_px):
            speed_mps = np.nan
        else:
            speed_mps = dist_px * M_PER_PX * fps

        time_s = frame_idx / fps

        metrics.append(
            FrameMetrics(
                frame=frame_idx,
                com_x=com_x,
                com_y=com_y,
                com_x_px=com_x_px,
                com_y_px=com_y_px,
                tilt_deg=tilt_deg,
                dx_px=dx_px,
                dy_px=dy_px,
                dist_px=dist_px,
                com_y_m=COM_y_m,
                speed_mps=speed_mps,
                time_s=time_s,
                pitch_hz=np.nan,
                stride_m=np.nan,
                stability_std_y_m=np.nan,
                stability_jerk=np.nan,
                stability_score=np.nan,
                COM_x=com_x,
                COM_y=com_y,
                COM_x_px=com_x_px,
                COM_y_px=com_y_px,
                COM_y_m=COM_y_m,
            )
        )

    cap.release()
    pose.close()

    df = pd.DataFrame([m.__dict__ for m in metrics])

    # ピッチ / ストライド / 安定性サマリ
    valid_y = df["com_y_m"].values
    valid_t = df["time_s"].values

    # 簡易ステップ検出
    step_indices = []
    for i in range(1, len(valid_y) - 1):
        if np.isnan(valid_y[i - 1]) or np.isnan(valid_y[i]) or np.isnan(valid_y[i + 1]):
            continue
        if valid_y[i] < valid_y[i - 1] and valid_y[i] < valid_y[i + 1]:
            step_indices.append(i)

    duration = valid_t[-1] - valid_t[0] if len(valid_t) >= 2 else np.nan
    step_count = len(step_indices)

    if duration > 0 and step_count > 0:
        pitch_hz = step_count / duration
    else:
        pitch_hz = np.nan

    dx = df["dx_px"].values
    dx_pos = dx[dx > 0]
    if len(dx_pos) > 0 and duration > 0:
        avg_speed_px_per_s = np.nansum(dx_pos) * (1.0 / duration)
    else:
        avg_speed_px_per_s = np.nan

    avg_speed_mps = avg_speed_px_per_s * M_PER_PX if not np.isnan(avg_speed_px_per_s) else np.nan
    if not np.isnan(avg_speed_mps) and not np.isnan(pitch_hz) and pitch_hz > 0:
        stride_m = avg_speed_mps / pitch_hz
    else:
        stride_m = np.nan

    if np.any(~np.isnan(valid_y)):
        stability_std_y_m = float(np.nanstd(valid_y))
        jerk = []
        for i in range(2, len(valid_y)):
            if np.isnan(valid_y[i]) or np.isnan(valid_y[i - 1]) or np.isnan(valid_y[i - 2]):
                continue
            jerk.append(valid_y[i] - 2 * valid_y[i - 1] + valid_y[i - 2])
        stability_jerk = float(np.nanmean(np.abs(jerk))) if len(jerk) > 0 else np.nan
    else:
        stability_std_y_m = np.nan
        stability_jerk = np.nan

    base_score = 10.0
    penalty = 0.0
    if not np.isnan(stability_std_y_m):
        penalty += stability_std_y_m * 100.0
    if not np.isnan(stability_jerk):
        penalty += stability_jerk * 10.0
    stability_score = max(0.0, base_score - penalty)

    summary_row = {
        "frame": "_summary_",
        "com_x": np.nan,
        "com_y": np.nan,
        "com_x_px": np.nan,
        "com_y_px": np.nan,
        "tilt_deg": np.nan,
        "dx_px": np.nan,
        "dy_px": np.nan,
        "dist_px": np.nan,
        "com_y_m": np.nan,
        "speed_mps": np.nan,
        "time_s": np.nan,
        "pitch_hz": pitch_hz,
        "stride_m": stride_m,
        "stability_std_y_m": stability_std_y_m,
        "stability_jerk": stability_jerk,
        "stability_score": stability_score,
        "COM_x": np.nan,
        "COM_y": np.nan,
        "COM_x_px": np.nan,
        "COM_y_px": np.nan,
        "COM_y_m": np.nan,
    }

    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="入力動画(mp4)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"動画が存在しません: {video_path}")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    athlete, video_id = parse_athlete_and_video_id(video_path)
    out_dirs = get_output_dirs(base_dir, athlete, video_id)

    print(f"🎥 動画: {video_path}")
    print(f"選手名: {athlete}, video_id: {video_id}")
    print("⏱ メトリクス解析を開始します...")

    df = analyze_video(video_path)

    csv_path = os.path.join(out_dirs["metrics"], f"{video_id}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ CSV出力完了: {csv_path}")
    print(df.head().to_string())
    print(f"[{len(df)} rows x {len(df.columns)} columns]")


if __name__ == "__main__":
    main()


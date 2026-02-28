# ============================================================
#  pose_angle_analyzer_v1_0.py  （A版正式対応）
#
#  MediaPipe Pose による角度解析スクリプト
#  出力:
#    outputs/{athlete}/{video_id}/angles/{video_id}.csv
#
#  呼び出し:
#    pose_reporter_pdf_ai_v5_5_3.py から subprocess で呼ばれる
# ============================================================

import os
import sys
import argparse
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np


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


def get_output_csv_path(athlete: str, video_id: str) -> str:
    root = os.path.join("outputs", athlete, video_id, "angles")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{video_id}.csv")


def angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cos_angle = (ba @ bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def analyze_angles(video_path: str) -> pd.DataFrame:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not results.pose_landmarks:
            frames.append(
                {
                    "frame": frame_idx,
                    "left_knee_deg": np.nan,
                    "right_knee_deg": np.nan,
                    "left_hip_deg": np.nan,
                    "right_hip_deg": np.nan,
                    "left_elbow_deg": np.nan,
                    "right_elbow_deg": np.nan,
                }
            )
            continue

        lm = results.pose_landmarks.landmark

        def pt(i):
            return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=float)

        left_knee = angle_3pt(
            pt(mp_pose.PoseLandmark.LEFT_HIP.value),
            pt(mp_pose.PoseLandmark.LEFT_KNEE.value),
            pt(mp_pose.PoseLandmark.LEFT_ANKLE.value),
        )
        right_knee = angle_3pt(
            pt(mp_pose.PoseLandmark.RIGHT_HIP.value),
            pt(mp_pose.PoseLandmark.RIGHT_KNEE.value),
            pt(mp_pose.PoseLandmark.RIGHT_ANKLE.value),
        )

        left_hip = angle_3pt(
            pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            pt(mp_pose.PoseLandmark.LEFT_HIP.value),
            pt(mp_pose.PoseLandmark.LEFT_KNEE.value),
        )
        right_hip = angle_3pt(
            pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            pt(mp_pose.PoseLandmark.RIGHT_HIP.value),
            pt(mp_pose.PoseLandmark.RIGHT_KNEE.value),
        )

        left_elbow = angle_3pt(
            pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
            pt(mp_pose.PoseLandmark.LEFT_ELBOW.value),
            pt(mp_pose.PoseLandmark.LEFT_WRIST.value),
        )
        right_elbow = angle_3pt(
            pt(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
            pt(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            pt(mp_pose.PoseLandmark.RIGHT_WRIST.value),
        )

        frames.append(
            {
                "frame": frame_idx,
                "left_knee_deg": left_knee,
                "right_knee_deg": right_knee,
                "left_hip_deg": left_hip,
                "right_hip_deg": right_hip,
                "left_elbow_deg": left_elbow,
                "right_elbow_deg": right_elbow,
            }
        )

    cap.release()
    pose.close()
    return pd.DataFrame(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="入力動画(mp4)")
    args = parser.parse_args()

    video_path = args.video
    print(f"🎥 角度解析開始: {video_path}")

    athlete, video_id = parse_athlete_and_video_id(video_path)
    out_csv = get_output_csv_path(athlete, video_id)

    df = analyze_angles(video_path)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"✅ 角度CSV出力: {out_csv}")


if __name__ == "__main__":
    main()



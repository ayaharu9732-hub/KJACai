# ==========================================================
# 🦶 pose_overlay_draw_feet_v2.py（修正版）
# ----------------------------------------------------------
# ✅ 改良点
#   - --video 引数で動画パスを正しく受け取る
#   - OpenCVで動画が開けない場合に詳細な警告
#   - 出力を outputs/images に自動保存
# ==========================================================

import cv2
import mediapipe as mp
import os
import argparse

OUTPUT_DIR = r"C:\Users\Futamura\KJACai\outputs\images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(video_path):
    """MediaPipe Pose を用いた足部骨格描画"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 動画が読み込めませんでした。パスを確認してください: {video_path}")
        return

    print(f"🎥 入力: {video_path}")
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{basename}_pose_overlay.mp4")

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                )
            out.write(frame)

        cap.release()
        out.release()

    print(f"✅ 骨格描画完了: {frame_count} フレーム出力 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipeで足部骨格を描画")
    parser.add_argument("--video", required=True, help="入力動画のパス")
    args = parser.parse_args()

    process_video(args.video)







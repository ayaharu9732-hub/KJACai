# ============================================================
#  pose_overlay_mediapipe_heavy.py
#
#  目的:
#    - MediaPipe Pose (model_complexity=2) で骨格つき overlay 動画を作る
#    - 被写体が小さくても検出されるように、いったん拡大してから解析
#    - A版ディレクトリ構造:
#        outputs/{athlete}/{video_id}/overlay/{video_id}.mp4
#
#  使い方 (PowerShell):
#    cd C:\Users\Futamura\KJACai
#    python .\scripts\pose_overlay_mediapipe_heavy.py `
#       --video "videos\岩本結々スタブロ11.15.mp4"
#
#  必要なら --output を指定すれば任意パスにも書けます。
# ============================================================

import os
import cv2
import argparse
from typing import Tuple

import mediapipe as mp


# ------------------------------------------------------------
#  共通: athlete / video_id の抽出（A版と合わせる）
# ------------------------------------------------------------
def parse_athlete_and_video_id(video_path: str) -> Tuple[str, str]:
    """
    例: 岩本結々スタブロ11.15.mp4 →
        athlete='岩本結々スタブロ', video_id='岩本結々スタブロ11_15'
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    # 末尾の数字とピリオドを取った部分を選手名とみなす
    i = len(stem) - 1
    while i >= 0 and (stem[i].isdigit() or stem[i] == "."):
        i -= 1

    if i < 0:
        athlete = stem
    else:
        athlete = stem[: i + 1]

    video_id = stem.replace(".", "_")

    athlete = athlete.strip() or stem
    return athlete, video_id


def get_overlay_path(athlete: str, video_id: str) -> str:
    """
    A版正式構造:
      outputs/{athlete}/{video_id}/overlay/{video_id}.mp4
    """
    root = os.path.join("outputs", athlete, video_id, "overlay")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{video_id}.mp4")


# ------------------------------------------------------------
#  メイン処理: Heavy モデルで骨格 overlay 作成
# ------------------------------------------------------------
def build_overlay(video_path: str, output_path: str, scale: float = 1.6) -> None:
    print(f"🎥 overlay 生成開始")
    print(f"  動画:  {video_path}")
    print(f"  出力:  {output_path}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # ★ ここがポイント：Heavyモデル & 閾値低め
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,          # Heavyモデル
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    frame_count = 0
    detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---- いったん拡大してから MediaPipe に渡す ----
        h, w = frame.shape[:2]
        if scale != 1.0:
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            resized = frame

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            detected_frames += 1
            # 太めの線で骨格を描画
            mp_drawing.draw_landmarks(
                resized,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=3, circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=2
                ),
            )

        # 元のサイズに戻して書き出し
        output_frame = cv2.resize(resized, (w0, h0))
        writer.write(output_frame)

        if frame_count % 100 == 0:
            print(f"  処理中... {frame_count}フレーム")

    cap.release()
    writer.release()
    pose.close()

    print(f"✅ overlay 出力完了: {output_path} (frames={frame_count})")
    print(f"   ↳ 骨格検出フレーム数: {detected_frames}")


# ------------------------------------------------------------
#  CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe Heavy 版 overlay 生成スクリプト"
    )
    parser.add_argument("--video", required=True, help="入力動画 (mp4)")
    parser.add_argument(
        "--output",
        help="出力 overlay 動画パス（省略時は A版構造に自動配置）",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.6,
        help="MediaPipe に渡す前の拡大率 (default: 1.6)",
    )
    args = parser.parse_args()

    video_path = args.video
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        athlete, video_id = parse_athlete_and_video_id(video_path)
        output_path = get_overlay_path(athlete, video_id)

    build_overlay(video_path, output_path, scale=args.scale)


if __name__ == "__main__":
    main()

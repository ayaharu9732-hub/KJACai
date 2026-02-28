# ============================================================
#  export_fullbody_frame.py
#
#  - YOLOv8x-pose で人物を検出
#  - バウンディングボックス「面積最大」のフレームを抽出
#  - 全身が最もよく映っているフレームを test_image.jpg として保存
# ============================================================

import os
import cv2
import argparse
from ultralytics import YOLO


def export_fullbody_frame(video_path: str, out_path: str = "test_image.jpg"):
    if not os.path.exists(video_path):
        print(f"[ERROR] 動画が見つかりません: {video_path}")
        return

    print("[INFO] モデル読み込み中: yolov8x-pose.pt")
    model = YOLO("yolov8x-pose.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 動画を開けません")
        return

    best_area = 0
    best_frame = None

    print("[INFO] 全身フレーム抽出開始...")

    # YOLO で1フレームずつ解析（高速・安定）
    for result in model.track(
        source=video_path,
        stream=True,
        persist=True,
        verbose=False,
        conf=0.5,
        iou=0.5,
    ):
        frame = result.orig_img
        boxes = result.boxes

        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        if len(xyxy) == 0:
            continue

        # 最大の人物を選択
        for (x1, y1, x2, y2) in xyxy:
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_frame = frame.copy()

    if best_frame is not None:
        cv2.imwrite(out_path, best_frame)
        print(f"[INFO] test_image.jpg を保存しました → {out_path}")
    else:
        print("[WARN] 全身フレームが見つかりませんでした。")

    print("[INFO] 抽出完了")


def main():
    parser = argparse.ArgumentParser(description="全身が映る最良フレーム抽出ツール")
    parser.add_argument("--video", required=True, help="動画ファイルパス")
    parser.add_argument("--out", default="test_image.jpg", help="保存ファイル名")
    args = parser.parse_args()

    export_fullbody_frame(args.video, args.out)


if __name__ == "__main__":
    main()


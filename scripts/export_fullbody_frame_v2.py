import os
import cv2
import argparse
from ultralytics import YOLO


def expand_bbox(x1, y1, x2, y2, w, h, ratio=0.4):
    """バウンディングボックスを上下左右に拡張"""
    bw = x2 - x1
    bh = y2 - y1

    expand_w = bw * ratio
    expand_h = bh * ratio

    nx1 = max(0, int(x1 - expand_w))
    ny1 = max(0, int(y1 - expand_h))
    nx2 = min(w - 1, int(x2 + expand_w))
    ny2 = min(h - 1, int(y2 + expand_h))

    return nx1, ny1, nx2, ny2


def run(video_path):
    print("[INFO] YOLOv8x-pose 読み込み中…")
    model = YOLO("yolov8x-pose.pt")

    print("[INFO] 動画を解析して人物フレームを抽出中…")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: 動画が開けません")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # YOLO tracking
    results = model.track(
        source=video_path,
        stream=True,
        conf=0.3,
        iou=0.5,
        persist=True,
        verbose=False
    )

    saved = False

    for result in results:
        frame = result.orig_img

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        if len(boxes) == 0:
            continue

        # 面積最大の人物を選ぶ
        areas = []
        for b in boxes:
            x1, y1, x2, y2 = b
            areas.append((x2 - x1) * (y2 - y1))

        best_idx = int(areas.index(max(areas)))
        x1, y1, x2, y2 = boxes[best_idx]

        # 拡張バウンディングボックス
        ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, width, height)

        cropped = frame[ey1:ey2, ex1:ex2]

        # 保存
        save_path = "test_image_cropped.jpg"
        cv2.imwrite(save_path, cropped)
        print(f"[INFO] 抽出成功 → {save_path}")
        saved = True
        break

    cap.release()

    if not saved:
        print("[WARN] 人物の抽出に失敗しました（人物が小さすぎる可能性）")
    else:
        print("[INFO] export_fullbody_frame_v2 完了")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    run(args.video)


if __name__ == "__main__":
    main()


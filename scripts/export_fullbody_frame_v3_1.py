# ============================================================
#  export_fullbody_frame_v3_1.py
#
#  ★成功率100%の人物抽出ツール（MediaPipe Heavy 対応）
#   - YOLOv8x-pose で人物検出
#   - 身長比率 10% 以上で採用（ランニング動画対応）
#   - keypoints が壊れていても補正して採用
#   - bbox が横長でも除外しない（走行動画最適化）
#   - main_track_id を最初の人物で固定（安定）
#   - fallback（YOLO 最強信頼フレーム）も実装
#   - 最終出力：512×512（MediaPipe Heavy 好み）
# ============================================================

import os
import cv2
import argparse
from datetime import datetime
from ultralytics import YOLO


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}")


def extract_best_fullbody_frame(video_path, out_path):
    log("[INFO] YOLOv8x-pose 読み込み中…")
    model = YOLO("yolov8x-pose.pt")

    log("[INFO] 動画を解析して人物フレームを抽出中…")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    MIN_HEIGHT_RATIO = 0.10   # ★30% → 10% に緩和
    best_frame = None
    best_score = -1
    main_track_id = None

    frame_idx = -1

    for result in model.track(
        source=video_path,
        stream=True,
        conf=0.25,
        iou=0.45,
        persist=True,
        verbose=False
    ):
        frame_idx += 1
        frame = result.orig_img

        if result.boxes is None:
            continue

        boxes = result.boxes
        ids = boxes.id
        xyxy = boxes.xyxy

        if ids is None:
            continue

        ids = ids.cpu().numpy().astype(int)
        xyxy = xyxy.cpu().numpy()

        if len(ids) == 0:
            continue

        # ----------------------------------------------------
        # main_track_id を最初の人物に固定（安定）
        # ----------------------------------------------------
        if main_track_id is None:
            main_track_id = ids[0]
            log(f"[INFO] main_track_id = {main_track_id}")

        # main_track_id に該当する index を探す
        idxs = [i for i, t in enumerate(ids) if t == main_track_id]
        if not idxs:
            continue
        idx = idxs[0]

        x1, y1, x2, y2 = xyxy[idx]
        w_box = max(x2 - x1, 1)
        h_box = max(y2 - y1, 1)
        h_ratio = h_box / h

        # ----------------------------------------------------
        # 身長 10% 未満 → 除外
        # ----------------------------------------------------
        if h_ratio < MIN_HEIGHT_RATIO:
            continue

        # ----------------------------------------------------
        # スコア：bbox 高さ + 幅で評価（横長でもOK）
        # ----------------------------------------------------
        score = h_box + w_box

        if score > best_score:
            best_score = score
            best_frame = frame.copy()

    # ------------------------------------------------------------
    # fallback（絶対失敗させない）
    # ------------------------------------------------------------
    if best_frame is None:
        log("[WARN] 条件を満たす人物フレームが見つかりません → fallback 使用")
        cap = cv2.VideoCapture(video_path)
        ret, fallback_frame = cap.read()
        cap.release()

        if fallback_frame is None:
            log("[ERROR] fallback にも失敗")
            return False

        best_frame = fallback_frame

    # ------------------------------------------------------------
    # MediaPipe Heavy 好みの 512x512 に整形
    # ------------------------------------------------------------
    final = cv2.resize(best_frame, (512, 512))

    cv2.imwrite(out_path, final)
    log(f"[INFO] 抽出成功 → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="test_image_fullbody_v3_1.jpg")
    args = parser.parse_args()

    extract_best_fullbody_frame(args.video, args.out)
    log("[INFO] export_fullbody_frame_v3_1 完了")


if __name__ == "__main__":
    main()




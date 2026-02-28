# ============================================================
#  yolo_pose_extractor_v2.py
#
#  役割:
#    - YOLOv8x-pose で全身キーポイント + bbox + トラッキングID を取得
#    - outputs/{athlete}/{video_id}/yolo_pose/ に CSV + overlay 動画を保存
#
#  仕様:
#    - モデル: yolov8x-pose.pt
#    - トラッカー: ByteTrack (ultralytics 標準)
#    - CSV:
#        frame, track_id, x1, y1, x2, y2, img_w, img_h,
#        kp0_x, kp0_y, kp0_conf, ..., kpN_x, kpN_y, kpN_conf
# ============================================================

import os
import sys
import argparse
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except Exception as e:
    print(f"[ERROR] ultralytics のインポートに失敗しました: {e}")
    _YOLO_OK = False


# ------------------------------------------------------------
# 共通: 選手名・video_id 抽出（A版と揃える）
# ------------------------------------------------------------
def parse_athlete_and_video_id(video_path: str):
    """
    例: 岩本結々スタブロ11.15.mp4
       → athlete='岩本結々スタブロ', video_id='岩本結々スタブロ11_15'
    """
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


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{now_str()}] {msg}")


# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
def run_yolo_pose(video_path: str, model_path: str = "yolov8x-pose.pt"):
    if not _YOLO_OK:
        log("ERROR: ultralytics が使えないため中断します。")
        return

    if not os.path.isfile(video_path):
        log(f"ERROR: video ファイルが存在しません → {video_path}")
        return

    athlete, video_id = parse_athlete_and_video_id(video_path)
    out_dir = get_yolo_output_dir(athlete, video_id)

    csv_path = os.path.join(out_dir, f"{video_id}_yolo_keypoints_v2.csv")
    overlay_path = os.path.join(out_dir, f"{video_id}_yolo_overlay_v2.mp4")

    log(f"動画: {video_path}")
    log(f"選手名: {athlete}, video_id: {video_id}")
    log(f"出力CSV : {csv_path}")
    log(f"出力overlay: {overlay_path}")

    # まず動画情報を取得
    cap_info = cv2.VideoCapture(video_path)
    if not cap_info.isOpened():
        log("ERROR: 動画を開けませんでした。")
        return

    fps = cap_info.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    img_w = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()

    # YOLO モデル
    log(f"[YOLO] モデル読み込み: {model_path}")
    model = YOLO(model_path)

    # overlay 動画用 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (img_w, img_h))

    rows = []
    frame_idx = 0
    num_kp = None

    # ByteTrack 付きトラッキング
    # tracker 引数は ultralytics の bytetrack.yaml に依存（標準インストール前提）
    log("[YOLO] トラッキング開始 (ByteTrack)")
    for result in model.track(
        source=video_path,
        stream=True,
        verbose=False,
        tracker="bytetrack.yaml",
    ):
        # result.orig_img: BGR
        frame_idx += 1
        frame_bgr = result.orig_img.copy()
        ih, iw = frame_bgr.shape[:2]

        if num_kp is None and result.keypoints is not None:
            num_kp = result.keypoints.xyn.shape[1]

        # overlay 描画 (ultralytics 標準描画)
        plot_img = result.plot()  # BGR
        writer.write(plot_img)

        if result.boxes is None or result.keypoints is None:
            continue

        boxes = result.boxes
        kpts = result.keypoints

        # tracking ID
        ids = boxes.id.cpu().numpy() if boxes.id is not None else np.arange(len(boxes))

        # bbox (xyxy)
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        # keypoints (x,y,conf) in pixel or normalized?
        # result.keypoints.xy: pixel座標 / xyn: 0-1正規化
        key_xy = kpts.xy.cpu().numpy()      # (N, K, 2)
        key_conf = kpts.conf.cpu().numpy()  # (N, K)

        for det_i in range(len(boxes)):
            tid = int(ids[det_i])
            x1, y1, x2, y2 = xyxy[det_i].tolist()

            row = {
                "frame": frame_idx,
                "track_id": tid,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "img_w": iw,
                "img_h": ih,
            }

            if num_kp is None:
                num_kp = key_xy.shape[1]

            for k in range(num_kp):
                kx, ky = key_xy[det_i, k]
                kc = key_conf[det_i, k]
                row[f"kp{k}_x"] = float(kx)
                row[f"kp{k}_y"] = float(ky)
                row[f"kp{k}_conf"] = float(kc)

            rows.append(row)

    writer.release()
    log(f"[overlay] 動画生成完了: {overlay_path}")

    if not rows:
        log("WARNING: YOLOの検出結果がありませんでした。")
        return

    df = pd.DataFrame(rows)
    log(f"[CSV] 書き出し開始: {csv_path}")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log(f"[CSV] 書き出し完了: {csv_path} (rows={len(df)})")
    log("=== yolo_pose_extractor_v2 完了 ===")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8x Pose + ByteTrack 抽出 (v2)")
    parser.add_argument("--video", required=True, help="入力動画パス (mp4)")
    parser.add_argument("--model", default="yolov8x-pose.pt", help="YOLO pose モデルパス")
    args = parser.parse_args()

    run_yolo_pose(args.video, args.model)


if __name__ == "__main__":
    main()

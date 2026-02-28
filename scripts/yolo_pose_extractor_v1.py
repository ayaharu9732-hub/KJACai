# ============================================================
#  yolo_pose_extractor_v1.py
#
#  役割:
#    - YOLOv8/YOLOv9 Pose で動画から骨格を推定
#    - 1人の選手をトラッキングして keypoints CSV + overlay動画 を出力
#
#  出力:
#    outputs/{athlete}/{video_id}/yolo_pose/{video_id}_keypoints.csv
#    outputs/{athlete}/{video_id}/yolo_pose/{video_id}_yolo_overlay.mp4
#
#  使い方:
#    python scripts/yolo_pose_extractor_v1.py ^
#       --video "C:/Users/Futamura/KJACai/videos/岩本結々スタブロ11.15.mp4"
# ============================================================

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    print("ultralytics がインポートできません。先に `pip install ultralytics` を実行してください。")
    raise


# ------------------------------------------------------------
#  共通ユーティリティ
# ------------------------------------------------------------

def parse_athlete_and_video_id(video_path: str) -> Tuple[str, str]:
    """
    例: 二村遥香10.5.mp4 → athlete='二村遥香', video_id='二村遥香10_5'
    （既存A版と同じロジック）
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    # 末尾の数字とピリオドを削る → 残りを選手名とみなす
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


def get_yolo_output_dirs(athlete: str, video_id: str) -> Dict[str, str]:
    """
    YOLO Pose 専用の出力ディレクトリ
    """
    root = os.path.join("outputs", athlete, video_id, "yolo_pose")
    os.makedirs(root, exist_ok=True)
    return {
        "root": root,
        "csv": os.path.join(root, f"{video_id}_keypoints.csv"),
        "overlay": os.path.join(root, f"{video_id}_yolo_overlay.mp4"),
    }


@dataclass
class PoseFrame:
    frame_idx: int
    time_s: float
    track_id: int
    keypoints: np.ndarray  # (num_kpts, 3) → x, y, conf
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2


# ------------------------------------------------------------
#  YOLO Pose 推論
# ------------------------------------------------------------

def run_yolo_pose(
    video_path: str,
    model_name: str = "yolov8n-pose.pt",
    device: str = "cpu",
) -> List[PoseFrame]:
    """
    YOLOv8/YOLOv9 Pose で 1人の選手をトラッキングした結果を返す。
    - 各フレームにつき、最も大きい bbox を持つ人物を対象選手とする簡易版。
      （必要なら将来トラッキングIDで安定化させる）
    """
    print(f"[YOLO] モデル読み込み: {model_name} (device={device})")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames: List[PoseFrame] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        time_s = frame_idx / fps

        # YOLO推論（ストリーム=Falseなので1フレーム毎）
        # conf は後でフィルタリングに使える
        results = model.predict(
            source=frame,
            save=False,
            verbose=False,
            device=device,
        )

        if len(results) == 0:
            frame_idx += 1
            continue

        r = results[0]

        if r.keypoints is None or len(r.keypoints) == 0:
            frame_idx += 1
            continue

        # このフレーム内の全検出を取得
        kpts_xyc = r.keypoints.xy.cpu().numpy()  # (n, num_kpts, 2)
        kpts_conf = r.keypoints.conf.cpu().numpy()  # (n, num_kpts)

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # (n, 4)
        track_ids = (
            r.boxes.id.cpu().numpy().astype(int)
            if r.boxes.id is not None
            else np.arange(len(boxes_xyxy))
        )

        # 一番大きなbbox（面積最大）を選手とみなす
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        best_idx = int(np.argmax(areas))

        box = boxes_xyxy[best_idx]
        tid = int(track_ids[best_idx])

        xy = kpts_xyc[best_idx]  # (num_kpts, 2)
        conf = kpts_conf[best_idx][:, None]  # (num_kpts, 1)
        kpts = np.concatenate([xy, conf], axis=1)  # (num_kpts, 3)

        frames.append(
            PoseFrame(
                frame_idx=frame_idx,
                time_s=time_s,
                track_id=tid,
                keypoints=kpts,
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            )
        )

        frame_idx += 1

    cap.release()
    print(f"[YOLO] 推論完了: {len(frames)} フレームで選手検出")
    return frames


# ------------------------------------------------------------
#  overlay動画生成
# ------------------------------------------------------------

def render_overlay_video(
    video_path: str,
    frames_pose: List[PoseFrame],
    out_path: str,
    model_skeleton: str = "coco",
) -> None:
    """
    元動画に YOLO Pose の骨格を描画して overlay動画を作る
    """
    if not frames_pose:
        print("[overlay] Poseデータが空のため、overlayは生成しません。")
        return

    print(f"[overlay] 動画生成開始: {out_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # フレーム index → PoseFrame の辞書（高速アクセス用）
    pose_by_frame: Dict[int, PoseFrame] = {pf.frame_idx: pf for pf in frames_pose}

    # COCOの簡易骨格（17点版を想定）
    # (ここはモデルに応じて調整可能：YOLOv8 PoseはCOCO17が標準）
    skeleton_pairs = [
        (5, 7), (7, 9),   # 左腕
        (6, 8), (8, 10),  # 右腕
        (11, 13), (13, 15),  # 左脚
        (12, 14), (14, 16),  # 右脚
        (5, 6),    # 肩
        (11, 12),  # 腰
        (5, 11), (6, 12),  # 体幹
    ]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pf = pose_by_frame.get(frame_idx, None)
        if pf is not None:
            kpts = pf.keypoints  # (num_kpts, 3) [x,y,conf]

            # 点を描画
            for i in range(kpts.shape[0]):
                x, y, conf = kpts[i]
                if conf < 0.3:
                    continue
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

            # 線（骨格）を描画
            for i, j in skeleton_pairs:
                if i >= kpts.shape[0] or j >= kpts.shape[0]:
                    continue
                x1, y1, c1 = kpts[i]
                x2, y2, c2 = kpts[j]
                if c1 < 0.3 or c2 < 0.3:
                    continue
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[overlay] 動画生成完了: {out_path}")


# ------------------------------------------------------------
#  CSV 保存
# ------------------------------------------------------------

def save_keypoints_csv(frames: List[PoseFrame], csv_path: str) -> None:
    """
    PoseFrame のリストを 1つの長いCSVに変換して保存する
    - frame, time_s, track_id, kpt_index, x, y, conf
    """
    print(f"[CSV] 書き出し開始: {csv_path}")

    rows = []
    for pf in frames:
        num_kpts = pf.keypoints.shape[0]
        for k in range(num_kpts):
            x, y, conf = pf.keypoints[k]
            rows.append(
                {
                    "frame": pf.frame_idx,
                    "time_s": pf.time_s,
                    "track_id": pf.track_id,
                    "kpt_index": k,
                    "x": float(x),
                    "y": float(y),
                    "conf": float(conf),
                    "bbox_x1": pf.bbox[0],
                    "bbox_y1": pf.bbox[1],
                    "bbox_x2": pf.bbox[2],
                    "bbox_y2": pf.bbox[3],
                }
            )

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] 書き出し完了: {csv_path} (rows={len(df)})")


# ------------------------------------------------------------
#  main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8/YOLOv9 Pose で骨格CSV＋overlay動画を出力するツール"
    )
    parser.add_argument("--video", required=True, help="入力動画(mp4)")
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO Pose モデルファイル（例: yolov8n-pose.pt / yolov9t-pose.pt）",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0 など")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"動画ファイルが見つかりません: {video_path}")
        sys.exit(1)

    athlete, video_id = parse_athlete_and_video_id(video_path)
    out_dirs = get_yolo_output_dirs(athlete, video_id)

    # 1) YOLO Pose 推論
    frames_pose = run_yolo_pose(
        video_path=video_path,
        model_name=args.model,
        device=args.device,
    )

    # 2) CSV 保存
    save_keypoints_csv(frames_pose, out_dirs["csv"])

    # 3) overlay 動画
    render_overlay_video(
        video_path=video_path,
        frames_pose=frames_pose,
        out_path=out_dirs["overlay"],
    )

    print("=== yolo_pose_extractor_v1 完了 ===")


if __name__ == "__main__":
    main()

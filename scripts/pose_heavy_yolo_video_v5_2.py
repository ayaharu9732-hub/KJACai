# ============================================================
# pose_heavy_yolo_video_v5_3.py  （mp_env / mediapipe 0.10.14 用）
#
# - YOLOv8x-pose でメイン選手のバウンディングボックスを取得
# - その領域を MediaPipe PoseLandmarker Heavy (3D) で解析
# - 4K 向け極太骨格 + 足部ラインをフルフレームに overlay
# - ランドマーク情報を CSV に出力
#
# 依存:
#   - mediapipe==0.10.14  （mp_env）
#   - ultralytics
#   - opencv-python / numpy / pandas
#
# 使い方:
#   (mp_env) python scripts/pose_heavy_yolo_video_v5_3.py --video "C:\...\sample.mp4"
# ============================================================

import os
import time
import math
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.framework.formats import landmark_pb2

# MediaPipe Image クラス
MPImage = mp.Image

# ------------------------------------------------------------
# 描画 & モデル設定
# ------------------------------------------------------------
POINT_SIZE = 20        # キーポイント半径（4K 用 極太）
LINE_THICKNESS = 10    # ライン太さ

# BGR 色
COLOR_LEFT = (0, 255, 0)        # 左側: 緑
COLOR_RIGHT = (0, 0, 255)       # 右側: 赤
COLOR_SKELETON = (0, 255, 255)  # 骨格ライン: シアン系
COLOR_HEEL = (255, 0, 0)        # かかと
COLOR_BALL = (0, 255, 255)      # 母指球
COLOR_TOE = (0, 0, 255)         # つま先

mp_drawing = mp.solutions.drawing_utils

POSE_MODEL_PATH = "pose_landmarker_heavy.task"

# ★ ここが 2.5 に固定（クロップ拡大倍率）
CROP_SCALE = 2.5


# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


def parse_athlete_and_video_id(video_path: str) -> Tuple[str, str]:
    """
    例:
      岩本結々スタブロ11.15.mp4 → athlete=岩本結々スタブロ, video_id=岩本結々スタブロ11_15
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    i = len(stem) - 1
    while i >= 0 and (stem[i].isdigit() or stem[i] == "."):
        i -= 1

    if i < 0:
        athlete = stem
    else:
        athlete = stem[: i + 1]

    athlete = athlete.strip() or stem
    video_id = stem.replace(".", "_")
    return athlete, video_id


def get_output_paths(athlete: str, video_id: str) -> Dict[str, str]:
    """
    出力ディレクトリ:
      outputs/{athlete}/{video_id}/pose_heavy_v5/
    """
    root = os.path.join("outputs", athlete, video_id, "pose_heavy_v5")
    os.makedirs(root, exist_ok=True)
    overlay_path = os.path.join(root, f"{video_id}_pose_heavy_v5_overlay.mp4")
    csv_path = os.path.join(root, f"{video_id}_pose_heavy_v5_landmarks.csv")
    return {"root": root, "overlay": overlay_path, "csv": csv_path}


# ------------------------------------------------------------
# 足部近似 & 描画
# ------------------------------------------------------------
def estimate_foot_points_from_pixels(
    ankle_x: float,
    ankle_y: float,
    bbox_width: float,
) -> Dict[str, Tuple[float, float]]:
    """
    ankle のピクセル位置と bbox の幅から、
    heel / ball / toe を同一水平線上に近似配置。
    """
    foot_len = bbox_width * 0.18
    if foot_len <= 0:
        foot_len = 20.0  # 下限

    offset_heel = -0.3 * foot_len
    offset_ball = +0.4 * foot_len
    offset_toe = +0.7 * foot_len

    heel_x = ankle_x + offset_heel
    ball_x = ankle_x + offset_ball
    toe_x = ankle_x + offset_toe

    return {
        "ankle": (ankle_x, ankle_y),
        "heel": (heel_x, ankle_y),
        "ball": (ball_x, ankle_y),
        "toe": (toe_x, ankle_y),
    }


def draw_feet(image: np.ndarray, foot_points: Dict[str, Tuple[float, float]]) -> None:
    """足部の点 + ラインを極太で描画"""
    if not foot_points:
        return

    ax, ay = foot_points["ankle"]
    hx, hy = foot_points["heel"]
    bx, by = foot_points["ball"]
    tx, ty = foot_points["toe"]

    # 点（20px）
    for (x, y, col) in [
        (ax, ay, (255, 255, 255)),
        (hx, hy, COLOR_HEEL),
        (bx, by, COLOR_BALL),
        (tx, ty, COLOR_TOE),
    ]:
        cv2.circle(image, (int(x), int(y)), POINT_SIZE, col, -1)

    # ライン（10px）
    cv2.line(image, (int(hx), int(hy)), (int(bx), int(by)), COLOR_SKELETON, LINE_THICKNESS)
    cv2.line(image, (int(bx), int(by)), (int(tx), int(ty)), COLOR_SKELETON, LINE_THICKNESS)


# ------------------------------------------------------------
# croppped → full frame 座標変換
# ------------------------------------------------------------
def to_fullframe_landmarks(
    pose_result,
    bbox: Tuple[int, int, int, int],
    full_w: int,
    full_h: int,
) -> landmark_pb2.NormalizedLandmarkList:
    """
    cropped 座標系の MediaPipe ランドマークを
    フル解像度(3840x2160 など)の正規化(0-1)へ変換
    """
    x1, y1, x2, y2 = bbox
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    # ★ ここが超重要：pose_landmarks[0] が NormalizedLandmarkList
    src_list = pose_result.pose_landmarks[0]

    dst_list = landmark_pb2.NormalizedLandmarkList()
    for lm in src_list.landmark:
        fx = (x1 + lm.x * crop_w) / full_w
        fy = (y1 + lm.y * crop_h) / full_h

        new_lm = landmark_pb2.NormalizedLandmark(
            x=float(fx),
            y=float(fy),
            z=float(lm.z),
            visibility=float(getattr(lm, "visibility", 0.0)),
        )
        dst_list.landmark.append(new_lm)

    return dst_list


def draw_heavy_skeleton_fullframe(
    frame: np.ndarray,
    full_lms: landmark_pb2.NormalizedLandmarkList,
) -> None:
    """フルフレーム正規化座標の LandmarkList を極太スタイルで描画"""
    landmark_spec = mp_drawing.DrawingSpec(
        color=(0, 255, 0), thickness=LINE_THICKNESS, circle_radius=POINT_SIZE
    )
    connection_spec = mp_drawing.DrawingSpec(
        color=COLOR_SKELETON, thickness=LINE_THICKNESS
    )

    mp_drawing.draw_landmarks(
        frame,
        full_lms,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=landmark_spec,
        connection_drawing_spec=connection_spec,
    )


# ------------------------------------------------------------
# メイン処理
# ------------------------------------------------------------
def run(video_path: str) -> None:
    log(f"動画: {video_path}")
    if not os.path.exists(video_path):
        log("[ERROR] 動画が存在しません")
        return

    if not os.path.exists(POSE_MODEL_PATH):
        log(f"[ERROR] PoseLandmarker モデルが見つかりません: {POSE_MODEL_PATH}")
        return

    athlete, video_id = parse_athlete_and_video_id(video_path)
    out_paths = get_output_paths(athlete, video_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("[ERROR] 動画を開けませんでした")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    log(f"選手名: {athlete}, video_id: {video_id}")
    log(f"出力Overlay: {out_paths['overlay']}")
    log(f"出力CSV     : {out_paths['csv']}")
    log(f"FPS = {fps}")
    log(f"解像度 = {width} x {height}, 総フレーム = {total_frames}")
    log(f"PoseLandmarker モデル: {POSE_MODEL_PATH}")

    # MediaPipe PoseLandmarker (IMAGE モード)
    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    pose_options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)

    # YOLO モデル
    log("YOLOv8x-pose 読み込み中…")
    yolo_model = YOLO("yolov8x-pose.pt")

    # Overlay VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_paths["overlay"]), exist_ok=True)
    writer = cv2.VideoWriter(out_paths["overlay"], fourcc, fps, (width, height))

    rows: List[Dict[str, float]] = []
    main_track_id: Optional[int] = None
    start_time = time.time()
    frame_idx = -1

    # YOLO track ストリーム
    for result in yolo_model.track(
        source=video_path,
        stream=True,
        conf=0.5,
        iou=0.5,
        persist=True,
        verbose=False,
    ):
        frame_idx += 1
        if frame_idx >= total_frames:
            break

        if frame_idx % 30 == 0:
            log(f"処理中… frame={frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")

        frame = result.orig_img  # BGR
        if frame is None:
            continue

        fh, fw = frame.shape[:2]

        # 人物がいないフレーム
        if result.boxes is None or result.boxes.id is None:
            writer.write(frame)
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        if len(ids) == 0:
            writer.write(frame)
            continue

        # main_track_id 未確定なら「面積最大の人」
        if main_track_id is None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))
            main_track_id = int(ids[best_idx])
            log(f"main_track_id = {main_track_id}")

        idxs = [i for i, tid in enumerate(ids) if tid == main_track_id]
        if not idxs:
            writer.write(frame)
            continue
        det_idx = idxs[0]

        x1, y1, x2, y2 = boxes[det_idx]
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), fw - 1)
        y2 = min(int(y2), fh - 1)
        if x2 <= x1 or y2 <= y1:
            writer.write(frame)
            continue

        bbox_w = x2 - x1

        # ROI 切り出し
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            writer.write(frame)
            continue

        # ★ ここで CROP_SCALE 倍に拡大してから MediaPipe へ
        crop_scaled = cv2.resize(
            crop,
            None,
            fx=CROP_SCALE,
            fy=CROP_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(crop_scaled, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=rgb)
        pose_result = landmarker.detect(mp_image)

        if pose_result.pose_landmarks is None or len(pose_result.pose_landmarks) == 0:
            writer.write(frame)
            continue

        # フルフレーム正規化座標へ変換
        full_lms = to_fullframe_landmarks(
            pose_result,
            (x1, y1, x2, y2),
            fw,
            fh,
        )

        # 骨格描画
        draw_heavy_skeleton_fullframe(frame, full_lms)

        # 左右足首ピクセル（index: 27=L ankle, 28=R ankle）
        left_ankle_px = None
        right_ankle_px = None
        if len(full_lms.landmark) > 28:
            l_ankle = full_lms.landmark[27]
            r_ankle = full_lms.landmark[28]
            left_ankle_px = (l_ankle.x * fw, l_ankle.y * fh)
            right_ankle_px = (r_ankle.x * fw, r_ankle.y * fh)

        # 足部ライン
        if left_ankle_px is not None:
            fp_l = estimate_foot_points_from_pixels(
                ankle_x=left_ankle_px[0],
                ankle_y=left_ankle_px[1],
                bbox_width=bbox_w,
            )
            draw_feet(frame, fp_l)

        if right_ankle_px is not None:
            fp_r = estimate_foot_points_from_pixels(
                ankle_x=right_ankle_px[0],
                ankle_y=right_ankle_px[1],
                bbox_width=bbox_w,
            )
            draw_feet(frame, fp_r)

        # バウンディングボックス & フレーム番号
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{frame_idx}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # Overlay 出力
        writer.write(frame)

        # CSV 行
        t_sec = frame_idx / fps
        row: Dict[str, float] = {"frame": frame_idx, "time_s": t_sec}

        for i, lm in enumerate(full_lms.landmark):
            row[f"lm{i}_x_norm"] = float(lm.x)
            row[f"lm{i}_y_norm"] = float(lm.y)
            row[f"lm{i}_z_rel"] = float(lm.z)
            row[f"lm{i}_vis"] = float(getattr(lm, "visibility", 0.0))
            row[f"lm{i}_x_px"] = float(lm.x * fw)
            row[f"lm{i}_y_px"] = float(lm.y * fh)

        if left_ankle_px is not None:
            row["left_ankle_x"] = float(left_ankle_px[0])
            row["left_ankle_y"] = float(left_ankle_px[1])
        else:
            row["left_ankle_x"] = math.nan
            row["left_ankle_y"] = math.nan

        if right_ankle_px is not None:
            row["right_ankle_x"] = float(right_ankle_px[0])
            row["right_ankle_y"] = float(right_ankle_px[1])
        else:
            row["right_ankle_x"] = math.nan
            row["right_ankle_y"] = math.nan

        rows.append(row)

    # 後処理
    writer.release()
    landmarker.close()

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(out_paths["csv"]), exist_ok=True)
        df.to_csv(out_paths["csv"], index=False, encoding="utf-8-sig")
        log(f"CSV 書き出し完了: {out_paths['csv']} (rows={len(df)})")
    else:
        log("[WARN] 有効なランドマーク行が 0 件でした")

    elapsed = time.time() - start_time
    log(f"Overlay 完了: {out_paths['overlay']}")
    log(f"=== pose_heavy_yolo_video_v5_3 完了 (処理時間: {elapsed/60:.1f} 分) ===")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8x-pose + MediaPipe Heavy PoseLandmarker 4K 極太骨格 overlay (v5.3, CROP_SCALE=2.5)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="入力動画パス (mp4)",
    )
    args = parser.parse_args()
    run(args.video)


if __name__ == "__main__":
    main()



# ============================================================
# pose_heavy_yolo_video_v5.py
#
# YOLOv8x-pose + MediaPipe Heavy PoseLandmarker (3D)
#  - YOLO: 主体選手のトラッキング & クロップ領域決定
#  - MediaPipe Heavy: 33ランドマーク(3D)を高精度推定
#  - 自前の POSE_CONNECTIONS で骨格を描画（MPスタイル風）
#  - 足部（ankle / heel / foot_index）を強調表示
#
# 出力（A版準拠）:
#   outputs/{athlete}/{video_id}/pose_heavy_v5/
#     - {video_id}_pose_heavy_v5_overlay.mp4
#     - {video_id}_pose_heavy_v5_landmarks.csv
#
# 使い方:
#   (mp_env) python scripts/pose_heavy_yolo_video_v5.py \
#       --video "C:\...\岩本結々スタブロ11.15.mp4"
# ============================================================

import os
import math
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ============================================================
# ログユーティリティ
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


# ============================================================
# A版と同じ 選手名 / video_id 抽出
# ============================================================

def parse_athlete_and_video_id(video_path: str) -> Tuple[str, str]:
    """
    ex: 岩本結々スタブロ11.15.mp4 →
        athlete  = 岩本結々スタブロ
        video_id = 岩本結々スタブロ11_15
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

    log(f"[INFO] 選手名: {athlete}, video_id: {video_id}")
    return athlete, video_id


def get_output_paths(athlete: str, video_id: str) -> Dict[str, str]:
    root = os.path.join("outputs", athlete, video_id, "pose_heavy_v5")
    os.makedirs(root, exist_ok=True)

    overlay_path = os.path.join(
        root, f"{video_id}_pose_heavy_v5_overlay.mp4"
    )
    csv_path = os.path.join(
        root, f"{video_id}_pose_heavy_v5_landmarks.csv"
    )

    return {
        "root": root,
        "overlay": overlay_path,
        "csv": csv_path,
    }


# ============================================================
# MediaPipe Pose Connections（33点版）自前定義
# （BlazePose Full Body と同等）
# ============================================================

POSE_CONNECTIONS = [
    # 顔〜上半身
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 肩
    (11, 12),
    # 左腕
    (11, 13), (13, 15),
    # 右腕
    (12, 14), (14, 16),
    # 体幹
    (11, 23), (12, 24), (23, 24),
    # 左脚
    (23, 25), (25, 27), (27, 29), (29, 31),
    # 右脚
    (24, 26), (26, 28), (28, 30), (30, 32),
]

# ランドマーク名（CSV用）
POSE_LM_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# 足部インデックス（BlazePose仕様）
ID_LEFT_ANKLE = 27
ID_RIGHT_ANKLE = 28
ID_LEFT_HEEL = 29
ID_RIGHT_HEEL = 30
ID_LEFT_TOE = 31
ID_RIGHT_TOE = 32

# 描画色（BGR）
COLOR_LANDMARK = (0, 255, 0)        # 緑
COLOR_CONNECTION = (0, 200, 255)    # 水色
COLOR_FEET = (0, 128, 255)          # 足部強調
COLOR_FEET_TOE = (0, 0, 255)        # つま先（赤）


# ============================================================
# MediaPipe Heavy PoseLandmarker 初期化
# ============================================================

def create_pose_landmarker(model_path: str) -> mp_vision.PoseLandmarker:
    """
    MediaPipe Heavy PoseLandmarker を IMAGE モードで初期化
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PoseLandmarker モデルが見つかりません: {model_path}")

    base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return landmarker


# ============================================================
# 1フレーム分の骨格描画
# ============================================================

def draw_pose_on_frame(
    frame: np.ndarray,
    crop_rect: Tuple[int, int, int, int],
    landmarks: List[mp_vision.PoseLandmarkerResult],
) -> None:
    """
    frame: 元のフルフレーム (BGR)
    crop_rect: (x1, y1, x2, y2) ※クロップの左上/右下（ピクセル）
    landmarks: PoseLandmarkerResult の pose_landmarks[0] を利用
    """
    h, w, _ = frame.shape
    x1, y1, x2, y2 = crop_rect
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    if not landmarks:
        return

    lm_list = landmarks[0].pose_landmarks
    if not lm_list:
        return

    # 33 点
    lm = lm_list[0]  # PoseLandmarkerResult -> pose_landmarks[0]

    # まずは全ランドマークを global 座標化
    pts = []
    vis = []
    for lm_i in lm:
        # 正規化座標 (0~1) → crop 内 → フルフレーム
        cx = lm_i.x * crop_w + x1
        cy = lm_i.y * crop_h + y1
        pts.append((int(cx), int(cy)))
        vis.append(lm_i.visibility)

    # 接続線を描く
    for i, j in POSE_CONNECTIONS:
        if i >= len(pts) or j >= len(pts):
            continue
        # visibility が低すぎる場合はスキップ
        if vis[i] < 0.3 or vis[j] < 0.3:
            continue
        x1g, y1g = pts[i]
        x2g, y2g = pts[j]
        cv2.line(frame, (x1g, y1g), (x2g, y2g), COLOR_CONNECTION, 2)

    # ランドマーク点
    for (cx, cy), v in zip(pts, vis):
        if v < 0.3:
            continue
        cv2.circle(frame, (cx, cy), 3, COLOR_LANDMARK, -1)

    # 足部強調（ankle / heel / toe）
    def safe_draw(idx: int, color: Tuple[int, int, int], radius: int = 5):
        if idx < len(pts) and vis[idx] >= 0.3:
            xg, yg = pts[idx]
            cv2.circle(frame, (xg, yg), radius, color, -1)

    safe_draw(ID_LEFT_ANKLE, COLOR_FEET, 6)
    safe_draw(ID_RIGHT_ANKLE, COLOR_FEET, 6)
    safe_draw(ID_LEFT_HEEL, COLOR_FEET, 5)
    safe_draw(ID_RIGHT_HEEL, COLOR_FEET, 5)
    safe_draw(ID_LEFT_TOE, COLOR_FEET_TOE, 5)
    safe_draw(ID_RIGHT_TOE, COLOR_FEET_TOE, 5)


# ============================================================
# 1フレーム分のランドマーク → CSV 行作成
# ============================================================

def landmarks_to_row(
    frame_idx: int,
    time_s: float,
    crop_rect: Tuple[int, int, int, int],
    result: Optional[mp_vision.PoseLandmarkerResult],
) -> Dict[str, float]:
    """
    PoseLandmarkerResult から、フルフレーム座標系に変換した
    ランドマーク情報を 1 行分の dict にまとめる
    """
    row: Dict[str, float] = {
        "frame": frame_idx,
        "time_s": time_s,
    }

    x1, y1, x2, y2 = crop_rect
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)

    if result is None or not result.pose_landmarks:
        # 全て NaN
        for i, name in enumerate(POSE_LM_NAMES):
            row[f"{name}_x"] = math.nan
            row[f"{name}_y"] = math.nan
            row[f"{name}_z"] = math.nan
            row[f"{name}_score"] = math.nan
        return row

    lm = result.pose_landmarks[0]  # 1人分
    n = min(len(lm), len(POSE_LM_NAMES))

    for i in range(len(POSE_LM_NAMES)):
        key = POSE_LM_NAMES[i]
        if i < n:
            lm_i = lm[i]
            cx = lm_i.x * crop_w + x1
            cy = lm_i.y * crop_h + y1
            row[f"{key}_x"] = float(cx)
            row[f"{key}_y"] = float(cy)
            row[f"{key}_z"] = float(lm_i.z)
            row[f"{key}_score"] = float(lm_i.visibility)
        else:
            row[f"{key}_x"] = math.nan
            row[f"{key}_y"] = math.nan
            row[f"{key}_z"] = math.nan
            row[f"{key}_score"] = math.nan

    return row


# ============================================================
# メイン処理
# ============================================================

def run(video_path: str) -> None:
    log(f"[INFO] 動画: {video_path}")
    if not os.path.exists(video_path):
        log("[ERROR] 動画ファイルが見つかりません。")
        return

    athlete, video_id = parse_athlete_and_video_id(video_path)
    paths = get_output_paths(athlete, video_id)
    log(f"[INFO] 出力Overlay: {paths['overlay']}")
    log(f"[INFO] 出力CSV     : {paths['csv']}")

    # 動画情報
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("[ERROR] 動画を開けませんでした。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"[INFO] FPS = {fps:.3f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(paths["overlay"], fourcc, fps, (width, height))

    # モデルロード
    log("[INFO] YOLOv8x-pose 読み込み中…")
    yolo_model = YOLO("yolov8x-pose.pt")

    pose_model_path = os.path.abspath("pose_landmarker_heavy.task")
    log(f"[INFO] PoseLandmarker モデル: {pose_model_path}")
    pose_landmarker = create_pose_landmarker(pose_model_path)

    # YOLO トラッキング
    log("[INFO] YOLO トラッキング開始 (ByteTrack)")
    main_track_id: Optional[int] = None

    rows: List[Dict[str, float]] = []
    frame_idx = -1

    # YOLO stream
    for result in yolo_model.track(
        source=video_path,
        stream=True,
        verbose=False,
        conf=0.5,
        iou=0.5,
        persist=True,
    ):
        frame_idx += 1
        time_s = frame_idx / fps

        frame = result.orig_img  # BGR
        if frame is None:
            # 念のためフォールバック
            ret, frm = cap.read()
            if not ret:
                break
            frame = frm

        # デフォルトクロップ = フルフレーム
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, width, height

        # YOLO 検出から main_track_id を決める / bbox を取得
        boxes = result.boxes
        if boxes is not None and boxes.id is not None and len(boxes.id) > 0:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()  # (n,4)

            # main_track_id 未確定なら、面積最大の人を採用
            if main_track_id is None:
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                best_idx = int(np.argmax(areas))
                main_track_id = int(ids[best_idx])
                log(f"[INFO] main_track_id = {main_track_id}")

            # このフレームに main_track_id がいるか
            idxs = [i for i, tid in enumerate(ids) if tid == main_track_id]
            if idxs:
                j = idxs[0]
                x1, y1, x2, y2 = xyxy[j]
                bw = x2 - x1
                bh = y2 - y1

                # 余裕をもたせて拡張
                margin = 0.4
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                half_w = bw * (0.5 + margin)
                half_h = bh * (0.5 + margin)

                crop_x1 = int(max(cx - half_w, 0))
                crop_x2 = int(min(cx + half_w, width))
                crop_y1 = int(max(cy - half_h, 0))
                crop_y2 = int(min(cy + half_h, height))

        # クロップを作成
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            crop = frame.copy()
            crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, width, height

        # MediaPipe に渡す画像（RGB）
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=crop_rgb,
        )

        # Heavy Pose 推論
        pose_result: Optional[mp_vision.PoseLandmarkerResult] = None
        try:
            pose_result = pose_landmarker.detect(mp_image)
        except Exception as e:
            log(f"[WARN] PoseLandmarker detect エラー: {e}")
            pose_result = None

        # オーバーレイ描画
        draw_pose_on_frame(
            frame,
            (crop_x1, crop_y1, crop_x2, crop_y2),
            [pose_result] if pose_result is not None else [],
        )

        # フレーム番号を左上に表示
        cv2.putText(
            frame,
            f"frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        # CSV 行
        row = landmarks_to_row(
            frame_idx,
            time_s,
            (crop_x1, crop_y1, crop_x2, crop_y2),
            pose_result,
        )
        rows.append(row)

    # 終了処理
    cap.release()
    writer.release()
    pose_landmarker.close()

    # CSV 保存
    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    log(f"[INFO] CSV 書き出し完了: {paths['csv']} (rows={len(df)})")
    log(f"[INFO] Overlay 動画生成完了: {paths['overlay']}")
    log("[INFO] === pose_heavy_yolo_video_v5 完了 ===")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8x-pose + MediaPipe Heavy PoseLandmarker v5"
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



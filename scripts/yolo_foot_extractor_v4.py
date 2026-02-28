# ============================================================
#  yolo_foot_extractor_v4.py  （修正版）
#
#  - YOLOv8x Pose で右足 / 左足の ankle を取得
#  - バウンディングボックス幅から足長を近似し
#    heel / ball / toe を水平線上に配置（見やすさ重視）
#  - ★ トラッキング(ByteTrack)は廃止
#      → 各フレームごとに「一番大きい人」を採用
#  - YOLO の骨格は result.plot() で毎フレーム描画
#
#  A版準拠ディレクトリ:
#      outputs/{athlete}/{video_id}/foot_v4/
#  出力:
#      CSV : {video_id}_foot_v4.csv
#      MP4 : {video_id}_foot_overlay_v4.mp4
# ============================================================

import os
import cv2
import math
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from ultralytics import YOLO


# ------------------------------------------------------------
# ログ
# ------------------------------------------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


# ------------------------------------------------------------
# A版と同じ 選手名 / video_id 抽出
# ------------------------------------------------------------
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
    log(f"選手名: {athlete}, video_id: {video_id}")
    return athlete, video_id


def get_output_paths(athlete: str, video_id: str) -> Dict[str, str]:
    root = os.path.join("outputs", athlete, video_id, "foot_v4")
    os.makedirs(root, exist_ok=True)
    csv_path = os.path.join(root, f"{video_id}_foot_v4.csv")
    overlay_path = os.path.join(root, f"{video_id}_foot_overlay_v4.mp4")
    return {
        "root": root,
        "csv": csv_path,
        "overlay": overlay_path,
    }


# ------------------------------------------------------------
# 足部ポイントの近似計算
# ------------------------------------------------------------
def estimate_foot_points(
    ankle_x: float,
    ankle_y: float,
    bbox_width: float,
    is_left: bool,
) -> Dict[str, Tuple[float, float]]:
    """
    ankle を基準に、水平ライン上に heel / ball / toe を並べる簡易近似。

    - 足長 ≒ bbox_width * 0.18（経験的な比率）
    - heel: ankle より少し後ろ
    - ball: ankle より少し前
    - toe : ankle よりさらに前
    左右で前後の向きは区別しない（カメラの向き次第で変わるため）。
    """
    foot_len = bbox_width * 0.18
    if foot_len <= 0:
        foot_len = 20.0  # 最低長さ（px）

    offset_heel = -0.3 * foot_len
    offset_ball = +0.4 * foot_len
    offset_toe = +0.7 * foot_len

    heel_x = ankle_x + offset_heel
    ball_x = ankle_x + offset_ball
    toe_x = ankle_x + offset_toe

    heel_y = ankle_y
    ball_y = ankle_y
    toe_y = ankle_y

    return {
        "ankle": (ankle_x, ankle_y),
        "heel": (heel_x, heel_y),
        "ball": (ball_x, ball_y),
        "toe": (toe_x, toe_y),
    }


# ------------------------------------------------------------
# メイン処理（★トラッキング無し）
# ------------------------------------------------------------
def run(video_path: str) -> None:
    log(f"動画: {video_path}")
    if not os.path.exists(video_path):
        log("ERROR: 動画ファイルが見つかりません。")
        return

    athlete, video_id = parse_athlete_and_video_id(video_path)
    paths = get_output_paths(athlete, video_id)
    log(f"出力CSV: {paths['csv']}")
    log(f"出力Overlay: {paths['overlay']}")

    # 元動画情報
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("ERROR: 動画を開けませんでした。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(paths["overlay"], fourcc, fps, (width, height))

    # YOLO モデル
    log("[YOLO] モデル読み込み: yolov8x-pose.pt")
    model = YOLO("yolov8x-pose.pt")

    rows: List[Dict[str, float]] = []
    frame_idx = -1

    # ★ track ではなく普通の推論（各フレーム独立）
    for result in model(
        source=video_path,
        stream=True,
        verbose=False,
        conf=0.25,   # 少し緩め
        iou=0.5,
    ):
        frame_idx += 1
        time_s = frame_idx / fps

        # YOLO の骨格付き画像をベースフレームにする
        frame = result.plot()  # BGR, boxes + keypoints 描画済み

        # 人が検出されていなければ、そのまま書き出し
        if result.boxes is None or result.keypoints is None:
            writer.write(frame)
            continue

        boxes = result.boxes
        kpts = result.keypoints

        xyxy = boxes.xyxy.cpu().numpy()  # (n, 4)
        n_det = xyxy.shape[0]
        if n_det == 0:
            writer.write(frame)
            continue

        # ★ 各フレームで「一番大きい人」を採用
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best_idx = int(np.argmax(areas))

        x1, y1, x2, y2 = xyxy[best_idx]
        bbox_w = max(float(x2 - x1), 1.0)

        # キーポイント座標
        k_xy = kpts.xy[best_idx].cpu().numpy()  # (K, 2)

        # 信頼度
        if hasattr(kpts, "conf") and kpts.conf is not None:
            k_conf_raw = kpts.conf[best_idx].cpu().numpy()
            if k_conf_raw.ndim == 2:
                k_conf = k_conf_raw[:, 0]
            else:
                k_conf = k_conf_raw
        else:
            k_conf = np.ones((k_xy.shape[0],), dtype=np.float32)

        K = k_xy.shape[0]

        left_ankle = None
        right_ankle = None

        # YOLOv8 COCO format:
        # 15: left ankle, 16: right ankle （K>=17想定）
        if K >= 17:
            if k_conf[15] > 0.15:
                left_ankle = (float(k_xy[15, 0]), float(k_xy[15, 1]))
            if k_conf[16] > 0.15:
                right_ankle = (float(k_xy[16, 0]), float(k_xy[16, 1]))

        # 足部ポイント初期化
        def init_points():
            return {
                "ankle": (math.nan, math.nan),
                "heel": (math.nan, math.nan),
                "ball": (math.nan, math.nan),
                "toe": (math.nan, math.nan),
            }

        left_points = init_points()
        right_points = init_points()

        if left_ankle is not None:
            left_points = estimate_foot_points(
                ankle_x=left_ankle[0],
                ankle_y=left_ankle[1],
                bbox_width=bbox_w,
                is_left=True,
            )
        if right_ankle is not None:
            right_points = estimate_foot_points(
                ankle_x=right_ankle[0],
                ankle_y=right_ankle[1],
                bbox_width=bbox_w,
                is_left=False,
            )

        # -------------- Overlay に足部を描画 --------------
        # バウンディングボックス
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            1,
        )

        def draw_foot(fp: Dict[str, Tuple[float, float]], color: Tuple[int, int, int]):
            ax, ay = fp["ankle"]
            hx, hy = fp["heel"]
            bx, by = fp["ball"]
            tx, ty = fp["toe"]

            if not math.isnan(ax):
                cv2.circle(frame, (int(ax), int(ay)), 4, color, -1)
            if not math.isnan(hx):
                cv2.circle(frame, (int(hx), int(hy)), 4, (255, 0, 0), -1)    # heel
            if not math.isnan(bx):
                cv2.circle(frame, (int(bx), int(by)), 4, (0, 255, 255), -1)  # ball
            if not math.isnan(tx):
                cv2.circle(frame, (int(tx), int(ty)), 4, (0, 0, 255), -1)    # toe

            if not math.isnan(hx) and not math.isnan(bx):
                cv2.line(frame, (int(hx), int(hy)), (int(bx), int(by)), color, 2)
            if not math.isnan(bx) and not math.isnan(tx):
                cv2.line(frame, (int(bx), int(by)), (int(tx), int(ty)), color, 2)

        # 左足は緑、右足はピンク
        draw_foot(left_points, (0, 255, 0))
        draw_foot(right_points, (255, 0, 255))

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

        # -------------- CSV 行を追加 --------------
        row = {
            "frame": frame_idx,
            "time_s": time_s,
        }
        for side, pts in [("left", left_points), ("right", right_points)]:
            for name in ["ankle", "heel", "ball", "toe"]:
                x, y = pts[name]
                row[f"{side}_{name}_x"] = float(x)
                row[f"{side}_{name}_y"] = float(y)
        rows.append(row)

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    log(f"[CSV] 書き出し完了: {paths['csv']} (rows={len(df)})")
    log(f"[overlay] 動画生成完了: {paths['overlay']}")
    log("=== yolo_foot_extractor_v4 完了 ===")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Pose による足部ポイント抽出 v4 (A版準拠 / トラッキング無し)"
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



# ============================================================
#  export_fullbody_frame_v4.py
#
#  - YOLOv8x-pose でメインの人物をトラッキング
#  - 1番良いフレームを1枚だけ選ぶ
#  - バウンディングボックス中心を基準に
#    「人物が中央に来る正方形クロップ画像」を作る
#  - 512x512 にリサイズして保存
#
#  使い方（mp_env を有効化してから）:
#    python scripts/export_fullbody_frame_v4.py ^
#       --video "C:\...\your_video.mp4" ^
#       --out "test_image_fullbody_v4.jpg"
#
#  出力画像は MediaPipe Heavy PoseLandmarker v3 の
#  テスト入力としてそのまま使えます。
# ============================================================

import os
import cv2
import argparse
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------
# ログ
# ------------------------------------------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


# ------------------------------------------------------------
# メイン人物（main_track_id）を決めて、ベストフレームを1枚選ぶ
# ------------------------------------------------------------
def find_best_frame_and_bbox(
    video_path: str,
    model: YOLO,
    min_height_ratio: float = 0.12,   # 画面高さに対する最小身長比
    min_conf: float = 0.35,           # 検出確度の下限
) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
    """
    戻り値:
        best_frame (BGR画像) or None
        best_bbox  (x1, y1, x2, y2) or None
    """
    log("[INFO] 動画を解析して人物フレームを抽出中…")

    main_track_id: Optional[int] = None
    best_frame: Optional[np.ndarray] = None
    best_bbox: Optional[Tuple[float, float, float, float]] = None
    best_score: float = -1.0

    frame_idx = -1

    for result in model.track(
        source=video_path,
        stream=True,
        verbose=False,
        conf=min_conf,
        iou=0.5,
        persist=True,
    ):
        frame_idx += 1
        frame = result.orig_img  # BGR
        if frame is None:
            continue

        h, w = frame.shape[:2]

        boxes = result.boxes
        kpts = result.keypoints

        # 人物検出が無いフレームはスキップ
        if boxes is None or boxes.id is None or len(boxes) == 0:
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()  # (n, 4)
        confs = boxes.conf.cpu().numpy()  # (n,)

        # main_track_id が未定なら「面積最大の人」を採用
        if main_track_id is None:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best_idx = int(np.argmax(areas))
            main_track_id = int(ids[best_idx])
            log(f"[INFO] main_track_id = {main_track_id}")

        # このフレームの中から main_track_id に対応する index を探す
        idx_list = [i for i, tid in enumerate(ids) if int(tid) == main_track_id]
        if not idx_list:
            continue
        idx = idx_list[0]

        x1, y1, x2, y2 = xyxy[idx]
        conf = float(confs[idx])

        # バウンディングボックス高さが小さすぎるものは除外
        box_h = max(float(y2 - y1), 1.0)
        height_ratio = box_h / float(h)

        if height_ratio < min_height_ratio:
            # 人が小さすぎて解析に向かない
            continue

        # 今回は「高さ比 × 確度」をスコアにする
        score = height_ratio * conf

        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_bbox = (float(x1), float(y1), float(x2), float(y2))

    return best_frame, best_bbox


# ------------------------------------------------------------
# 中心を基準にした正方形クロップ
# ------------------------------------------------------------
def crop_center_square(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    scale: float = 1.5,
    out_size: int = 512,
) -> np.ndarray:
    """
    bbox の中心を基準に、人物が中央に来る正方形クロップを行い、
    out_size x out_size にリサイズして返す。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # バウンディングボックス中心
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    side = max(box_w, box_h) * scale

    # 正方形の半分
    half = side / 2.0

    # 左上 & 右下を計算（画面外にはみ出さないようにクリップ）
    x0 = int(max(cx - half, 0))
    y0 = int(max(cy - half, 0))
    x1 = int(min(cx + half, w - 1))
    y1 = int(min(cy + half, h - 1))

    # 万一サイズが小さすぎる場合は補正
    if x1 <= x0 or y1 <= y0:
        return cv2.resize(frame, (out_size, out_size), interpolation=cv2.INTER_AREA)

    crop = frame[y0:y1, x0:x1].copy()

    # 正方形でない場合は、短い辺に合わせてパディングしてからリサイズしてもいいが、
    # ここではシンプルにリサイズだけ行う
    crop_resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop_resized


# ------------------------------------------------------------
# メイン
# ------------------------------------------------------------
def run(video_path: str, out_path: str) -> None:
    log(f"[INFO] 動画: {video_path}")
    if not os.path.exists(video_path):
        log("[ERROR] 動画ファイルが存在しません。")
        return

    # YOLO モデル読み込み
    log("[INFO] YOLOv8x-pose 読み込み中…")
    model = YOLO("yolov8x-pose.pt")

    # ベストフレーム探索
    best_frame, best_bbox = find_best_frame_and_bbox(video_path, model)

    if best_frame is None or best_bbox is None:
        log("[WARN] 条件を満たす人物フレームが見つかりませんでした。")
        log("[ERROR] 有効なフレームが得られなかったため、画像は出力されませんでした。")
        return

    log("[INFO] ベストフレームから中央クロップを実行します…")
    fullbody_img = crop_center_square(best_frame, best_bbox, scale=1.6, out_size=512)

    cv2.imwrite(out_path, fullbody_img)
    log(f"[INFO] 全身クロップ画像を書き出しました → {out_path}")
    log("[INFO] export_fullbody_frame_v4 完了")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8x-pose を用いて人物を中央に配置したフルボディ画像を1枚生成 (v4)"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="入力動画パス (mp4)",
    )
    parser.add_argument(
        "--out",
        default="test_image_fullbody_v4.jpg",
        help="出力画像ファイル名 (既定: test_image_fullbody_v4.jpg)",
    )
    args = parser.parse_args()

    run(args.video, args.out)


if __name__ == "__main__":
    main()





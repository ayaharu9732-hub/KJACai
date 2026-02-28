# ============================================================
#  export_fullbody_frame_v3.py
#
#  - YOLOv8x-pose で人物を検出・トラッキング
#  - 「頭〜足」がきちんと写っているフレームだけを候補にする
#  - 人物が一番大きく見えるフレームを 1 枚だけ採用
#  - bbox を上下左右に拡張し、512x512 の正方形にクロップ＆リサイズ
#  - MediaPipe Heavy PoseLandmarker 用のテスト画像を自動生成
#
#  使い方（mp_env 内）:
#    python scripts/export_fullbody_frame_v3.py \
#        --video "C:\path\to\video.mp4" \
#        --out   "test_image_fullbody_v3.jpg"   # 省略時デフォルト
#
# ============================================================

import os
import math
import argparse
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_best_person_frame(
    video_path: str,
    model_path: str = "yolov8x-pose.pt",
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    動画を YOLOv8x-pose で走査し、
    - 頭〜足（nose と ankle）が検出されている
    - 人物がフレームの中でなるべく大きい
    という条件を満たす「最良フレーム」とその bbox を返す。

    戻り値:
        best_frame: np.ndarray (BGR) もしくは None
        best_bbox : (x1, y1, x2, y2) もしくは None
    """
    log("[INFO] YOLOv8x-pose 読み込み中…")
    model = YOLO(model_path)

    # FPS 取得（time_s スコアにちょっと使うだけ）
    tmp_cap = cv2.VideoCapture(video_path)
    if not tmp_cap.isOpened():
        log("[ERROR] 動画を開けませんでした。")
        return None, None
    fps = tmp_cap.get(cv2.CAP_PROP_FPS) or 30.0
    tmp_cap.release()

    log("[INFO] 動画を解析して人物フレームを抽出中…")

    best_score: float = -1.0
    best_frame: Optional[np.ndarray] = None
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    frame_idx = -1
    main_track_id: Optional[int] = None

    for result in model.track(
        source=video_path,
        stream=True,
        verbose=False,
        conf=0.5,
        iou=0.5,
        persist=True,
    ):
        frame_idx += 1
        frame = result.orig_img  # BGR (H, W, C)

        if result.boxes is None or result.keypoints is None:
            continue

        boxes = result.boxes
        kpts = result.keypoints

        if boxes.id is None:
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()  # (n, 4)
        n_det = len(ids)
        if n_det == 0:
            continue

        # track_id が未決定なら「面積最大の人」を main_track_id に採用
        if main_track_id is None:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best_idx = int(np.argmax(areas))
            main_track_id = int(ids[best_idx])
            log(f"[INFO] main_track_id = {main_track_id}")

        # このフレームで main_track_id に対応する index を探す
        idxs = [i for i, tid in enumerate(ids) if tid == main_track_id]
        if not idxs:
            continue
        i = idxs[0]

        x1, y1, x2, y2 = xyxy[i]
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)

        H, W = frame.shape[:2]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # ---- キーポイント情報 ----
        k_xy = kpts.xy[i].cpu().numpy()  # (K, 2)

        if hasattr(kpts, "conf") and kpts.conf is not None:
            k_conf_raw = kpts.conf[i].cpu().numpy()
            if k_conf_raw.ndim == 2:
                k_conf = k_conf_raw[:, 0]
            else:
                k_conf = k_conf_raw
        else:
            k_conf = np.ones((k_xy.shape[0],), dtype=np.float32)

        K = k_xy.shape[0]

        # COCO keypoint index:
        # 0: nose, 15: left ankle, 16: right ankle
        nose_y = None
        ankle_y_list: List[float] = []

        if K > 0 and k_conf[0] > 0.3:  # nose
            nose_y = float(k_xy[0, 1])

        if K > 15 and k_conf[15] > 0.3:
            ankle_y_list.append(float(k_xy[15, 1]))
        if K > 16 and k_conf[16] > 0.3:
            ankle_y_list.append(float(k_xy[16, 1]))

        # 頭と足首が取れていないフレームはスキップ
        if nose_y is None or not ankle_y_list:
            continue

        ankle_y = max(ankle_y_list)
        person_height_px = ankle_y - nose_y
        if person_height_px <= 0:
            continue

        # 「画面のどれくらいの高さを人物が占めているか」
        height_ratio = person_height_px / float(H)

        # 全身の最低条件：画面高さの 30% 以上を占める（小さすぎるフレーム除外）
        if height_ratio < 0.30:
            continue

        # 画面中心とのズレ（左右に寄りすぎ対策）
        center_offset = abs((cx / W) - 0.5)  # 0〜0.5程度

        # time_s はなるべく関与させない（A:人物が一番大きいフレーム重視）
        time_s = frame_idx / fps

        # スコア：人物の高さを最優先しつつ、中心からのズレを軽めにペナルティ
        score = (
            height_ratio * 1.0           # 人物が大きいほど高スコア
            - center_offset * 0.3        # 画面中心に近いほど高スコア
        )

        # 少しだけ時系列の早さも加点（同点のときに先に出た方を優先）
        score += -0.0001 * time_s

        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_bbox = (int(x1), int(y1), int(x2), int(y2))

    if best_frame is None or best_bbox is None:
        log("[WARN] 条件を満たす人物フレームが見つかりませんでした。")
        return None, None

    log(f"[INFO] 最良フレームを決定 (score={best_score:.3f})")
    return best_frame, best_bbox


def crop_to_square_around_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_size: int = 512,
    margin_scale: float = 0.30,
) -> np.ndarray:
    """
    bbox を中心に上下左右に margin_scale 分だけ拡張し、
    その領域をベースにした正方形を切り出して target_size にリサイズ。
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))

    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # まずは bbox を上下左右に margin_scale 分だけ拡張
    half_w = w * (0.5 + margin_scale)
    half_h = h * (0.5 + margin_scale)

    # 正方形にするため、縦横の大きい方に合わせる
    half_side = max(half_w, half_h)

    # 正方形の中心は bbox 中心
    sx1 = int(round(cx - half_side))
    sy1 = int(round(cy - half_side))
    sx2 = int(round(cx + half_side))
    sy2 = int(round(cy + half_side))

    # 画面外に出ないように補正
    if sx1 < 0:
        shift = -sx1
        sx1 += shift
        sx2 += shift
    if sy1 < 0:
        shift = -sy1
        sy1 += shift
        sy2 += shift
    if sx2 > W:
        shift = sx2 - W
        sx1 -= shift
        sx2 -= shift
    if sy2 > H:
        shift = sy2 - H
        sy1 -= shift
        sy2 -= shift

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(W, sx2)
    sy2 = min(H, sy2)

    if sx2 <= sx1 or sy2 <= sy1:
        # 万一おかしな場合は元 frame をリサイズ
        log("[WARN] 正方形クロップが無効になったため、フルフレームをリサイズしました。")
        return cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)

    crop = frame[sy1:sy2, sx1:sx2].copy()
    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return crop_resized


def run(video_path: str, out_path: str) -> None:
    if not os.path.exists(video_path):
        log(f"[ERROR] 動画ファイルが存在しません: {video_path}")
        return

    best_frame, best_bbox = choose_best_person_frame(video_path)
    if best_frame is None or best_bbox is None:
        log("[ERROR] 有効なフレームが得られなかったため、画像は出力されませんでした。")
        return

    img = crop_to_square_around_bbox(best_frame, best_bbox, target_size=512, margin_scale=0.30)

    cv2.imwrite(out_path, img)
    log(f"[INFO] 抽出画像を書き出しました → {out_path}")
    log("[INFO] export_fullbody_frame_v3 完了")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8x-pose による全身フレーム自動抽出ツール v3"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="入力動画パス (mp4)",
    )
    parser.add_argument(
        "--out",
        default="test_image_fullbody_v3.jpg",
        help="出力画像パス（デフォルト: test_image_fullbody_v3.jpg）",
    )
    args = parser.parse_args()
    run(args.video, args.out)


if __name__ == "__main__":
    main()



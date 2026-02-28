# ============================================================
#  pose_overlay_heavy_crop.py
#
#  目的:
#    - YOLOv8 で人を検出 → その周辺だけをクロップ
#    - クロップ部分に MediaPipe Pose (Heavy) で骨格を描画
#    - A版ディレクトリ構造:
#        outputs/{athlete}/{video_id}/overlay/{video_id}.mp4
#
#  使い方 (PowerShell):
#    cd C:\Users\Futamura\KJACai
#    python .\scripts\pose_overlay_heavy_crop.py `
#        --video "videos\岩本結々スタブロ11.15.mp4"
#
#    ※ YOLO が無い / うまく動かない場合は --no_yolo で強制オフ
# ============================================================

import os
import argparse
from typing import Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np

# YOLOv8（ultralytics）が使えるかどうか
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False


# ------------------------------------------------------------
#  共通: athlete / video_id の抽出（A版と合わせる）
# ------------------------------------------------------------
def parse_athlete_and_video_id(video_path: str) -> Tuple[str, str]:
    """
    例: 岩本結々スタブロ11.15.mp4 →
        athlete='岩本結々スタブロ', video_id='岩本結々スタブロ11_15'
    """
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    # 末尾の数字とピリオドを取った部分を選手名とみなす
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


def get_overlay_path(athlete: str, video_id: str) -> str:
    """
    A版正式構造:
      outputs/{athlete}/{video_id}/overlay/{video_id}.mp4
    """
    root = os.path.join("outputs", athlete, video_id, "overlay")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{video_id}.mp4")


# ------------------------------------------------------------
#  YOLO で人の領域を検出 → クロップ範囲として返す
# ------------------------------------------------------------
def detect_person_bbox_yolo(
    model: "YOLO",
    frame: np.ndarray,
    margin: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    """
    frame から「人(person)」の最大バウンディングボックスを返す。
    返り値: (x1, y1, x2, y2) / 見つからなければ None
    """
    h, w = frame.shape[:2]

    results = model(frame, imgsz=640, conf=0.25, verbose=False)
    if not results:
        return None
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return None

    # person (クラス0) の中で最大のものを採用
    boxes = r.boxes
    best_area = 0.0
    best_xyxy = None

    for b in boxes:
        cls = int(b.cls[0].item()) if b.cls is not None else -1
        if cls != 0:  # 0: person
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_xyxy = (x1, y1, x2, y2)

    if best_xyxy is None:
        return None

    x1, y1, x2, y2 = best_xyxy

    # マージンを追加して少し広めにクロップ
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * (1.0 + margin)
    bh = (y2 - y1) * (1.0 + margin)

    x1 = int(max(0, cx - bw / 2.0))
    x2 = int(min(w, cx + bw / 2.0))
    y1 = int(max(0, cy - bh / 2.0))
    y2 = int(min(h, cy + bh / 2.0))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ------------------------------------------------------------
#  メイン処理: Heavy + Crop overlay
# ------------------------------------------------------------
def build_overlay(
    video_path: str,
    output_path: str,
    use_yolo: bool = True,
) -> None:
    print("🎥 overlay(Heavy-CROP) 生成開始")
    print(f"  動画:  {video_path}")
    print(f"  出力:  {output_path}")
    print(f"  YOLO使用: {use_yolo and _YOLO_AVAILABLE}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("動画を開けませんでした。")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w0, h0))

    # YOLO モデルの準備
    yolo_model = None
    if use_yolo and _YOLO_AVAILABLE:
        yolo_model = YOLO("yolov8n-pose.pt")  # 自動でDLされる
        print("  YOLOv8n-pose モデル読み込み完了")

    # MediaPipe Pose (Heavy)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,          # Heavy モデル
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    frame_count = 0
    detected_frames = 0
    used_yolo_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        base_frame = frame.copy()

        x1 = y1 = x2 = y2 = None
        roi = base_frame

        # ---- 1) YOLO で選手を見つけてクロップ ----
        if yolo_model is not None:
            bbox = detect_person_bbox_yolo(yolo_model, base_frame)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                roi = base_frame[y1:y2, x1:x2].copy()
                used_yolo_frames += 1

        # ---- 2) クロップした ROI に MediaPipe Heavy をかける ----
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            detected_frames += 1
            mp_drawing.draw_landmarks(
                roi,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=3,
                    circle_radius=2,
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=2,
                ),
            )

        # ---- 3) ROI を元フレームへ戻す ----
        if x1 is not None:
            base_frame[y1:y2, x1:x2] = roi
        else:
            base_frame = roi

        writer.write(base_frame)

        if frame_count % 100 == 0:
            print(f"  処理中... {frame_count}フレーム")

    cap.release()
    writer.release()
    pose.close()

    print(f"✅ overlay 出力完了: {output_path}")
    print(f"   総フレーム数: {frame_count}")
    print(f"   YOLO でクロップしたフレーム数: {used_yolo_frames}")
    print(f"   骨格検出できたフレーム数: {detected_frames}")


# ------------------------------------------------------------
#  CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLO + MediaPipe Heavy CROP overlay 生成スクリプト"
    )
    parser.add_argument("--video", required=True, help="入力動画 (mp4)")
    parser.add_argument(
        "--output",
        help="出力 overlay 動画パス（省略時は A版構造に自動配置）",
    )
    parser.add_argument(
        "--no_yolo",
        action="store_true",
        help="YOLOを使わず、フルフレームをそのまま MediaPipe に渡す",
    )
    args = parser.parse_args()

    video_path = args.video
    if args.output:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        athlete, video_id = parse_athlete_and_video_id(video_path)
        output_path = get_overlay_path(athlete, video_id)

    use_yolo = not args.no_yolo
    if use_yolo and not _YOLO_AVAILABLE:
        print("⚠ ultralytics がインポートできませんでした。YOLO を無効化して実行します。")
        use_yolo = False

    build_overlay(video_path, output_path, use_yolo=use_yolo)


if __name__ == "__main__":
    main()



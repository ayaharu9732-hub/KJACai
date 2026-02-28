import cv2
import os
import csv
import argparse
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def log(msg: str):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}")


# =============================
#  抽出するランドマーク ID
# =============================
LM_ANKLE = 0        # 足首（ankle）
LM_HEEL = 1         # かかと（heel）
LM_BIGTOE = 3       # 母指球（ball / big toe）
LM_TOE_TIP = 4      # つま先（toe tip）

TARGET_LMS = {
    "ankle": LM_ANKLE,
    "heel": LM_HEEL,
    "bigtoe": LM_BIGTOE,
    "toetip": LM_TOE_TIP,
}


# =============================
#  座標を OpenCV 描画用に変換
# =============================
def to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


# =============================
#  フレームごとに描画
# =============================
def draw_landmarks(frame, lm_list, w, h):
    # 点の色とサイズ
    for name, idx in TARGET_LMS.items():
        lm = lm_list[idx]
        x, y = to_pixel(lm, w, h)
        cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)  # 黄色の点

    # 簡易ライン
    # 足首 → かかと
    cv2.line(frame,
             to_pixel(lm_list[LM_ANKLE], w, h),
             to_pixel(lm_list[LM_HEEL], w, h),
             (0, 255, 0), 2)

    # かかと → 母指球
    cv2.line(frame,
             to_pixel(lm_list[LM_HEEL], w, h),
             to_pixel(lm_list[LM_BIGTOE], w, h),
             (255, 0, 0), 2)

    # 母指球 → つま先
    cv2.line(frame,
             to_pixel(lm_list[LM_BIGTOE], w, h),
             to_pixel(lm_list[LM_TOE_TIP], w, h),
             (0, 128, 255), 2)


# =============================
#  メイン処理
# =============================
def run(video_path):
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    athlete = stem.rstrip("0123456789.")
    video_id = stem.replace(".", "_")

    out_dir = os.path.join("outputs", athlete, video_id, "foot")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{video_id}_foot_keypoints.csv")
    overlay_path = os.path.join(out_dir, f"{video_id}_foot_overlay.mp4")

    log(f"動画: {video_path}")
    log(f"出力CSV: {csv_path}")
    log(f"出力Overlay: {overlay_path}")

    # =============================
    #      MediaPipe FootLandmarker
    # =============================
    log("FootLandmarker 初期化中...")

    base_options = python.BaseOptions(model_asset_path=None)
    options = vision.FootLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.FootLandmarker.create_from_options(options)

    # =============================
    #          Video IO
    # =============================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("ERROR: 動画を開けませんでした。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    # CSV 書き込み準備
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer_csv = csv.writer(csv_file)
    writer_csv.writerow(["frame", "foot", "name", "x", "y"])

    frame_idx = 0

    # =============================
    #          解析ループ
    # =============================
    log("解析開始...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=frame)

        result = landmarker.detect_async(mp_image, frame_idx)

        # 右足・左足の2つまで出る可能性
        if result and result.foot_landmarks:
            for foot_id, lm_list in enumerate(result.foot_landmarks):
                foot_name = "right" if foot_id == 0 else "left"

                # 描画
                draw_landmarks(frame, lm_list, w, h)

                # CSV 出力
                for name, idx in TARGET_LMS.items():
                    lm = lm_list[idx]
                    x, y = lm.x, lm.y  # 0〜1 の正規化座標
                    writer_csv.writerow([frame_idx, foot_name, name, x, y])

        writer.write(frame)
        frame_idx += 1

    # クリーンアップ
    csv_file.close()
    writer.release()
    cap.release()

    log(f"完了: {video_path}")
    log(f"出力: {csv_path}")
    log(f"出力: {overlay_path}")
    log("=== foot_landmarks_extractor_v1 完了 ===")


# =============================
#  CLI エントリポイント
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="動画パス（mp4）")
    args = parser.parse_args()

    run(args.video)

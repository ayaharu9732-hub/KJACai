import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from datetime import datetime


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# YOLO foot key indices
KP = {
    "L_ANKLE": 15,
    "R_ANKLE": 16,
    "L_HEEL": 17,
    "R_HEEL": 18,
    "L_TOE": 19,
    "R_TOE": 20,
}


def bigtoe_point(heel, toe):
    """母指球（big toe ball）を補間で生成"""
    return heel * 0.2 + toe * 0.8


def draw_foot_overlay(frame, kps):
    """足部 4 点 + ライン表示"""

    def draw_point(pt, color, r=6):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), r, color, -1)

    def draw_line(p1, p2, color, t=2):
        cv2.line(frame, (int(p1[0]), int(p1[1])),
                 (int(p2[0]), int(p2[1])), color, t)

    # RGB
    C_ANKLE = (0, 255, 255)
    C_HEEL = (0, 128, 255)
    C_BIGTOE = (0, 255, 0)
    C_TOE = (255, 0, 0)

    for side in ["L", "R"]:
        ankle = kps[f"{side}_ANKLE"]
        heel = kps[f"{side}_HEEL"]
        toe = kps[f"{side}_TOE"]
        bigtoe = bigtoe_point(heel, toe)

        # plot
        draw_point(ankle, C_ANKLE)
        draw_point(heel, C_HEEL)
        draw_point(bigtoe, C_BIGTOE, r=7)
        draw_point(toe, C_TOE)

        # lines
        draw_line(heel, bigtoe, C_BIGTOE)
        draw_line(bigtoe, toe, C_BIGTOE)
        draw_line(ankle, bigtoe, (255, 255, 255))

    return frame


def run(video_path: str):
    log(f"動画: {video_path}")

    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)

    athlete = stem.split("スタブロ")[0]
    video_id = stem.replace(".", "_")

    out_dir = os.path.join("outputs", athlete, video_id, "foot_v3")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{video_id}_foot_v3.csv")
    out_video = os.path.join(out_dir, f"{video_id}_foot_overlay_v3.mp4")

    log(f"出力CSV: {csv_path}")
    log(f"出力Overlay: {out_video}")

    # YOLO model
    log("[YOLO] モデル読み込み: yolov8x-pose.pt")
    model = YOLO("yolov8x-pose.pt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    csv_lines = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)[0]
        persons = result.keypoints

        if len(persons) > 0:
            # assume single athlete → ID0
            k = persons[0].data.cpu().numpy()[0]  # (21,3)

            kps = {}

            for name, idx in KP.items():
                x, y, conf = k[idx]          # ← 修正ポイント（3値展開）
                kps[name] = np.array([x, y])  # conf は無視

            # overlay描画
            frame = draw_foot_overlay(frame, kps)

            # CSV保存
            line = [frame_idx]
            for name in KP.keys():
                line += [kps[name][0], kps[name][1]]

            # bigtoe（母指球） 追加
            L_bigtoe = bigtoe_point(kps["L_HEEL"], kps["L_TOE"])
            R_bigtoe = bigtoe_point(kps["R_HEEL"], kps["R_TOE"])

            line += [L_bigtoe[0], L_bigtoe[1], R_bigtoe[0], R_bigtoe[1]]

            csv_lines.append(line)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # CSV 書き出し
    header = ["frame"]
    for name in KP.keys():
        header += [f"{name}_x", f"{name}_y"]
    header += ["L_BIGTOE_x", "L_BIGTOE_y", "R_BIGTOE_x", "R_BIGTOE_y"]

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for line in csv_lines:
            f.write(",".join(map(str, line)) + "\n")

    log(f"[CSV] 完了: {csv_path} (rows={len(csv_lines)})")
    log("=== YOLO Foot Extractor v3 完了 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    run(args.video)



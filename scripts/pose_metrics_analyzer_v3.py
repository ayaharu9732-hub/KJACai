# ==========================================================
# ⚡ pose_metrics_analyzer_v3.py  v3.3
# - MediaPipe Pose でCOMを取得
# - 速度(m/s)、傾き(簡易)、time_s を算出
# - ピッチ(歩/秒)・ストライド(m/歩)・安定性スコアをCSV末尾に_summary_行で付与
# - 校正 m/px は outputs/jsonl/calibration_result.json から読み込み
# ==========================================================

import os
import cv2
import math
import json
import argparse
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = r"C:\Users\Futamura\KJACai"
CALIB_JSON = os.path.join(BASE_DIR, "outputs", "jsonl", "calibration_result.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs", "koike_pose_metrics_v3.csv")

def load_m_per_px(fallback=0.0020):
    try:
        with open(CALIB_JSON, "r", encoding="utf-8") as f:
            d = json.load(f)
        v = float(d.get("m_per_px", fallback))
        return v
    except Exception:
        return fallback

def local_minima_indices(y, min_dist_frames=5):
    """単純な局所最小検出（COM_yの極小）"""
    y = np.asarray(y)
    n = len(y)
    if n < 3:
        return []
    mins = []
    last = -10**9
    for i in range(1, n-1):
        if y[i] <= y[i-1] and y[i] <= y[i+1]:
            if i - last >= min_dist_frames:
                mins.append(i)
                last = i
    return mins

def stability_indices(df, fps):
    """COM_yの揺れと傾き角のjerkから安定性を0-10で評価（簡易）。"""
    if len(df) < 5 or fps <= 0:
        return 0.0, 0.0, 0.0
    win = max(3, int(0.25 * fps))
    std_y = df["COM_y"].rolling(win, center=True).std().mean()
    jerk = df["tilt_deg"].diff().diff().abs().mean()
    # 経験則でスコア化（小さいほど不安定 → 10に近いほど安定）
    score = max(0.0, 10.0 - ( (std_y or 0)*120.0 + (jerk or 0)*0.3 ))
    return float(std_y or 0), float(jerk or 0), float(round(score, 2))

def analyze_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 動画を開けませんでした: {video_path}")
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"🎥 読み込み: {video_path}")
    print(f"⏱ FPS={fps:.2f}, フレーム数={frame_count}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    results_list = []
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            # 腰中心（左右HIPの中点）をCOM近似とする（正規化座標）
            lx, ly = lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y
            rx, ry = lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y
            cx, cy = (lx + rx) / 2.0, (ly + ry) / 2.0

            # 簡易傾き角：COMのフレーム差からの進行方向角（deg）
            # （本格的には肩↔腰ベクトルなどを推奨。ここでは簡易で十分）
            results_list.append({"frame": f_idx, "COM_x": cx, "COM_y": cy})
        f_idx += 1

    cap.release()

    if not results_list or fps <= 0:
        print("⚠️ 検出結果がありません。")
        return

    df = pd.DataFrame(results_list)
    # 差分・距離・速度
    df["dx"] = df["COM_x"].diff().fillna(0)
    df["dy"] = df["COM_y"].diff().fillna(0)
    df["dist_px"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

    m_per_px = load_m_per_px()
    df["speed_mps"] = df["dist_px"] * m_per_px * fps

    # 傾き角（COM差分から進行方向角を推定）
    df["tilt_deg"] = np.degrees(np.arctan2(df["dy"], df["dx"])).fillna(0)

    # 時間[s]
    df["time_s"] = df["frame"] / fps

    # ピッチ（COM_y の極小をステップ接地の代替指標）
    step_idx = local_minima_indices(df["COM_y"].values, min_dist_frames=max(3, int(0.2 * fps)))
    duration = len(df) / fps if fps > 0 else 0.0
    pitch_hz = (len(step_idx) / duration) if duration > 0 else 0.0

    # ストライド（水平移動距離 / 歩数）
    total_dx_m = (df["COM_x"].iloc[-1] - df["COM_x"].iloc[0]) * m_per_px
    stride_m = (total_dx_m / max(1, len(step_idx))) if len(step_idx) > 0 else 0.0

    # 安定性
    std_y, jerk, stab = stability_indices(df, fps)

    # メタを最後に1行追加
    meta = pd.DataFrame([{
        "frame": "_summary_",
        "pitch_hz": pitch_hz,
        "stride_m": stride_m,
        "stability_std_y": std_y,
        "stability_jerk": jerk,
        "stability_score": stab
    }])

    out = pd.concat([df, meta], ignore_index=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV出力完了: {OUTPUT_CSV}")
    print(out.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="骨格メトリクス解析")
    parser.add_argument("--video", required=True, help="入力動画パス")
    args = parser.parse_args()
    analyze_pose(args.video)







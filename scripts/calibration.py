# ==========================================================
# 📏 calibration.py  v2.1
# 距離スケール校正スクリプト（リサイズ対応版）
# ----------------------------------------------------------
# ✅ 改良点
#   - 表示ウィンドウを自動リサイズ（幅960px程度）
#   - クリック位置は元解像度にスケーリング補正して計算
# ==========================================================

import cv2
import json
import os
import argparse

OUTPUT_DIR = r"C:\Users\Futamura\KJACai\outputs\jsonl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calibrate_interactive(video_path):
    """校正用動画から m/px を算出（縮小表示対応）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません（校正）: {video_path}")

    # 最初のフレーム取得
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("最初のフレームを取得できません。")

    h, w = frame.shape[:2]

    # 画面が大きすぎる場合は縮小（幅960px基準）
    target_width = 960
    scale = 1.0
    if w > target_width:
        scale = target_width / w
        frame_display = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        frame_display = frame.copy()

    display_clone = frame_display.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display_clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("calibration", display_clone)

    cv2.imshow("calibration", display_clone)
    cv2.setMouseCallback("calibration", click_event)

    print("🖱 マウスで校正用の2点をクリックしてください。")
    print("    例：スタートライン〜10mラインなど")
    print(f"🔍 表示スケール: {scale:.2f}（クリック位置は自動補正されます）")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 2 or key == 27:
            break

    cv2.destroyAllWindows()

    if len(points) < 2:
        raise RuntimeError("2点が選択されていません。")

    # スケール補正して実座標化
    scaled_points = [(int(x / scale), int(y / scale)) for (x, y) in points]
    (x1, y1), (x2, y2) = scaled_points

    px_dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
    print(f"📏 ピクセル距離（補正後）: {px_dist:.2f}px")

    real_m = float(input("実際の距離を[m]単位で入力してください: "))
    m_per_px = real_m / px_dist
    print(f"✅ 校正完了: 1px = {m_per_px:.6f} m")

    cap.release()

    result = {
        "video_path": video_path,
        "px_dist": px_dist,
        "real_m": real_m,
        "m_per_px": m_per_px,
        "display_scale": scale
    }
    json_path = os.path.join(OUTPUT_DIR, "calibration_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"💾 校正結果を保存しました: {json_path}")
    return m_per_px, px_dist, real_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="距離スケール校正（縮小表示対応）")
    parser.add_argument("--video", required=True, help="解析する動画ファイルのパス")
    args = parser.parse_args()

    video_path = args.video
    print(f"🎥 校正対象動画: {video_path}")

    m_per_px, px_dist, real_m = calibrate_interactive(video_path)

    print("\n📘 校正結果")
    print(f" - ピクセル距離: {px_dist:.2f}px")
    print(f" - 実距離: {real_m:.3f} m")
    print(f" - 1px あたり: {m_per_px:.6f} m")
    print("✅ calibration.py（リサイズ対応版）正常終了")








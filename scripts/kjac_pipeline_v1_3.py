# ==========================================================
# 🚀 KJACテクニカルAI フル統合パイプライン v1.3（最終化）
# - 常に .venv の Python (sys.executable) でサブスクリプトを実行
# - 最新MP4自動選択
# - v5.0 PDFレポートを出力
# ==========================================================

import os
import sys
import glob
import subprocess
from datetime import datetime

BASE_DIR = r"C:\Users\Futamura\KJACai"
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
IMAGES_DIR = os.path.join(OUTPUTS_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_latest_video(videos_dir: str):
    mp4_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    if not mp4_files:
        raise FileNotFoundError("🎥 videos フォルダに MP4 ファイルが見つかりません。")
    latest_file = max(mp4_files, key=os.path.getmtime)
    return latest_file

def run_script(script_name: str, *args):
    path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(path):
        print(f"⚠️ {script_name} が見つかりません。スキップします。")
        return
    cmd = [sys.executable, path] + list(args)
    print(f"[▶] {script_name} を実行中...")
    subprocess.run(cmd, check=False)

def main():
    print("=" * 60)
    print("🚀 KJACテクニカルAI フル統合パイプライン v1.3 起動")
    print("=" * 60)

    try:
        video_path = get_latest_video(VIDEOS_DIR)
        print(f"🎥 対象動画: {os.path.basename(video_path)}")
    except Exception as e:
        print(f"❌ 動画選択エラー: {e}")
        return

    steps = [
        ("calibration.py",         ["--video", video_path]),
        ("pose_overlay_draw_feet_v2.py", ["--video", video_path]),
        ("pose_metrics_analyzer_v3.py",  ["--video", video_path]),
        # v5.0 PDF（overlay は推定、video を渡しておく）
        ("pose_reporter_pdf_ai_v5_0.py", ["--csv", os.path.join(OUTPUTS_DIR, "koike_pose_metrics_v3.csv"),
                                          "--video", video_path,
                                          "--athlete", "二村 遥香"]),
    ]

    for script, args in steps:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ {script} 実行中...")
        run_script(script, *args)

    print("=" * 60)
    print("✅ 全処理完了 — outputs/pdf 内を確認してください。")
    print("=" * 60)

if __name__ == "__main__":
    main()








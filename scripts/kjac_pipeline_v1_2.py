# ============================================
# 🚀 KJACテクニカルAI 統合パイプライン v1.2
# ✅ 最新動画自動選択＋フル自動解析版
# ============================================

import os
import subprocess
import glob
from datetime import datetime

BASE_DIR = r"C:\Users\Futamura\KJACai"
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

def get_latest_video(videos_dir: str):
    """videosフォルダ内の最新MP4を取得"""
    mp4_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    if not mp4_files:
        raise FileNotFoundError("🎥 videos フォルダに MP4 ファイルが見つかりません。")
    latest_file = max(mp4_files, key=os.path.getmtime)
    return latest_file

def run_script(script_name: str, video_path: str):
    """個別スクリプトを実行"""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"⚠️ {script_name} が見つかりません。スキップします。")
        return
    print(f"[▶] {script_name} を実行中...")
    subprocess.run(["python", script_path, "--video", video_path], check=False)

def main():
    print("=" * 60)
    print("🚀 KJACテクニカルAI フル統合パイプライン v1.2 起動")
    print("=" * 60)

    try:
        video_path = get_latest_video(VIDEOS_DIR)
        print(f"🎥 対象動画: {os.path.basename(video_path)}")
    except Exception as e:
        print(f"❌ 動画選択エラー: {e}")
        return

    # ステップ実行
    steps = [
        "calibration.py",
        "pose_overlay_draw_feet_v2.py",
        "pose_metrics_analyzer_v3.py",
        "pose_reporter_pdf_ai_v4_4.py",
    ]

    for step in steps:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚙️ {step} 実行中...")
        run_script(step, video_path)

    print("=" * 60)
    print(f"✅ 全処理完了 — PDF出力は outputs/pdf 内を確認してください。")
    print("=" * 60)

if __name__ == "__main__":
    main()







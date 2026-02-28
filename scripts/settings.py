import os
BASE = r"C:\Users\Futamura\KJACai"
OUT = os.path.join(BASE, "outputs")
CALIB_JSON = os.path.join(OUT, "jsonl", "calibration_result.json")
IMAGES_DIR = os.path.join(OUT, "images")
PDF_DIR = os.path.join(OUT, "pdf")

# フォント（日本語ラベル用）：無ければ DejaVu にフォールバック
REPORT_FONT = "IPAexGothic"  # Windowsに導入推奨
M_PER_PX_FALLBACK = 0.0020   # 校正が無いときの応急値（m/px）

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)








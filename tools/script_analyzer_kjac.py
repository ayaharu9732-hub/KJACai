import os
import ast
from textwrap import shorten

TARGET_DIR = r"C:\Users\Futamura\KJACai"

KEYWORDS = {
    "pdf": ["reportlab", "FPDF", "canvas", "pdf", "drawString", "platypus"],
    "video": ["cv2", "save_video", "pose", "AlphaPose"],
    "image": ["matplotlib", "plt", "savefig", "PIL"],
    "csv": ["csv", "pandas", "read_csv", "to_csv"],
    "analysis": ["stride", "pitch", "acceleration", "biomechanics", "angle", "speed"],
    "ai": ["openai", "gpt", "message", "client"],
}

def classify_script(source):
    """スクリプト内容から役割を推測する"""
    roles = []
    text = source.lower()

    for role, words in KEYWORDS.items():
        if any(w in text for w in words):
            roles.append(role)

    return roles or ["unknown"]

def analyze_script(path):
    """各ファイルの解析"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
    except:
        return None

    try:
        tree = ast.parse(src)
    except:
        return None

    functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    doc = ast.get_docstring(tree)
    doc_preview = shorten(doc, 150) if doc else "（docstringなし）"

    roles = classify_script(src)

    return {
        "file": os.path.basename(path),
        "functions": functions,
        "classes": classes,
        "doc": doc_preview,
        "roles": roles,
    }

def scan_kjac(path=TARGET_DIR):
    results = []

    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(root, f)
                info = analyze_script(fp)
                if info:
                    results.append(info)

    return results


if __name__ == "__main__":
    results = scan_kjac()

    print("\n===== ✅ KJACai スクリプト解析レポート =====")
    for r in results:
        print(f"\n📄 {r['file']}")
        print(f"   役割推測: {', '.join(r['roles'])}")
        print(f"   関数数: {len(r['functions'])} / クラス数: {len(r['classes'])}")
        print(f"   docstring: {r['doc']}")




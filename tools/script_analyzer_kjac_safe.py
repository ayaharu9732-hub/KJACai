import os
import ast
from textwrap import shorten

TARGET_DIR = r"C:\Users\Futamura\KJACai"

KEYWORDS = {
    "pdf": ["reportlab", "fpdf", "canvas", "pdf", "drawstring", "platypus"],
    "video": ["cv2", "save_video", "pose", "alphapose"],
    "image": ["matplotlib", "plt", "savefig", "pil"],
    "csv": ["csv", "pandas", "read_csv", "to_csv"],
    "analysis": ["stride", "pitch", "acceleration", "biomechanics", "angle", "speed"],
    "ai": ["openai", "gpt", "message", "client"],
}

def safe_print(text: str):
    """cp932 で出力できない文字を全部 ? に置き換えて print する"""
    try:
        encoded = text.encode("cp932", errors="replace")
        print(encoded.decode("cp932"))
    except:
        print("<?>")

def classify_script(source):
    roles = []
    text = source.lower()
    for role, words in KEYWORDS.items():
        if any(w in text for w in words):
            roles.append(role)
    return roles or ["unknown"]

def analyze_script(path):
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
    doc_preview = shorten(doc, 150) if doc else "(docstringなし)"

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

    safe_print("\n===== KJACai Script Report =====")
    for r in results:
        safe_print(f"\nFILE: {r['file']}")
        safe_print(f"  roles: {', '.join(r['roles'])}")
        safe_print(f"  functions: {len(r['functions'])}")
        safe_print(f"  classes: {len(r['classes'])}")
        safe_print(f"  doc: {r['doc']}")






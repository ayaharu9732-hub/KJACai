# -*- coding: utf-8 -*-
"""
KJAC AI / easyrev_sports 棚卸しスクリプト
- scripts/ の一覧化
- 主要データファイルの存在確認
- 主要スクリプトの import / help 起動の簡易チェック
- 結果を reports/_inventory/ に出力

使い方:
  python scripts/audit_kjac_ai_state.py
"""

from __future__ import annotations

import os
import re
import csv
import sys
import json
import time
import traceback
import subprocess
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCRIPTS_DIR = os.path.join(ROOT, "scripts")
DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
OUT_DIR = os.path.join(REPORTS_DIR, "_inventory")

TARGET_SCRIPTS = [
    "build_athlete_carte.py",
    "normalize_googleform_to_forms_clean.py",
    "normalize_googleform_to_forms_clean.py",  # 念のため重複OK
    "analyze_relay_video.py",                  # 想定名（無ければ検出される）
    "relay_analyzer.py",                       # 想定名（無ければ検出される）
]

DATA_TARGETS = [
    os.path.join(ROOT, "data", "input", "googleform_responses.csv"),
    os.path.join(ROOT, "data", "processed", "forms_clean.csv"),
    os.path.join(ROOT, "masters", "pain_locations.csv"),
    os.path.join(ROOT, "masters", "athletes_master.csv"),
    os.path.join(ROOT, "assets", "kjac_logo.jpg"),
    os.path.join(ROOT, "fonts", "ipaexg.ttf"),
]

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def read_head_comment(path: str, max_lines: int = 40) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
        head = "\n".join(lines)
        # docstringっぽいのだけ抜く
        m = re.search(r'"""(.*?)"""', head, flags=re.S)
        if m:
            return m.group(1).strip()
        m2 = re.search(r"'''(.*?)'''", head, flags=re.S)
        if m2:
            return m2.group(1).strip()
        return head.strip()
    except Exception:
        return ""

def has_main_guard(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        return "__name__" in txt and "== \"__main__\"" in txt
    except Exception:
        return False

def run_subprocess(cmd: list[str], timeout_sec: int = 25) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        return p.returncode, (p.stdout or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\n[timeout]\n"
        return 124, out
    except Exception as e:
        return 1, f"[exception] {e}\n"

def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def list_files_recursive(base: str, exts: tuple[str, ...] = (".py",)) -> list[str]:
    out = []
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    return sorted(out)

def relpath(p: str) -> str:
    try:
        return os.path.relpath(p, ROOT)
    except Exception:
        return p

def profile_forms_clean(path: str) -> str:
    # pandas無しでも最低限の列名だけ読む
    if not os.path.exists(path):
        return f"[missing] {path}\n"
    try:
        import pandas as pd  # noqa
        df = pd.read_csv(path, encoding="utf-8-sig")
        cols = list(df.columns)
        lines = []
        lines.append(f"[OK] forms_clean.csv rows={len(df)} cols={len(cols)}")
        # 重要そうな列だけ表示
        must = [
            "practice_date", "timestamp",
            "athlete_id", "athlete_name",
            "menu", "drills", "theme",
            "pain_flag", "pain_location",
            "good", "bad", "cause", "next",
            "achievement", "satisfaction", "condition",
            "ai_request",
            "measured_events", "times_raw",
            "track_type", "meet_name",
            "goal_event", "goal_time", "goal_target_date", "goal_focus",
        ]
        present = [c for c in must if c in cols]
        missing = [c for c in must if c not in cols]
        lines.append("present_important_cols=" + ", ".join(present))
        lines.append("missing_important_cols=" + ", ".join(missing))
        # athlete_id 上位
        if "athlete_id" in df.columns:
            vc = df["athlete_id"].astype(str).value_counts().head(20)
            lines.append("top_athlete_id_counts:")
            lines.append(vc.to_string())
        return "\n".join(lines) + "\n"
    except Exception as e:
        return f"[error] forms_clean profile failed: {e}\n"

def main():
    safe_mkdir(OUT_DIR)

    # 0) tree snapshot
    tree_lines = []
    for base in ["scripts", "data", "masters", "assets", "fonts", "reports"]:
        p = os.path.join(ROOT, base)
        if os.path.exists(p):
            tree_lines.append(f"[DIR] {base}")
            for root, dirs, files in os.walk(p):
                depth = relpath(root).count(os.sep) - relpath(p).count(os.sep)
                if depth > 3:
                    dirs[:] = []
                    continue
                indent = "  " * depth
                tree_lines.append(f"{indent}{os.path.basename(root)}/")
                for fn in sorted(files)[:80]:
                    tree_lines.append(f"{indent}  {fn}")
        else:
            tree_lines.append(f"[missing dir] {base}")
    write_text(os.path.join(OUT_DIR, "project_tree.txt"), "\n".join(tree_lines) + "\n")

    # 1) scripts index
    py_files = list_files_recursive(SCRIPTS_DIR, (".py",))
    idx_csv = os.path.join(OUT_DIR, "scripts_index.csv")
    with open(idx_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "size", "mtime", "has_main", "head_comment"])
        for p in py_files:
            st = os.stat(p)
            w.writerow([
                relpath(p),
                st.st_size,
                datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "yes" if has_main_guard(p) else "no",
                (read_head_comment(p)[:300].replace("\n", " / ")),
            ])

    # 2) data targets check
    data_lines = [f"Inventory generated at {now_str()}", ""]
    data_lines.append("Data/Assets existence check:")
    for p in DATA_TARGETS:
        data_lines.append(f"- {'OK' if os.path.exists(p) else 'MISSING'} : {p}")
    data_lines.append("")
    data_lines.append("forms_clean profile:")
    data_lines.append(profile_forms_clean(os.path.join(ROOT, "data", "processed", "forms_clean.csv")))
    write_text(os.path.join(OUT_DIR, "data_profile.txt"), "\n".join(data_lines))

    # 3) quick run checks (import / --help)
    results = []
    for name in sorted(set(TARGET_SCRIPTS)):
        sp = os.path.join(SCRIPTS_DIR, name)
        if not os.path.exists(sp):
            results.append((name, "missing", "", ""))
            continue

        # import check (SyntaxError/IndentationError がここで捕まる)
        mod = f"scripts.{os.path.splitext(name)[0]}"
        cmd_import = [sys.executable, "-c", f"import {mod} as m; print('import_ok:', getattr(m,'__file__',''))"]
        rc1, out1 = run_subprocess(cmd_import, timeout_sec=25)

        # help check (argparse の壊れ具合も拾える)
        cmd_help = [sys.executable, sp, "--help"]
        rc2, out2 = run_subprocess(cmd_help, timeout_sec=25)

        status = "ok"
        if rc1 != 0:
            status = "import_failed"
        elif rc2 != 0:
            status = "help_failed"

        results.append((name, status, out1.strip()[-800:], out2.strip()[-800:]))

    out_csv = os.path.join(OUT_DIR, "quick_run_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["script", "status", "import_tail", "help_tail"])
        for r in results:
            w.writerow(list(r))

    # 4) summary md
    md = []
    md.append("# KJAC AI 棚卸しサマリー")
    md.append("")
    md.append(f"- Generated: {now_str()}")
    md.append(f"- Project: {ROOT}")
    md.append("")
    md.append("## 主要ファイルの存在確認")
    for p in DATA_TARGETS:
        md.append(f"- {'OK' if os.path.exists(p) else 'MISSING'}: `{p}`")
    md.append("")
    md.append("## scripts クイックチェック結果")
    for name, status, _, _ in results:
        md.append(f"- `{name}` : {status}")
    md.append("")
    md.append("## 次にやるべき優先順位（棚卸し後の定石）")
    md.append("1. import_failed / help_failed のスクリプトを先に直す（Syntax/Indent/NameError を潰す）")
    md.append("2. 成長カルテ：`build_athlete_carte.py` が動く状態を固定（PDF出力が1つでも出ればOK）")
    md.append("3. データ正規化：`normalize_googleform_to_forms_clean.py` の入力パス統一（data/input を正とする）")
    md.append("4. リレー：現状は“分析テンプレ運用”か“自動化”かを分けて整理（スクリプト名を決める）")
    md.append("")
    write_text(os.path.join(OUT_DIR, "inventory_summary.md"), "\n".join(md))

    print(f"[OK] Inventory written to: {OUT_DIR}")
    print(f"- inventory_summary.md")
    print(f"- scripts_index.csv")
    print(f"- quick_run_results.csv")
    print(f"- project_tree.txt")
    print(f"- data_profile.txt")

if __name__ == "__main__":
    main()

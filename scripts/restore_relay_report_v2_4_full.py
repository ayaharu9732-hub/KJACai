#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
restore_relay_report_v2_4_full.py
- relay_report_generator_ai_v2_4.py を「完全に」安全版へ書き戻す（自己改変を根絶）
- --out 対応
- Event/Time(s) だけのCSVも簡易モードで通す（Start/Pass1/Pass2/Pass3/Finish）
- matplotlib の日本語フォントを IPAexGothic に寄せる（DejaVu glyph warning軽減）
- fpdf2/旧fpdf両対応（headerのcellを旧式に）
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

TARGET = Path("relay_report_generator_ai_v2_4.py")

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak_fullrestore_{ts}")
    if path.exists():
        bak.write_bytes(path.read_bytes())
    return bak

CODE = r'''# -*- coding: utf-8 -*-
"""
Relay Report Generator AI v2.4 (SAFE FULL)
- Japanese PDF supported (IPAexGothic)
- Robust CSV loader
- Speed & Acceleration charts
- Rule-based technical comments
- Simple mode: [Event, Time(s)] only -> auto split(4) + speed/accel
- Output path controllable by --out
"""

import os
import argparse
from pathlib import Path
from io import StringIO

import pandas as pd
import matplotlib.pyplot as plt

from fpdf import FPDF

# ==============================
# Font settings (PDF)
# ==============================
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "ipaexg.ttf")
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError("fonts/ipaexg.ttf が見つかりません。IPAexGothic を fonts フォルダに入れてください。")

# ==============================
# matplotlib (graph) font settings (Japanese)
# ==============================
import matplotlib
from matplotlib import font_manager
if os.path.exists(FONT_PATH):
    try:
        font_manager.fontManager.addfont(FONT_PATH)
        ipa_name = font_manager.FontProperties(fname=FONT_PATH).get_name()
        matplotlib.rcParams["font.family"] = ipa_name
    except Exception:
        matplotlib.rcParams["font.family"] = "IPAexGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ==============================
# Utility
# ==============================
def smart_read_csv(csv_path):
    """
    文字コード自動判定が失敗しても落とさず、既知エンコーディングを順に試して読み込む。
    """
    p = str(csv_path)
    encs = ["utf-8-sig", "utf-8", "utf-16", "cp932", "shift_jis"]

    last_err = None
    for enc in encs:
        try:
            df = pd.read_csv(p, encoding=enc, engine="python", on_bad_lines="skip")
            return df
        except Exception as e:
            last_err = e

    # 最後の手段：replace で文字列化
    try:
        data = Path(p).read_bytes()
        text = data.decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")
        return df
    except Exception as e:
        last_err = e

    raise RuntimeError(f"CSVの文字コードを読み取れませんでした（fallback全滅）: {last_err}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes to:
      Section, Time(s), Speed(m/s), Accel(m/s^2)

    ✅ 簡易モード:
      - columns が [Event, Time(s)] だけでも通す
      - Event が Start/Pass1/Pass2/Pass3/Finish の累積時刻なら
        4区間(100m×4)に変換して Speed/Accel を自動生成
    """
    cols = list(df.columns)
    mapping = {}

    for c in cols:
        cl = str(c).lower()
        if ("section" in cl) or ("split" in cl) or ("区間" in str(c)):
            mapping[c] = "Section"
        elif ("event" in cl) or ("イベント" in str(c)):
            mapping[c] = "Event"
        elif ("time" in cl) or ("duration" in cl) or ("秒" in str(c)):
            mapping[c] = "Time(s)"
        elif ("speed" in cl) or ("v=" in cl) or ("速度" in str(c)):
            mapping[c] = "Speed(m/s)"
        elif ("accel" in cl) or ("加速" in str(c)):
            mapping[c] = "Accel(m/s^2)"

    df = df.rename(columns=mapping)

    # ✅ 簡易モード（Event/Time(s)のみ）
    if ("Section" not in df.columns) and ("Event" in df.columns) and ("Time(s)" in df.columns):
        work = df[["Event", "Time(s)"]].copy()
        work["Event"] = work["Event"].astype(str)
        work["Time(s)"] = pd.to_numeric(work["Time(s)"], errors="coerce")
        work = work.dropna(subset=["Time(s)"])

        order = ["Start", "Pass1", "Pass2", "Pass3", "Finish"]
        key = {e.lower(): e for e in order}

        def norm_event(x: str) -> str:
            xl = x.strip().lower()
            for k, v in key.items():
                if (k == xl) or (k in xl):
                    return v
            return x.strip()

        work["EventN"] = work["Event"].map(norm_event)

        if set(order).issubset(set(work["EventN"].tolist())):
            t = {}
            for e in order:
                t[e] = float(work.loc[work["EventN"] == e, "Time(s)"].iloc[0])

            base = t["Start"]
            splits = [
                ("1→2", t["Pass1"] - base),
                ("2→3", t["Pass2"] - t["Pass1"]),
                ("3→4", t["Pass3"] - t["Pass2"]),
                ("4→G", t["Finish"] - t["Pass3"]),
            ]

            rows = []
            v_prev = 0.0
            for sec, dt in splits:
                dt = max(float(dt), 1e-6)
                v = 100.0 / dt
                a = (v - v_prev) / dt
                rows.append({"Section": sec, "Time(s)": dt, "Speed(m/s)": v, "Accel(m/s^2)": a})
                v_prev = v

            return pd.DataFrame(rows)

        raise ValueError("簡易CSVは Event列に Start/Pass1/Pass2/Pass3/Finish の5点が必要です。")

    # 通常モード
    required = ["Section", "Time(s)", "Speed(m/s)", "Accel(m/s^2)"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"CSVに必要な列がありません: {r}")

    return df


# ==============================
# Graph generation
# ==============================
def make_graphs(df: pd.DataFrame, out_prefix: str):
    work = df.iloc[:4].copy()

    sections = work["Section"].astype(str)
    speeds = pd.to_numeric(work["Speed(m/s)"], errors="coerce").fillna(0)
    accels = pd.to_numeric(work["Accel(m/s^2)"], errors="coerce").fillna(0)

    # Speed
    plt.figure()
    plt.bar(sections, speeds)
    plt.title("区間別 平均速度 (m/s)")
    plt.ylabel("m/s")
    plt.tight_layout()
    speed_png = out_prefix + "_speed.png"
    plt.savefig(speed_png, dpi=160)
    plt.close()

    # Accel
    plt.figure()
    plt.bar(sections, accels)
    plt.title("区間別 平均加速度 (m/s²)")
    plt.ylabel("m/s²")
    plt.tight_layout()
    accel_png = out_prefix + "_accel.png"
    plt.savefig(accel_png, dpi=160)
    plt.close()

    return speed_png, accel_png


# ==============================
# Technical Comment Generator
# ==============================
def generate_technical_comment(df: pd.DataFrame) -> str:
    work = df.iloc[:4].copy()
    work["Speed(m/s)"] = pd.to_numeric(work["Speed(m/s)"], errors="coerce")
    fastest = work.loc[work["Speed(m/s)"].idxmax()]
    slowest = work.loc[work["Speed(m/s)"].idxmin()]

    comment = []
    comment.append("【技術所見（フォーム・動作観点）】")
    comment.append("")
    comment.append(f"・最もスピード効率が良かった区間は「{fastest['Section']}」で、加速のロスが少なくフォームの安定性が高いと推測されます。")
    comment.append(f"・一方「{slowest['Section']}」は相対的にスピード効率が低く、接地や姿勢制御に改善余地がある可能性があります。")
    comment.append("・バトン受けからトップスピードまでの姿勢変化と、接地位置のブレを重点的に確認すると改善効果が高いと考えられます。")
    comment.append("・特に『腰高の維持』『接地の真下化』『腕振りの左右差』の3点を次戦のチェックポイントに推奨します。")
    return "\n".join(comment)


# ==============================
# PDF class
# ==============================
class ReportPDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.title_text = title
        # fpdf2/旧fpdf互換：uni引数は使わない
        self.add_font("IPA", "", FONT_PATH)
        self.add_font("IPA", "B", FONT_PATH)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("IPA", "B", 16)
        # 旧fpdf互換：new_x/new_y を使わない
        self.cell(0, 10, self.title_text, 0, 1, "C")
        self.ln(4)


# ==============================
# Main PDF generation
# ==============================
def create_pdf(csv_path: str, title: str, out_pdf: str | None = None):
    df_raw = smart_read_csv(csv_path)
    df = normalize_columns(df_raw)

    # 出力先制御（--out 指定優先）
    if out_pdf:
        out_pdf = str(out_pdf)
        out_prefix = os.path.splitext(out_pdf)[0]
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    else:
        out_prefix = os.path.splitext(csv_path)[0]
        out_pdf = out_prefix + "_AI_report.pdf"

    speed_png, accel_png = make_graphs(df, out_prefix)
    tech_comment = generate_technical_comment(df)

    pdf = ReportPDF(title)
    pdf.add_page()

    pdf.set_font("IPA", "", 11)
    pdf.cell(0, 8, "【区間別数値サマリー】", 0, 1)

    pdf.set_font("IPA", "", 10)
    for i in range(min(4, len(df))):
        r = df.iloc[i]
        line = f"{r['Section']} : {float(r['Time(s)']):.3f}s / {float(r['Speed(m/s)']):.3f} m/s / {float(r['Accel(m/s^2)']):.3f} m/s²"
        pdf.cell(0, 6, line, 0, 1)

    pdf.ln(4)

    pdf.set_font("IPA", "", 11)
    pdf.cell(0, 8, "【速度・加速 分析グラフ】", 0, 1)
    pdf.image(speed_png, w=170)
    pdf.ln(3)
    pdf.image(accel_png, w=170)
    pdf.ln(6)

    pdf.set_font("IPA", "", 11)
    pdf.multi_cell(0, 7, tech_comment)

    pdf.output(out_pdf)
    print("Saved PDF:", out_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--title", default="4x100m Relay Analysis Report")
    ap.add_argument("--out", default=None, help="出力PDFパス（未指定なら <csv>_AI_report.pdf）")
    args = ap.parse_args()
    create_pdf(args.csv, args.title, out_pdf=args.out)


if __name__ == "__main__":
    main()
'''

def main():
    bak = backup(TARGET)
    TARGET.write_text(CODE, encoding="utf-8", newline="\n")
    print("[OK] restored full:", TARGET)
    if bak.exists():
        print("[OK] backup saved:", bak.name)

if __name__ == "__main__":
    main()

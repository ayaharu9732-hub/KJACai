# -*- coding: utf-8 -*-
"""
成長カルテ Page3+（日別カルテ）生成
- 期間内の日付ごとにページを作る（1日=1ページ基本、溢れたら増ページOK）
"""

import os
import re
import argparse
from typing import List

import pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def safe_mkdir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def normalize_athlete_id(aid: str) -> str:
    s = str(aid).strip()
    m = re.match(r"^([A-Za-z])[-_]?0*(\d+)$", s)
    if not m:
        return s
    return f"{m.group(1).upper()}-{int(m.group(2)):05d}"


def parse_date(s):
    return pd.to_datetime(str(s).strip(), errors="coerce")


def is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    t = str(x).strip()
    return (t == "" or t.lower() == "nan")


def to_str(x) -> str:
    return "" if is_blank(x) else str(x).strip()


def to_int_safe(x, default=None):
    try:
        if is_blank(x):
            return default
        return int(float(x))
    except Exception:
        return default


def register_jp_font(base_dir: str) -> str:
    font_path = os.path.join(base_dir, "fonts", "ipaexg.ttf")
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("IPAexGothic", font_path))
            return "IPAexGothic"
        except Exception:
            pass
    return "Helvetica"


def load_forms_clean(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "data", "processed", "forms_clean.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "practice_date" not in df.columns:
        raise ValueError("forms_clean.csv に practice_date 列がありません")
    if "athlete_id" not in df.columns:
        raise ValueError("forms_clean.csv に athlete_id 列がありません")
    df["practice_date"] = df["practice_date"].apply(parse_date)
    df["athlete_id"] = df["athlete_id"].apply(normalize_athlete_id)
    return df


def risk_bar_color(label: str):
    if label == "CRITICAL":
        return colors.HexColor("#B71C1C")
    if label == "HIGH":
        return colors.HexColor("#E65100")
    if label == "MID":
        return colors.HexColor("#F9A825")
    return colors.HexColor("#2E7D32")


def daily_risk_label(row) -> str:
    """
    日別の簡易リスク：痛みがあればMID以上、自己評価低ければ加点
    """
    pain_flag = to_str(row.get("pain_flag"))
    pain_loc = to_str(row.get("pain_location"))
    has_pain = (pain_loc != "") or (pain_flag not in ("", "0", "なし", "無", "いいえ", "No", "NO"))

    # 自己評価
    ach = to_int_safe(row.get("achievement"), default=3)
    sat = to_int_safe(row.get("satisfaction"), default=3)
    cond = to_int_safe(row.get("condition"), default=3)

    low_cnt = sum(1 for v in [ach, sat, cond] if v is not None and v <= 2)

    if has_pain and low_cnt >= 2:
        return "HIGH"
    if has_pain:
        return "MID"
    if low_cnt >= 2:
        return "MID"
    return "LOW"


def make_table(rows, col_widths, font, fontsize=9.5):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("FONTSIZE", (0, 0), (-1, -1), fontsize),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build_daily_pages_pdf(base_dir: str, athlete_id: str, audience: str, start: str, end: str) -> str:
    df = load_forms_clean(base_dir)
    aid = normalize_athlete_id(athlete_id)

    s_dt = pd.to_datetime(start, errors="coerce")
    e_dt = pd.to_datetime(end, errors="coerce")

    df_a = df[df["athlete_id"] == aid].copy()
    df_p = df_a[(df_a["practice_date"] >= s_dt) & (df_a["practice_date"] <= e_dt)].copy()
    df_p = df_p.sort_values("practice_date")

    if len(df_a) == 0:
        raise ValueError(f"athlete_id={aid} が見つかりません")
    if len(df_p) == 0:
        raise ValueError("指定期間にデータがありません")

    font = register_jp_font(base_dir)
    styles = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=styles["BodyText"], fontName=font, fontSize=10, leading=13, wordWrap="CJK")
    h = ParagraphStyle("h", parent=styles["Heading3"], fontName=font, fontSize=12, leading=14)
    title = ParagraphStyle("title", parent=styles["Title"], fontName=font, fontSize=16, leading=18)

    athlete_name = to_str(df_a.iloc[-1].get("athlete_name")) or aid

    out_dir = os.path.join(base_dir, "reports", "athletes", "_tmp_parts")
    safe_mkdir(out_dir)
    out_path = os.path.join(out_dir, f"{aid}_{start}_to_{end}_{audience}__p3plus.pdf")

    doc = SimpleDocTemplate(
        out_path,
        pagesize=landscape(A4),
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=12 * mm, bottomMargin=10 * mm,
    )

    story: List = []

    # 日付ごと（基本：1日=1行想定。複数行ある場合は最終行を日別として扱う）
    df_p["_d"] = df_p["practice_date"].dt.date
    for i, (d, g) in enumerate(df_p.groupby("_d")):
        row = g.sort_values("practice_date").iloc[-1]
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        risk = daily_risk_label(row)

        if i == 0:
            # Page3からスタートする想定なので最初にPageBreak入れてもOK（結合時に自然に続く）
            pass
        else:
            story.append(PageBreak())

        story.append(Paragraph(f"日別詳細：{date_str}", title))
        story.append(Spacer(1, 6))

        # リスク帯
        bar = Table([[Paragraph(f"<b>{'良好' if risk=='LOW' else '注意'}（{risk}）</b>　フォーム精度を保ち段階的に積み上げ",
                                ParagraphStyle("rb", fontName=font, fontSize=11, leading=13, textColor=colors.white, wordWrap="CJK"))]],
                    colWidths=[doc.width])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), risk_bar_color(risk)),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.black),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(bar)
        story.append(Spacer(1, 8))

        # 練習の要約
        story.append(Paragraph("練習の要約", h))
        menu = to_str(row.get("menu"))
        drills = to_str(row.get("drills"))
        theme = to_str(row.get("theme"))
        pain = to_str(row.get("pain_location")) or "なし"

        t1 = make_table([
            ["メニュー", Paragraph(menu.replace("\n", "<br/>"), base)],
            ["ドリル/内容", Paragraph(drills.replace("\n", "<br/>"), base)],
            ["テーマ", Paragraph(theme.replace("\n", "<br/>"), base)],
            ["痛み", Paragraph(pain.replace("\n", "<br/>"), base)],
        ], col_widths=[doc.width * 0.18, doc.width * 0.82], font=font)
        story.append(t1)
        story.append(Spacer(1, 8))

        # 自己評価
        story.append(Paragraph("自己評価（1〜5）", h))
        ach = to_str(row.get("achievement"))
        sat = to_str(row.get("satisfaction"))
        cond = to_str(row.get("condition"))
        t2 = make_table([
            ["達成度", ach],
            ["満足度", sat],
            ["コンディション", cond],
        ], col_widths=[doc.width * 0.18, doc.width * 0.82], font=font)
        story.append(t2)
        story.append(Spacer(1, 8))

        # 振り返り
        story.append(Paragraph("振り返り（Good / Bad / Cause / Next）", h))
        good = to_str(row.get("good"))
        bad = to_str(row.get("bad"))
        cause = to_str(row.get("cause"))
        nxt = to_str(row.get("next"))

        t3 = make_table([
            ["Good", Paragraph(good.replace("\n", "<br/>"), base)],
            ["Bad", Paragraph(bad.replace("\n", "<br/>"), base)],
            ["Cause", Paragraph(cause.replace("\n", "<br/>"), base)],
            ["Next", Paragraph(nxt.replace("\n", "<br/>"), base)],
        ], col_widths=[doc.width * 0.18, doc.width * 0.82], font=font)
        story.append(t3)
        story.append(Spacer(1, 8))

        # 計測・環境
        story.append(Paragraph("計測・環境", h))
        measured = to_str(row.get("measured_events")) or "ない"
        times = to_str(row.get("times"))
        surface = to_str(row.get("surface"))
        location = to_str(row.get("location"))

        t4 = make_table([
            ["測った種目", measured],
            ["記録", times if times else "（記入なし）"],
            ["路面", surface if surface else "（記入なし）"],
            ["場所/大会名", location if location else "（記入なし）"],
        ], col_widths=[doc.width * 0.18, doc.width * 0.82], font=font)
        story.append(t4)
        story.append(Spacer(1, 8))

        # 目標（変更時）
        story.append(Paragraph("目標（変更時）", h))
        ge = to_str(row.get("goal_event"))
        gt = to_str(row.get("goal_time"))
        gd = to_str(row.get("goal_target_date"))
        gf = to_str(row.get("goal_focus"))

        t5 = make_table([
            ["目標種目", ge if ge else "（記入なし）"],
            ["目標タイム", gt if gt else "（記入なし）"],
            ["目標期限", gd if gd else "（記入なし）"],
            ["理由/重点", gf if gf else "（記入なし）"],
        ], col_widths=[doc.width * 0.18, doc.width * 0.82], font=font)
        story.append(t5)

    doc.build(story)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("athlete_id")
    ap.add_argument("--audience", choices=["coach", "parent"], default="coach")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    saved = build_daily_pages_pdf(base_dir, args.athlete_id, args.audience, args.start, args.end)
    print(f"[OK] Page3+ PDF: {saved}")


if __name__ == "__main__":
    main()

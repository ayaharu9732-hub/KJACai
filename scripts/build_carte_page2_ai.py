# -*- coding: utf-8 -*-
"""
成長カルテ Page2（AIコメント）生成（Python3.9 / APIなしテンプレ版）
"""

import os
import re
import argparse
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
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


def compute_simple_risk(df_period: pd.DataFrame) -> Dict[str, object]:
    """
    Page2用：簡易リスク（あなたのpage1と完全一致でなくてOKならこれで十分）
    - pain_days: pain_location or pain_flag がある日の数
    """
    if df_period is None or len(df_period) == 0:
        return {"pain_days": 0, "risk_label": "LOW", "risk_reason": "データなし"}

    pain_days = set()
    for _, r in df_period.iterrows():
        d = r.get("practice_date")
        if pd.isna(d):
            continue
        pf = to_str(r.get("pain_flag"))
        pl = to_str(r.get("pain_location"))
        has_pain = (pl != "") or (pf not in ("", "0", "なし", "無", "いいえ", "No", "NO"))
        if has_pain:
            pain_days.add(d.date())

    pain_days = len(pain_days)

    if pain_days >= 3:
        return {"pain_days": pain_days, "risk_label": "HIGH", "risk_reason": "痛みが複数日"}
    if pain_days >= 1:
        return {"pain_days": pain_days, "risk_label": "MID", "risk_reason": "軽微な痛みあり"}
    return {"pain_days": 0, "risk_label": "LOW", "risk_reason": "痛み記録なし"}


def pick_goal(df_athlete: pd.DataFrame, df_period: pd.DataFrame) -> Dict[str, str]:
    cols = ["goal_event", "goal_time", "goal_target_date", "goal_focus"]

    def last_non_empty(dfx: pd.DataFrame) -> Optional[pd.Series]:
        if dfx is None or len(dfx) == 0:
            return None
        d = dfx.sort_values("practice_date")
        for i in range(len(d) - 1, -1, -1):
            row = d.iloc[i]
            if any(to_str(row.get(c)) != "" for c in cols):
                return row
        return None

    r = last_non_empty(df_period)
    if r is None:
        r = last_non_empty(df_athlete)

    out = {c: "" for c in cols}
    if r is not None:
        for c in cols:
            out[c] = to_str(r.get(c))
    return out


def build_page2_pdf(base_dir: str, athlete_id: str, audience: str, start: str, end: str) -> str:
    df = load_forms_clean(base_dir)
    aid = normalize_athlete_id(athlete_id)

    s_dt = pd.to_datetime(start, errors="coerce")
    e_dt = pd.to_datetime(end, errors="coerce")
    df_a = df[df["athlete_id"] == aid].copy()
    df_p = df_a[(df_a["practice_date"] >= s_dt) & (df_a["practice_date"] <= e_dt)].copy().sort_values("practice_date")

    if len(df_a) == 0:
        raise ValueError(f"athlete_id={aid} が見つかりません")
    if len(df_p) == 0:
        raise ValueError("指定期間にデータがありません")

    athlete_name = to_str(df_a.iloc[-1].get("athlete_name")) or aid
    goal = pick_goal(df_a, df_p)
    risk = compute_simple_risk(df_p)
    pain_days = int(risk["pain_days"])

    # 直近ログ（最後の日）
    last = df_p.iloc[-1]
    last_good = to_str(last.get("good"))
    last_bad = to_str(last.get("bad"))
    last_next = to_str(last.get("next"))
    last_theme = to_str(last.get("theme"))
    last_pain = to_str(last.get("pain_location"))

    out_dir = os.path.join(base_dir, "reports", "athletes", "_tmp_parts")
    safe_mkdir(out_dir)
    out_path = os.path.join(out_dir, f"{aid}_{start}_to_{end}_{audience}__p2.pdf")

    font = register_jp_font(base_dir)
    styles = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=styles["BodyText"], fontName=font, fontSize=10, leading=14, wordWrap="CJK")
    title = ParagraphStyle("title", parent=styles["Title"], fontName=font, fontSize=18, leading=22, alignment=1)

    doc = SimpleDocTemplate(
        out_path,
        pagesize=landscape(A4),
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=12 * mm, bottomMargin=10 * mm,
    )

    story = []
    story.append(Paragraph("AIコメント", title))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"【対象】{athlete_name}（{aid}）", base))
    story.append(Paragraph(f"【期間】{start} ～ {end}", base))
    story.append(Spacer(1, 8))

    story.append(Paragraph(f"■ リスク判定：<b>{risk['risk_label']}</b>", base))
    story.append(Paragraph(f"- 理由：{risk['risk_reason']}（痛み日数={pain_days}）", base))
    if last_pain:
        story.append(Paragraph(f"- 直近の痛み：{last_pain}", base))
    story.append(Spacer(1, 8))

    story.append(Paragraph("■ 目標（登録内容）", base))
    story.append(Paragraph(f"- 種目：{goal.get('goal_event','')}", base))
    story.append(Paragraph(f"- タイム：{goal.get('goal_time','')}", base))
    story.append(Paragraph(f"- 期限：{goal.get('goal_target_date','')}", base))
    story.append(Paragraph(f"- 重点：{goal.get('goal_focus','')}", base))
    story.append(Spacer(1, 10))

    story.append(Paragraph("■ 直近ログの要点", base))
    if last_theme:
        story.append(Paragraph(f"- テーマ：{last_theme}", base))
    if last_good:
        story.append(Paragraph(f"- Good：{last_good}", base))
    if last_bad:
        story.append(Paragraph(f"- Bad：{last_bad}", base))
    if last_next:
        story.append(Paragraph(f"- Next：{last_next}", base))
    story.append(Spacer(1, 10))

    story.append(Paragraph("■ 次の一手（コーチ向け提案）", base))
    if risk["risk_label"] in ("HIGH",):
        tips = [
            "痛みが複数日ある間は“強度アップ”より“痛みゼロで終える設計”を優先。",
            "跳躍/反発系の量を抑え、技術ドリル＋補強＋可動域に寄せる。",
            "翌日の痛み残りが無いことをトリガーに、強度を段階的に戻す。",
        ]
    elif risk["risk_label"] == "MID":
        tips = [
            "軽微な痛みがあるので、強度は維持〜微調整で悪化させない運用。",
            "接地の再現性（引っかかり/骨盤維持）を軸に“量より質”へ。",
            "違和感が翌日に残る場合は、反発系の量を一段落とす。",
        ]
    else:
        tips = [
            "痛み記録なし。フォーム精度を保ち、段階的に積み上げ。",
            "計測がある日は前後のケア/補強を固定ルーティン化。",
        ]
    for t in tips:
        story.append(Paragraph(f"- {t}", base))

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
    saved = build_page2_pdf(base_dir, args.athlete_id, args.audience, args.start, args.end)
    print(f"[OK] Page2 PDF: {saved}")


if __name__ == "__main__":
    main()

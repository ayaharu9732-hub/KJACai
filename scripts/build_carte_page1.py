# -*- coding: utf-8 -*-
"""
成長カルテ Page1（サマリー）生成
= あなたが貼ったスクリプトを「ページ1専用」として固定し、関数として呼べる形にしたもの

使い方（単体）:
  python scripts/build_carte_page1.py A-0001 --audience coach --start 2026-01-12 --end 2026-01-14
"""

import os
import argparse
from datetime import datetime

import pandas as pd

# -*- coding: utf-8 -*-
"""
EasyRev Sports / KJAC
成長カルテ（COACH / PARENT） 1ページ目 生成スクリプト（Python 3.9互換・A4横固定）

✅ A4 Landscape 固定
✅ IPAexGothic 強制（白豆腐回避）
✅ A-0001 -> A-00001 自動補正
✅ forms_clean.csv の practice_date を日付列として使用（date列は不要）
✅ 目標が空欄にならない（goal_* を期間内 -> 期間外最新 の順に拾う）
✅ 期間サマリーはみ出し防止（Paragraph + colWidth最適化 + rowHeights自動）
✅ サマリーヘッダー塗りつぶしの文字は白
✅ 痛み部位別重み付きリスク判定（賢い判定）
✅ pain_log_count = risk_metrics["pain_days"]（痛みログ数は「痛みがあった日数」）
✅ unknown pain は reports/athletes/_unknown_pain_values.csv へログ

入力:
  data/processed/forms_clean.csv

痛みマスタ（任意）:
  masters/pain_locations.csv
  ※ 存在すれば raw->normalized の補正に使う（列名は柔軟対応）

出力:
  reports/athletes/<athlete_id>_<start>_to_<end>_<audience>.pdf
"""

import os
import re
import csv
import argparse
from datetime import datetime

import pandas as pd

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# -----------------------------
# Utilities
# -----------------------------
def safe_mkdir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def normalize_athlete_id(aid: str) -> str:
    """
    A-0001 -> A-00001 など 5桁へ正規化
    """
    if aid is None:
        return ""
    s = str(aid).strip()
    m = re.match(r"^([A-Za-z])[-_]?0*(\d+)$", s)
    if not m:
        return s
    prefix = m.group(1).upper()
    num = int(m.group(2))
    return f"{prefix}-{num:05d}"


def parse_date(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return pd.NaT
    ss = str(s).strip()
    if ss == "":
        return pd.NaT
    # 例: 2026/01/14, 2026-01-14
    return pd.to_datetime(ss, errors="coerce")


def is_blank(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return True
    return False


def to_int_safe(x, default=None):
    try:
        if is_blank(x):
            return default
        return int(float(x))
    except Exception:
        return default


def to_str_safe(x):
    if is_blank(x):
        return ""
    return str(x).strip()


# -----------------------------
# Pain master / normalization
# -----------------------------
def load_pain_master(master_path: str):
    """
    masters/pain_locations.csv があれば raw->normalized の辞書を作る
    列名が違っても、先頭2列を raw / normalized として扱う
    """
    if not os.path.exists(master_path):
        return None

    try:
        dfm = pd.read_csv(master_path, encoding="utf-8-sig")
    except Exception:
        try:
            dfm = pd.read_csv(master_path, encoding="cp932")
        except Exception:
            return None

    if dfm.shape[1] < 2:
        return None

    cols = list(dfm.columns)
    # 候補列名（あれば優先）
    raw_candidates = ["raw", "input", "original", "pain_raw", "value", "before"]
    norm_candidates = ["normalized", "norm", "standard", "pain_location", "after"]

    raw_col = None
    norm_col = None

    for c in cols:
        if c in raw_candidates:
            raw_col = c
            break
    for c in cols:
        if c in norm_candidates:
            norm_col = c
            break

    # 見つからなければ先頭2列
    if raw_col is None or norm_col is None:
        raw_col = cols[0]
        norm_col = cols[1]

    mp = {}
    for _, r in dfm.iterrows():
        a = to_str_safe(r.get(raw_col))
        b = to_str_safe(r.get(norm_col))
        if a != "" and b != "":
            mp[a] = b
    return mp if len(mp) > 0 else None


def split_locations(s: str):
    """
    痛み部位が "左ハムストリング, 右ふくらはぎ" 等の可能性があるので分割
    """
    if is_blank(s):
        return []
    text = str(s)
    # 区切り候補: 、,，,;,/,改行,空白(連続)
    parts = re.split(r"[、，,;/\n]+", text)
    out = []
    for p in parts:
        t = str(p).strip()
        if t:
            out.append(t)
    return out


def normalize_pain_location(raw: str, norm_map, unknown_log_path: str, athlete_id: str, practice_date: str):
    """
    1つの部位を正規化
    - master辞書にあれば置換
    - なければそのまま返しつつ unknown に記録
    """
    if is_blank(raw):
        return ""
    s = str(raw).strip()

    if norm_map is not None and s in norm_map:
        return norm_map[s]

    # 未知はログ
    safe_mkdir(os.path.dirname(unknown_log_path))
    existed = os.path.exists(unknown_log_path)
    try:
        with open(unknown_log_path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if not existed:
                w.writerow(["timestamp", "athlete_id", "practice_date", "unknown_value"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), athlete_id, practice_date, s])
    except Exception:
        pass

    return s


# -----------------------------
# Risk logic (weighted)
# -----------------------------
def pain_weight(loc: str) -> int:
    """
    痛み部位の重み（賢くする：典型的な故障リスクを高めに）
    ※ 文字列包含で判定
    """
    if is_blank(loc):
        return 0
    s = str(loc)

    # 高リスク（腱・関節・骨系）
    high4 = ["アキレス", "膝", "腰", "足首", "股関節", "すね", "シンスプリント", "疲労骨折", "骨"]
    for k in high4:
        if k in s:
            return 4

    # 中〜高（ハム・ふくらはぎ・大腿など）
    mid3 = ["ハム", "ハムストリング", "ふくらはぎ", "腓腹", "大腿", "鼠径", "臀部", "お尻"]
    for k in mid3:
        if k in s:
            return 3

    # 低〜中（違和感・張り・軽微）
    low2 = ["張り", "違和感", "軽い", "筋肉痛"]
    for k in low2:
        if k in s:
            return 2

    # 不明は最低1
    return 1


def compute_risk_metrics(df_period: pd.DataFrame, norm_map, unknown_log_path: str, athlete_id: str):
    """
    期間内のリスク指標を計算
    """
    if df_period is None or len(df_period) == 0:
        return {
            "log_count": 0,
            "pain_days": 0,
            "pain_weight_score": 0,
            "low_condition_days": 0,
            "low_satisfaction_days": 0,
            "low_achievement_days": 0,
            "risk_label": "LOW",
            "risk_message": "良好（LOW） 継続OK（油断せず）",
        }

    dfp = df_period.copy()

    # pain_flag / pain_location を見る（pain_flagが空でも pain_location があれば痛み扱い）
    pain_days_set = set()
    day_weight = {}  # date -> max weight that day

    for _, r in dfp.iterrows():
        d = to_str_safe(r.get("practice_date"))
        pain_flag = to_str_safe(r.get("pain_flag"))
        pain_loc_raw = to_str_safe(r.get("pain_location"))

        has_pain = False
        if pain_flag != "" and pain_flag not in ["0", "なし", "無", "いいえ", "No", "NO"]:
            has_pain = True
        if pain_loc_raw != "":
            has_pain = True

        if has_pain and d != "":
            pain_days_set.add(d)

            # 部位ごと正規化→重み
            locs = split_locations(pain_loc_raw) if pain_loc_raw != "" else []
            wmax = 1
            if len(locs) > 0:
                wvals = []
                for loc in locs:
                    nloc = normalize_pain_location(loc, norm_map, unknown_log_path, athlete_id, d)
                    wvals.append(pain_weight(nloc))
                wmax = max(wvals) if len(wvals) > 0 else 1
            day_weight[d] = max(day_weight.get(d, 0), wmax)

    pain_days = len(pain_days_set)
    pain_weight_score = sum(day_weight.values())  # 日ごとの最大重みを合算

    # 体調/満足/達成（数値が入る前提：<=2 を注意）
    def count_low(colname: str):
        c = 0
        for _, r in dfp.iterrows():
            v = to_int_safe(r.get(colname), default=None)
            if v is not None and v <= 2:
                c += 1
        return c

    low_condition_days = count_low("condition")
    low_satisfaction_days = count_low("satisfaction")
    low_achievement_days = count_low("achievement")

    # スコア化（痛みが中心、次に主観指標）
    score = 0
    # 痛み日数
    if pain_days >= 3:
        score += 6
    elif pain_days == 2:
        score += 4
    elif pain_days == 1:
        score += 2

    # 痛み重み（腱/関節系は強く反映）
    score += pain_weight_score  # 0〜(4*日数) 程度

    # 主観指標（低い日が多いほど加点）
    score += min(low_condition_days, 3)
    score += min(low_satisfaction_days, 3)
    score += min(low_achievement_days, 3)

    # ラベル決定（賢い閾値）
    # CRITICAL: 高重み痛みが続く / 合計スコアが大きい
    if pain_weight_score >= 9 or score >= 14 or pain_days >= 4:
        label = "CRITICAL"
        msg = "要注意（CRITICAL） 直ちに負荷調整・状態確認が必要"
    elif pain_weight_score >= 6 or score >= 10 or pain_days >= 3:
        label = "HIGH"
        msg = "注意（HIGH） 負荷注意・痛み/疲労の確認推奨"
    elif pain_weight_score >= 3 or score >= 6 or pain_days >= 1:
        label = "MID"
        msg = "注意（MID） 疲労/痛みの兆候に注意しつつ継続"
    else:
        label = "LOW"
        msg = "良好（LOW） 継続OK（油断せず）"

    return {
        "log_count": int(len(dfp)),
        "pain_days": int(pain_days),
        "pain_weight_score": int(pain_weight_score),
        "low_condition_days": int(low_condition_days),
        "low_satisfaction_days": int(low_satisfaction_days),
        "low_achievement_days": int(low_achievement_days),
        "risk_label": label,
        "risk_message": msg,
    }


def risk_colors(label: str):
    """
    リスク帯の色
    """
    if label == "CRITICAL":
        return colors.HexColor("#B71C1C")  # deep red
    if label == "HIGH":
        return colors.HexColor("#E65100")  # deep orange
    if label == "MID":
        return colors.HexColor("#F9A825")  # amber
    return colors.HexColor("#2E7D32")      # green


# -----------------------------
# PDF Parts
# -----------------------------
def register_jp_font(base_dir: str):
    """
    IPAexGothic を fonts/ipaexg.ttf から登録
    """
    font_path = os.path.join(base_dir, "fonts", "ipaexg.ttf")
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("IPAexGothic", font_path))
            return "IPAexGothic"
        except Exception:
            pass
    # fallback
    return "Helvetica"


def make_styles(font_name: str):
    styles = getSampleStyleSheet()

    h = ParagraphStyle(
        "h",
        parent=styles["Heading3"],
        fontName=font_name,
        fontSize=12,
        leading=14,
        spaceAfter=4,
    )
    p = ParagraphStyle(
        "p",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9.5,
        leading=11.5,
    )
    small = ParagraphStyle(
        "small",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=8.5,
        leading=10.5,
    )
    tiny = ParagraphStyle(
        "tiny",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=7.5,
        leading=9.5,
    )
    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=16,
        leading=18,
        alignment=1,  # center
    )
    return {"h": h, "p": p, "small": small, "tiny": tiny, "title": title}


def cell_para(text, style):
    t = to_str_safe(text)
    if t == "":
        t = ""
    # CJK wrap を効かせるために Paragraph 化
    return Paragraph(t.replace("\n", "<br/>"), style)


def build_goal_table(goal, S, width):
    data = [
        [cell_para("目標種目", S["p"]), cell_para(goal.get("goal_event", ""), S["p"])],
        [cell_para("目標タイム", S["p"]), cell_para(goal.get("goal_time", ""), S["p"])],
        [cell_para("目標期限", S["p"]), cell_para(goal.get("goal_target_date", ""), S["p"])],
        [cell_para("理由／重点", S["p"]), cell_para(goal.get("goal_focus", ""), S["p"])],
    ]
    colw = [width * 0.18, width * 0.82]
    t = Table(data, colWidths=colw)
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.black),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return t


def build_dashboard_table(risk_metrics, S, width, start_date: str, end_date: str):
    # 重要: pain_log_count は pain_days
    pain_log_count = risk_metrics["pain_days"]

    data = [
        [cell_para("対象期間", S["p"]), cell_para(f"{start_date} ～ {end_date}", S["p"]),
         cell_para("痛みログ数", S["p"]), cell_para(str(pain_log_count), S["p"])],
        [cell_para("ログ件数", S["p"]), cell_para(str(risk_metrics["log_count"]), S["p"]),
         cell_para("", S["p"]), cell_para("", S["p"])],
        [cell_para("最大RISK", S["p"]), cell_para(risk_metrics["risk_label"], S["p"]),
         cell_para("", S["p"]), cell_para("", S["p"])],
    ]
    colw = [width * 0.12, width * 0.48, width * 0.12, width * 0.28]
    t = Table(data, colWidths=colw)
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.black),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (0,-1), colors.whitesmoke),
        ("BACKGROUND", (2,0), (2,-1), colors.whitesmoke),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return t


def build_period_summary_table(df_period: pd.DataFrame, S, width, font_name: str, audience: str):
    """
    期間サマリー：はみ出し防止優先
    - header 背景紫 + 文字白
    - 各セルは Paragraph
    """
    # 列構成（今の画面に合わせて）
    headers = ["日付", "メニュー", "テーマ", "痛み", "Good", "Bad", "Next", "計測", "タイム"]

    # colWidths: 合計=width になるよう配分
    colw = [
        width * 0.10,  # 日付
        width * 0.25,  # メニュー
        width * 0.12,  # テーマ
        width * 0.10,  # 痛み
        width * 0.13,  # Good
        width * 0.13,  # Bad
        width * 0.13,  # Next
        width * 0.08,  # 計測
        width * 0.06,  # タイム
    ]

    data = [headers]

    # 期間内を日付昇順
    dfp = df_period.copy()
    dfp["practice_date"] = pd.to_datetime(dfp["practice_date"], errors="coerce")
    dfp = dfp.sort_values("practice_date")

    for _, r in dfp.iterrows():
        d = r.get("practice_date")
        d_str = ""
        if pd.notna(d):
            d_str = d.strftime("%Y-%m-%d")

        menu = to_str_safe(r.get("menu"))
        theme = to_str_safe(r.get("theme"))
        pain = to_str_safe(r.get("pain_location"))
        good = to_str_safe(r.get("good"))
        bad = to_str_safe(r.get("bad"))
        nxt = to_str_safe(r.get("next"))
        measured = to_str_safe(r.get("measured_events"))
        times = to_str_safe(r.get("times"))

        # 文字が長いところは tiny で安全に
        menu_p = cell_para(menu, S["tiny"])
        theme_p = cell_para(theme, S["small"])
        pain_p = cell_para(pain, S["small"])
        good_p = cell_para(good, S["small"])
        bad_p = cell_para(bad, S["small"])
        next_p = cell_para(nxt, S["small"])
        meas_p = cell_para(measured, S["small"])
        time_p = cell_para(times, S["small"])

        data.append([
            cell_para(d_str, S["small"]),
            menu_p,
            theme_p,
            pain_p,
            good_p,
            bad_p,
            next_p,
            meas_p,
            time_p,
        ])

    t = Table(data, colWidths=colw, repeatRows=1)
    header_bg = colors.HexColor("#3D2A5E")  # 紫
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.black),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),  # ★ヘッダー白文字
        ("FONTNAME", (0,0), (-1,0), font_name),
        ("FONTSIZE", (0,0), (-1,0), 9.5),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    return t


# -----------------------------
# Data extract
# -----------------------------
def load_forms_clean(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "data", "processed", "forms_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"forms_clean.csv が見つかりません: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    # 必須列チェック
    if "practice_date" not in df.columns:
        raise ValueError("forms_clean.csv に practice_date 列がありません（列名を確認してください）")
    if "athlete_id" not in df.columns:
        raise ValueError("forms_clean.csv に athlete_id 列がありません（列名を確認してください）")

    # 正規化
    df["practice_date"] = df["practice_date"].apply(parse_date)
    df["athlete_id"] = df["athlete_id"].apply(normalize_athlete_id)

    # 文字列系が NaN のまま出るのを防ぐ
    for c in ["menu", "drills", "theme", "pain_flag", "pain_location", "good", "bad", "cause", "next",
              "measured_events", "times", "goal_event", "goal_time", "goal_target_date", "goal_focus",
              "athlete_name"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


def pick_goal(df_athlete: pd.DataFrame, df_period: pd.DataFrame):
    """
    目標が空欄にならないように
    1) 期間内で goal_* が埋まっている最新
    2) 期間外含め athlete 全体で goal_* が埋まっている最新
    """
    cols = ["goal_event", "goal_time", "goal_target_date", "goal_focus"]

    def extract_latest(dfx):
        if dfx is None or len(dfx) == 0:
            return None
        d = dfx.copy()
        d = d.sort_values("practice_date")
        # 「少なくとも1項目埋まっている」行を候補
        mask = None
        for c in cols:
            if c in d.columns:
                m = ~d[c].astype(str).str.strip().isin(["", "nan", "None"])
                mask = m if mask is None else (mask | m)
        if mask is None:
            return None
        cand = d[mask]
        if len(cand) == 0:
            return None
        r = cand.iloc[-1]
        out = {}
        for c in cols:
            out[c] = to_str_safe(r.get(c))
        return out

    g = extract_latest(df_period)
    if g is not None:
        return g

    g = extract_latest(df_athlete)
    if g is not None:
        return g

    # それでもなければ空
    return {"goal_event": "", "goal_time": "", "goal_target_date": "", "goal_focus": ""}


# -----------------------------
# Main build
# -----------------------------
def build_pdf(base_dir: str, athlete_id_arg: str, audience: str, start_date: str, end_date: str,
              df_all: pd.DataFrame, out_path: str) -> str:

    athlete_id = normalize_athlete_id(athlete_id_arg)

    # 対象 athlete 抽出
    df_a = df_all[df_all["athlete_id"] == athlete_id].copy()
    if len(df_a) == 0:
        # A-0001 を入れてきたときに、df側が A-00001 であるのは既に正規化で吸収済み
        # それでも無い場合は候補提示
        raise ValueError(f"athlete_id={athlete_id} のデータが forms_clean.csv に見つかりません")

    s_dt = pd.to_datetime(start_date, errors="coerce")
    e_dt = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(s_dt) or pd.isna(e_dt):
        raise ValueError("start/end の日付が不正です。例: 2026-01-12")

    df_p = df_a[(df_a["practice_date"] >= s_dt) & (df_a["practice_date"] <= e_dt)].copy()

    # pain master
    master_path = os.path.join(base_dir, "masters", "pain_locations.csv")
    norm_map = load_pain_master(master_path)
    unknown_log_path = os.path.join(base_dir, "reports", "athletes", "_unknown_pain_values.csv")

    # リスク
    risk_metrics = compute_risk_metrics(df_p, norm_map, unknown_log_path, athlete_id)

    # 目標（空欄にならない）
    goal = pick_goal(df_a, df_p)

    # PDF
    safe_mkdir(os.path.dirname(out_path))

    # フォント
    font_name = register_jp_font(base_dir)
    S = make_styles(font_name)

    pagesize = landscape(A4)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=pagesize,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    story = []

    # ヘッダー（左右に薄いテキスト）
    header_left = Paragraph("EasyRev Sports / KJAC AI", ParagraphStyle("hl", fontName=font_name, fontSize=9, leading=10))
    header_right = Paragraph(f"成長カルテ（{audience.upper()}） | p.1", ParagraphStyle("hr", fontName=font_name, fontSize=9, leading=10, alignment=2))
    ht = Table([[header_left, header_right]], colWidths=[doc.width*0.5, doc.width*0.5])
    ht.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.grey),
    ]))
    story.append(ht)

    # タイトル
    athlete_name = to_str_safe(df_a.iloc[-1].get("athlete_name"))
    title_text = f"【成長カルテ】  {athlete_name}（{athlete_id}）"
    story.append(Spacer(1, 2))
    story.append(Paragraph(title_text, S["title"]))
    story.append(Spacer(1, 6))

    # リスク帯
    bar_color = risk_colors(risk_metrics["risk_label"])
    risk_bar = Table([[cell_para(risk_metrics["risk_message"], ParagraphStyle("rb", fontName=font_name, fontSize=11, leading=13, textColor=colors.white))]],
                     colWidths=[doc.width])
    risk_bar.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bar_color),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("BOX", (0,0), (-1,-1), 0.8, colors.black),
    ]))
    story.append(risk_bar)
    story.append(Spacer(1, 10))

    # 目標
    story.append(Paragraph("【目標】", S["h"]))
    story.append(build_goal_table(goal, S, doc.width))
    story.append(Spacer(1, 10))

    # ダッシュボード
    story.append(Paragraph("【コーチ用ダッシュボード（最速把握）】", S["h"]))
    story.append(build_dashboard_table(risk_metrics, S, doc.width, start_date, end_date))
    story.append(Spacer(1, 10))

    # 期間サマリー
    story.append(Paragraph("■ 期間サマリー一覧", S["h"]))
    story.append(build_period_summary_table(df_p, S, doc.width, font_name, audience))

    # ビルド（PermissionError対策：開いてると失敗する）
    try:
        doc.build(story)
        return out_path
    except PermissionError:
        # 開いてるPDFがある場合はファイル名を変えて逃がす
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = out_path.replace(".pdf", f"_{ts}.pdf")
        doc = SimpleDocTemplate(
            alt,
            pagesize=pagesize,
            leftMargin=14 * mm,
            rightMargin=14 * mm,
            topMargin=10 * mm,
            bottomMargin=10 * mm,
        )
        doc.build(story)
        return alt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("athlete_id", help="例: A-0001 / A-00001")
    ap.add_argument("--audience", choices=["coach", "parent"], default="coach")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"[INFO] base_dir: {base_dir}")

    df = load_forms_clean(base_dir)

    aid = normalize_athlete_id(args.athlete_id)
    out_dir = os.path.join(base_dir, "reports", "athletes")
    safe_mkdir(out_dir)

    out_path = os.path.join(out_dir, f"{aid}_{args.start}_to_{args.end}_{args.audience}.pdf")

    saved = build_pdf(base_dir, aid, args.audience, args.start, args.end, df, out_path)
    print(f"[OK] PDF出力: {saved}")


if __name__ == "__main__":
    main()







def build_page1_pdf(base_dir: str, athlete_id: str, audience: str, start: str, end: str) -> str:
    """
    runner から呼ばれる想定：ページ1のPDFを生成してパスを返す
    """
    df = load_forms_clean(base_dir)
    aid = normalize_athlete_id(athlete_id)

    out_dir = os.path.join(base_dir, "reports", "athletes", "_tmp_parts")
    safe_mkdir(out_dir)

    out_path = os.path.join(out_dir, f"{aid}_{start}_to_{end}_{audience}__p1.pdf")
    saved = build_pdf(base_dir, aid, audience, start, end, df, out_path)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("athlete_id")
    ap.add_argument("--audience", choices=["coach", "parent"], default="coach")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    saved = build_page1_pdf(base_dir, args.athlete_id, args.audience, args.start, args.end)
    print(f"[OK] Page1 PDF: {saved}")


if __name__ == "__main__":
    main()

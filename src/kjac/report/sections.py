from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.units import mm

from src.kjac.report.layout import draw_lines, draw_table, draw_title


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_metrics(metrics_csv: str | Path) -> tuple[list[dict[str, str]], dict[str, float], dict[str, str] | None]:
    rows: list[dict[str, str]] = []
    with Path(metrics_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})

    data_rows = [r for r in rows if (r.get("section", "") or "").lower() != "summary"]
    summary_rows = [r for r in rows if (r.get("section", "") or "").lower() == "summary"]
    summary = summary_rows[0] if summary_rows else None

    def pick(col: str) -> float:
        if summary:
            sv = _to_float(summary.get(col))
            if not math.isnan(sv):
                return sv
        vals = [_to_float(r.get(col)) for r in data_rows]
        vals = [v for v in vals if not math.isnan(v)]
        return float(sum(vals) / len(vals)) if vals else math.nan

    stats = {
        "pitch_hz": pick("pitch_hz"),
        "stride_m": pick("stride_m"),
        "avg_speed_mps": pick("avg_speed_mps"),
    }
    return data_rows, stats, summary


def _fmt(v: float, unit: str = "") -> str:
    if math.isnan(v):
        return "N/A"
    if unit:
        return f"{v:.2f} {unit}"
    return f"{v:.2f}"


def _pick_first(summary: dict[str, str] | None, candidates: list[str]) -> str:
    if not summary:
        return "N/A"
    lowered = {str(k).lower(): (v or "").strip() for k, v in summary.items()}
    for key in candidates:
        value = lowered.get(key.lower(), "")
        if value:
            return value
    return "N/A"


def draw_page_1(
    c,
    data_rows: list[dict[str, str]],
    stats: dict[str, float],
    summary: dict[str, str] | None,
    font_name: str,
) -> None:
    y = draw_title(c, "1. 大会情報とピッチ／ストライド概要", font_name)
    info_table = [
        ["Meet / Competition name", _pick_first(summary, ["meet_name", "competition", "meet", "大会名"])],
        ["Date", _pick_first(summary, ["date", "race_date", "日付"])],
        ["Lane", _pick_first(summary, ["lane", "レーン"])],
        ["Round (Heat/Final)", _pick_first(summary, ["round", "race_round", "予選決勝", "heat_final"])],
        ["Official / Wind-legal flag", _pick_first(summary, ["official", "wind_legal", "公認", "追い風参考"])],
        ["Wind (m/s)", _pick_first(summary, ["wind_mps", "wind", "風", "風速"])],
        ["Time (s)", _pick_first(summary, ["time_s", "time", "タイム"])],
        ["Steps", _pick_first(summary, ["steps", "step_count", "歩数"])],
    ]
    y = draw_table(c, info_table, y, font_name, col_widths=[70 * mm, 180 * mm])

    summary_lines = [
        f"Input rows: {len(data_rows)}",
        f"Pitch: {_fmt(stats['pitch_hz'], 'steps/s')}",
        f"Stride: {_fmt(stats['stride_m'], 'm')}",
        f"Average speed: {_fmt(stats['avg_speed_mps'], 'm/s')}",
    ]
    draw_lines(c, summary_lines, y, font_name, size=12)
    c.showPage()


def draw_page_2(c, rows: list[dict[str, str]], font_name: str) -> None:
    y = draw_title(c, "2. 区間別フォーム指標（一覧）", font_name)
    candidates = ["section", "time_s", "avg_speed_mps", "pitch_hz", "stride_m", "trunk_tilt_deg", "knee_angle_deg"]
    headers = [h for h in candidates if rows and any((r.get(h) or "") for r in rows)]
    if not headers:
        headers = ["section"]
    table_data = [headers]
    for row in rows[:10]:
        table_data.append([row.get(h, "") for h in headers])
    draw_table(c, table_data, y, font_name)
    c.showPage()


def draw_page_3(c, stats: dict[str, float], data_rows: list[dict[str, str]], font_name: str) -> None:
    y = draw_title(c, "3. 区間評価と今日の良かったポイント", font_name)

    pitch = stats.get("pitch_hz", math.nan)
    stride = stats.get("stride_m", math.nan)
    speed = stats.get("avg_speed_mps", math.nan)

    eval_lines = []
    if math.isnan(pitch):
        eval_lines.append("ピッチ評価：データ不足のため判定できません。")
    elif pitch < 3.5:
        eval_lines.append("ピッチ評価：リズムは安定しています。このまま、少しテンポを上げられるとさらに良くなります。")
    elif pitch < 4.5:
        eval_lines.append("ピッチ評価：今の局面として、とても良いリズムで走れています。")
    else:
        eval_lines.append("ピッチ評価：脚の回転がとても力強く、スピードにつながっています。")

    if math.isnan(stride):
        eval_lines.append("ストライド評価：データ不足のため判定できません。")
    elif stride < 1.2:
        eval_lines.append("ストライド評価：一歩をもう少し前に運べる余地があります。地面を後ろに押す意識を高めていきましょう。")
    elif stride < 1.8:
        eval_lines.append("ストライド評価：バランスが良く、コントロールされた走りができています。")
    else:
        eval_lines.append("ストライド評価：一歩の伸びが大きく、とても効果的です。")

    eval_lines += [
        "",
        "ポジティブまとめ：",
        f"- 現在の平均スピードは {_fmt(speed, 'm/s')} です。",
        "- リズムとストライドの形がそろってきており、良い走りの土台ができています。",
        "- 加速中の姿勢をこのまま意識すると、さらに安定して伸びていけます。",
        "",
        "区間別評価：",
    ]
    y = draw_lines(c, eval_lines, y, font_name, size=11)
    legend_lines = [
        "凡例：◎ = 区間内で最速（最大速度）",
        "　　　○ = 最大速度の95%以上",
        "　　　△ = それ以外",
    ]
    y = draw_lines(c, legend_lines, y, font_name, size=10, line_h=5.0 * mm)

    section_speeds: list[tuple[str, float]] = []
    for row in data_rows:
        section = (row.get("section", "") or "").strip()
        row_speed = _to_float(row.get("avg_speed_mps"))
        if section and not math.isnan(row_speed):
            section_speeds.append((section, row_speed))

    max_speed = max((s for _, s in section_speeds), default=math.nan)
    eps = 1e-9
    section_rows = [r for r in data_rows if (r.get("section", "") or "").strip()]

    table_data = [["区間", "平均速度(m/s)", "評価"]]
    cell_bg_map: dict[tuple[int, int], colors.Color] = {}
    for row in section_rows:
        section = (row.get("section", "") or "").strip()
        row_speed = _to_float(row.get("avg_speed_mps"))
        if math.isnan(row_speed) or math.isnan(max_speed):
            speed_text = "N/A"
            rating = "—"
            bg = colors.whitesmoke
        elif abs(row_speed - max_speed) <= eps:
            speed_text = f"{row_speed:.2f}"
            rating = "◎"
            bg = colors.lightgreen
        elif row_speed >= 0.95 * max_speed:
            speed_text = f"{row_speed:.2f}"
            rating = "○"
            bg = colors.whitesmoke
        else:
            speed_text = f"{row_speed:.2f}"
            rating = "△"
            bg = colors.mistyrose
        table_data.append([section, speed_text, rating])
        cell_bg_map[(len(table_data) - 1, 2)] = bg

    y = draw_table(c, table_data, y, font_name, col_widths=[70 * mm, 80 * mm, 50 * mm], cell_bg_map=cell_bg_map)
    c.showPage()


def draw_page_4(c, stats: dict[str, float], font_name: str) -> None:
    y = draw_title(c, "4. 冬季トレーニング提案", font_name)
    lines = [
        "1) 加速ドリル（10〜20m×6本）：1本ごとにしっかり休んで、質を高く行いましょう。",
        "2) 補強トレーニング：ブルガリアンスクワット＋ヒップヒンジを週2〜3回行いましょう。",
        "3) ウィケット走：ピッチとストライドのタイミングをそろえる練習をしましょう。",
        "4) 体幹・姿勢づくり：スプリント前に体幹を固めるドリルを入れましょう。",
        "5) 週1回チェック：metrics CSVのピッチ/ストライドの変化を見比べましょう。",
        "",
        f"現在の参考値：ピッチ={_fmt(stats.get('pitch_hz', math.nan))}、ストライド={_fmt(stats.get('stride_m', math.nan), 'm')}",
    ]
    draw_lines(c, lines, y, font_name, size=11)
    c.showPage()

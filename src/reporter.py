# src/reporter.py
import os, numpy as np, matplotlib.pyplot as plt
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def setup_font(font_path):
    try:
        pdfmetrics.registerFont(TTFont("IPAexGothic", font_path))
    except Exception:
        pass  # フォント未配置でも英字フォントで継続

def _set_font(c, size=12):
    try:
        c.setFont("IPAexGothic", size)
    except:
        c.setFont("Helvetica", size)

def draw_paragraphs(c, x, y, width, lines, size=12, leading=16):
    """
    簡易段落描画：widthに合わせて手動折返し（句読点優先）
    """
    _set_font(c, size)
    avg_w = pdfmetrics.stringWidth("あ", c._fontname, size) or 6
    max_chars = max(1, int(width / avg_w))

    cy = y
    for raw in lines:
        text = (str(raw) or "").replace("\u200b","").strip()
        if not text:
            cy -= leading
            continue
        buff = ""
        for ch in text:
            buff += ch
            if ch in "。、「」、・,; " and len(buff) >= int(max_chars*0.7):
                c.drawString(x, cy, buff.strip())
                cy -= leading
                buff = ""
            elif len(buff) >= max_chars:
                c.drawString(x, cy, buff.strip())
                cy -= leading
                buff = ""
        if buff:
            c.drawString(x, cy, buff.strip())
            cy -= leading
    return cy

def make_pdf(pages, out_pdf, times, speeds_px, speeds_ms, ai_summary_text=None):
    c = canvas.Canvas(out_pdf, pagesize=landscape(A4))
    img_x, img_y, img_w, img_h = 50, 180, 740, 360

    # --- 各フレームページ ---
    for t, phase, img_path, v_px_s, v_ms, v_kmh, tilt, notes in pages:
        _set_font(c, 18)
        c.drawString(50, 560, f"t = {t:.1f}s  |  Phase: {phase}")

        _set_font(c, 11)
        meta = f"Speed proxy: {v_px_s:.1f} px/s  |  {v_ms:.2f} m/s  |  {v_kmh:.1f} km/h"
        if tilt is not None:
            meta += f"  |  Trunk tilt: {tilt:.1f}°"
        c.drawString(50, 540, meta)

        c.drawImage(img_path, img_x, img_y, width=img_w, height=img_h, preserveAspectRatio=True)

        _set_font(c, 12)
        c.drawString(50, 150, "Notes")
        y = 130
        for line in notes:
            c.drawString(50, y, line)
            y -= 16
        c.showPage()

    # --- 速度プロット ---
    try:
        plot_px = os.path.join(os.path.dirname(out_pdf), "speed_plot_px.png")
        x = np.array(times, dtype=float); y = np.array(speeds_px, dtype=float)
        plt.figure(figsize=(8,4.5), dpi=150)
        plt.plot(x, y, label="Speed proxy (px/s)")
        plt.xlabel("Time (s)"); plt.ylabel("px/s"); plt.title("Speed proxy over time")
        plt.tight_layout(); plt.legend(); plt.savefig(plot_px); plt.close()

        _set_font(c, 18)
        c.drawString(50, 560, "Speed proxy summary (px/s)")
        c.drawImage(plot_px, 50, 160, width=760, height=360, preserveAspectRatio=True)
        c.showPage()
    except Exception as e:
        _set_font(c, 12)
        c.drawString(50, 560, f"速度プロット失敗: {e}")
        c.showPage()

    # --- AI分析レポート（常に2ページ構成で描画） ---
    v_ms_arr = list(speeds_ms)
    vmax = max(v_ms_arr) if v_ms_arr else 0.0
    t_at = times[v_ms_arr.index(vmax)] if v_ms_arr else 0.0
    vavg = sum(v_ms_arr)/len(v_ms_arr) if v_ms_arr else 0.0

    # ai_summary_text は list[list[str]]（2ページ）を想定。なければフォールバック
    if isinstance(ai_summary_text, (list, tuple)) and ai_summary_text and all(isinstance(p, (list, tuple)) for p in ai_summary_text):
        pages_ai = ai_summary_text
    else:
        pages_ai = [[
            "（AIサマリーの入力が空でした。フォールバック表示）",
            f"ピーク速度：{vmax:.2f} m/s（t ≈ {t_at:.1f}s）  平均速度：{vavg:.2f} m/s",
            "終盤の失速抑制とピッチ維持に取り組みましょう。"
        ], [
            "【効果的なトレーニング】",
            "・30mダッシュ / ミニハードル走 / 体幹補強（プランク系）",
        ]]

    # 1ページ目
    _set_font(c, 18); c.drawString(50, 560, "AI分析レポート（サマリー）")
    _set_font(c, 12)
    c.drawString(50, 530, f"ピーク速度：{vmax:.2f} m/s（t ≈ {t_at:.1f}s）  平均速度：{vavg:.2f} m/s")
    draw_paragraphs(c, 50, 500, 760, list(pages_ai[0]), size=12, leading=16)
    c.showPage()

    # 2ページ目
    _set_font(c, 18); c.drawString(50, 560, "AI分析レポート（サマリー） ②")
    draw_paragraphs(c, 50, 520, 760, list(pages_ai[1]), size=12, leading=16)
    c.showPage()

    c.save()

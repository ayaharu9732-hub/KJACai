# -*- coding: utf-8 -*-
"""
notes.py
AIサマリーを生成して常に「整形済みの文字列」で返す安全版
"""

from typing import List, Optional, Sequence, Union
import os

def _coerce_text_to_str(text: Union[str, Sequence[str], None]) -> str:
    """リスト/タプルなら段落結合、Noneでも空文字、常にstrへ"""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return "\n\n".join([t if isinstance(t, str) else str(t) for t in text])
    if not isinstance(text, str):
        return str(text)
    return text

def _format_numbers(times: List[float], speeds_ms: List[float]) -> str:
    if not speeds_ms:
        return "ピーク速度：N/A　平均速度：N/A"
    v_peak = max(speeds_ms)
    v_avg  = sum(speeds_ms)/len(speeds_ms)
    # ざっくりピーク時刻（速度最大のインデックス）
    t_peak = 0.0
    try:
        idx = max(range(len(speeds_ms)), key=lambda i: speeds_ms[i])
        t_peak = times[idx] if 0 <= idx < len(times) else 0.0
    except Exception:
        pass
    return f"ピーク速度は {v_peak:.2f} m/s（t ≈ {t_peak:.1f}s）、平均速度は {v_avg:.2f} m/s でした。"

def _build_prompt(times, speeds_ms, tilts, phases) -> str:
    nums = _format_numbers(times, speeds_ms)
    return f"""あなたは中学短距離のコーチです。以下の時系列データ（m/s）から、1人の走者の走りを簡潔に評価してください。
出力は日本語。見出し＋短い段落＆箇条書きで、読みやすい構成にしてください。

【数値要約】
{nums}

【データ概要】
- データ点数: {len(speeds_ms)}
- フェーズ例: {', '.join(list(dict.fromkeys([p for p in (phases or []) if p]))[:5]) or 'N/A'}

【出力フォーマット（厳守）】
1) 概要（2〜3文）
2) 良い点（箇条書き 3項目）
3) 改善点（箇条書き 3項目）
4) 推奨トレーニング（箇条書き 3項目：具体名＋狙い）
※ 主語は「選手」ではなく「走り」や「フォーム」を使い、1人の走者向けに書く。
※ 数字は本文中に自然に織り込む。
"""

def generate_ai_summary(
    times: List[float],
    speeds_ms: List[float],
    tilts: Optional[List[float]] = None,
    phases: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    max_tokens: int = 700,
) -> str:
    """
    AIサマリーを生成して必ず「段落文字列」で返す。
    失敗時は簡易テンプレを返す。
    """
    prompt = _build_prompt(times, speeds_ms, tilts, phases)

    # OPENAI_API_KEY が無ければテンプレ
    if not os.environ.get("OPENAI_API_KEY"):
        fallback = [
            "【概要】\n加速局面の立ち上がりは概ねスムーズ。ピーク以降の速度維持に改善の余地があります。",
            "【良い点】\n・体幹が比較的安定\n・接地が軽い区間が見られる\n・腕振りリズムが一定",
            "【改善点】\n・終盤でピッチ低下傾向\n・接地時の沈み込みがやや大きい\n・最後の2歩で力みが出やすい",
            "【推奨トレーニング】\n・30mダッシュ（加速質の向上）\n・ミニハードル走（ピッチ改善）\n・体幹補強（姿勢維持）",
        ]
        return "\n\n".join(fallback)

    try:
        from openai import OpenAI
        client = OpenAI()

        r = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        text = r.choices[0].message.content
    except Exception as e:
        text = None

    # 必ず文字列化
    text = _coerce_text_to_str(text).strip()

    # 最後の保険：空ならテンプレ
    if not text:
        text = _coerce_text_to_str(_format_numbers(times, speeds_ms))
        text += "\n\n良い点\n・スタートからの加速がスムーズ\n・体幹が安定\n・中盤のリズムが一定"
        text += "\n\n改善点\n・平均速度の底上げ\n・終盤の失速抑制\n・腕振りの前後強調"
        text += "\n\n推奨トレーニング\n・30mダッシュ\n・ミニハードル\n・体幹補強"

    return text

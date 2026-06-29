"""
Text Rendering — Routing & Classification
Determines which pipeline to use for a given prompt.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any
from .config import REVIEW_MODEL, load_prompt
from .utils import _openai_chat_completion, _weighted_text_units


MAX_CANVAS_SIDE = 4096


def _clamp_canvas(width: int, height: int) -> tuple[int, int]:
    width = max(768, int(width))
    height = max(768, int(height))
    longest = max(width, height)
    if longest > MAX_CANVAS_SIDE:
        scale = MAX_CANVAS_SIDE / float(longest)
        width = int(round(width * scale))
        height = int(round(height * scale))
    return width, height


def _round_canvas_side(value: float) -> int:
    return int(round(float(value) / 64.0) * 64)


def _canvas_from_long_side(long_side: int, aspect_ratio: float) -> tuple[int, int]:
    """Build a canvas from longest side and width/height aspect ratio."""
    aspect_ratio = max(0.35, min(float(aspect_ratio or 1.0), 2.4))
    long_side = max(768, min(int(long_side), MAX_CANVAS_SIDE))
    if aspect_ratio >= 1.0:
        width = long_side
        height = long_side / aspect_ratio
    else:
        height = long_side
        width = long_side * aspect_ratio
    return _clamp_canvas(_round_canvas_side(width), _round_canvas_side(height))


def _infer_canvas_aspect(prompt: str, category: str, render_type: str, base_w: int, base_h: int) -> tuple[float, str]:
    """Infer width/height ratio while keeping routing semantics untouched."""
    text = f"{prompt or ''} {category or ''} {render_type or ''}".lower()
    if re.search(r"(横版|横幅|宽屏|风景画幅|landscape|widescreen|wallpaper|banner|scoreboard|subtitle|caption|电影|字幕)", text):
        return 16 / 9, "landscape prompt/category"
    if re.search(r"(竖版|纵向|竖向|portrait|卷轴|scroll|长卷)", text):
        return 0.62, "explicit portrait/scroll prompt/category"
    if re.search(r"(document|paper|page|书页|信纸|宣纸|米纸|article|poem|prose|赋|经文|长文|全文|古文|古风|poster|海报)", text):
        return 4 / 3, "wide text/page prompt/category"
    if base_w and base_h:
        aspect = base_w / base_h
        if render_type in {"layered", "pure_code"} and aspect < 1.0:
            return 4 / 3, "wide default for text rendering"
        return aspect, "category default aspect"
    return 4 / 3, "wide default aspect"


def _dynamic_text_canvas(
    total_units: float,
    prompt: str,
    category: str,
    render_type: str,
    base_w: int,
    base_h: int,
) -> tuple[int, int, str]:
    """Scale canvas continuously from text volume instead of fixed buckets."""
    aspect, aspect_reason = _infer_canvas_aspect(prompt, category, render_type, base_w, base_h)
    if total_units >= 420:
        long_side = MAX_CANVAS_SIDE
    elif total_units >= 200:
        progress = (total_units - 200) / 220.0
        long_side = 3072 + progress * (MAX_CANVAS_SIDE - 3072)
    elif total_units >= 80:
        progress = (total_units - 80) / 120.0
        long_side = 1536 + progress * (2560 - 1536)
    elif total_units >= 20:
        progress = (total_units - 20) / 60.0
        long_side = 1152 + progress * (1536 - 1152)
    else:
        long_side = max(base_w, base_h, 1024)

    width, height = _canvas_from_long_side(math.ceil(long_side), aspect)
    reason = (
        f"dynamic canvas from {total_units:.1f} text units; "
        f"long side {max(width, height)}; aspect {aspect:.2f} ({aspect_reason})"
    )
    return width, height, reason


def get_canvas_size(category: str, render_type: str) -> tuple[int, int]:
    category = (category or "").lower()
    render_type = (render_type or "").lower()

    if render_type == "pure_code":
        mapping = {
            "slide": (1600, 900),
            "webpage": (1440, 1100),
            "document": (1240, 1754),
            "certificate": (1240, 1754),
            "business_card": (1050, 600),
            "card": (1050, 600),
        }
        return mapping.get(category, (1024, 1024))

    if render_type in ("embedded", "embedded_svg"):
        mapping = {
            "sign": (1408, 960),
            "dialogue": (1280, 960),
            "print": (1024, 1408),
            "poster": (768, 1024),
            "cover": (768, 1024),
            "banner": (1408, 768),
        }
        return mapping.get(category, (1024, 1024))

    return (1024, 1024)


def plan_canvas_size(
    prompt: str,
    expected_texts: list[str],
    category: str,
    render_type: str,
) -> dict[str, Any]:
    """Choose a recommended render canvas from text volume and route.

    This is intentionally deterministic. Routing decides composition strategy;
    this planner only picks a practical canvas and records why.
    """
    category_l = (category or "").lower()
    render_l = (render_type or "").lower()
    total_chars = sum(len(str(text)) for text in expected_texts or [])
    total_units = sum(_weighted_text_units(str(text)) for text in expected_texts or [])
    prompt_l = prompt or ""
    long_text_intent = bool(
        re.search(
            r"(完整|全文|全篇|长文|文章|诗文|赋|经文|信件|full|complete|entire|long[- ]form|article|poem|prose)",
            prompt_l,
            flags=re.IGNORECASE,
        )
    )
    base_w, base_h = get_canvas_size(category_l, render_l)
    reason = f"default {render_l or 'unknown'} / {category_l or 'generic'} canvas"

    if total_units >= 20:
        width, height, reason = _dynamic_text_canvas(
            total_units, prompt_l, category_l, render_l, base_w, base_h
        )
    elif long_text_intent:
        width, height, reason = _dynamic_text_canvas(
            80, prompt_l, category_l, render_l, base_w, base_h
        )
        reason = f"{reason}; prompt requests complete/long-form text"
    else:
        width, height = base_w, base_h

    if category_l in {"slide", "banner", "webpage"} and total_units < 80:
        width, height = base_w, base_h
        reason = f"preserve {category_l} aspect ratio for short text"
    elif category_l in {"document", "certificate"} and total_units < 200:
        width, height = max(width, 1240), max(height, 1754)
        reason = f"{reason}; respect document-like portrait aspect"
    elif category_l == "poster" and total_units < 80 and render_l == "pure_code":
        width, height = max(base_w, 1536), max(base_h, 2048)
        reason = "poster should not fallback to 1024 square; use portrait poster canvas"

    width, height = _clamp_canvas(width, height)
    return {
        "width": width,
        "height": height,
        "total_chars": total_chars,
        "total_text_units": round(total_units, 1),
        "long_text_intent": long_text_intent,
        "source": "heuristic",
        "reason": reason,
    }


def _build_retry_prompt(base_prompt: str, feedback: str, extra_context: str = "") -> str:
    prompt = base_prompt.strip()
    if extra_context:
        prompt += f"\n\nAdditional context:\n{extra_context.strip()}"
    if feedback:
        prompt += (
            "\n\nThe previous draft failed review for these reasons:\n"
            f"{feedback.strip()}\n\n"
            "Regenerate the entire layout from scratch and fix every issue above. "
            "Do not explain. Output code only."
        )
    return prompt


def _build_embedded_category_guidance(category: str) -> str:
    category = (category or "").lower()
    guidance = {
        "sign": (
            "Create one or more wide, front-facing sign boards with clear inner padding. "
            "Support posts or frames must stay outside the text area and must not occlude any characters."
        ),
        "label": (
            "Create a large front-facing label panel on the package or bottle. "
            "Keep all text inside the label panel and do not wrap text around curved edges. "
            "Labels must read as attached stickers or printed panels on visible containers, not floating cards."
        ),
        "dialogue": (
            "Create a clean subtitle box, speech bubble, or screen panel with enough padding. "
            "The text area must remain separate from character silhouettes."
        ),
    }
    return guidance.get(
        category,
        "Create one obvious text-bearing surface or region with stable geometry, enough padding, and no occlusion over the text."
    )


def classify_text_render_type(prompt: str) -> str:
    system = load_prompt("text_render_classify")
    response = _openai_chat_completion(
        timeout=30.0,
        model=REVIEW_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=64,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"\s*```", "", raw)
    valid_types = {"pure_code", "layered", "embedded_svg", "direct_gen"}
    try:
        result = json.loads(raw)
        rt = result.get("text_render_type", "embedded_svg")
        if rt in ("embedded", "label"):
            rt = "embedded_svg"
        return rt if rt in valid_types else "embedded_svg"
    except json.JSONDecodeError:
        if "pure_code" in raw:
            return "pure_code"
        elif "layered" in raw:
            return "layered"
        elif "direct_gen" in raw:
            return "direct_gen"
        return "embedded_svg"

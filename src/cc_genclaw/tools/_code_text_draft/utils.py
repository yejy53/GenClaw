"""Text Rendering — Utility Functions

Text metrics, parsing, font helpers, and common text-part operations.
"""
from __future__ import annotations

import re
import json
import time
from types import SimpleNamespace

from cc_genclaw.llm import LLMClient, LLMConfig

from .config import REVIEW_MODEL


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    tokens = [
        "429",
        "rate limit",
        "upstream",
        "temporarily",
        "timeout",
        "timed out",
        "saturated",
        "饱和",
        "overloaded",
        "model_not_found",
    ]
    return any(token in msg for token in tokens)


def _openai_chat_completion(timeout: float, retries: int = 3, initial_delay: float = 5.0, **kwargs):
    delay = initial_delay
    last_exc = None
    for attempt in range(retries):
        try:
            cfg = LLMConfig.for_code_draft()
            cfg.timeout = timeout
            response = LLMClient(cfg).create(
                messages=kwargs.get("messages") or [],
                tools=kwargs.get("tools"),
                model=kwargs.get("model"),
            )
            return _to_attr_tree(response)
        except Exception as exc:
            last_exc = exc
            if attempt == retries - 1 or not _is_transient_error(exc):
                raise
            print(f"  OpenAI request transient failure: {exc}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2
    raise last_exc


def _to_attr_tree(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_attr_tree(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_tree(item) for item in value]
    return value


def _parse_numeric_attr(value, default=0.0):
    if value is None:
        return float(default)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else float(default)


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return text[start:]


def _extract_quoted_strings(text: str) -> list[str]:
    patterns = [
        r"“([^”]{1,200})”",
        r'"([^"\n]{1,200})"',
        r"「([^」]{1,200})」",
        r"『([^』]{1,200})』",
    ]
    seen = set()
    items = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            item = match.strip()
            if item and item not in seen:
                seen.add(item)
                items.append(item)
    return items


def _fallback_split_from_prompt(prompt: str) -> dict:
    quoted = _extract_quoted_strings(prompt)
    first_placement = "primary text-bearing surface near the visual focus"
    second_placement = "secondary supporting text-bearing surface near the primary one"
    other_placement = "additional supporting text-bearing surface"

    text_parts = []
    for idx, item in enumerate(quoted):
        if idx == 0:
            placement = first_placement
        elif idx == 1:
            placement = second_placement
        else:
            placement = other_placement
        text_parts.append({"text": item, "placement": placement})

    bg_desc = re.sub(r"[“\"「『][^”\"」』]{1,200}[”\"」』]", " ", prompt)
    bg_desc = re.sub(r"\s+", " ", bg_desc).strip()
    if not bg_desc:
        bg_desc = prompt.strip()
    bg_desc += " Leave clear, front-facing space for the quoted text content."
    return {
        "text_parts": _dedupe_text_parts(text_parts),
        "background": bg_desc,
    }


def _weighted_text_units(text: str) -> float:
    units = 0.0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            units += 1.1
        elif ch.isupper():
            units += 0.72
        elif ch.islower():
            units += 0.58
        elif ch.isdigit():
            units += 0.56
        elif ch.isspace():
            units += 0.32
        else:
            units += 0.45
    return max(units, 1.0)


def _dedupe_text_parts(text_parts: list) -> list:
    deduped = []
    seen = set()
    for item in text_parts:
        text = (item.get("text") or "").strip()
        placement = (item.get("placement") or "").strip()
        if not text or text in seen:
            continue
        deduped.append({"text": text, "placement": placement})
        seen.add(text)
    return deduped


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


def _font_family_for_text(text: str) -> str:
    if _contains_cjk(text):
        return "'PingFang SC', 'Noto Sans SC', SimHei, STHeiti, 'Microsoft YaHei', sans-serif"
    return "'Arial Black', Arial, Helvetica, sans-serif"


def _format_svg_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _estimate_panel_width_for_text(
    text: str,
    base_width: float,
    max_width: float,
    panel_height: float,
    height_ratio: float,
    max_font: float,
) -> float:
    target_font = min(panel_height * height_ratio, max_font)
    required_width = (_weighted_text_units(text) * target_font) / 0.82 + panel_height * 0.30
    return round(_clamp(max(base_width, required_width), base_width, max_width), 1)


def _estimate_panel_font_size(
    text: str,
    panel_width: float,
    panel_height: float,
    height_ratio: float,
    min_font: float,
    max_font: float,
) -> float:
    width_fit = (panel_width * 0.80) / max(_weighted_text_units(text), 1.0)
    height_fit = panel_height * height_ratio
    return round(_clamp(min(width_fit, height_fit, max_font), min_font, max_font), 1)


def _default_expected_placements(category: str, count: int) -> list[str]:
    _ = category
    placements = [
        "primary text-bearing surface near the visual center",
        "secondary supporting text-bearing surface near the primary one",
        "left-side supporting text-bearing surface",
        "right-side supporting text-bearing surface",
        "lower supporting text-bearing surface",
    ]
    if count <= len(placements):
        return placements[:count]
    extra = [f"additional supporting panel {i + 1}" for i in range(count - len(placements))]
    return placements + extra


def _normalize_for_match(s: str) -> str:
    return s.strip().rstrip(".,;:!?。，；：！？")


def _compact_for_containment_match(s: str) -> str:
    text = str(s or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[《》“”\"'‘’「」『』\[\]（）(){}<>.,;:!?。，；：！？、·\-—_]", "", text)
    return text


def _ensure_text_parts_cover_expected(text_parts: list, expected_texts: list[str], category: str) -> list:
    expected_clean = [str(text).strip() for text in expected_texts if str(text).strip()]
    if not expected_clean:
        return _dedupe_text_parts(text_parts)

    # expected_long_texts is the authoritative text source. The split LLM is
    # allowed to help with background/layout hints, but it must not introduce
    # visible text that was not explicitly passed as expected text.
    split_parts = _dedupe_text_parts(text_parts)
    expected_compact = [
        compact for compact in (_compact_for_containment_match(text) for text in expected_clean)
        if compact
    ]
    exact_placements: dict[str, str] = {}
    split_placements: list[str] = []
    for item in split_parts:
        item_text = (item.get("text") or "").strip()
        item_compact = _compact_for_containment_match(item_text)
        placement = (item.get("placement") or "").strip()
        if placement:
            split_placements.append(placement)
        if item_compact in expected_compact and placement:
            exact_placements[item_compact] = placement

    fallback_placements = _default_expected_placements(category, len(expected_texts))
    use_split_placements_by_order = len(split_parts) == len(expected_clean)
    merged: list[dict] = []
    for idx, clean in enumerate(expected_clean):
        compact = _compact_for_containment_match(clean)
        placement = exact_placements.get(compact) or ""
        if not placement and use_split_placements_by_order and idx < len(split_placements):
            placement = split_placements[idx]
        if not placement and fallback_placements:
            placement = fallback_placements[min(idx, len(fallback_placements) - 1)]
        merged.append({"text": clean, "placement": placement})
    return merged


def _pick_first_matching_text_part(text_parts: list[dict], keywords: list[str]) -> dict | None:
    for idx, item in enumerate(text_parts):
        placement = (item.get("placement") or "").lower()
        if any(keyword in placement for keyword in keywords):
            return text_parts.pop(idx)
    return None


def _order_text_parts_for_layout(category: str, text_parts: list[dict]) -> list[dict]:
    items = [{"text": item["text"], "placement": item.get("placement", "")} for item in text_parts if item.get("text")]
    category = (category or "").lower()
    ordered = []

    if category == "sign":
        title = _pick_first_matching_text_part(items, ["banner", "main", "headline", "central", "large"])
        subtitle = _pick_first_matching_text_part(items, ["below", "subtitle", "under", "smaller", "directly below"])
        if title:
            ordered.append(title)
        if subtitle:
            ordered.append(subtitle)
        ordered.extend(items)
        return ordered

    if category == "label":
        main = _pick_first_matching_text_part(items, ["prominent", "central", "main", "focal"])
        if main:
            ordered.append(main)
        ordered.extend(items)
        return ordered

    return items


def _make_centered_text_panel(
    panel_id: str,
    text: str,
    placement: str,
    canvas_width: int,
    center_x: float,
    top_y: float,
    base_width: float,
    max_width: float,
    panel_height: float,
    corner_radius: float,
    fill: str,
    stroke: str,
    text_color: str,
    height_ratio: float,
    min_font: float,
    max_font: float,
    font_weight: str = "bold",
) -> dict:
    panel_width = _estimate_panel_width_for_text(
        text,
        base_width=base_width,
        max_width=max_width,
        panel_height=panel_height,
        height_ratio=height_ratio,
        max_font=max_font,
    )
    x = _clamp(center_x - panel_width / 2.0, canvas_width * 0.04, canvas_width - panel_width - canvas_width * 0.04)
    font_size = _estimate_panel_font_size(
        text,
        panel_width=panel_width,
        panel_height=panel_height,
        height_ratio=height_ratio,
        min_font=min_font,
        max_font=max_font,
    )
    return {
        "id": panel_id,
        "shape": "rounded_rect",
        "text": text,
        "placement": placement,
        "x": round(x, 1),
        "y": round(top_y, 1),
        "width": round(panel_width, 1),
        "height": round(panel_height, 1),
        "rx": round(corner_radius, 1),
        "fill": fill,
        "stroke": stroke,
        "stroke_width": 3 if panel_height >= 110 else 2,
        "text_color": text_color,
        "font_family": _font_family_for_text(text),
        "font_weight": font_weight,
        "font_size": font_size,
        "text_anchor": "middle",
        "text_x": round(x + panel_width / 2.0, 1),
        "text_y": round(top_y + panel_height / 2.0, 1),
        "max_text_width": round(panel_width * 0.80, 1),
        "keep_single_line": True,
    }


def split_prompt_text_and_background(prompt: str) -> dict:
    """Split user prompt into structured text parts and a background-only description."""
    response = _openai_chat_completion(
        timeout=30.0,
        model=REVIEW_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a prompt analyst. The user's prompt describes an image containing text.\n"
                "Split it into two parts:\n"
                "1. text_parts: A list of objects, each with 'text' (exact string) and 'placement' (where it goes).\n"
                "2. background: A rich description of ONLY the visual scene/background, with NO text content.\n\n"
                "Output JSON like:\n"
                '{"text_parts": [{"text": "HELLO", "placement": "centered at top"}, ...],'
                ' "background": "a dramatic sunset over mountains with warm orange tones"}\n\n'
                "Rules:\n"
                "- Extract ALL text strings CHARACTER-PERFECTLY as the user specified.\n"
                "- Do NOT duplicate the same text string multiple times in text_parts.\n"
                "- The background description must be vivid and detailed enough to generate a standalone image.\n"
                "- Carry over style/mood/color information from the original prompt into the background.\n"
                "- Keep the background visual-only: do NOT include the quoted text, invented text, subtitles, "
                "caption bars, speech bubbles, danmaku boxes, label panels, blank cards, UI text containers, "
                "or placeholder areas that look meant for text.\n"
                "- You may describe natural empty composition or clean negative space, but do not ask for a visible text box.\n"
                "Output ONLY valid JSON, no markdown fences, no explanation."
            )},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        try:
            result = json.loads(_extract_first_json_object(raw))
        except json.JSONDecodeError:
            result = _fallback_split_from_prompt(prompt)
    result["text_parts"] = _dedupe_text_parts(result.get("text_parts", []))
    return result

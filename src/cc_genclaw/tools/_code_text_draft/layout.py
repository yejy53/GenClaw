"""
Text Rendering — Layout Planning
Generates structured JSON layout plans for embedded text placement,
plus helpers that turn a plan into an HTML blueprint draft.
"""
from __future__ import annotations

import json
import re

from .config import REVIEW_MODEL, load_prompt
from .utils import (
    _weighted_text_units,
    _clamp,
    _openai_chat_completion,
    _extract_first_json_object,
)


# ============================================================
# Embedded structured plan (JSON-based)
# ============================================================

def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_plan_region(region: dict | None, default_region: dict) -> dict:
    region = region or {}
    x = _safe_float(region.get("x", region.get("left", default_region["x"])), default_region["x"])
    y = _safe_float(region.get("y", region.get("top", default_region["y"])), default_region["y"])
    w = _safe_float(region.get("w", region.get("width", default_region["w"])), default_region["w"])
    h = _safe_float(region.get("h", region.get("height", default_region["h"])), default_region["h"])

    w = _clamp(w, 0.10, 0.92)
    h = _clamp(h, 0.07, 0.86)
    x = _clamp(x, 0.02, 0.96 - w)
    y = _clamp(y, 0.02, 0.96 - h)
    return {
        "x": round(x, 3),
        "y": round(y, 3),
        "w": round(w, 3),
        "h": round(h, 3),
    }


def _default_plan_regions(count: int) -> list[dict]:
    presets = [
        {"x": 0.14, "y": 0.12, "w": 0.72, "h": 0.18},
        {"x": 0.24, "y": 0.34, "w": 0.52, "h": 0.12},
        {"x": 0.08, "y": 0.58, "w": 0.24, "h": 0.14},
        {"x": 0.38, "y": 0.56, "w": 0.24, "h": 0.14},
        {"x": 0.68, "y": 0.58, "w": 0.24, "h": 0.14},
    ]
    if count <= len(presets):
        return presets[:count]

    extra = []
    remaining = count - len(presets)
    for idx in range(remaining):
        row = idx // 3
        col = idx % 3
        extra.append({
            "x": round(0.08 + col * 0.30, 3),
            "y": round(0.74 + row * 0.14, 3),
            "w": 0.24,
            "h": 0.12,
        })
    return presets + extra


def _importance_for_index(idx: int) -> str:
    if idx == 0:
        return "primary"
    if idx == 1:
        return "secondary"
    return "supporting"


def _infer_wrap_mode(text: str, notes: str = "", surface_type: str = "") -> str:
    lower = f"{notes} {surface_type}".lower()
    text_units = _weighted_text_units(text)
    single_line_tokens = [
        "banner", "headline", "title", "header", "top", "strip",
        "hanging", "overhead", "signboard", "plaque",
    ]
    if any(token in lower for token in single_line_tokens) and text_units <= 42:
        return "single_line"
    if text_units >= 34:
        return "multiline"
    if text_units >= 20:
        return "auto"
    return "single_line"


def _infer_max_lines(text: str, wrap_mode: str) -> int:
    if wrap_mode == "single_line":
        return 1
    text_units = _weighted_text_units(text)
    if text_units >= 44:
        return 3
    return 2


def _normalize_text_surface(
    item: dict,
    placement_hint: str,
    default_region: dict,
    idx: int,
) -> dict:
    text = (item.get("text") or "").strip()
    notes = (item.get("notes") or item.get("placement") or placement_hint or "").strip()
    wrap_mode = (item.get("wrap_mode") or "").strip().lower()
    if wrap_mode not in {"single_line", "multiline", "auto"}:
        wrap_mode = _infer_wrap_mode(text, notes=notes, surface_type=item.get("surface_type", ""))
    max_lines = int(_safe_float(item.get("max_lines"), _infer_max_lines(text, wrap_mode)))
    max_lines = max(1, min(max_lines, 3))
    alignment = (item.get("alignment") or item.get("align") or "center").strip().lower()
    if alignment not in {"center", "left", "right"}:
        alignment = "center"
    attachment = (item.get("attachment") or "integrated").strip().lower()
    if attachment not in {"attached", "free-standing", "hanging", "integrated"}:
        attachment = "integrated"
    importance = (item.get("importance") or _importance_for_index(idx)).strip().lower()
    if importance not in {"primary", "secondary", "supporting"}:
        importance = _importance_for_index(idx)

    carrier_id = item.get("carrier_id")
    if carrier_id is not None:
        carrier_id = str(carrier_id).strip() or None

    return {
        "id": str(item.get("id") or f"text_surface_{idx + 1}"),
        "text": text,
        "importance": importance,
        "surface_type": str(item.get("surface_type") or "text_surface"),
        "attachment": attachment,
        "carrier_id": carrier_id,
        "shape_hint": str(item.get("shape_hint") or "rounded_panel"),
        "region": _normalize_plan_region(item.get("region"), default_region),
        "alignment": alignment,
        "wrap_mode": wrap_mode,
        "max_lines": max_lines,
        "notes": notes,
    }


def _normalize_supporting_object(item: dict, idx: int) -> dict:
    return {
        "id": str(item.get("id") or f"object_{idx + 1}"),
        "object_type": str(item.get("object_type") or "simple carrier silhouette"),
        "region": _normalize_plan_region(item.get("region"), {"x": 0.18, "y": 0.24, "w": 0.44, "h": 0.42}),
        "notes": str(item.get("notes") or "").strip(),
    }


def _carrier_region_from_surface(surface_region: dict) -> dict:
    return _normalize_plan_region(
        {
            "x": surface_region["x"] - surface_region["w"] * 0.10,
            "y": surface_region["y"] - surface_region["h"] * 0.55,
            "w": surface_region["w"] * 1.20,
            "h": surface_region["h"] * 2.40,
        },
        surface_region,
    )


def _fallback_embedded_layout_plan(text_parts: list[dict], expected_texts: list[str], canvas_width: int, canvas_height: int) -> dict:
    placement_map = {(item.get("text") or "").strip(): (item.get("placement") or "").strip() for item in text_parts}
    default_regions = _default_plan_regions(len(expected_texts))
    text_surfaces = []
    supporting_objects = []

    for idx, text in enumerate(expected_texts):
        placement_hint = placement_map.get((text or "").strip(), "")
        surface = _normalize_text_surface(
            {
                "text": text,
                "surface_type": "text_surface",
                "attachment": "integrated",
                "region": default_regions[min(idx, len(default_regions) - 1)],
                "notes": placement_hint,
            },
            placement_hint=placement_hint,
            default_region=default_regions[min(idx, len(default_regions) - 1)],
            idx=idx,
        )
        text_surfaces.append(surface)

    return {
        "planner_source": "fallback",
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "layout_summary": "",
        "supporting_objects": supporting_objects,
        "text_surfaces": text_surfaces,
    }


def _normalize_embedded_layout_plan(
    raw_plan: dict,
    text_parts: list[dict],
    expected_texts: list[str],
    canvas_width: int,
    canvas_height: int,
) -> dict:
    placement_map = {(item.get("text") or "").strip(): (item.get("placement") or "").strip() for item in text_parts}
    raw_surfaces = raw_plan.get("text_surfaces") or raw_plan.get("surfaces") or []
    raw_by_text = {}
    for item in raw_surfaces:
        text = (item.get("text") or "").strip()
        if text and text not in raw_by_text:
            raw_by_text[text] = item

    default_regions = _default_plan_regions(len(expected_texts))
    text_surfaces = []
    for idx, text in enumerate(expected_texts):
        clean = (text or "").strip()
        raw_item = raw_by_text.get(clean, {})
        placement_hint = placement_map.get(clean, "")
        if not raw_item:
            raw_item = {"text": clean, "notes": placement_hint}
        if "region" not in raw_item:
            raw_item = dict(raw_item)
            raw_item["region"] = default_regions[min(idx, len(default_regions) - 1)]
        surface = _normalize_text_surface(
            raw_item,
            placement_hint=placement_hint,
            default_region=default_regions[min(idx, len(default_regions) - 1)],
            idx=idx,
        )
        text_surfaces.append(surface)

    supporting_objects = []
    raw_objects = raw_plan.get("supporting_objects") or raw_plan.get("objects") or []
    known_object_ids = set()
    for idx, item in enumerate(raw_objects[:6]):
        obj = _normalize_supporting_object(item, idx)
        supporting_objects.append(obj)
        known_object_ids.add(obj["id"])

    for surface in text_surfaces:
        if surface["attachment"] != "attached":
            continue
        carrier_id = surface.get("carrier_id")
        if carrier_id and carrier_id in known_object_ids:
            continue
        if not carrier_id:
            carrier_id = f"{surface['id']}_carrier"
            surface["carrier_id"] = carrier_id
        supporting_objects.append({
            "id": carrier_id,
            "object_type": "simple carrier silhouette",
            "region": _carrier_region_from_surface(surface["region"]),
            "notes": "minimal carrier silhouette behind the attached text surface",
        })
        known_object_ids.add(carrier_id)

    layout_plan = {
        "planner_source": raw_plan.get("planner_source", "llm"),
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "layout_summary": str(raw_plan.get("layout_summary") or "").strip(),
        "supporting_objects": supporting_objects,
        "text_surfaces": text_surfaces,
    }
    layout_plan["layout_summary"] = _summarize_text_layout_plan(layout_plan)
    return layout_plan


def _region_anchor_description(region: dict) -> str:
    x_mid = region["x"] + region["w"] / 2.0
    y_mid = region["y"] + region["h"] / 2.0
    if x_mid < 0.35:
        horizontal = "left"
    elif x_mid > 0.65:
        horizontal = "right"
    else:
        horizontal = "center"
    if y_mid < 0.30:
        vertical = "upper"
    elif y_mid > 0.68:
        vertical = "lower"
    else:
        vertical = "middle"
    return f"{vertical}-{horizontal}"


def _summarize_text_layout_plan(layout_plan: dict) -> str:
    if layout_plan.get("layout_summary"):
        return layout_plan["layout_summary"]
    parts = []
    for obj in layout_plan.get("supporting_objects", []):
        parts.append(
            f'{obj["id"]}: {obj.get("object_type", "object")} near {_region_anchor_description(obj["region"])}.'
        )
    for surface in layout_plan.get("text_surfaces", []):
        extra = f' attached to {surface["carrier_id"]}' if surface.get("carrier_id") else ""
        parts.append(
            f'{surface["id"]}: "{surface["text"]}" on a {surface.get("surface_type", "surface")} near '
            f'{_region_anchor_description(surface["region"])}, wrap={surface.get("wrap_mode", "auto")}{extra}.'
        )
    return " ".join(parts) or "N/A"


def plan_embedded_layout(
    prompt: str,
    expected_texts: list[str],
    text_parts: list[dict],
    canvas_width: int,
    canvas_height: int,
    review_feedback: str = "",
) -> dict:
    system = load_prompt("embedded_layout_plan")
    system = system.replace("{canvas_width}", str(canvas_width))
    system = system.replace("{canvas_height}", str(canvas_height))

    planner_request = [
        f"Canvas size: {canvas_width} x {canvas_height}",
        f"Original prompt:\n{prompt}",
        "Expected texts (must all appear exactly once):",
        json.dumps(expected_texts, ensure_ascii=False, indent=2),
        "Extracted text parts and loose placement hints:",
        json.dumps(text_parts, ensure_ascii=False, indent=2),
    ]
    if review_feedback:
        planner_request.extend([
            "Previous draft feedback that should be fixed in the new plan:",
            review_feedback,
        ])

    try:
        response = _openai_chat_completion(
            timeout=120.0,
            model=REVIEW_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "\n".join(planner_request)},
            ],
            max_tokens=4096,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = json.loads(_extract_first_json_object(raw))
        return _normalize_embedded_layout_plan(parsed, text_parts, expected_texts, canvas_width, canvas_height)
    except Exception:
        return _fallback_embedded_layout_plan(text_parts, expected_texts, canvas_width, canvas_height)


def _build_html_from_plan(layout_plan: dict, expected_texts: list[str], review_feedback: str = "") -> str:
    layout_json = json.dumps(layout_plan, indent=2, ensure_ascii=False)
    parts = [
        f'Canvas size: {layout_plan.get("canvas_width")} x {layout_plan.get("canvas_height")}',
        "Expected texts (must all appear exactly once):",
        json.dumps(expected_texts, indent=2, ensure_ascii=False),
        "Structured layout plan (JSON):",
        layout_json,
        "Interpretation rules:",
        "- Treat each region as a soft layout boundary, not a rigid forced rectangle.",
        "- Keep the scene minimal and focused on the text-bearing surfaces.",
        "- Use only the minimum supporting objects needed to make the text placement plausible.",
        "- Fill the canvas; avoid large empty areas, especially at the bottom.",
        "- Use large, legible fonts; let CSS wrap long text instead of forcing line breaks.",
    ]
    if review_feedback:
        parts.extend([
            "Previous draft feedback that must be fixed:",
            review_feedback,
        ])
    return "\n".join(parts)


def _validate_generated_html(html_code: str, expected_texts: list[str]) -> list[str]:
    issues = []
    low = html_code.lower()
    if "<html" not in low and "<!doctype" not in low and "<body" not in low:
        issues.append("Output is missing an HTML document root.")
    raw_norm = re.sub(r"\s+", " ", html_code)
    for text in expected_texts:
        norm = re.sub(r"\s+", " ", text or "").strip()
        if norm and norm not in raw_norm:
            issues.append(f'Missing expected text "{text}".')
    return issues


def _render_debug_html_from_layout_plan(layout_plan: dict) -> str:
    """Fallback HTML blueprint built directly from the layout plan regions."""
    cw = layout_plan.get("canvas_width", 1024)
    ch = layout_plan.get("canvas_height", 1024)
    blocks = []
    for obj in layout_plan.get("supporting_objects", []):
        r = obj["region"]
        blocks.append(
            f'<div style="position:absolute;left:{r["x"]*100:.2f}%;top:{r["y"]*100:.2f}%;'
            f'width:{r["w"]*100:.2f}%;height:{r["h"]*100:.2f}%;background:#E7E1D4;'
            f'border:2px solid #9C8C76;border-radius:12px;"></div>'
        )
    for surface in layout_plan.get("text_surfaces", []):
        r = surface["region"]
        text = surface.get("text", "")
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        blocks.append(
            f'<div style="position:absolute;left:{r["x"]*100:.2f}%;top:{r["y"]*100:.2f}%;'
            f'width:{r["w"]*100:.2f}%;height:{r["h"]*100:.2f}%;background:#F7F3EA;'
            f'border:3px solid #8C765C;border-radius:14px;display:flex;align-items:center;'
            f'justify-content:center;text-align:center;padding:1%;box-sizing:border-box;'
            f"font-family:'PingFang SC','Noto Sans SC','Microsoft YaHei',sans-serif;"
            f'font-weight:bold;color:#2E2418;font-size:42px;overflow-wrap:break-word;">{safe}</div>'
        )
    body = "\n".join(blocks)
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        '*{margin:0;padding:0;box-sizing:border-box}'
        f'html,body{{width:{cw}px;height:{ch}px;overflow:hidden;background:#FBF8F1;position:relative}}'
        f'</style></head><body>{body}</body></html>'
    )



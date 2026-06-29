"""HTML layout repair for code_text_draft.

This module edits an existing HTML/CSS artifact after deterministic DOM geometry
checks fail. It is intentionally narrower than the initial code generator: keep
the content and background, repair layout constraints.
"""

from __future__ import annotations

import json
from typing import Any

from .codegen import _parse_generated_code
from .config import SVG_MODEL, load_prompt
from .utils import _openai_chat_completion


def _compact_geometry(geometry: dict | None) -> dict:
    """Keep the repair prompt focused on actionable geometry data."""
    if not isinstance(geometry, dict):
        return {}
    layout_usage = geometry.get("layout_usage") if isinstance(geometry.get("layout_usage"), dict) else {}
    return {
        "viewport": geometry.get("viewport"),
        "text_node_count": geometry.get("text_node_count"),
        "layout_usage": {
            "text_area_bbox": layout_usage.get("text_area_bbox"),
            "body_text_bbox": layout_usage.get("body_text_bbox"),
            "coverage": layout_usage.get("coverage"),
            "font": layout_usage.get("font"),
            "body_font": layout_usage.get("body_font"),
            "total_text_units": layout_usage.get("total_text_units"),
            "issues": layout_usage.get("issues") or [],
        },
        "issues": geometry.get("issues") or [],
    }


def _truncate_text(value: str, limit: int = 120_000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<!-- truncated for repair prompt -->"


def repair_html_layout(
    *,
    html_code: str,
    canvas_width: int,
    canvas_height: int,
    expected_texts: list[str],
    geometry: dict | None,
    review_feedback: str = "",
    layout_recommendation: str = "",
    background_placeholder: str = "__BG_IMAGE__",
) -> dict[str, Any]:
    """Return a repaired complete HTML document, or a failed repair result."""
    if not str(html_code or "").strip():
        return {"status": "failed", "error": "html_code is empty"}

    system = load_prompt("html_layout_repair")
    system = system.replace("{canvas_width}", str(canvas_width))
    system = system.replace("{canvas_height}", str(canvas_height))

    user_content = (
        "Repair the existing HTML below. Return only the full repaired HTML.\n\n"
        f"Expected text strings that must be preserved exactly:\n"
        f"{json.dumps(expected_texts or [], ensure_ascii=False, indent=2)}\n\n"
        f"Review feedback:\n{review_feedback or '(none)'}\n\n"
        f"Target layout recommendation that the repaired HTML should satisfy:\n"
        f"{layout_recommendation or '(none)'}\n\n"
        "DOM geometry report:\n"
        f"{json.dumps(_compact_geometry(geometry), ensure_ascii=False, indent=2)}\n\n"
        "Existing HTML:\n"
        "```html\n"
        f"{_truncate_text(html_code)}\n"
        "```"
    )

    try:
        response = _openai_chat_completion(
            timeout=180.0,
            model=SVG_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=16384,
            temperature=0.15,
        )
        raw = response.choices[0].message.content.strip()
        parsed = _parse_generated_code(raw)
        repaired = parsed.get("code", "").strip()
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if "<html" not in repaired.lower() and "<!doctype" not in repaired.lower():
        return {"status": "failed", "error": "repair output was not complete HTML"}
    if background_placeholder in html_code and background_placeholder not in repaired:
        return {"status": "failed", "error": f"repair output removed {background_placeholder}"}

    return {
        "status": "success",
        "html_code": repaired,
        "repair_summary": "Repaired existing HTML layout from geometry/review feedback.",
        "code_type": parsed.get("code_type", "html"),
    }

"""Input/output normalization for format_prompt."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .utils import truncate_text, unique_preserve_order

_FACTS_LIMIT = 2000
_TEXT_LIMIT = 3000


@dataclass
class FormatPromptBrief:
    original_user_prompt: str
    base_prompt: str = ""
    image_intent: str = ""
    painter_thoughts: str = ""
    planning_needs: List[Any] = field(default_factory=list)
    notes_for_planner: List[str] = field(default_factory=list)
    search_facts_summary: str = ""
    reasoning_output: List[str] = field(default_factory=list)
    svg_description: str = ""
    draft_png_path: str = ""
    text_template_path: str = ""
    expected_texts_used: List[str] = field(default_factory=list)
    category: str = ""
    render_type: str = ""
    user_image_path: str = ""
    source_image_path: str = ""
    reference_image_paths: List[str] = field(default_factory=list)
    target_tool: str = ""

    def to_model_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data["search_facts_summary"] = truncate_text(data["search_facts_summary"], _FACTS_LIMIT)
        data["svg_description"] = truncate_text(data["svg_description"], _TEXT_LIMIT)
        data["painter_thoughts"] = truncate_text(data["painter_thoughts"], _TEXT_LIMIT)
        data["reasoning_output"] = [
            truncate_text(item, _TEXT_LIMIT) for item in self.reasoning_output
        ]
        data["expected_texts_used"] = [
            truncate_text(item, _TEXT_LIMIT) for item in self.expected_texts_used
        ]
        return data


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str_list(value: Any) -> List[str]:
    return [str(v) for v in _as_list(value) if str(v or "").strip()]


def normalize_brief(args: Dict[str, Any]) -> FormatPromptBrief:
    """Map new and legacy tool args into one explicit brief."""
    original = str(
        args.get("original_user_prompt")
        or args.get("base_prompt")
        or args.get("prompt")
        or ""
    ).strip()
    base_prompt = str(args.get("base_prompt") or original).strip()

    facts = str(args.get("search_facts_summary") or args.get("verified_facts") or "").strip()
    if not facts:
        # Legacy field. Treat it as a summary if the caller did not provide
        # the new explicit field; cap before sending to the composer.
        facts = str(args.get("facts") or "").strip()

    reasoning = _as_str_list(args.get("reasoning_output") or args.get("reasoning"))
    notes = _as_str_list(args.get("notes_for_planner"))
    refs = unique_preserve_order(_as_str_list(args.get("reference_image_paths")))

    source = str(args.get("source_image_path") or "").strip()
    text_template = str(args.get("text_template_path") or "").strip()
    draft = str(args.get("draft_png_path") or "").strip()
    user_image = str(args.get("user_image_path") or "").strip()
    if not source:
        source = text_template or draft or user_image

    return FormatPromptBrief(
        original_user_prompt=original,
        base_prompt=base_prompt,
        image_intent=str(args.get("image_intent") or "").strip(),
        painter_thoughts=str(args.get("painter_thoughts") or "").strip(),
        planning_needs=_as_list(args.get("planning_needs")),
        notes_for_planner=notes,
        search_facts_summary=facts,
        reasoning_output=reasoning,
        svg_description=str(args.get("svg_description") or "").strip(),
        draft_png_path=draft,
        text_template_path=text_template,
        expected_texts_used=_as_str_list(args.get("expected_texts_used")),
        category=str(args.get("category") or "").strip(),
        render_type=str(args.get("render_type") or "").strip(),
        user_image_path=user_image,
        source_image_path=source,
        reference_image_paths=refs,
        target_tool=str(args.get("target_tool") or "").strip(),
    )


def brief_to_user_text(brief: FormatPromptBrief) -> str:
    payload = brief.to_model_payload()
    return (
        "Compose the final generation prompt from this trajectory brief.\n"
        "Use only these fields and the attached images.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_fallback_prompt(brief: FormatPromptBrief) -> str:
    parts: List[str] = []
    anchor = brief.original_user_prompt or brief.base_prompt
    if anchor:
        parts.append(anchor)
    if brief.image_intent and brief.image_intent not in parts:
        parts.append(f"Visual intent: {brief.image_intent}")
    if brief.search_facts_summary:
        parts.append(f"Verified visible facts: {truncate_text(brief.search_facts_summary, _FACTS_LIMIT)}")
    if brief.reasoning_output:
        parts.append("Reasoning conclusions: " + "; ".join(brief.reasoning_output))
    if brief.svg_description:
        parts.append("Preserve this layout/composition: " + truncate_text(brief.svg_description, _TEXT_LIMIT))
    if brief.expected_texts_used or brief.text_template_path:
        if brief.render_type.lower() in {"embedded", "embedded_svg"}:
            parts.append(
                "If using the embedded text reference image, preserve the exact "
                "text strings and which object/surface each label belongs to, "
                "but treat UI-like boxes, bars, borders, background fills, and "
                "placeholder panels as position/content placeholders only. "
                "Convert them into scene-native physical carriers such as paper "
                "or acrylic museum plaques, storefront signs, invitation cards, "
                "handwritten notes, sticky notes, speech bubbles, product tags, "
                "menu boards, chalkboards, display-case labels, or printed cards. "
                "For signs and plaques, include real physical construction: board "
                "thickness, bevels or edges, frames, poles, brackets, overhead "
                "gantry, wall mounts, bolts, screws, hanging wires, tape, shadows, "
                "reflections, and perspective attachment to the environment. "
                "Naturalize carrier size, perspective, materials, and scene "
                "proportions; do not rigidly copy oversized flat blueprint text boxes, "
                "floating rounded UI cards, or screen-overlay panels."
            )
        else:
            parts.append(
                "If using the text template image, preserve all rendered text exactly "
                "as shown; do not rewrite, omit, or add text."
            )
    return "\n\n".join(parts).strip() or "Generate the requested image faithfully."

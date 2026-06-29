"""VLM review tool for final image constraint checking.

This is deliberately an atomic checker, not a generator. It reviews the final
image produced by t2i/i2i against the original user request and returns
structured, actionable feedback that the main agent can feed into one bounded
i2i retry.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ..config import load_genclaw_config
from ..runtime.artifacts import record_artifact, record_artifacts_in_dir
from ..runtime.session_dir import tool_output_dir
from .base import BaseTool


_DEFAULT_CRITERIA = [
    "exact_count",
    "object_presence",
    "color_fidelity",
    "spatial_relation",
    "relative_size",
]


_REVIEW_SYSTEM_PROMPT = """You are a strict visual QA reviewer for an image-generation agent.

Your job is to check whether the FINAL generated image satisfies the user's hard visual constraints AND any concrete commitments already made by upstream tools (structural drafts, expected long text, reference images, derived conclusions).

Focus ONLY on objective, verifiable constraints. Choose the subset relevant to this task from:
- exact object counts and required objects being present and visible
- specified colors
- spatial relations (left/right/top/bottom/center/grid/near/far) and relative sizes
- preservation of a supplied structural draft when draft information is provided
- readable text fidelity:
  - For short text strings such as titles, labels, dates, prices, buttons, and short slogans, check obvious spelling/character errors, legibility, order, and placement.
  - For long-form text such as poems, articles, menus with many items, paragraphs, documents, or dense multi-line text, DO NOT attempt character-perfect OCR. Exact text content is assumed to be validated by the source/DOM/code renderer. Review only visual readability and layout risks: clipping, occlusion, overlap, text running outside the frame, extremely small font, poor contrast, unreadable blur, distorted glyphs after restyling, or a layout that no longer looks like readable text.
  - Preserve the template/layout when expected_texts come from a code_text_draft or HTML/CSS renderer.
  - If the source is code_text_draft embedded_svg, also check whether text carriers look naturally embedded in the scene. Small labels, exhibit plaques, signs, and bubbles should remain proportionate to their objects; flag oversized poster-like panels or obvious blueprint remnants as layout/text integration issues. For signs/plaques/cards, require evidence of physical carrier construction or scene attachment: board thickness, frame, pole, bracket, gantry, wall mount, bolts, tape, hanging wire, shadows, reflections, or perspective attachment. Flag floating rounded rectangles, flat app-like cards, screen-overlay panels, or sign boxes with no physical support/attachment as objective embedded-text integration failures.
- identity / reference fidelity: when reference images and/or a search facts summary are supplied, visually compare the final image against each attached reference according to its stated role. The image must still read as the named entity, object, place, product, or design. Check distinctive visible structure, not just a generic category match: silhouette, proportions, material, surface layout, major components, attachment points, logos/marks when relevant, signature colors, costume/outfit details, and relative placement of distinctive parts. When multiple named entities or reference roles must coexist, judge each independently. Do not pass a generic similar object when the referenced structure is visibly different.
- scientific or geometric accuracy: when reasoning conclusions are supplied, the image must agree with them (correct structure, correct option, correct relation)

Do NOT grade aesthetics, realism, lighting, camera quality, or artistic taste unless those affect a hard constraint in the user request.

Return ONLY one valid JSON object. No markdown fences, no prose outside JSON.

Schema:
{
  "passed": boolean,
  "verdict": "PASS" | "NEEDS_FIX" | "FAIL",
  "issues": [
    {
      "type": "missing_object" | "count_mismatch" | "color_mismatch" | "position_mismatch" | "size_mismatch" | "draft_drift" | "text_mismatch" | "identity_drift" | "reference_mismatch" | "scientific_mismatch" | "other",
      "severity": "minor" | "major",
      "description": "short visible issue",
      "fix_instruction": "one direct instruction useful for the next i2i prompt"
    }
  ],
  "suggested_fix_prompt": "single prompt for revising the current image while preserving correct parts",
  "repair_strategy": "none" | "edit_previous" | "regenerate_from_source",
  "should_retry": boolean
}

Rules:
- If all relevant hard constraints are satisfied, use verdict PASS, passed true, issues [], should_retry false.
- If there are fixable issues, use verdict NEEDS_FIX, passed false, should_retry true.
- If the image is too unrelated or cannot be judged, use verdict FAIL, passed false, should_retry false.
- suggested_fix_prompt must be concise and directly usable as an i2i prompt.
- Choose repair_strategy:
  - edit_previous for local, simple edits where the current image is a good canvas (minor color, small missing detail, slight position adjustment, text clarity).
  - regenerate_from_source for major global failures where the current image is a bad canvas: split-screen / stacked panels / collage, duplicated main scene, wrong canvas or aspect ratio, missing primary subject, object relationships distributed across separate panels, or severe composition drift.
  - none for PASS or FAIL.
- When repair_strategy is regenerate_from_source, suggested_fix_prompt should describe a fresh single coherent generation from the original source/reference images, not an edit of the failed image.
- When suggesting fixes, preserve all correct objects, text, identity, and layout; do not ask to regenerate from scratch unless repair_strategy is regenerate_from_source or verdict is FAIL.
"""


def _review_model() -> str:
    return (
        os.environ.get("VLM_REVIEW_MODEL")
        or os.environ.get("SCENE_DRAFT_REVIEW_MODEL")
        or os.environ.get("OPENAI_MODEL_NAME")
        or "gpt-5.4"
    )


def _encode_image(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _reference_label(index: int) -> str:
    return chr(ord("A") + index) if 0 <= index < 26 else str(index + 1)


def _review_message_content(user_text: str, final_path: Path, reference_paths: List[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_text},
        {"type": "text", "text": "Final image: generated image to review."},
        {"type": "image_url", "image_url": {"url": _encode_image(final_path)}},
    ]
    for idx, ref in enumerate(reference_paths):
        ref_path = Path(ref)
        if not ref_path.is_absolute() or not ref_path.is_file():
            continue
        label = _reference_label(idx)
        role = "primary source/canvas/reference" if idx == 0 else "secondary visual reference"
        content.append({"type": "text", "text": f"Reference {label}: {role}. Compare only according to its intended role in the prompt/context."})
        content.append({"type": "image_url", "image_url": {"url": _encode_image(ref_path)}})
    return content


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("review response is not a JSON object")
    return obj


def _normalize_review(obj: Dict[str, Any]) -> Dict[str, Any]:
    issues_raw = obj.get("issues")
    issues: List[Dict[str, Any]] = []
    if isinstance(issues_raw, list):
        for item in issues_raw:
            if isinstance(item, dict):
                issues.append(
                    {
                        "type": str(item.get("type") or "other"),
                        "severity": str(item.get("severity") or "major"),
                        "description": str(item.get("description") or "").strip(),
                        "fix_instruction": str(item.get("fix_instruction") or "").strip(),
                    }
                )

    verdict = str(obj.get("verdict") or "").upper()
    if verdict not in {"PASS", "NEEDS_FIX", "FAIL"}:
        verdict = "PASS" if bool(obj.get("passed")) and not issues else "NEEDS_FIX"

    passed = bool(obj.get("passed")) if "passed" in obj else verdict == "PASS"
    if verdict == "PASS":
        passed = True
        issues = []

    suggested = str(obj.get("suggested_fix_prompt") or "").strip()
    should_retry = bool(obj.get("should_retry"))
    if verdict == "NEEDS_FIX" and suggested:
        should_retry = True
    if verdict != "NEEDS_FIX":
        should_retry = False

    repair_strategy = str(obj.get("repair_strategy") or "").strip().lower()
    if repair_strategy not in {"none", "edit_previous", "regenerate_from_source"}:
        repair_strategy = _infer_repair_strategy(verdict, issues, should_retry)
    if verdict != "NEEDS_FIX" or not should_retry:
        repair_strategy = "none"

    return {
        "status": "success",
        "passed": passed,
        "verdict": verdict,
        "issues": issues,
        "suggested_fix_prompt": suggested,
        "repair_strategy": repair_strategy,
        "should_retry": should_retry,
    }


def _infer_repair_strategy(
    verdict: str,
    issues: List[Dict[str, Any]],
    should_retry: bool,
) -> str:
    if verdict != "NEEDS_FIX" or not should_retry:
        return "none"

    global_failure_markers = (
        "split",
        "stacked",
        "panel",
        "collage",
        "duplicate",
        "duplicated",
        "two separate",
        "separate views",
        "separate product shots",
        "wrong canvas",
        "aspect ratio",
        "long strip",
        "main subject",
        "primary subject",
        "not clearly hanging",
        "distributed across",
    )
    for issue in issues:
        if str(issue.get("severity") or "").lower() != "major":
            continue
        text = " ".join(
            [
                str(issue.get("type") or ""),
                str(issue.get("description") or ""),
                str(issue.get("fix_instruction") or ""),
            ]
        ).lower()
        if any(marker in text for marker in global_failure_markers):
            return "regenerate_from_source"

    return "edit_previous"


class VLMReviewTool(BaseTool):
    name = "vlm_review"
    description = ""  # filled from yaml card at runtime
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Absolute path to the final image to review.",
            },
            "original_user_prompt": {
                "type": "string",
                "description": "The original user request; source of truth for hard constraints.",
            },
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional criteria to check, e.g. exact_count, color_fidelity, spatial_relation.",
            },
            "draft_png_path": {
                "type": "string",
                "description": "Optional structural draft image path used before i2i.",
            },
            "svg_description": {
                "type": "string",
                "description": "Optional layout JSON/string from code_scene_draft.",
            },
            "expected_texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of verbatim text strings that must remain readable in the image (typically from code_text_draft.expected_texts_used).",
            },
            "source_render_type": {
                "type": "string",
                "description": "Optional upstream render source, e.g. code_text_draft:pure_code, code_text_draft:layered, code_text_draft:embedded_svg, code_scene_draft:i2i.",
            },
            "reference_image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of absolute reference image paths (typically from search.reference_paths) the final image must remain consistent with.",
            },
            "search_facts_summary": {
                "type": "string",
                "description": "Optional free-text summary of distinguishing facts about named entities / IPs / products from search (e.g. signature colors, logo shape, costume details, product model, era, brand mark).",
            },
            "reasoning_conclusions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of factual / geometric / scientific conclusions from reason that the final image must agree with.",
            },
            "max_issues": {
                "type": "integer",
                "description": "Maximum number of issues to return (default 6).",
            },
        },
        "required": ["image_path", "original_user_prompt"],
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        image_path = str(args.get("image_path") or "").strip()
        original_prompt = str(args.get("original_user_prompt") or "").strip()
        if not image_path:
            return self.err(json.dumps({"status": "failed", "error": "image_path is required"}, ensure_ascii=False))
        if not original_prompt:
            return self.err(json.dumps({"status": "failed", "error": "original_user_prompt is required"}, ensure_ascii=False))

        path = Path(image_path)
        if not path.is_absolute() or not path.is_file():
            return self.err(
                json.dumps(
                    {
                        "status": "failed",
                        "error": f"image_path must be an existing absolute file path, got {image_path!r}",
                    },
                    ensure_ascii=False,
                )
            )

        try:
            load_genclaw_config()
        except FileNotFoundError:
            # Unit tests and some local integrations may provide env directly.
            pass

        criteria = args.get("criteria") or _DEFAULT_CRITERIA
        if not isinstance(criteria, list):
            criteria = _DEFAULT_CRITERIA
        max_issues = int(args.get("max_issues") or 6)
        draft_png_path = str(args.get("draft_png_path") or "").strip()
        svg_description = str(args.get("svg_description") or "").strip()
        source_render_type = str(args.get("source_render_type") or "").strip()

        def _coerce_str_list(value: Any) -> List[str]:
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            return []

        expected_texts = _coerce_str_list(args.get("expected_texts"))
        reference_paths = _coerce_str_list(args.get("reference_image_paths"))
        reasoning_conclusions = _coerce_str_list(args.get("reasoning_conclusions"))
        search_facts_summary = str(args.get("search_facts_summary") or "").strip()

        expected_texts_block = (
            json.dumps(expected_texts, ensure_ascii=False)
            if expected_texts
            else "(none)"
        )
        reference_paths_block = (
            json.dumps(reference_paths, ensure_ascii=False)
            if reference_paths
            else "(none)"
        )
        reasoning_block = (
            json.dumps(reasoning_conclusions, ensure_ascii=False)
            if reasoning_conclusions
            else "(none)"
        )
        search_facts_block = search_facts_summary[:4000] or "(none)"

        user_text = (
            f"Original user prompt:\n{original_prompt}\n\n"
            f"Review criteria:\n{json.dumps(criteria, ensure_ascii=False)}\n\n"
            f"Max issues to report: {max_issues}\n\n"
            f"Upstream render source, if known: {source_render_type or '(none)'}\n\n"
            f"Draft image path, if any: {draft_png_path or '(none)'}\n\n"
            f"Structural draft description, if any:\n{svg_description[:4000] or '(none)'}\n\n"
            f"Expected verbatim texts that must remain readable, if any:\n{expected_texts_block}\n\n"
            f"Reference image paths the final image must stay consistent with, if any:\n{reference_paths_block}\n\n"
            f"Search-derived facts about named entities / IPs / products, if any:\n{search_facts_block}\n\n"
            f"Reasoning conclusions the final image must agree with, if any:\n{reasoning_block}\n\n"
            "Attached images are ordered as: Final image first, then Reference A, "
            "Reference B, Reference C, etc. When reference_match is relevant, "
            "compare the final image visually against the attached references "
            "according to their roles. Return the JSON object."
        )

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                timeout=120.0,
            )
            response = client.with_options(timeout=90.0).chat.completions.create(
                model=_review_model(),
                messages=[
                    {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _review_message_content(user_text, path, reference_paths),
                    },
                ],
                max_tokens=1200,
            )
            raw = response.choices[0].message.content or ""
            payload = _normalize_review(_extract_json_object(raw))
        except Exception as e:  # noqa: BLE001
            return self.err(
                json.dumps(
                    {"status": "failed", "error": f"vlm_review failed: {type(e).__name__}: {e}"},
                    ensure_ascii=False,
                )
            )

        payload["image_path"] = str(path.resolve())
        payload["model"] = _review_model()

        try:
            out_dir = Path(tool_output_dir("vlm_review"))
            (out_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "args": args,
                        "result": payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            record_artifacts_in_dir(out_dir, tool_name="vlm_review")
            record_artifact(
                path,
                tool_name="vlm_review",
                role="review_target",
                kind="image",
                label="Reviewed image",
                out_dir=str(out_dir),
            )
        except Exception:
            pass

        return self.ok(json.dumps(payload, ensure_ascii=False))

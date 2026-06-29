"""SVG generation / review / revision (self-contained).

Notes:
- Config (api_key / base_url / models) is read from os.environ, not a config
  dict. Populate env via cc_genclaw.config.load_genclaw_config().
- Logging uses the stdlib logging module.
- No ``temperature`` is sent (newer default models reject it).
- Canvas size is dynamic: the overflow clamp reads the SVG's declared size.

Public API: generate / review / revise.
Internal helpers: _parse_response / _clamp_svg_to_canvas / _apply_viewbox_crop.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

from cc_genclaw.llm import LLMClient, LLMConfig

from .prompts import REVIEW_PROMPT, SVG_REVISE_SYSTEM_PROMPT, SVG_SYSTEM_PROMPT
from .render import parse_canvas_size

logger = logging.getLogger("cc_genclaw.scene_draft")

_PLANNING_DELIMITER = "===PLANNING_START==="
_LEGEND_DELIMITER = "===LEGEND_START==="
_SVG_DELIMITER = "===SVG_CODE_START==="
MIN_SVG_CHARS = 400
MAX_RETRIES = 3

# Default models. Generation can be set separately; review defaults to the
# main agent's model (OPENAI_MODEL_NAME), then a safe fallback.
_DEFAULT_MODEL_FALLBACK = "api_naci_azure_gpt-5.5"
_DEFAULT_REVIEW_FALLBACK = "api_naci_azure_gpt-5.5"


# ==============================================================================
# Config helpers (env-based, self-contained)
# ==============================================================================

def _api_key() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


def _base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


def gen_model() -> str:
    if (os.environ.get("CODE_LLM_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return (
            os.environ.get("SCENE_DRAFT_GEN_MODEL")
            or os.environ.get("SVG_GEN_MODEL_NAME")
            or os.environ.get("CODE_LLM_MODEL_NAME")
            or LLMConfig.for_code_draft().model
            or _DEFAULT_MODEL_FALLBACK
        )
    return LLMConfig.for_code_draft().model or os.environ.get("MAIN_LLM_MODEL_NAME") or _DEFAULT_MODEL_FALLBACK


def review_model() -> str:
    if (os.environ.get("CODE_LLM_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return (
            os.environ.get("SCENE_DRAFT_REVIEW_MODEL")
            or os.environ.get("SVG_REVIEW_MODEL_NAME")
            or os.environ.get("CODE_LLM_MODEL_NAME")
            or LLMConfig.for_code_draft().model
            or _DEFAULT_REVIEW_FALLBACK
        )
    return LLMConfig.for_code_draft().model or _DEFAULT_REVIEW_FALLBACK


def _chat_completion(*, timeout: float, **kwargs):
    cfg = LLMConfig.for_code_draft()
    cfg.timeout = timeout
    response = LLMClient(cfg).create(
        messages=kwargs.get("messages") or [],
        tools=kwargs.get("tools"),
        model=kwargs.get("model"),
    )
    return _to_attr_tree(response)


def _to_attr_tree(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_attr_tree(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_attr_tree(item) for item in value]
    return value


# ==============================================================================
# Internal Helpers - Canvas Clamp (safety net for overflow)
# ==============================================================================

def _clamp_svg_to_canvas(
    svg_code: str,
    canvas_w: Optional[int] = None,
    canvas_h: Optional[int] = None,
) -> str:
    """Expand viewBox when shapes overflow the declared canvas.

    Canvas dimensions default to the SVG's own declared width/height (dynamic),
    not a fixed 1024x1024. HTML documents and unparseable SVG are returned
    unchanged; no-op when all shapes fit.
    """
    import xml.etree.ElementTree as ET

    if "<html" in svg_code.lower() or "<!doctype" in svg_code.lower():
        return svg_code

    if canvas_w is None or canvas_h is None:
        canvas_w, canvas_h = parse_canvas_size(svg_code)

    try:
        root = ET.fromstring(svg_code)
    except ET.ParseError:
        return svg_code

    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    shape_tags = {f"{ns}circle", f"{ns}ellipse", f"{ns}rect", f"{ns}line",
                  f"{ns}polygon", f"{ns}polyline", f"{ns}path"}

    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")
    shape_count = 0
    bg_rect = None

    for elem in root.iter():
        tag = elem.tag
        if tag not in shape_tags:
            continue
        local_tag = tag.replace(ns, "")

        ex1, ey1, ex2, ey2 = None, None, None, None

        if local_tag == "circle":
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            r = float(elem.get("r", 0))
            ex1, ey1, ex2, ey2 = cx - r, cy - r, cx + r, cy + r
        elif local_tag == "ellipse":
            cx = float(elem.get("cx", 0))
            cy = float(elem.get("cy", 0))
            rx = float(elem.get("rx", 0))
            ry = float(elem.get("ry", 0))
            ex1, ey1, ex2, ey2 = cx - rx, cy - ry, cx + rx, cy + ry
        elif local_tag == "rect":
            x = float(elem.get("x", 0))
            y = float(elem.get("y", 0))
            w = float(elem.get("width", 0))
            h = float(elem.get("height", 0))
            if w >= canvas_w * 0.9 and h >= canvas_h * 0.9:
                if bg_rect is None:
                    bg_rect = elem
                continue
            ex1, ey1, ex2, ey2 = x, y, x + w, y + h
        elif local_tag == "line":
            x1 = float(elem.get("x1", 0))
            y1 = float(elem.get("y1", 0))
            x2 = float(elem.get("x2", 0))
            y2 = float(elem.get("y2", 0))
            ex1, ey1 = min(x1, x2), min(y1, y2)
            ex2, ey2 = max(x1, x2), max(y1, y2)
        elif local_tag in ("polygon", "polyline"):
            points_str = elem.get("points", "")
            try:
                coords = [float(v) for v in re.split(r"[,\s]+", points_str.strip()) if v]
                xs, ys = coords[0::2], coords[1::2]
                if xs and ys:
                    ex1, ey1, ex2, ey2 = min(xs), min(ys), max(xs), max(ys)
            except (ValueError, IndexError):
                continue
        elif local_tag == "path":
            d = elem.get("d", "")
            nums = re.findall(r"[-+]?\d*\.?\d+", d)
            if len(nums) >= 2:
                coords = [float(n) for n in nums]
                xs, ys = coords[0::2], coords[1::2]
                if xs and ys:
                    ex1, ey1, ex2, ey2 = min(xs), min(ys), max(xs), max(ys)

        if ex1 is not None:
            min_x, min_y = min(min_x, ex1), min(min_y, ey1)
            max_x, max_y = max(max_x, ex2), max(max_y, ey2)
            shape_count += 1

    if shape_count == 0 or min_x == float("inf"):
        return svg_code

    overflow_tolerance = 5
    overflows = (min_x < -overflow_tolerance or min_y < -overflow_tolerance or
                 max_x > canvas_w + overflow_tolerance or max_y > canvas_h + overflow_tolerance)
    if not overflows:
        return svg_code

    logger.warning(
        "[scene_draft] content bbox (%.0f,%.0f)-(%.0f,%.0f) overflows canvas %dx%d, adjusting viewBox",
        min_x, min_y, max_x, max_y, canvas_w, canvas_h,
    )

    expand_padding = 40
    vb_x = min(0, min_x - expand_padding)
    vb_y = min(0, min_y - expand_padding)
    vb_max_x = max(canvas_w, max_x + expand_padding)
    vb_max_y = max(canvas_h, max_y + expand_padding)
    vb_w = vb_max_x - vb_x
    vb_h = vb_max_y - vb_y

    new_vb = f"{int(vb_x)} {int(vb_y)} {int(vb_w)} {int(vb_h)}"

    if re.search(r'viewBox\s*=\s*"[^"]*"', svg_code):
        svg_code = re.sub(r'viewBox\s*=\s*"[^"]*"', f'viewBox="{new_vb}"', svg_code, count=1)
    elif re.search(r"viewBox\s*=\s*'[^']*'", svg_code):
        svg_code = re.sub(r"viewBox\s*=\s*'[^']*'", f"viewBox='{new_vb}'", svg_code, count=1)

    if bg_rect is not None:
        bg_fill = bg_rect.get("fill", "white")
        old_rect_pattern = re.compile(
            r'<rect\s[^>]*(?:width\s*=\s*["\'][0-9.]+["\'])[^>]*/?\s*>',
            re.IGNORECASE,
        )
        match = old_rect_pattern.search(svg_code)
        if match:
            new_rect = (
                f'<rect x="{int(vb_x)}" y="{int(vb_y)}" '
                f'width="{int(vb_w)}" height="{int(vb_h)}" fill="{bg_fill}" />'
            )
            svg_code = svg_code[:match.start()] + new_rect + svg_code[match.end():]

    logger.info("[scene_draft] viewBox adjusted to: %s", new_vb)
    return svg_code


# ==============================================================================
# Internal Helpers - ViewBox Crop (reviewer-driven)
# ==============================================================================

def _apply_viewbox_crop(
    svg_code: str,
    viewbox_params: Union[str, Tuple[int, int, int, int]],
) -> str:
    """Replace the viewBox attribute to zoom into the reviewer-recommended bbox."""
    if isinstance(viewbox_params, (tuple, list)):
        if len(viewbox_params) != 4:
            logger.warning("[scene_draft] invalid viewBox tuple: %s, skip crop", viewbox_params)
            return svg_code
        parts = list(viewbox_params)
    else:
        parts = str(viewbox_params).split()
        if len(parts) != 4:
            logger.warning("[scene_draft] invalid viewBox format: %s, skip crop", viewbox_params)
            return svg_code

    try:
        vals = [int(float(v)) for v in parts]
    except (ValueError, TypeError):
        logger.warning("[scene_draft] non-numeric viewBox: %s, skip crop", viewbox_params)
        return svg_code

    new_vb_str = f"{vals[0]} {vals[1]} {vals[2]} {vals[3]}"

    if re.search(r'viewBox\s*=\s*"[^"]*"', svg_code):
        cropped = re.sub(r'viewBox\s*=\s*"[^"]*"', f'viewBox="{new_vb_str}"', svg_code, count=1)
    elif re.search(r"viewBox\s*=\s*'[^']*'", svg_code):
        cropped = re.sub(r"viewBox\s*=\s*'[^']*'", f"viewBox='{new_vb_str}'", svg_code, count=1)
    else:
        logger.warning("[scene_draft] no viewBox attribute, skip crop")
        return svg_code

    logger.info("[scene_draft] viewBox cropped to: %s", new_vb_str)
    return cropped


# ==============================================================================
# Internal Helpers - Response Parsing
# ==============================================================================

def _parse_response(raw: str) -> dict:
    """Parse raw LLM response into planning / layout(JSON) / svg code.

    Supports the 3-part CoT format PLANNING / LEGEND / SVG_CODE, with a 2-part
    fallback. PART 2 is now a layout JSON string; it is returned as
    ``svg_description`` (raw string) plus ``layout`` (parsed dict or None).
    """
    planning = ""
    svg_description = ""
    svg_code = ""

    has_planning = _PLANNING_DELIMITER in raw
    has_legend = _LEGEND_DELIMITER in raw
    has_svg = _SVG_DELIMITER in raw

    if has_planning and has_svg:
        after_planning = raw.split(_PLANNING_DELIMITER, 1)[1]
        if has_legend:
            planning_part, after_legend = after_planning.split(_LEGEND_DELIMITER, 1)
            legend_part, code_part = after_legend.split(_SVG_DELIMITER, 1)
            planning = planning_part.strip()
            svg_description = legend_part.strip()
            svg_code = code_part.strip()
        else:
            planning_part, code_part = after_planning.split(_SVG_DELIMITER, 1)
            planning = planning_part.strip()
            svg_code = code_part.strip()
    elif has_svg:
        desc_part, code_part = raw.split(_SVG_DELIMITER, 1)
        svg_description = desc_part.strip()
        svg_code = code_part.strip()
    else:
        svg_code = raw

    # Strip fences / leftover delimiters from the layout JSON.
    svg_description = re.sub(r"^```(?:json)?\s*", "", svg_description)
    svg_description = re.sub(r"\s*```$", "", svg_description).strip()

    svg_code = re.sub(r"^```(?:svg|html|xml)?\s*", "", svg_code)
    svg_code = re.sub(r"\s*```$", "", svg_code).strip()

    for delim in (_PLANNING_DELIMITER, _LEGEND_DELIMITER, _SVG_DELIMITER):
        svg_code = svg_code.replace(delim, "")
    svg_code = re.sub(
        r"={2,}(?:PLANNING|LEGEND|SVG)[_\s]*(?:START|CODE)[_\s]*={2,}",
        "",
        svg_code,
        flags=re.IGNORECASE,
    )

    svg_start = re.search(r"<svg[\s>]", svg_code, re.IGNORECASE)
    html_start = re.search(r"(?:<html|<!doctype)", svg_code, re.IGNORECASE)
    code_start = None
    if svg_start and html_start:
        code_start = min(svg_start.start(), html_start.start())
    elif svg_start:
        code_start = svg_start.start()
    elif html_start:
        code_start = html_start.start()

    if code_start and code_start > 0:
        preamble = svg_code[:code_start].strip()
        if preamble and not svg_description:
            svg_description = preamble
        svg_code = svg_code[code_start:]

    if svg_start and not html_start:
        svg_end = re.search(r"</svg\s*>", svg_code, re.IGNORECASE)
        if svg_end:
            svg_code = svg_code[:svg_end.end()]
    else:
        html_end = re.search(r"</html\s*>", svg_code, re.IGNORECASE)
        if html_end:
            svg_code = svg_code[:html_end.end()]

    svg_code = re.sub(r"<text[^>]*>.*?</text>", "", svg_code, flags=re.DOTALL | re.IGNORECASE)

    svg_code = _clamp_svg_to_canvas(svg_code)

    code_type = "html" if "<html" in svg_code.lower() or "<!doctype" in svg_code.lower() else "svg"

    # Best-effort parse the layout JSON; keep raw string regardless.
    layout = None
    if svg_description:
        try:
            import json
            layout = json.loads(svg_description)
        except Exception:
            layout = None

    if planning:
        logger.info("[scene_draft] CoT planning (%d chars)", len(planning))

    return {
        "svg_code": svg_code,
        "svg_description": svg_description,
        "layout": layout,
        "svg_planning": planning,
        "code_type": code_type,
    }


# ==============================================================================
# Public API - Generation
# ==============================================================================

def generate(prompt: str, model: Optional[str] = None) -> dict:
    """Generate SVG/HTML from a text prompt using the 3-part CoT prompt.

    Retries up to MAX_RETRIES when the SVG body is shorter than MIN_SVG_CHARS.
    """
    resolved_model = model or gen_model()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _chat_completion(
                timeout=180.0,
                model=resolved_model,
                messages=[
                    {"role": "system", "content": SVG_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate SVG code for:\n{prompt}"},
                ],
                max_tokens=16384,
            )

            raw = response.choices[0].message.content.strip()
            parsed = _parse_response(raw)

            if len(parsed["svg_code"]) < MIN_SVG_CHARS:
                logger.warning(
                    "[scene_draft] generate attempt %d: too short (%d < %d), retrying",
                    attempt, len(parsed["svg_code"]), MIN_SVG_CHARS,
                )
                last_error = f"SVG too short: {len(parsed['svg_code'])} chars"
                continue

            parsed["model"] = resolved_model
            return parsed

        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            logger.error("[scene_draft] generate attempt %d failed: %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                continue
            raise

    raise RuntimeError(f"generate failed after {MAX_RETRIES} attempts: {last_error}")


# ==============================================================================
# Public API - Vision Review (PASS / OPTIMIZE / FAIL)
# ==============================================================================

def review(prompt: str, png_path: str, model: Optional[str] = None, svg_code: str = "") -> dict:
    """Vision-review a rendered SVG against the prompt.

    Returns {passed, verdict, feedback, viewbox_params}.
    """
    resolved_model = model or review_model()

    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    review_text = REVIEW_PROMPT.format(prompt=prompt, svg_code=svg_code[:8000])

    response = _chat_completion(
        timeout=60.0,
        model=resolved_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": review_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=512,
    )

    reply = response.choices[0].message.content.strip()

    if reply.startswith("PASS"):
        return {"passed": True, "verdict": "PASS", "feedback": None, "viewbox_params": None}

    if reply.startswith("OPTIMIZE"):
        viewbox_params: Optional[Tuple[int, int, int, int]] = None
        vb_match = re.search(r"<viewBox>\s*(.+?)\s*</viewBox>", reply)
        if vb_match:
            parts = vb_match.group(1).strip().split()
            if len(parts) == 4:
                try:
                    viewbox_params = tuple(int(float(v)) for v in parts)  # type: ignore[assignment]
                except ValueError:
                    viewbox_params = None
        logger.info("[scene_draft] review OPTIMIZE viewBox: %s", viewbox_params)
        return {
            "passed": False,
            "verdict": "OPTIMIZE",
            "feedback": "Objects too small, needs viewBox crop",
            "viewbox_params": viewbox_params,
        }

    if "<feedback>" in reply and "</feedback>" in reply:
        feedback = reply.split("<feedback>")[1].split("</feedback>")[0].strip()
    else:
        feedback = reply.replace("FAIL", "").strip()

    return {"passed": False, "verdict": "FAIL", "feedback": feedback, "viewbox_params": None}


# ==============================================================================
# Public API - Revision
# ==============================================================================

def revise(prompt: str, old_svg_code: str, feedback: str, model: Optional[str] = None) -> dict:
    """Revise an SVG based on reviewer feedback. Same shape as generate()."""
    resolved_model = model or gen_model()

    user_msg = (
        f"Original prompt: {prompt}\n\n"
        f"Previous SVG code:\n```\n{old_svg_code}\n```\n\n"
        f"Review feedback: {feedback}\n\n"
        f"Please generate a corrected SVG that fixes all the issues."
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _chat_completion(
                timeout=180.0,
                model=resolved_model,
                messages=[
                    {"role": "system", "content": SVG_REVISE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=16384,
            )

            raw = response.choices[0].message.content.strip()
            parsed = _parse_response(raw)

            if len(parsed["svg_code"]) < MIN_SVG_CHARS:
                logger.warning(
                    "[scene_draft] revise attempt %d: too short (%d), retrying",
                    attempt, len(parsed["svg_code"]),
                )
                last_error = f"Revised SVG too short: {len(parsed['svg_code'])} chars"
                continue

            parsed["model"] = resolved_model
            return parsed

        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            logger.error("[scene_draft] revise attempt %d failed: %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                continue
            raise

    raise RuntimeError(f"revise failed after {MAX_RETRIES} attempts: {last_error}")

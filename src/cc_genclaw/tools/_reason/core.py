"""Reasoning core — the single implementation behind the `reason` tool.

This module is the **atomic** multimodal-reasoning primitive:

    run_reasoning(prompt, user_image_path=None, reference_image_paths=None) -> dict

It calls one chat LLM with the 4-step CoT system prompt
(``reasoning_prompt.yaml`` next to this file) plus the prompt and any
images, then returns the parsed JSON dict (``{"reasoning_knowledge":
[...]}``) with defensive fallbacks.

Design (mirrors tools/_image_gen):
- Self-contained — the helpers it needs (image encoding, robust JSON
  parsing) live in this module.
- Credentials/endpoint/model come from ``os.environ`` (populated by
  ``cc_genclaw.config.load_genclaw_config`` before this runs):
  ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``OPENAI_MODEL_NAME``.
- ``temperature=0.0`` for stable reasoning, exactly as before.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger("cc_genclaw.reason")

_PROMPT_PATH = Path(__file__).resolve().parent / "reasoning_prompt.yaml"
_MAX_REF_IMAGES = 5  # limit to prevent token explosion


# ==============================================================================
# System prompt
# ==============================================================================

def _load_system_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("system_prompt", "")


# ==============================================================================
# Image encoding (encode an image file as an OpenAI data URI)
# ==============================================================================

_MIME_BY_EXT = {
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}


def _encode_image(path: str) -> str:
    """Return a data-URI for an image file, or '' on failure.

    Returning '' (rather than raising) matches the legacy truthiness
    checks (``if base64_img:``) so a bad path simply drops the image
    instead of aborting the whole reasoning call.
    """
    try:
        if not path or not os.path.exists(path):
            return ""
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = _MIME_BY_EXT.get(os.path.splitext(path)[1].lower(), "image/jpeg")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:  # noqa: BLE001
        logger.warning("[reason] failed to encode image %s: %s", path, e)
        return ""


# ==============================================================================
# Robust JSON parsing
# ==============================================================================

def _clean_json_markdown(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    if content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    return content


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete ``{...}`` block (brace-matched, nesting-aware)."""
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text


def _fix_common_json_issues(text: str) -> str:
    """Repair the most common illegal characters in LLM JSON.

    1. Strip stray backslashes from non-standard escapes (LLMs often write
       LaTeX like ``\\(`` ``\\)`` ``\\,`` inside JSON strings).
    2. Drop trailing commas (``,}`` / ``,]``).
    """
    if not text:
        return text

    def _repl_bad_escape(m: "re.Match[str]") -> str:
        seq = m.group(0)
        nxt = seq[1] if len(seq) > 1 else ""
        if nxt in '"\\/bfnrtu':
            return seq  # legal escape, keep
        return nxt  # illegal escape (likely LaTeX), drop the backslash

    text = re.sub(r"\\.", _repl_bad_escape, text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def _parse_json_response(content: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """Parse JSON from an LLM response with multi-layer fallback.

    L1 clean fence + json.loads → L2 extract first {...} → L3 repair
    escapes/trailing commas → L4 default (or raise).
    """
    if not content:
        if default is not None:
            return default
        raise json.JSONDecodeError("empty content", "", 0)

    cleaned = _clean_json_markdown(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    extracted = _extract_first_json_object(cleaned)
    if extracted and extracted != cleaned:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    candidate = extracted if extracted else cleaned
    fixed = _fix_common_json_issues(candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as final_err:
        if default is not None:
            return default
        raise final_err


# ==============================================================================
# Public API
# ==============================================================================

def run_reasoning(
    prompt: str,
    user_image_path: Optional[str] = None,
    reference_image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Reason over ``prompt`` (+ optional images) and return the parsed dict.

    Returns ``{"reasoning_knowledge": List[str], ...}``. On a transport
    failure returns ``{"reasoning_knowledge": [], "error": ...}``; on an
    unparseable-but-non-empty model reply, preserves the raw content as a
    single reasoning_knowledge entry rather than silently dropping it.
    """
    from openai import OpenAI  # local import keeps module import cheap

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL_NAME", "gpt-5.4")

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Same composite text context the legacy tool built, so model behaviour
    # is identical: user_intent == prompt, need_process_problem == [prompt].
    need_process_problem = [prompt]
    text_context = f"""
    === User Input (Fact-Enriched) ===
    Request: {prompt}
    
    === Problems to Solve (Target Output) ===
    {json.dumps(need_process_problem, indent=2)}
    """

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": text_context}]

    # Primary user image — high detail for reasoning fidelity.
    if user_image_path and os.path.exists(user_image_path):
        data_uri = _encode_image(user_image_path)
        if data_uri:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_uri, "detail": "high"},
            })
            user_content.append({
                "type": "text",
                "text": "[System Note: The image above is the User Input Image.]",
            })

    # Reference images (capped to avoid token explosion).
    if reference_image_paths:
        valid_count = 0
        for img_path in reference_image_paths:
            if valid_count >= _MAX_REF_IMAGES:
                break
            if img_path and os.path.exists(img_path):
                data_uri = _encode_image(img_path)
                if data_uri:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
                    valid_count += 1
        if valid_count > 0:
            user_content.append({
                "type": "text",
                "text": "[System Note: The images above are Reference Images.]",
            })

    messages = [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip()

        unparseable_sentinel: Dict[str, Any] = {"_unparseable": True}
        parsed = _parse_json_response(content, default=unparseable_sentinel)
        if parsed.get("_unparseable"):
            logger.warning(
                "[reason] JSON unparseable after fallbacks; preserving raw "
                "content as single reasoning_knowledge entry (len=%d)",
                len(content),
            )
            return {
                "reasoning_knowledge": [content] if content else [],
                "parse_error": True,
                "raw_content_preview": content[:500],
            }
        rk = parsed.get("reasoning_knowledge")
        if not isinstance(rk, list):
            parsed["reasoning_knowledge"] = [str(rk)] if rk else []
        return parsed

    except Exception as e:  # noqa: BLE001
        return {"reasoning_knowledge": [], "error": f"Reasoning failed: {str(e)}"}

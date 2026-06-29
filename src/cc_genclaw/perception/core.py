"""Perception sub-LLM caller.

This replaces the legacy `tools.Intent_Analysis.intent_analyzer` dependency
for runtime use. It keeps the same behavior shape: one chat completion over
the perception system prompt plus user text / optional image, returning a raw
dict for `parse_perception()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config import load_genclaw_config
from ..llm import LLMClient
from .prompts import load_perception_prompt
from .utils import encode_image_data_uri, parse_json_response


def _build_user_content(user_input: str, user_image_path: Optional[str]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if user_input:
        content.append({"type": "text", "text": user_input})
    if user_image_path and Path(user_image_path).is_file():
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_data_uri(user_image_path)},
            }
        )
    return content or [{"type": "text", "text": ""}]


def run_perception(
    user_input: str,
    user_image_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run the perception sub-LLM and return its raw JSON object."""
    load_genclaw_config()
    system_prompt = load_perception_prompt()
    if not system_prompt.strip():
        raise RuntimeError("perception prompt is empty")

    response = LLMClient().create(
        messages=[{"role": "user", "content": _build_user_content(user_input, user_image_path)}],
        system=system_prompt,
    )
    message = response.get("choices", [{}])[0].get("message", {})
    content = str(message.get("content") or "").strip()
    return parse_json_response(content)

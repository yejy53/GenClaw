"""Small utilities for the perception sub-LLM."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any


def encode_image_data_uri(path: str) -> str:
    """Encode a local image file as a data URI for OpenAI multimodal input."""
    p = Path(path)
    mime, _ = mimetypes.guess_type(str(p))
    if not mime:
        mime = "image/png"
    with open(p, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating markdown fences."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Perception response JSON was not an object")
    return value

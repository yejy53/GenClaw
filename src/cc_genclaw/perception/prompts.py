"""Perception prompt loader."""

from __future__ import annotations

from pathlib import Path

import yaml

PERCEPTION_PROMPT_YAML = (
    Path(__file__).resolve().parents[1] / "prompts" / "perception_prompt.yaml"
)


def load_perception_prompt() -> str:
    """Read the `system_prompt` field from the src perception prompt YAML."""
    with open(PERCEPTION_PROMPT_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return str(data.get("system_prompt", "") or "")

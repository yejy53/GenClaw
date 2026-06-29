"""Capability YAML loader.

Each tool's prompt comes from prompts/tool_cards/<name>.yaml in the 7-section
format (what / when_to_use / when_not_to_use / usage / failure_modes /
cross_tool / examples). This loader renders them into the description string
that gets passed to the model.

The 7 sections are concatenated in a stable order with Markdown headers so
the model sees a clear, predictable structure (cc style).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


SECTION_ORDER = [
    "what",
    "when_to_use",
    "when_not_to_use",
    "usage",
    "failure_modes",
    "cross_tool",
    "examples",
]


_SECTION_TITLES = {
    "what": "## What",
    "when_to_use": "## When to use",
    "when_not_to_use": "## When NOT to use",
    "usage": "## Usage notes",
    "failure_modes": "## Failure modes",
    "cross_tool": "## Cross-tool constraints",
    "examples": "## Examples",
}


def render_capability(yaml_path: str | Path) -> str:
    """Read a capability YAML and render to a description string."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    parts: List[str] = []
    name = data.get("name", "(unnamed tool)")
    one_liner = data.get("description", "")
    if one_liner:
        parts.append(one_liner)

    for key in SECTION_ORDER:
        body = data.get(key)
        if not body:
            continue
        parts.append(_SECTION_TITLES[key])
        if isinstance(body, str):
            parts.append(body.strip())
        elif isinstance(body, list):
            for item in body:
                parts.append(f"- {item}".rstrip())
        else:
            parts.append(str(body))

    return "\n\n".join(parts).strip()


def _yaml_files(card_dir: Path):
    """List card yaml files, skipping macOS AppleDouble companions (._*)."""
    return [p for p in sorted(card_dir.glob("*.yaml")) if not p.name.startswith("._")]


def render_tools_listing(card_dir: str | Path) -> str:
    """Render a compact list of all tools (name + one-liner description) for
    inclusion in the main system prompt's `# Using your tools` section.
    """
    card_dir = Path(card_dir)
    out: List[str] = []
    for path in _yaml_files(card_dir):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
        name = data.get("name", path.stem)
        desc = data.get("description", "").splitlines()[0]
        out.append(f"- `{name}`: {desc}")
    return "\n".join(out)


def load_all_descriptions(card_dir: str | Path) -> Dict[str, str]:
    """Return {tool_name: rendered_description} for all yaml cards."""
    card_dir = Path(card_dir)
    out: Dict[str, str] = {}
    for path in _yaml_files(card_dir):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
        name = data.get("name", path.stem)
        out[name] = render_capability(path)
    return out


def load_short_descriptions(card_dir: str | Path) -> Dict[str, str]:
    """Return {tool_name: short_description} for cards that define one.

    Used to override a tool's short_description from yaml (Phase-6).
    """
    card_dir = Path(card_dir)
    out: Dict[str, str] = {}
    for path in _yaml_files(card_dir):
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
        name = data.get("name", path.stem)
        sd = data.get("short_description")
        if sd:
            out[name] = sd.strip() if isinstance(sd, str) else str(sd)
    return out


# ---- helper to inject loaded descriptions into tool instances ----


def attach_descriptions_to_tools(tools: list, card_dir: str | Path) -> None:
    """Replace each tool's `.description` (and optionally `.short_description`)
    with values rendered from yaml cards.

    Tools that don't have a matching YAML card keep their default values.
    """
    descs = load_all_descriptions(card_dir)
    short_descs = load_short_descriptions(card_dir)
    for t in tools:
        if t.name in descs:
            t.description = descs[t.name]
        if t.name in short_descs:
            t.short_description = short_descs[t.name]

"""Load and template-fill the main system prompt.

The system.md template uses {{var}} placeholders for L2 Session info.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Optional


_DEFAULT_PATH = Path(__file__).parent / "system.md"


def render_system_prompt(
    cwd: Optional[str] = None,
    model: str = "gpt-5.4",
    template_path: Optional[Path] = None,
) -> str:
    """Render the main system prompt by filling L2 Session placeholders."""
    path = Path(template_path) if template_path else _DEFAULT_PATH
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    cwd = cwd or os.getcwd()

    substitutions = {
        "cwd": cwd,
        "platform": platform.system(),
        "shell": os.environ.get("SHELL", "unknown"),
        "os_version": platform.platform(),
        "session_date": time.strftime("%Y-%m-%d"),
        "model": model,
    }

    for k, v in substitutions.items():
        text = text.replace("{{" + k + "}}", str(v))

    return text


def render_system_prompt_with_tools(
    tool_card_dir: Path,
    cwd: Optional[str] = None,
    model: str = "gpt-5.4",
    template_path: Optional[Path] = None,
) -> str:
    """Same as render_system_prompt but appends a compact tool listing.

    Note: each tool's full 7-section description is sent via the OpenAI
    `tools` parameter (see messages.convert_tools_to_llm), so the tool
    listing in the system prompt is intentionally just a compact index.
    """
    base = render_system_prompt(cwd=cwd, model=model, template_path=template_path)
    from .capability_loader import render_tools_listing

    listing = render_tools_listing(tool_card_dir)
    return base + "\n\n## Available tools (full schemas in tools parameter)\n\n" + listing

"""FORMAT_PROMPT tool — compose trajectory context into a final image prompt."""
from __future__ import annotations

import json
from typing import Any, Dict

from ..runtime.artifacts import record_artifacts_in_dir
from ..runtime.session_dir import tool_output_dir
from .base import BaseTool
from ._format_prompt.core import compose_generation_prompt, write_artifacts


class FormatPromptTool(BaseTool):
    name = "format_prompt"
    description = ""
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            # New explicit trajectory brief fields.
            "original_user_prompt": {"type": "string"},
            "image_intent": {"type": "string"},
            "painter_thoughts": {"type": "string"},
            "planning_needs": {"type": "array"},
            "notes_for_planner": {"type": "array", "items": {"type": "string"}},
            "search_facts_summary": {"type": "string"},
            "verified_facts": {"type": "string"},
            "reasoning_output": {"type": "array", "items": {"type": "string"}},
            "draft_png_path": {"type": "string"},
            "text_template_path": {"type": "string"},
            "expected_texts_used": {"type": "array", "items": {"type": "string"}},
            "category": {"type": "string"},
            "render_type": {"type": "string"},
            "source_image_path": {"type": "string"},
            "target_tool": {"type": "string", "enum": ["", "t2i", "i2i"]},
            # Backward-compatible legacy fields.
            "base_prompt": {"type": "string"},
            "facts": {"type": "string"},
            "reasoning": {"type": "array", "items": {"type": "string"}},
            "reference_image_paths": {"type": "array", "items": {"type": "string"}},
            "user_image_path": {"type": "string"},
            "svg_description": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        has_anchor = any(
            str(args.get(key) or "").strip()
            for key in ("original_user_prompt", "base_prompt", "image_intent")
        )
        if not has_anchor:
            return self.err(
                "format_prompt requires 'original_user_prompt', 'base_prompt', "
                "or 'image_intent'."
            )

        try:
            payload = compose_generation_prompt(args)
        except Exception as e:  # noqa: BLE001
            return self.err(f"format_prompt failed: {type(e).__name__}: {e}")

        out_dir = tool_output_dir("format_prompt")
        try:
            write_artifacts(out_dir, payload, args)
            record_artifacts_in_dir(out_dir, tool_name="format_prompt")
        except Exception:
            pass

        return self.ok(json.dumps(payload, ensure_ascii=False))

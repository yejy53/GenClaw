"""i2i — generate a new image guided by an existing image (atomic tool).

Thin wrapper over ``tools._image_gen.generate_image``. The backend,
endpoint, and model are decided by ``_image_gen/providers.yaml`` — the
agent does not choose them.

D-15: image_path MUST be:
  1) present in args (not derived from any global state)
  2) an absolute path (Path(p).is_absolute())
  3) an existing file (Path(p).is_file())
On any violation we return is_error=True with a message that lists the
legal sources of image_path values, so the LLM learns to extract from a
previous tool_result rather than invent paths.

Args (from LLM):
- image_path: str (required, absolute, file must exist)
- prompt: str (required)
- reference_image_paths: List[str] (optional, extra multi-image guidance;
  do NOT repeat image_path here)
- quality: 'fast' | 'balanced' | 'best' (optional, reserved — no effect)

ToolResult.content (JSON):
- status:     'success' | 'failed'
- final_path: absolute path to the generated PNG (success only)
- error:      error message (failed only)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .base import BaseTool


_LEGAL_SOURCES_HINT = (
    "Legal sources for image_path (extract from a previous tool_result.content):\n"
    "  - search.reference_paths[i]          (when refining a web reference)\n"
    "  - code_scene_draft.draft_png_path    (when upgrading an SVG draft to photoreal)\n"
    "  - code_text_draft.final_path         (when restyling a long-text image)\n"
    "  - t2i.final_path                     (when iterating on a prior generation)\n"
    "  - i2i.final_path                     (when chaining multiple i2i refinements)\n"
    "  - user_image_path from painter notes (when the user uploaded an image)"
)


def _reference_label(index: int) -> str:
    return chr(ord("A") + index) if 0 <= index < 26 else str(index + 1)


def _with_reference_header(prompt: str, ref_count: int) -> str:
    if ref_count < 2 or "Reference A" in prompt or "Reference use:" in prompt:
        return prompt

    lines = [
        "Reference use:",
        "- Reference A: the first image input and main canvas / composition anchor.",
    ]
    for index in range(1, ref_count):
        lines.append(
            f"- Reference {_reference_label(index)}: additional visual reference for identity, outfit, logo, detail, material, style, location, or object-specific traits."
        )
    return "\n".join(lines) + "\n\n" + prompt


class I2ITool(BaseTool):
    name = "i2i"
    description = ""  # filled from yaml card at runtime
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": (
                    "ABSOLUTE path to an existing image file. Must be extracted "
                    "from a previous tool_result.content — do NOT invent paths."
                ),
            },
            "prompt": {"type": "string"},
            "reference_image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Additional reference images for multi-image guidance "
                    "(do NOT include image_path here)."
                ),
            },
            "quality": {
                "type": "string",
                "enum": ["fast", "balanced", "best"],
                "description": "Reserved. Currently has no effect.",
            },
        },
        "required": ["image_path", "prompt"],
        "additionalProperties": False,
    }

    def _fail(self, error: str):
        return self.err(json.dumps({"status": "failed", "error": error}, ensure_ascii=False))

    def execute(self, args: Dict[str, Any], signal: Any = None):
        # D-15: image_path must be provided
        image_path = args.get("image_path")
        if not image_path or not isinstance(image_path, str):
            return self._fail(
                "i2i requires arg 'image_path' (absolute path to an existing image).\n"
                + _LEGAL_SOURCES_HINT
            )

        p = Path(image_path)

        # D-15: must be absolute
        if not p.is_absolute():
            return self._fail(
                f"i2i: image_path '{image_path}' is not an absolute path. "
                f"Use the absolute path from a previous tool_result.\n"
                + _LEGAL_SOURCES_HINT
            )

        # D-15: must be an existing file
        if not p.is_file():
            return self._fail(
                f"i2i: image_path '{image_path}' is not a valid file "
                f"(does not exist or is a directory).\n"
                + _LEGAL_SOURCES_HINT
            )

        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return self._fail("i2i requires a non-empty 'prompt'.")

        # quality is accepted for compatibility but intentionally ignored.
        quality = args.get("quality")
        if quality:
            import logging
            logging.getLogger("cc_genclaw.image_gen").info(
                "[i2i] quality=%r received but ignored (no effect).", quality
            )

        abs_image_path = str(p.resolve())
        extra_refs = [
            r for r in (args.get("reference_image_paths") or [])
            if r and str(r) != image_path
        ]
        # image_path always first; extra refs follow.
        refs = [abs_image_path] + [str(r) for r in extra_refs]
        generation_prompt = _with_reference_header(prompt, len(refs))

        from ._image_gen import generate_image

        try:
            final_path = generate_image(generation_prompt, refs)
        except Exception as e:  # noqa: BLE001
            return self._fail(f"{type(e).__name__}: {e}")

        return self.ok(
            json.dumps(
                {"status": "success", "final_path": final_path},
                ensure_ascii=False,
            )
        )

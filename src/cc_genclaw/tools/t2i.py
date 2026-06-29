"""t2i — generate a new image from a text prompt only (atomic tool).

Thin wrapper over ``tools._image_gen.generate_image``. The backend,
endpoint, and model are decided by ``_image_gen/providers.yaml`` — the
agent does not choose them.

Args (from LLM):
- prompt: str (required) — image description
- quality: 'fast' | 'balanced' | 'best' (optional, reserved — no effect)

ToolResult.content (JSON):
- status:     'success' | 'failed'
- final_path: absolute path to the generated PNG (success only)
- error:      error message (failed only)
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .base import BaseTool


class T2ITool(BaseTool):
    name = "t2i"
    description = ""  # filled from yaml card at runtime
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Text description of the image to generate.",
            },
            "quality": {
                "type": "string",
                "enum": ["fast", "balanced", "best"],
                "description": "Reserved. Currently has no effect.",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        prompt = args.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            return self.err(
                json.dumps(
                    {"status": "failed", "error": "t2i requires a non-empty 'prompt'."},
                    ensure_ascii=False,
                )
            )

        # quality is accepted for compatibility but intentionally ignored.
        quality = args.get("quality")
        if quality:
            import logging
            logging.getLogger("cc_genclaw.image_gen").info(
                "[t2i] quality=%r received but ignored (no effect).", quality
            )

        from ._image_gen import generate_image

        try:
            final_path = generate_image(prompt)
        except Exception as e:  # noqa: BLE001
            return self.err(
                json.dumps(
                    {"status": "failed", "error": f"{type(e).__name__}: {e}"},
                    ensure_ascii=False,
                )
            )

        return self.ok(
            json.dumps(
                {"status": "success", "final_path": final_path},
                ensure_ascii=False,
            )
        )

"""REASON tool — general-purpose LLM reasoning before image tools.

A single, simple reasoning call: give it a prompt (optionally with images
to interpret) and it thinks the problem through, returning the conclusions
as plain text. Use it whenever a request is not directly paintable as-is
and needs prior thinking — multi-hop fact chains, geometry/physics
derivation, interpreting a supplied figure, deciding what to search for,
or any other reasoning the downstream image tools cannot do reliably on
their own. There are no modes: describe what you want in `prompt`.

Args:
- prompt: str (required). The reasoning question.
- reference_image_paths: List[str] (optional, multi-modal grounding).
- user_image_path: str (optional, multi-modal user input).

ToolResult.content (JSON):
- reasoning_output: List[str] — the reasoning conclusions
- status: 'success'|'failed'
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..runtime.artifacts import record_artifacts_in_dir
from ..runtime.session_dir import tool_output_dir
from .base import BaseTool
from ._reason import run_reasoning


class ReasonTool(BaseTool):
    name = "reason"
    description = ""
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "reference_image_paths": {"type": "array", "items": {"type": "string"}},
            "user_image_path": {"type": "string"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        from cc_genclaw.config import load_genclaw_config
        load_genclaw_config()

        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return self.err("reason requires non-empty 'prompt'")

        user_image = str(args.get("user_image_path", "") or "")
        refs = list(args.get("reference_image_paths") or [])

        try:
            raw = run_reasoning(
                prompt=prompt,
                user_image_path=user_image or None,
                reference_image_paths=refs or None,
            )
        except Exception as e:  # noqa: BLE001
            return self.err(f"reason: run_reasoning failed: {type(e).__name__}: {e}")

        if not isinstance(raw, dict):
            raw = {}

        def _as_list(v) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            if isinstance(v, str) and v.strip():
                return [v]
            return []

        reasoning = _as_list(raw.get("reasoning_knowledge"))

        out_dir = tool_output_dir("reason")
        try:
            Path(out_dir, "meta.json").write_text(json.dumps({
                "raw": raw, "prompt_preview": prompt[:1000]
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            record_artifacts_in_dir(out_dir, tool_name="reason")
        except Exception:
            pass

        payload: Dict[str, Any] = {
            "reasoning_output": reasoning,
            "status": "success" if reasoning else "failed",
        }
        if payload["status"] != "success":
            payload["error"] = "reason returned empty reasoning_output"

        return self.ok(json.dumps(payload, ensure_ascii=False)) if payload["status"] == "success" \
            else self.err(json.dumps(payload, ensure_ascii=False))

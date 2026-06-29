"""CODE_SCENE_DRAFT tool — code-draws SVG geometry (no text rendering).

Thin wrapper over ``tools._code_scene_draft.draft_scene``. The whole SVG
pipeline (generate -> render -> review -> revise) lives self-contained in the
``_code_scene_draft`` package, configured via env (OPENAI_* + optional
SCENE_DRAFT_* model overrides).

The i2i refine is intentionally NOT done here: the agent should call the
separate ``i2i`` tool with ``image_path=draft_png_path`` to upgrade to
photoreal.

Args:
- prompt: str (required). What to draft.
- self_review: bool = True. Whether to run the VLM self-review + revise loop.

ToolResult.content (JSON):
- draft_png_path: str (absolute)
- svg_description: str (layout JSON string, caption for downstream i2i)
- review_verdict: 'PASS'|'OPTIMIZE'|'FAIL'|'SKIPPED'
- review_passed: bool
- status: 'success'|'failed'
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..runtime.artifacts import record_artifact, record_artifacts_in_dir
from ..runtime.session_dir import tool_output_dir
from .base import BaseTool


class CodeSceneDraftTool(BaseTool):
    name = "code_scene_draft"
    description = ""  # filled from yaml card at runtime
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "self_review": {"type": "boolean", "default": True},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        from cc_genclaw.config import load_genclaw_config
        load_genclaw_config()  # prime OPENAI_* into os.environ

        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return self.err("code_scene_draft requires non-empty 'prompt'")
        self_review = bool(args.get("self_review", True))

        from ._code_scene_draft import draft_scene

        out_dir = tool_output_dir("code_scene_draft")
        try:
            inner = draft_scene(prompt=prompt, output_dir=out_dir, self_review=self_review)
        except Exception as e:  # noqa: BLE001
            return self.err(
                f"code_scene_draft: draft_scene failed: {type(e).__name__}: {e}"
            )

        if not isinstance(inner, dict):
            inner = {}

        draft_path = inner.get("draft_png_path") or ""
        if draft_path:
            draft_path = str(Path(draft_path).resolve())

        svg_desc = str(inner.get("svg_description", "") or "")
        verdict = str(inner.get("review_verdict", "SKIPPED"))
        passed = bool(inner.get("review_passed", verdict in ("PASS", "OPTIMIZE", "SKIPPED")))
        status = "success" if draft_path and Path(draft_path).is_file() else "failed"

        try:
            Path(out_dir, "meta.json").write_text(json.dumps({
                "prompt": prompt,
                "self_review": self_review,
                "review_verdict": verdict,
                "review_feedback": str(inner.get("review_feedback", ""))[:1000],
                "draft_png_path": draft_path,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            record_artifacts_in_dir(out_dir, tool_name="code_scene_draft")
            if draft_path:
                record_artifact(
                    draft_path,
                    tool_name="code_scene_draft",
                    role="draft",
                    kind="image",
                    label="Structural draft",
                    out_dir=out_dir,
                )
        except Exception:
            pass

        payload: Dict[str, Any] = {
            "draft_png_path": draft_path,
            "svg_description": svg_desc[:2000],
            "review_verdict": verdict,
            "review_passed": passed,
            "status": status,
        }
        if status != "success":
            payload["error"] = inner.get("error", "no draft_png produced")

        body = json.dumps(payload, ensure_ascii=False)
        return self.ok(body) if status == "success" else self.err(body)

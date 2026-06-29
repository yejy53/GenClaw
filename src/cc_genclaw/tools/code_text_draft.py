"""CODE_TEXT_DRAFT tool — render exact long text into an image.

Wraps the self-contained ._code_text_draft.render_text_image(). The
module routes internally to 3 sub-pipelines (pure_code / layered /
embedded_svg), all of which now render text via HTML/CSS through a
headless Chromium (the legacy SVG/resvg path has been removed):
- pure_code: HTML design rendered straight to PNG.
- layered:   T2I background + HTML text overlay, one screenshot.
- embedded:  JSON plan -> HTML blueprint reference for downstream i2i.
Scenes that are unsuitable for code-based rendering (complex curved/3D
surfaces, perspective distortion) are rejected with a guiding error
telling the agent to fall back to t2i or i2i.

For pure_code/layered, this is usually a one-step final output. For
embedded_svg, final_path is a text/structure reference image; follow the
returned next_step_hint and run format_prompt -> i2i for the final natural
image.

Args:
- prompt: str (required). Describes the image (style, background, etc.)
- expected_long_texts: List[str] (required, non-empty). Verbatim text
  strings that must appear in the final image.
- category: '' | 'menu' | 'scoreboard' | 'poster' | 'sign' (default '')
- gen_backend: 'gemini' | 'qwen' (default 'gemini')

ToolResult.content (JSON):
- final_path: str (absolute, this IS the final image — not a draft)
- render_type: 'pure_code'|'layered'|'embedded_svg' actually used
- expected_texts_used: List[str]
- status: 'success'|'failed'
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..runtime.artifacts import record_artifact, record_artifacts_in_dir
from ..runtime.session_dir import tool_output_dir
from .base import BaseTool


class CodeTextDraftTool(BaseTool):
    name = "code_text_draft"
    description = ""
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "expected_long_texts": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "Verbatim strings that must appear in the image. "
                    "At least 1 required (otherwise use t2i)."
                ),
            },
            "category": {
                "type": "string",
                "enum": ["", "menu", "scoreboard", "poster", "sign"],
                "default": "",
            },
            "gen_backend": {
                "type": "string",
                "enum": ["gemini", "qwen"],
                "default": "gemini",
            },
        },
        "required": ["prompt", "expected_long_texts"],
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        from cc_genclaw.config import load_genclaw_config
        load_genclaw_config()

        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return self.err("code_text_draft requires non-empty 'prompt'")
        expected = list(args.get("expected_long_texts") or [])
        expected = [str(s) for s in expected if str(s).strip()]
        if not expected:
            return self.err(
                "code_text_draft requires non-empty 'expected_long_texts'. "
                "If no specific text needs to be rendered, use 't2i' instead."
            )
        category = str(args.get("category", "") or "")
        gen_backend = str(args.get("gen_backend", "gemini") or "gemini")

        try:
            from ._code_text_draft import render_text_image
        except Exception as e:  # noqa: BLE001
            return self.err(f"code_text_draft: import failed: {e}")

        out_dir = tool_output_dir("code_text_draft")
        try:
            inner = render_text_image(
                prompt=prompt,
                expected_texts=expected,
                output_dir=out_dir,
                category=category,
                gen_backend=gen_backend,
            )
        except Exception as e:  # noqa: BLE001
            return self.err(
                f"code_text_draft: render_text_image failed: "
                f"{type(e).__name__}: {e}"
            )

        if not isinstance(inner, dict):
            inner = {}
        final_path = inner.get("final_path") or inner.get("final_image") or ""
        if final_path:
            final_path = str(Path(final_path).resolve())
        render_type = str(inner.get("render_type", "unknown"))
        status = "success" if final_path and Path(final_path).is_file() else "failed"
        run_log = inner.get("log") if isinstance(inner.get("log"), dict) else {}
        steps = run_log.get("steps", {}) if isinstance(run_log, dict) else {}

        def _image_item(name: str, role: str, label: str) -> Dict[str, Any] | None:
            path = Path(out_dir, name)
            if not path.is_file():
                return None
            return {
                "path": str(path.resolve()),
                "role": role,
                "label": label,
            }

        process_images: List[Dict[str, Any]] = []
        output_role = "final_image"
        requires_i2i = False
        if render_type == "layered":
            process_images.extend(filter(None, [
                _image_item("background.png", "background", "Generated background without text"),
                _image_item("final.png", "final", "Final background + exact text"),
            ]))
        elif render_type in {"embedded", "embedded_svg"}:
            output_role = "text_structure_reference"
            requires_i2i = True
            process_images.extend(filter(None, [
                _image_item("draft.png", "draft", "Text layout draft"),
                _image_item("final.png", "reference", "Text/structure reference for format_prompt → i2i"),
            ]))
        else:
            process_images.extend(filter(None, [
                _image_item("final.png", "final", "Final HTML-rendered image"),
            ]))

        process_summary: Dict[str, Any] = {
            "render_type": render_type,
            "canvas_decision": steps.get("canvas_decision") or steps.get("canvas_size"),
            "selected_attempt": steps.get("selected_attempt") or steps.get("selected_draft_attempt"),
            "review": steps.get("review"),
            "repair": steps.get("repair"),
        }
        if render_type == "layered":
            process_summary.update({
                "background_requested": (steps.get("gen_bg") or {}).get("requested_size"),
                "background_raw_size": steps.get("bg_size_raw"),
                "background_resize": steps.get("background_resize"),
                "background_final_size": steps.get("bg_size"),
            })
        elif render_type in {"embedded", "embedded_svg"}:
            layout_plan = steps.get("layout_plan") or {}
            process_summary.update({
                "surface_count": layout_plan.get("surface_count"),
                "supporting_object_count": layout_plan.get("supporting_object_count"),
                "output_role": output_role,
                "requires_i2i": requires_i2i,
            })

        next_step_hint: Dict[str, Any] | None = None
        if render_type in {"embedded", "embedded_svg"} and status == "success":
            next_step_hint = {
                "recommended": "format_prompt_then_i2i",
                "reason": (
                    "embedded_svg output is a structural/text reference for local "
                    "text carriers, not the final image and not a rigid template."
                ),
                "instruction": (
                    "Next call format_prompt, then call i2i using the returned "
                    "i2i_image_path_hint/final_prompt. Preserve exact text strings "
                    "and their object associations, but naturalize carrier size, "
                    "perspective, materials, and proportions. Do not rigidly copy "
                    "oversized flat text boxes from the draft."
                ),
                "format_prompt_args": {
                    "text_template_path": final_path,
                    "expected_texts_used": expected,
                    "render_type": render_type,
                    "category": category,
                    "target_tool": "i2i",
                },
            }

        try:
            Path(out_dir, "meta.json").write_text(json.dumps({
                "prompt": prompt,
                "expected_long_texts": expected,
                "category": category,
                "gen_backend": gen_backend,
                "render_type": render_type,
                "final_path": final_path,
                "process_summary": process_summary,
                "process_images": process_images,
                "next_step_hint": next_step_hint,
                "output_role": output_role,
                "requires_i2i": requires_i2i,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            record_artifacts_in_dir(out_dir, tool_name="code_text_draft")
            for image in process_images:
                record_artifact(
                    image["path"],
                    tool_name="code_text_draft",
                    role=image["role"],
                    kind="image",
                    label=image["label"],
                    out_dir=out_dir,
                    metadata={"render_type": render_type},
                )
        except Exception:
            pass

        payload: Dict[str, Any] = {
            "final_path": final_path,
            "render_type": render_type,
            "expected_texts_used": expected,
            "category": category,
            "process_summary": process_summary,
            "process_images": process_images,
            "output_role": output_role,
            "requires_i2i": requires_i2i,
            "review_passed": (steps.get("review") or {}).get("passed"),
            "review_feedback": (steps.get("review") or {}).get("feedback"),
            "repair_summary": steps.get("repair"),
            "status": status,
        }
        if next_step_hint:
            payload["next_step_hint"] = next_step_hint
        if status != "success":
            payload["error"] = inner.get("error", "no final image produced")

        body = json.dumps(payload, ensure_ascii=False)
        return self.ok(body) if status == "success" else self.err(body)

"""code_scene_draft pipeline (self-contained).

This tool produces a structural SVG draft PNG only (no i2i refinement step):
Photoreal
refinement is the job of the separate ``i2i`` tool, which the agent calls
explicitly with ``image_path=<draft_png_path>``.

Flow: generate SVG -> render PNG -> (optional) vision review with
  - OPTIMIZE: viewBox crop + re-render + re-review (up to 2 crops)
  - FAIL: revise + re-render + re-review (up to 3 rounds)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from . import generate as G
from .render import render_to_png

logger = logging.getLogger("cc_genclaw.scene_draft")

_MAX_CROP_ROUNDS = 2
_MAX_REVISE_ROUNDS = 3


def draft_scene(
    prompt: str,
    output_dir: str,
    self_review: bool = True,
) -> Dict[str, Any]:
    """Run the SVG draft pipeline. Returns a result dict.

    Keys: status, svg_code, svg_description (layout JSON string), layout
    (parsed dict|None), code_type, draft_png_path, review_passed,
    review_verdict, review_feedback, timing, error?.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total_t0 = time.time()
    result: Dict[str, Any] = {"status": "success", "timing": {}}

    try:
        # --- Step 1: generate SVG ---
        t0 = time.time()
        gen = G.generate(prompt)
        svg_code = gen["svg_code"]
        svg_description = gen["svg_description"]
        layout = gen.get("layout")
        code_type = gen["code_type"]
        result["timing"]["generate"] = round(time.time() - t0, 1)
        logger.info("[scene_draft] SVG generated: %d chars, type=%s", len(svg_code), code_type)

        svg_ext = "html" if code_type == "html" else "svg"
        (Path(output_dir) / f"step1.{svg_ext}").write_text(svg_code, encoding="utf-8")

        # --- Step 2: render PNG ---
        t0 = time.time()
        draft_png_path = str(Path(output_dir) / "step2_draft.png")
        render_to_png(svg_code, draft_png_path, code_type=code_type)
        result["timing"]["render"] = round(time.time() - t0, 1)

        # --- Step 3: vision review (+ OPTIMIZE / FAIL handling) ---
        review_passed = True
        review_verdict = "SKIPPED"
        review_feedback = ""

        if self_review:
            t0 = time.time()
            rev = G.review(prompt, draft_png_path, svg_code=svg_code)
            review_passed = rev["passed"]
            review_verdict = rev.get("verdict", "FAIL")
            review_feedback = rev.get("feedback") or ""

            # OPTIMIZE: crop the viewBox, re-render, re-review (bounded).
            crop_rounds = 0
            while (not review_passed) and review_verdict == "OPTIMIZE" and crop_rounds < _MAX_CROP_ROUNDS:
                crop_rounds += 1
                svg_code = G._apply_viewbox_crop(svg_code, rev["viewbox_params"])
                draft_png_path = str(Path(output_dir) / f"step3_crop{crop_rounds}.png")
                render_to_png(svg_code, draft_png_path, code_type=code_type)
                rev = G.review(prompt, draft_png_path, svg_code=svg_code)
                review_passed = rev["passed"]
                review_verdict = rev.get("verdict", "FAIL")
                review_feedback = rev.get("feedback") or ""
                if review_passed:
                    break
            # Structurally-correct-but-small after max crops -> accept.
            if (not review_passed) and review_verdict == "OPTIMIZE":
                review_passed = True
                review_feedback = ""

            # FAIL: revise, re-render, re-review (up to 3 rounds).
            revise_rounds = 0
            while (not review_passed) and review_verdict == "FAIL" and revise_rounds < _MAX_REVISE_ROUNDS:
                revise_rounds += 1
                logger.info("[scene_draft] FAIL revise round %d: %s", revise_rounds, review_feedback)
                revised = G.revise(prompt, svg_code, review_feedback)
                svg_code = revised["svg_code"]
                svg_description = revised["svg_description"]
                layout = revised.get("layout")
                code_type = revised["code_type"]

                draft_png_path = str(Path(output_dir) / f"step3_revise{revise_rounds}.png")
                render_to_png(svg_code, draft_png_path, code_type=code_type)
                rev = G.review(prompt, draft_png_path, svg_code=svg_code)
                review_passed = rev["passed"]
                review_verdict = rev.get("verdict", "FAIL")
                review_feedback = rev.get("feedback") or ""

                if review_passed:
                    break
                if review_verdict == "OPTIMIZE":
                    svg_code = G._apply_viewbox_crop(svg_code, rev["viewbox_params"])
                    draft_png_path = str(Path(output_dir) / f"step3_revise{revise_rounds}_crop.png")
                    render_to_png(svg_code, draft_png_path, code_type=code_type)
                    review_passed = True
                    review_verdict = "OPTIMIZE"
                    review_feedback = ""
                    break

            result["timing"]["review"] = round(time.time() - t0, 1)

        # --- archive layout JSON + final svg ---
        (Path(output_dir) / f"final.{('html' if code_type == 'html' else 'svg')}").write_text(
            svg_code, encoding="utf-8"
        )
        if svg_description:
            try:
                if layout is not None:
                    (Path(output_dir) / "layout.json").write_text(
                        json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                else:
                    (Path(output_dir) / "layout.json").write_text(svg_description, encoding="utf-8")
            except Exception:
                pass

        result.update({
            "svg_code": svg_code,
            "svg_description": svg_description,
            "layout": layout,
            "code_type": code_type,
            "draft_png_path": draft_png_path,
            "review_passed": review_passed,
            "review_verdict": review_verdict,
            "review_feedback": review_feedback,
        })
        result["timing"]["total"] = round(time.time() - total_t0, 1)
        logger.info("[scene_draft] complete in %ss", result["timing"]["total"])

    except Exception as e:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error("[scene_draft] failed: %s", e)

    return result

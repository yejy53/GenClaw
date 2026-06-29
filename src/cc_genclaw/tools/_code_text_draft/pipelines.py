"""
Text Rendering — Pipeline Implementations
Three HTML-based rendering pipelines:
- pure_code: HTML/CSS design rendered directly to PNG (text IS the design).
- layered:   T2I background + HTML text overlay (bg as CSS background), one screenshot.
- embedded:  JSON layout plan -> HTML blueprint draft -> Gemini I2I refinement.
All rendering is HTML via Playwright/Chromium (SVG/resvg path removed).
"""
from __future__ import annotations

import json
import math
import time
import shutil
from pathlib import Path

from .config import REVIEW_MODEL, load_prompt
from .router import (
    plan_canvas_size,
    classify_text_render_type,
    _build_retry_prompt,
)
from .utils import (
    _openai_chat_completion,
    _extract_first_json_object,
    _ensure_text_parts_cover_expected,
    _weighted_text_units,
    split_prompt_text_and_background,
)
from .layout import (
    _fallback_embedded_layout_plan,
    plan_embedded_layout,
    _validate_generated_html,
    _render_debug_html_from_layout_plan,
    _build_html_from_plan,
)
from .renderer import render_to_png, render_html_and_extract_text, inspect_html_geometry
from .review import review_text, review_html_layout
from .codegen import generate_code, generate_code_from_content
from .repair import repair_html_layout
from .backends import (
    t2i_generate,
)


def _format_geometry_feedback(geometry: dict, max_items: int = 4) -> str:
    issues = list((geometry or {}).get("issues") or [])
    if not issues:
        return ""
    parts = []
    for issue in issues[:max_items]:
        text = (issue.get("text") or "").replace("\n", " ").strip()
        if len(text) > 120:
            text = text[:117] + "..."
        parts.append(f"{issue.get('type', 'geometry_issue')}: {text}")
    more = "" if len(issues) <= max_items else f" (+{len(issues) - max_items} more)"
    return "DOM geometry issues: " + "; ".join(parts) + more


def _score_attempt(
    *,
    review: dict,
    geometry: dict | None = None,
    validation_issues: list[str] | None = None,
    missing_count: int = 0,
) -> float:
    """Lower is better. Used to keep the best failed attempt, not just the last."""
    issue_count = len((geometry or {}).get("issues") or [])
    validation_count = len(validation_issues or [])
    score = float(missing_count * 100 + validation_count * 80 + issue_count * 25)
    layout_usage = (geometry or {}).get("layout_usage") if isinstance(geometry, dict) else {}
    coverage = layout_usage.get("coverage") if isinstance(layout_usage, dict) else {}
    if isinstance(coverage, dict) and layout_usage.get("coverage_enforced", True):
        width_ratio = coverage.get("width_ratio")
        height_ratio = coverage.get("height_ratio")
        area_ratio = coverage.get("area_ratio")
        total_units = float(layout_usage.get("total_text_units") or 0)
        target_width = 0.70 if total_units >= 80 else 0.35
        target_height = 0.55 if total_units >= 80 else 0.20
        target_area = 0.24 if total_units >= 80 else 0.07
        if isinstance(width_ratio, (int, float)):
            score += max(0.0, target_width - float(width_ratio)) * 100
        if isinstance(height_ratio, (int, float)):
            score += max(0.0, target_height - float(height_ratio)) * 60
        if isinstance(area_ratio, (int, float)):
            score += max(0.0, target_area - float(area_ratio)) * 80
    if not review.get("passed"):
        score += 10
    return score


def _select_best_attempt(attempts: list[dict]) -> dict | None:
    if not attempts:
        return None
    passing = [a for a in attempts if a.get("passed")]
    pool = passing or attempts
    source_rank = {"repaired": 0, "generated": 1}
    return min(
        pool,
        key=lambda a: (
            float(a.get("score", 10_000)),
            source_rank.get(str(a.get("source") or ""), 2),
            int(a.get("attempt", 99)),
        ),
    )


def _device_scale_factor_for_canvas(width: int, height: int) -> int:
    """Avoid exploding memory on 3K/4K canvases."""
    return 1 if max(width, height) >= 2500 else 2


def _resolution_label(width: int, height: int) -> str:
    longest = max(width, height)
    if longest >= 3800:
        return "4K"
    if longest >= 1900:
        return "2K"
    return "high-resolution"


def _text_layout_recommendation(
    expected_texts: list[str],
    canvas_w: int,
    canvas_h: int,
    *,
    mode: str,
    prompt: str = "",
) -> str:
    """Deterministic layout hints so HTML generation starts with a balanced text plan."""
    total_units = sum(_weighted_text_units(str(text)) for text in expected_texts or [])
    if total_units < 12:
        return ""

    # Keep margins readable and leave room for titles/decorative background.
    x_pct = 8
    y_pct = 10
    w_pct = 84
    h_pct = 78
    box_w = canvas_w * (w_pct / 100.0)
    box_h = canvas_h * (h_pct / 100.0)
    line_height = 1.65
    target_fill = 0.80 if total_units >= 200 else (0.72 if total_units >= 80 else 0.55)
    estimated_font = math.sqrt(max(box_w * box_h * target_fill, 1.0) / max(total_units * line_height, 1.0))

    longest = max(canvas_w, canvas_h)
    if total_units >= 200:
        min_font = 52 if longest >= 3800 else (40 if longest >= 3000 else 28)
    elif total_units >= 80:
        min_font = 36 if longest >= 3800 else (28 if longest >= 2500 else 18)
    else:
        min_font = 14
    font_px = int(round(max(min_font, float(estimated_font))))

    contains_cjk = any("\u4e00" <= ch <= "\u9fff" for text in expected_texts or [] for ch in str(text))
    target_chars_per_line = 24 if contains_cjk else 34
    target_col_width = max(font_px * target_chars_per_line, 260)
    columns = max(1, min(6, int(round(box_w / target_col_width))))
    if total_units < 80:
        columns = min(columns, 3)
    if canvas_w / max(canvas_h, 1) > 1.25:
        columns = max(columns, 3 if total_units >= 200 else (2 if total_units >= 80 else 1))

    panel_guidance = ""
    if mode == "layered":
        panel_guidance = (
            "\n- For layered backgrounds, put long text on a translucent paper/scrim panel inside this region "
            "if the background is busy; keep the panel subtle but readable."
        )
    prompt_l = (prompt or "").lower()
    explicit_vertical = bool(
        any(token in prompt_l for token in ["竖排", "纵排", "从右到左", "自右向左", "vertical writing", "vertical text", "竖排书法"])
        or (("卷轴" in prompt_l or "长卷" in prompt_l or "scroll" in prompt_l) and any(token in prompt_l for token in ["竖排", "纵排", "书法", "calligraphy"]))
    )
    writing_guidance = (
        "- Writing flow: use normal horizontal text flow (`writing-mode: horizontal-tb`) and horizontal CSS columns. "
        "Do NOT use vertical-rl / upright glyph columns unless the user explicitly asks for vertical calligraphy, scroll, or portrait text."
        if not explicit_vertical
        else "- Writing flow: vertical writing is allowed because the prompt explicitly requests a vertical/scroll-like layout; still balance columns and fill the region."
    )

    return (
        "\n\nDETERMINISTIC TEXT LAYOUT RECOMMENDATION (follow unless the user explicitly requests a sparse layout):\n"
        f"- Estimated text volume: {total_units:.1f} display units.\n"
        f"- Main readable text region: left {x_pct}%, top {y_pct}%, width {w_pct}%, height {h_pct}% of the canvas.\n"
        f"- Target text coverage: fill roughly {int(target_fill * 100)}% of that region; avoid leaving the lower half or one side empty.\n"
        f"- Calculated body font size: {font_px}px. Use this as the starting font size; do NOT choose smaller upfront.\n"
        f"- Minimum acceptable body font size: {min_font}px. Only reduce below the calculated size after rendering/overflow pressure makes it necessary, and never below this minimum unless the text cannot otherwise fit.\n"
        f"- Suggested line-height: {line_height:.2f}; suggested horizontal CSS columns: {columns}; balance columns to similar heights.\n"
        f"{writing_guidance}\n"
        "- If text overflows, add/balance columns or slightly tighten line-height before shrinking below the calculated font size.\n"
        "- If text occupies much less than the region, increase font size above the calculated value, expand the region, or reduce unused margins."
        f"{panel_guidance}"
    )


def _resize_background_keep_aspect(bg_path: str, target_w: int, target_h: int, log: dict) -> tuple[int, int]:
    """Scale background up toward target size without crop or distortion."""
    from PIL import Image as PILImage

    path = Path(bg_path)
    with PILImage.open(path) as img:
        orig_w, orig_h = img.size
        scale = max(target_w / orig_w, target_h / orig_h, 1.0)
        # Keep background manageable. Rendering text over >4K backgrounds is costly.
        scale = min(scale, 4096 / max(orig_w, orig_h))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        resize_info = {
            "original": {"width": orig_w, "height": orig_h},
            "target": {"width": target_w, "height": target_h},
            "scale": scale,
            "mode": "none",
        }
        if (new_w, new_h) != (orig_w, orig_h):
            resized = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            resized.save(path)
            resize_info["mode"] = "scale_up_keep_aspect"
        resize_info["final"] = {"width": new_w, "height": new_h}
        log["steps"]["background_resize"] = resize_info
        return new_w, new_h


# ============================================================
# Path A: Pure Code (HTML/CSS → render → review)
# ============================================================

def _plan_pure_code_layout(prompt, expected_texts, canvas_w, canvas_h, category, case_dir, log):
    """CoT Step: generate a structured layout plan before writing HTML."""
    text_list = "\n".join(f'  {i+1}. "{t}"' for i, t in enumerate(expected_texts))
    plan_user_prompt = (
        f"Design brief: {prompt}\n\n"
        f"Required text strings (must appear exactly):\n{text_list}\n\n"
        f"Create a detailed visual layout plan for a {category} design on a {canvas_w}x{canvas_h} canvas."
    )
    print("  Step 0: Layout planning (CoT)...")
    t0 = time.time()
    plan_response = _openai_chat_completion(
        timeout=60.0,
        model=REVIEW_MODEL,
        messages=[
            {"role": "system", "content": load_prompt("pure_code_layout_plan").replace(
                "{canvas_width}", str(canvas_w)).replace(
                "{canvas_height}", str(canvas_h)).replace(
                "{category}", category or "generic")},
            {"role": "user", "content": plan_user_prompt},
        ],
        max_tokens=4096,
        temperature=0.4,
    )
    plan_time = time.time() - t0
    raw_plan = plan_response.choices[0].message.content.strip()

    layout_plan = None
    try:
        clean = _extract_first_json_object(raw_plan)
        layout_plan = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        print(f"  Layout plan parse failed, proceeding without plan")
        layout_plan = None

    if layout_plan:
        with open(case_dir / "layout_plan.json", "w", encoding="utf-8") as f:
            json.dump(layout_plan, f, indent=2, ensure_ascii=False)
        print(f"  Layout plan: {len(layout_plan.get('regions', []))} regions, "
              f"{len(layout_plan.get('decorations', []))} decorations ({plan_time:.1f}s)")

    log["steps"]["layout_plan"] = {
        "time": plan_time,
        "regions": len(layout_plan.get("regions", [])) if layout_plan else 0,
        "decorations": len(layout_plan.get("decorations", [])) if layout_plan else 0,
        "style": layout_plan.get("style", {}) if layout_plan else {},
    }
    return layout_plan


def run_pure_code(prompt, expected_texts, case_dir, log, category="generic"):
    """Path A: CoT layout plan → HTML/CSS → render + DOM extraction → review → retry."""
    print("  [Path A: pure_code]")
    canvas_plan = plan_canvas_size(prompt, expected_texts, category, "pure_code")
    canvas_w, canvas_h = canvas_plan["width"], canvas_plan["height"]
    log["steps"]["canvas_decision"] = canvas_plan
    log["steps"]["canvas_size"] = {"width": canvas_w, "height": canvas_h, "category": category}
    print(f"  Canvas: {canvas_w} x {canvas_h} ({category})")

    layout_plan = _plan_pure_code_layout(prompt, expected_texts, canvas_w, canvas_h, category, case_dir, log)
    layout_recommendation = _text_layout_recommendation(
        expected_texts,
        canvas_w,
        canvas_h,
        mode="pure_code",
        prompt=prompt,
    )

    text_list = "\n".join(f'  {i+1}. "{t}"' for i, t in enumerate(expected_texts))
    nav_guidance = ""
    if category in ("webpage", "slide"):
        nav_guidance = (
            "\n\nCRITICAL UI RULES:\n"
            "- All navigation items, footer links, tab labels, and action buttons MUST use "
            "VISIBLE TEXT LABELS — never use icon-only elements.\n"
            "- Minimum font size for any text is 14px. Footer and nav text must be at least 14px.\n"
            "- Use high-contrast colors for all text (dark text on light background or vice versa).\n"
            "- Ensure Chinese text uses 'Microsoft YaHei', 'PingFang SC', 'Noto Sans SC', sans-serif.\n"
        )

    plan_guidance = ""
    if layout_plan:
        style = layout_plan.get("style", {})
        bg = style.get("background", {})
        plan_parts = [
            f"\n\n**LAYOUT PLAN (follow this design closely):**",
            f"Visual mood: {style.get('mood', 'professional')}",
            f"Background: {bg.get('type', 'solid')} with colors {bg.get('colors', ['#FFFFFF'])}",
            f"Accent color: {style.get('accent_color', '#333')}",
        ]
        for region in layout_plan.get("regions", []):
            typo = region.get("typography", {})
            pos = region.get("position", {})
            plan_parts.append(
                f"- [{region.get('id', '?')}] \"{region.get('text', '')}\" at "
                f"({pos.get('x_pct', 0.5):.0%}, {pos.get('y_pct', 0.5):.0%}), "
                f"font {typo.get('font_size_px', 36)}px {typo.get('font_weight', '400')}, "
                f"color {typo.get('color', '#000')}, align {typo.get('alignment', 'left')}"
                f"{', vertical' if typo.get('transform') == 'vertical' else ''}"
            )
        for deco in layout_plan.get("decorations", []):
            plan_parts.append(
                f"- [decoration] {deco.get('type', '?')}: {deco.get('description', '')}, "
                f"color {deco.get('color', '#ccc')}"
            )
        summary = layout_plan.get("layout_summary", "")
        if summary:
            plan_parts.append(f"Summary: {summary}")
        plan_guidance = "\n".join(plan_parts)

    base_prompt = (
        f"Category: {category}\n"
        f"Canvas size: {canvas_w} x {canvas_h}\n"
        f"Original request: {prompt}\n\n"
        f"The following text strings MUST appear CHARACTER-PERFECT in the HTML output — "
        f"do NOT paraphrase, reword, or substitute any of them:\n{text_list}\n\n"
        "STRICT RULES:\n"
        "1. Every text string above MUST exist as a visible text node in the HTML DOM.\n"
        "2. Do NOT replace any text with icons, images, or SVG symbols.\n"
        "3. Do NOT rephrase, translate, or abbreviate any required text.\n"
        "4. All text must be fully visible within the viewport (no clipping, no overflow:hidden).\n"
        f"{nav_guidance}"
        f"{layout_recommendation}"
        f"{plan_guidance}\n\n"
        "Follow the layout plan above for positioning, colors, and decorative elements. "
        "Make the design look polished and visually engaging while keeping every required text fully visible."
    )

    attempts = []
    total_generate = 0.0
    total_render = 0.0
    total_review = 0.0
    final_review = {"passed": False, "feedback": "No review executed."}
    final_html = ""

    for attempt_idx in range(2):
        attempt_no = attempt_idx + 1
        print(f"  Generating HTML/CSS (attempt {attempt_no})...")
        prompt_for_attempt = base_prompt
        if attempt_idx > 0:
            prompt_for_attempt = _build_retry_prompt(
                base_prompt,
                final_review.get("feedback"),
                extra_context="Fix text clipping, visibility, and layout hierarchy issues in the regenerated HTML/CSS.",
            )

        t0 = time.time()
        result = generate_code(
            prompt_for_attempt,
            "html_pure_render",
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            category=category or "generic",
        )
        gen_time = time.time() - t0
        total_generate += gen_time
        final_html = result["code"]

        attempt_code = case_dir / f"code_attempt{attempt_no}.html"
        with open(attempt_code, "w", encoding="utf-8") as f:
            f.write(final_html)

        print(f"  Rendering + extracting DOM text (attempt {attempt_no})...")
        t0 = time.time()
        attempt_png = case_dir / f"final_attempt{attempt_no}.png"
        dom_text = render_html_and_extract_text(
            final_html,
            str(attempt_png),
            viewport_width=canvas_w,
            viewport_height=canvas_h,
            device_scale_factor=_device_scale_factor_for_canvas(canvas_w, canvas_h),
        )
        render_time = time.time() - t0
        total_render += render_time

        attempt_dom = case_dir / f"dom_text_attempt{attempt_no}.txt"
        with open(attempt_dom, "w", encoding="utf-8") as f:
            f.write(dom_text)

        # DOM post-validation: check if expected_texts are present in DOM
        dom_lower = dom_text.lower()
        missing_in_dom = [t for t in expected_texts if t.lower() not in dom_lower]
        if missing_in_dom:
            print(f"  DOM missing {len(missing_in_dom)} texts, injecting...")
            inject_html_parts = []
            for mt in missing_in_dom:
                safe = mt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                inject_html_parts.append(
                    f'<div style="position:relative;margin:4px 12px;padding:4px 8px;'
                    f'font-size:15px;color:#222;font-family:Arial,\'Microsoft YaHei\',sans-serif;'
                    f'word-wrap:break-word">{safe}</div>'
                )
            inject_block = "\n".join(inject_html_parts)
            if "</body>" in final_html:
                final_html = final_html.replace(
                    "</body>", inject_block + "\n</body>", 1
                )
            else:
                final_html += "\n" + inject_block
            with open(attempt_code, "w", encoding="utf-8") as f:
                f.write(final_html)
            dom_text = render_html_and_extract_text(
                final_html, str(attempt_png),
                viewport_width=canvas_w, viewport_height=canvas_h,
                device_scale_factor=_device_scale_factor_for_canvas(canvas_w, canvas_h),
            )
            with open(attempt_dom, "w", encoding="utf-8") as f:
                f.write(dom_text)
            print(f"  Re-rendered after injection")

        print(f"  Checking DOM geometry (attempt {attempt_no})...")
        geometry = inspect_html_geometry(
            final_html,
            canvas_w,
            canvas_h,
            expected_texts=expected_texts,
            render_type="pure_code",
            category=category,
        )
        geometry_feedback = _format_geometry_feedback(geometry)
        if geometry_feedback:
            print(f"  Geometry: FAIL ({len(geometry.get('issues') or [])} issues)")
        else:
            print("  Geometry: PASS")

        print(f"  Reviewing HTML layout (attempt {attempt_no})...")
        t0 = time.time()
        final_review = review_html_layout(
            prompt,
            str(attempt_png),
            expected_texts,
            html_code=final_html,
            dom_text=dom_text,
        )
        review_time = time.time() - t0
        total_review += review_time
        if geometry_feedback:
            final_review = {
                "passed": False,
                "feedback": (
                    f"{geometry_feedback} "
                    f"{final_review.get('feedback') or ''}"
                ).strip(),
            }
        score = _score_attempt(
            review=final_review,
            geometry=geometry,
            missing_count=len(missing_in_dom),
        )

        attempts.append({
            "attempt": attempt_no,
            "generate_time": gen_time,
            "render_time": render_time,
            "review_time": review_time,
            "passed": final_review["passed"],
            "feedback": final_review["feedback"],
            "geometry": geometry,
            "score": score,
            "code_path": str(attempt_code),
            "png_path": str(attempt_png),
            "dom_path": str(attempt_dom),
        })
        print(f"  Review: {'PASS' if final_review['passed'] else 'FAIL'}")

        if final_review["passed"]:
            break

    best_attempt = _select_best_attempt(attempts)
    if best_attempt:
        shutil.copyfile(best_attempt["code_path"], case_dir / "code.html")
        shutil.copyfile(best_attempt["png_path"], case_dir / "final.png")
        shutil.copyfile(best_attempt["dom_path"], case_dir / "dom_text.txt")
        final_review = {
            "passed": bool(best_attempt.get("passed")),
            "feedback": best_attempt.get("feedback"),
        }
        log["steps"]["selected_attempt"] = {
            "attempt": best_attempt.get("attempt"),
            "source": best_attempt.get("source"),
            "score": best_attempt.get("score"),
            "passed": best_attempt.get("passed"),
        }

    log["steps"]["attempts"] = attempts
    log["steps"]["generate"] = {"status": "ok", "chars": len(final_html), "time": total_generate}
    log["steps"]["render"] = {"status": "ok", "time": total_render}
    log["steps"]["review"] = {
        "passed": final_review["passed"],
        "feedback": final_review["feedback"],
        "time": total_review,
    }


# ============================================================
# Path B: Layered (T2I background → HTML text overlay → one screenshot)
# ============================================================

def run_layered(prompt, expected_texts, case_dir, log, category=""):
    """Path B: split prompt → T2I background → embed bg as CSS background in HTML → one screenshot.

    The background image is generated first; the HTML canvas matches the background
    image size exactly and references it via the `__BG_IMAGE__` placeholder (replaced
    with a base64 data URI). Text is laid out over it by CSS, then rendered in a single
    Playwright screenshot. No transparent overlay + alpha composite step.
    """
    print("  [Path B: layered — HTML background-embed v3]")
    import base64
    from PIL import Image as PILImage

    canvas_plan = plan_canvas_size(prompt, expected_texts, category, "layered")
    target_w, target_h = canvas_plan["width"], canvas_plan["height"]
    log["steps"]["canvas_decision"] = canvas_plan

    # Step 1: Split prompt → text parts + background description
    print("  Step 1: Splitting prompt into text + background...")
    t0 = time.time()
    split = split_prompt_text_and_background(prompt)
    text_parts = _ensure_text_parts_cover_expected(split["text_parts"], expected_texts, category)
    bg_desc = split["background"]
    log["steps"]["split_prompt"] = {
        "text_parts": text_parts, "bg_description": bg_desc,
        "time": time.time() - t0,
    }
    print(f"    {len(text_parts)} text parts extracted")
    for tp in text_parts:
        print(f"      \"{tp['text']}\" → {tp['placement']}")
    print(f"    BG: {bg_desc[:100]}...")

    # Step 2: Generate background (Gemini T2I, no text)
    print("  Step 2: Generating background (Gemini T2I)...")
    t0 = time.time()
    bg_path = str(case_dir / "background.png")
    bg_gen_prompt = (
        f"{bg_desc}\n\n"
        f"Generate a {_resolution_label(target_w, target_h)} background suitable for a final "
        f"{target_w}x{target_h} text layout. Prefer the same general aspect ratio. "
        "Use rich visual texture and enough quiet negative space for later text overlay. "
        "CRITICAL: Do NOT include ANY text, letters, words, numbers, or typography in the image. "
        "Generate ONLY the visual scene/background. No readable characters whatsoever. "
        "Do NOT create subtitle bars, danmaku/comment boxes, speech bubbles, label panels, blank cards, "
        "UI containers, lower-third graphics, or any visible placeholder area meant to hold text. "
        "Any text box or caption treatment will be added later by HTML/CSS."
    )
    t2i_generate(bg_gen_prompt, bg_path)
    log["steps"]["gen_bg"] = {
        "status": "ok",
        "time": time.time() - t0,
        "requested_size": {"width": target_w, "height": target_h},
        "resolution_label": _resolution_label(target_w, target_h),
    }

    # Step 3: Canvas size follows the background image (no fixed canvas)
    bg_img = PILImage.open(bg_path)
    raw_bg_w, raw_bg_h = bg_img.size
    bg_img.close()
    log["steps"]["bg_size_raw"] = {"width": raw_bg_w, "height": raw_bg_h}
    canvas_w, canvas_h = _resize_background_keep_aspect(bg_path, target_w, target_h, log)
    print(f"  Step 3: Background size = {canvas_w} x {canvas_h}")
    log["steps"]["bg_size"] = {"width": canvas_w, "height": canvas_h}

    with open(bg_path, "rb") as f:
        bg_data_uri = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    text_desc_lines = "\n".join(
        f'- "{tp["text"]}" → {tp["placement"]}' for tp in text_parts
    )
    layout_recommendation = _text_layout_recommendation(
        expected_texts, canvas_w, canvas_h, mode="layered", prompt=prompt
    )
    base_text_prompt = (
        f"Place these text elements over a {canvas_w}x{canvas_h} background photo:\n"
        f"{text_desc_lines}\n\n"
        f"Original context: {prompt}\n"
        f"Background description (for choosing text color/contrast): {bg_desc}"
        f"{layout_recommendation}"
    )

    final_path = str(case_dir / "final.png")
    attempts = []
    repairs = []
    total_generate = 0.0
    total_render = 0.0
    total_review = 0.0
    total_repair = 0.0
    review = {"passed": False, "feedback": "No review executed."}
    html_code = ""
    dom_text = ""

    for attempt_idx in range(2):
        attempt_no = attempt_idx + 1
        prompt_for_attempt = base_text_prompt
        if attempt_idx > 0:
            prompt_for_attempt = _build_retry_prompt(
                base_text_prompt,
                review.get("feedback"),
                extra_context="Fix text visibility, contrast, clipping, or wrong/missing text in the regenerated HTML overlay.",
            )

        print(f"  Step 4: Generating text-overlay HTML ({canvas_w}x{canvas_h}, attempt {attempt_no})...")
        t0 = time.time()
        result = generate_code(
            prompt_for_attempt, "html_text_layered",
            canvas_width=canvas_w, canvas_height=canvas_h,
        )
        raw_html = result["code"]
        html_code = raw_html.replace("__BG_IMAGE__", bg_data_uri)
        total_generate += time.time() - t0
        attempt_html = case_dir / f"code_attempt{attempt_no}.html"
        with open(attempt_html, "w", encoding="utf-8") as f:
            f.write(html_code)

        print(f"  Step 5: Rendering HTML + extracting DOM (attempt {attempt_no})...")
        t0 = time.time()
        attempt_png = case_dir / f"final_attempt{attempt_no}.png"
        dom_text = render_html_and_extract_text(
            html_code, str(attempt_png),
            viewport_width=canvas_w, viewport_height=canvas_h,
            device_scale_factor=_device_scale_factor_for_canvas(canvas_w, canvas_h),
        )
        total_render += time.time() - t0

        print(f"  Step 6: Reviewing final image (DOM-aware, attempt {attempt_no})...")
        t0 = time.time()
        # Review sees the structural HTML with the bg placeholder, NOT the megabyte
        # base64 data URI (which would bloat the prompt and stall the request).
        html_for_review = raw_html.replace(
            bg_data_uri, "__BG_IMAGE__"
        ) if bg_data_uri in raw_html else raw_html
        review = review_html_layout(
            prompt, str(attempt_png), expected_texts,
            html_code=html_for_review, dom_text=dom_text,
        )
        total_review += time.time() - t0
        geometry = inspect_html_geometry(
            html_code,
            canvas_w,
            canvas_h,
            expected_texts=expected_texts,
            render_type="layered",
            category=category,
        )
        geometry_feedback = _format_geometry_feedback(geometry)
        if geometry_feedback:
            review = {
                "passed": False,
                "feedback": (
                    f"{geometry_feedback} "
                    f"{review.get('feedback') or ''}"
                ).strip(),
            }
            print(f"  Geometry: FAIL ({len(geometry.get('issues') or [])} issues)")
        else:
            print("  Geometry: PASS")
        score = _score_attempt(review=review, geometry=geometry)
        attempts.append({
            "attempt": attempt_no,
            "source": "generated",
            "passed": review["passed"],
            "feedback": review["feedback"],
            "geometry": geometry,
            "score": score,
            "code_path": str(attempt_html),
            "png_path": str(attempt_png),
            "dom_text": dom_text,
        })
        print(f"  Review: {'PASS' if review['passed'] else 'FAIL'}")

        if review["passed"]:
            break

        print(f"  Step 7: Repairing existing HTML layout (attempt {attempt_no})...")
        repair_t0 = time.time()
        repair_result = repair_html_layout(
            html_code=raw_html,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            expected_texts=expected_texts,
            geometry=geometry,
            review_feedback=review.get("feedback") or "",
            layout_recommendation=layout_recommendation,
        )
        repair_time = time.time() - repair_t0
        total_repair += repair_time
        repair_record = {
            "attempt": attempt_no,
            "status": repair_result.get("status"),
            "time": repair_time,
            "error": repair_result.get("error"),
            "summary": repair_result.get("repair_summary"),
        }

        if repair_result.get("status") != "success":
            repairs.append(repair_record)
            print(f"  Repair: FAIL ({repair_result.get('error')})")
            continue

        repaired_raw_html = str(repair_result.get("html_code") or "")
        repaired_html_code = repaired_raw_html.replace("__BG_IMAGE__", bg_data_uri)
        repaired_html = case_dir / f"code_repair_attempt{attempt_no}.html"
        repaired_png = case_dir / f"final_repair_attempt{attempt_no}.png"
        with open(repaired_html, "w", encoding="utf-8") as f:
            f.write(repaired_html_code)

        print(f"  Step 8: Rendering repaired HTML (attempt {attempt_no})...")
        t0 = time.time()
        repaired_dom_text = render_html_and_extract_text(
            repaired_html_code, str(repaired_png),
            viewport_width=canvas_w, viewport_height=canvas_h,
            device_scale_factor=_device_scale_factor_for_canvas(canvas_w, canvas_h),
        )
        repaired_render_time = time.time() - t0
        total_render += repaired_render_time

        print(f"  Step 9: Reviewing repaired HTML (attempt {attempt_no})...")
        t0 = time.time()
        repaired_review = review_html_layout(
            prompt, str(repaired_png), expected_texts,
            html_code=repaired_raw_html, dom_text=repaired_dom_text,
        )
        repaired_review_time = time.time() - t0
        total_review += repaired_review_time
        repaired_geometry = inspect_html_geometry(
            repaired_html_code,
            canvas_w,
            canvas_h,
            expected_texts=expected_texts,
            render_type="layered",
            category=category,
        )
        repaired_geometry_feedback = _format_geometry_feedback(repaired_geometry)
        if repaired_geometry_feedback:
            repaired_review = {
                "passed": False,
                "feedback": (
                    f"{repaired_geometry_feedback} "
                    f"{repaired_review.get('feedback') or ''}"
                ).strip(),
            }
            print(f"  Repaired geometry: FAIL ({len(repaired_geometry.get('issues') or [])} issues)")
        else:
            print("  Repaired geometry: PASS")

        repaired_score = _score_attempt(review=repaired_review, geometry=repaired_geometry)
        repair_record.update({
            "passed": repaired_review["passed"],
            "feedback": repaired_review["feedback"],
            "geometry_before": geometry,
            "geometry_after": repaired_geometry,
            "code_path": str(repaired_html),
            "png_path": str(repaired_png),
            "render_time": repaired_render_time,
            "review_time": repaired_review_time,
            "score": repaired_score,
        })
        repairs.append(repair_record)
        attempts.append({
            "attempt": attempt_no,
            "source": "repaired",
            "passed": repaired_review["passed"],
            "feedback": repaired_review["feedback"],
            "geometry": repaired_geometry,
            "score": repaired_score,
            "code_path": str(repaired_html),
            "png_path": str(repaired_png),
            "dom_text": repaired_dom_text,
        })
        review = repaired_review
        print(f"  Repair review: {'PASS' if repaired_review['passed'] else 'FAIL'}")

        if repaired_review["passed"]:
            break

    best_attempt = _select_best_attempt(attempts)
    if best_attempt:
        shutil.copyfile(best_attempt["code_path"], case_dir / "code.html")
        shutil.copyfile(best_attempt["png_path"], final_path)
        with open(case_dir / "dom_text.txt", "w", encoding="utf-8") as f:
            f.write(best_attempt.get("dom_text") or "")
        review = {
            "passed": bool(best_attempt.get("passed")),
            "feedback": best_attempt.get("feedback"),
        }
        log["steps"]["selected_attempt"] = {
            "attempt": best_attempt.get("attempt"),
            "source": best_attempt.get("source"),
            "score": best_attempt.get("score"),
            "passed": best_attempt.get("passed"),
        }

    log["steps"]["attempts"] = attempts
    log["steps"]["repairs"] = repairs
    log["steps"]["generate_text"] = {"status": "ok", "chars": len(html_code), "time": total_generate}
    log["steps"]["render"] = {"status": "ok", "time": total_render}
    log["steps"]["repair"] = {
        "status": (
            "passed"
            if any(item.get("passed") for item in repairs)
            else ("attempted" if repairs else "skipped")
        ),
        "attempt_count": len(repairs),
        "time": total_repair,
    }
    log["steps"]["review"] = {
        "passed": review["passed"], "feedback": review["feedback"],
        "time": total_review,
    }


# ============================================================
# Path C: Embedded (layout plan → HTML codegen → reviewed reference image)
# ============================================================

def run_embedded(prompt, expected_texts, case_dir, log, category="embedded"):
    """Embedded reference pipeline: layout planning → HTML codegen → review.

    The result is a text/structure reference image for the main agent to pass
    through format_prompt -> i2i. We intentionally do not run I2I here; doing so
    makes downstream naturalization preserve blueprint-like boxes too strongly.
    """
    print("  [Embedded: plan -> code]")
    canvas_plan = plan_canvas_size(prompt, expected_texts, category, "embedded")
    canvas_w, canvas_h = canvas_plan["width"], canvas_plan["height"]
    log["steps"]["canvas_decision"] = canvas_plan
    log["steps"]["canvas_size"] = {"width": canvas_w, "height": canvas_h, "category": category}
    print(f"  Canvas: {canvas_w} x {canvas_h} ({category})")

    print("  Step 1: Extracting text + placement...")
    t0 = time.time()
    split = split_prompt_text_and_background(prompt)
    text_parts = _ensure_text_parts_cover_expected(split["text_parts"], expected_texts, category)
    split_time = time.time() - t0
    log["steps"]["split_prompt"] = {
        "text_parts": text_parts,
        "time": split_time,
    }

    draft_attempts = []
    total_plan = 0.0
    total_generate = 0.0
    total_render = 0.0
    total_draft_review = 0.0
    draft_review = {"passed": False, "feedback": "No draft review executed."}
    final_layout_plan = None
    draft_code = ""
    draft_description = ""
    draft_path = case_dir / "draft.png"

    for attempt_idx in range(2):
        attempt_no = attempt_idx + 1
        t0 = time.time()
        print(f"  Step 2: Planning layout (attempt {attempt_no})...")
        layout_plan = plan_embedded_layout(
            prompt=prompt,
            expected_texts=expected_texts,
            text_parts=text_parts,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
            review_feedback=draft_review.get("feedback", "") if attempt_idx > 0 else "",
        )
        plan_time = time.time() - t0
        total_plan += plan_time
        final_layout_plan = layout_plan
        with open(case_dir / f"layout_plan_attempt{attempt_no}.json", "w", encoding="utf-8") as f:
            json.dump(layout_plan, f, indent=2, ensure_ascii=False)

        print(f"  Step 3: Generating HTML draft from plan (attempt {attempt_no})...")
        user_content = _build_html_from_plan(
            layout_plan,
            expected_texts,
            review_feedback=draft_review.get("feedback", "") if attempt_idx > 0 else "",
        )
        user_content += _text_layout_recommendation(
            expected_texts,
            canvas_w,
            canvas_h,
            mode="embedded",
            prompt=prompt,
        )
        t0 = time.time()
        result = generate_code_from_content(
            user_content,
            "html_from_plan",
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )
        gen_time = time.time() - t0
        total_generate += gen_time

        validation_issues = _validate_generated_html(result["code"], expected_texts)
        if result["code_type"] != "html":
            validation_issues.append("Generated output was not HTML.")

        used_fallback = False
        if validation_issues and attempt_idx == 1:
            used_fallback = True
            draft_code = _render_debug_html_from_layout_plan(layout_plan)
            draft_description = layout_plan.get("layout_summary", "") or "Fallback HTML draft from layout plan."
        elif validation_issues:
            draft_review = {"passed": False, "feedback": " ".join(validation_issues)}
            draft_attempts.append({
                "attempt": attempt_no,
                "planner_time": plan_time,
                "generate_time": gen_time,
                "render_time": 0.0,
                "review_time": 0.0,
                "passed": False,
                "feedback": draft_review["feedback"],
                "validation_issues": validation_issues,
                "used_fallback": False,
            })
            print("  Draft validation: FAIL")
            print(f"    Feedback: {draft_review['feedback']}")
            continue
        else:
            draft_code = result["code"]
            draft_description = result["description"] or layout_plan.get("layout_summary", "")

        attempt_code = case_dir / f"code_attempt{attempt_no}.html"
        with open(attempt_code, "w", encoding="utf-8") as f:
            f.write(draft_code)

        print(f"  Step 4: Rendering draft (attempt {attempt_no})...")
        t0 = time.time()
        attempt_draft = case_dir / f"draft_attempt{attempt_no}.png"
        render_to_png(
            draft_code,
            "html",
            str(attempt_draft),
            viewport_width=canvas_w,
            viewport_height=canvas_h,
        )
        render_time = time.time() - t0
        total_render += render_time

        print(f"  Step 4b: Checking DOM geometry (attempt {attempt_no})...")
        geometry = inspect_html_geometry(
            draft_code,
            canvas_w,
            canvas_h,
            expected_texts=expected_texts,
            render_type="embedded_svg",
            category=category,
        )
        geometry_feedback = _format_geometry_feedback(geometry)
        if geometry_feedback:
            print(f"  Geometry: FAIL ({len(geometry.get('issues') or [])} issues)")
        else:
            print("  Geometry: PASS")

        print(f"  Step 5: Reviewing draft (attempt {attempt_no})...")
        t0 = time.time()
        draft_review = review_text(prompt, str(attempt_draft), expected_texts, code=draft_code)
        review_time = time.time() - t0
        total_draft_review += review_time
        if geometry_feedback:
            draft_review = {
                "passed": False,
                "feedback": (
                    f"{geometry_feedback} "
                    f"{draft_review.get('feedback') or ''}"
                ).strip(),
            }
        score = _score_attempt(
            review=draft_review,
            geometry=geometry,
            validation_issues=validation_issues,
        )

        draft_attempts.append({
            "attempt": attempt_no,
            "planner_time": plan_time,
            "generate_time": gen_time,
            "render_time": render_time,
            "review_time": review_time,
            "passed": draft_review["passed"],
            "feedback": draft_review["feedback"],
            "validation_issues": validation_issues,
            "geometry": geometry,
            "score": score,
            "layout_plan_path": str(case_dir / f"layout_plan_attempt{attempt_no}.json"),
            "code_path": str(attempt_code),
            "png_path": str(attempt_draft),
            "description": draft_description,
            "used_fallback": used_fallback,
        })
        print(f"  Draft review: {'PASS' if draft_review['passed'] else 'FAIL'}")

        if draft_review["passed"]:
            break

    best_draft = _select_best_attempt(draft_attempts)
    if best_draft:
        try:
            shutil.copyfile(best_draft["layout_plan_path"], case_dir / "layout_plan.json")
        except Exception:
            with open(case_dir / "layout_plan.json", "w", encoding="utf-8") as f:
                json.dump(final_layout_plan or {}, f, indent=2, ensure_ascii=False)
        shutil.copyfile(best_draft["code_path"], case_dir / "code.html")
        shutil.copyfile(best_draft["png_path"], draft_path)
        draft_description = best_draft.get("description") or draft_description
        draft_review = {
            "passed": bool(best_draft.get("passed")),
            "feedback": best_draft.get("feedback"),
        }
        log["steps"]["selected_draft_attempt"] = {
            "attempt": best_draft.get("attempt"),
            "score": best_draft.get("score"),
            "passed": best_draft.get("passed"),
        }

    if final_layout_plan is None:
        final_layout_plan = _fallback_embedded_layout_plan(text_parts, expected_texts, canvas_w, canvas_h)
    log["steps"]["layout_plan"] = {
        "surface_count": len(final_layout_plan.get("text_surfaces", [])),
        "supporting_object_count": len(final_layout_plan.get("supporting_objects", [])),
        "text_surfaces": final_layout_plan.get("text_surfaces", []),
        "supporting_objects": final_layout_plan.get("supporting_objects", []),
    }

    reference_path = case_dir / "final.png"
    if draft_path.exists():
        shutil.copyfile(draft_path, reference_path)

    log["steps"]["draft_attempts"] = draft_attempts
    log["steps"]["plan"] = {"status": "ok", "time": total_plan}
    log["steps"]["generate"] = {"status": "ok", "chars": len(draft_code), "time": total_generate}
    log["steps"]["render"] = {"status": "ok", "time": total_render}
    log["steps"]["draft_review"] = {
        "passed": draft_review["passed"],
        "feedback": draft_review["feedback"],
        "time": total_draft_review,
    }
    log["steps"]["i2i"] = {
        "status": "skipped",
        "reason": "embedded_svg returns a text/structure reference; main agent should call format_prompt -> i2i",
    }
    log["steps"]["review"] = {
        "passed": draft_review["passed"],
        "feedback": draft_review["feedback"],
        "time": total_draft_review,
        "target": "embedded_reference",
    }
    log["steps"]["reference_output"] = {
        "path": str(reference_path),
        "source_draft_path": str(draft_path),
        "requires_i2i": True,
        "recommended_next": "format_prompt -> i2i",
    }


# ============================================================
# Router: run_single_case
# ============================================================

def run_single_case(case, output_dir):
    case_id = case["id"]
    category = case.get("category", "")
    prompt = case["prompt"]
    expected_texts = case.get("expected_texts", case.get("text", []))
    case_dir = Path(output_dir) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    log = {"case_id": case_id, "category": category, "prompt": prompt, "expected_texts": expected_texts, "steps": {}}

    print(f"\n{'='*60}")
    print(f"Case: {case_id}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'='*60}")

    print("  Classifying text render type...")
    t0 = time.time()
    render_type = classify_text_render_type(prompt)
    log["render_type"] = render_type
    log["steps"]["classify"] = {"type": render_type, "time": time.time() - t0}
    print(f"  → {render_type}")

    try:
        if render_type == "pure_code":
            run_pure_code(prompt, expected_texts, case_dir, log, category=category)
        elif render_type == "layered":
            run_layered(prompt, expected_texts, case_dir, log, category=category)
        elif render_type in ("embedded_svg", "embedded"):
            log["pipeline_variant"] = "embedded_plan_then_code"
            run_embedded(prompt, expected_texts, case_dir, log, category=category)
        else:
            raise ValueError(
                f"The text_render_type '{render_type}' involves highly complex curved/3D surfaces, labels, or perspective distortion "
                "that is not suitable for code-based text rendering. Please fallback to 't2i' (for general text generation) or "
                "'i2i' (for image-to-image styling) instead."
            )
    except Exception as e:
        print(f"  ERROR: {e}")
        log["error"] = str(e)

    with open(case_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n  📁 Output files:")
    for f in sorted(case_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            label = ""
            if f.name == "final.png":
                label = " ← FINAL"
            elif f.name == "log.json":
                label = " (log)"
            print(f"     {f.name:30s}  {size/1024:>7.1f} KB{label}")
    print(f"  📂 Dir: {case_dir}")
    return log

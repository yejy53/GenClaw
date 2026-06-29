"""
One-call entry point for code_text_draft.
"""
from __future__ import annotations

from pathlib import Path

from .config import set_gen_backend
from .router import classify_text_render_type as classify  # noqa: F401  re-exported via package __init__
from .pipelines import run_single_case


def render_text_image(
    prompt: str,
    expected_texts: list[str],
    output_dir: str | Path,
    category: str = "",
    gen_backend: str = "gemini",
) -> dict:
    """
    One-call entry point for text rendering.

    Args:
        prompt:         Scene description with quoted text to render.
        expected_texts: Ground-truth text strings that must appear in the image.
        output_dir:     Directory for intermediate and final outputs.
        category:       Optional hint (e.g. "sign", "caption", "dialogue").
        gen_backend:    "gemini" or "qwen" for image generation backend.

    Returns:
        dict with keys: case_id, render_type, final_path, log, ...
        For embedded_svg, final_path is a text/structure reference image that
        still requires downstream format_prompt -> i2i to become the final image.
    """
    set_gen_backend(gen_backend)

    case = {
        "id": Path(output_dir).name,
        "prompt": prompt,
        "expected_texts": expected_texts,
        "category": category,
    }

    log = run_single_case(case, Path(output_dir).parent)

    final_path = Path(output_dir) / "final.png"
    return {
        "case_id": case["id"],
        "render_type": log.get("render_type", "unknown"),
        "final_path": str(final_path) if final_path.exists() else None,
        "error": log.get("error"),
        "log": log,
    }

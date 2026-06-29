"""SVG/HTML -> PNG rendering (self-contained).

Playwright-based, wrapped in a ThreadPoolExecutor so it is safe to call from
async contexts.

Dynamic canvas: the viewport size is parsed from the SVG's declared
width/height (falling back to viewBox, then 1024x1024), so non-square drafts
render at their natural aspect ratio.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
import concurrent.futures
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("cc_genclaw.scene_draft")

_MAX_SIDE = 2048  # clamp viewport so a runaway size can't explode the screenshot
_MIN_SIDE = 64


def parse_canvas_size(code: str, default: Tuple[int, int] = (1024, 1024)) -> Tuple[int, int]:
    """Best-effort extract (width, height) from SVG/HTML code.

    Priority: explicit width/height on the first <svg> tag -> viewBox w/h ->
    ``default``. Values are clamped to [_MIN_SIDE, _MAX_SIDE].
    """
    svg_tag = re.search(r"<svg\b[^>]*>", code, re.IGNORECASE)
    tag = svg_tag.group(0) if svg_tag else code

    def _num(attr: str):
        m = re.search(rf'{attr}\s*=\s*["\']?\s*([0-9]+(?:\.[0-9]+)?)', tag, re.IGNORECASE)
        return float(m.group(1)) if m else None

    w = _num("width")
    h = _num("height")

    if not w or not h:
        vb = re.search(r'viewBox\s*=\s*["\']\s*[-0-9.]+\s+[-0-9.]+\s+([0-9.]+)\s+([0-9.]+)', tag, re.IGNORECASE)
        if vb:
            w = w or float(vb.group(1))
            h = h or float(vb.group(2))

    if not w or not h:
        return default

    wi = max(_MIN_SIDE, min(int(round(w)), _MAX_SIDE))
    hi = max(_MIN_SIDE, min(int(round(h)), _MAX_SIDE))
    return wi, hi


def _wrap_svg_in_html(svg_code: str) -> str:
    """Wrap standalone SVG in an HTML page for Playwright rendering."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{
    margin: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background: white;
}}
svg {{
    max-width: 100%;
    max-height: 100vh;
}}
</style>
</head><body>
{svg_code}
</body></html>"""


def _render_to_png_sync(
    code: str,
    output_path: str,
    code_type: str,
    width: int,
    height: int,
    timeout: int,
) -> str:
    from playwright.sync_api import sync_playwright

    html_content = code if code_type == "html" else _wrap_svg_in_html(code)

    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tmp_html = os.path.join(
        tempfile.gettempdir(), f"scene_draft_render_{int(time.time() * 1000)}.html"
    )
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                timeout=timeout,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                ],
            )
            page = browser.new_page(viewport={"width": width, "height": height})
            page.set_default_timeout(timeout)
            page.goto(f"file://{tmp_html}", timeout=timeout, wait_until="networkidle")
            page.wait_for_timeout(1000)
            page.screenshot(path=output_path, full_page=False, timeout=timeout)
            browser.close()
    finally:
        if os.path.exists(tmp_html):
            try:
                os.remove(tmp_html)
            except Exception:
                pass

    return output_path


def render_to_png(
    code: str,
    output_path: str,
    code_type: str = "svg",
    width: int = 0,
    height: int = 0,
    timeout: int = 30000,
) -> str:
    """Render SVG or HTML code to a PNG image.

    When ``width``/``height`` are 0 (default) the viewport is derived from the
    SVG's declared size via :func:`parse_canvas_size`, so drafts render at
    their natural aspect ratio rather than a forced square.
    """
    if not width or not height:
        width, height = parse_canvas_size(code)

    start_time = time.time()
    logger.info("[scene_draft] render -> %s (%s, %dx%d)", output_path, code_type, width, height)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _render_to_png_sync, code, output_path, code_type, width, height, timeout
        )
        result = future.result(timeout=90)

    logger.info("[scene_draft] render done in %.2fs: %s", time.time() - start_time, result)
    return result

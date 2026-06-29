"""
Text Rendering — Rendering Engine
Playwright/Chromium-based HTML-to-PNG rendering.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
import time


def render_to_png(code: str, code_type: str, output_path: str,
                  viewport_width: int = 1024, viewport_height: int = 1024,
                  transparent: bool = False,
                  device_scale_factor: int = 2) -> str:
    """Render a self-contained HTML document to PNG via headless Chromium.

    `code_type` is kept for call-site compatibility but is always treated as HTML.
    """
    import concurrent.futures

    def _sync():
        from playwright.sync_api import sync_playwright
        html = code

        tmp = os.path.join(tempfile.gettempdir(), f"text_render_{int(time.time())}.html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=[
                    "--no-sandbox",
                    "--font-render-hinting=none",
                    "--disable-font-subpixel-positioning",
                    "--disable-lcd-text",
                ])
                page = browser.new_page(
                    viewport={"width": viewport_width, "height": viewport_height},
                    device_scale_factor=device_scale_factor,
                )
                page.goto(f"file://{tmp}", wait_until="networkidle")
                page.wait_for_timeout(1000)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=output_path, full_page=False,
                                omit_background=transparent)
                browser.close()

            if device_scale_factor > 1:
                from PIL import Image as _PILImg
                _img = _PILImg.open(output_path)
                target_size = (viewport_width, viewport_height)
                if _img.size != target_size:
                    if transparent and _img.mode == "RGBA":
                        import numpy as np
                        arr = np.array(_img, dtype=np.float64)
                        alpha = arr[:, :, 3:4] / 255.0
                        arr[:, :, :3] *= alpha
                        premul = _PILImg.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
                        resized = premul.resize(target_size, _PILImg.Resampling.LANCZOS)
                        arr2 = np.array(resized, dtype=np.float64)
                        a2 = arr2[:, :, 3:4] / 255.0
                        safe_a = np.where(a2 > 1e-6, a2, 1.0)
                        arr2[:, :, :3] /= safe_a
                        _img = _PILImg.fromarray(np.clip(arr2, 0, 255).astype(np.uint8), "RGBA")
                    else:
                        _img = _img.resize(target_size, _PILImg.Resampling.LANCZOS)
                    _img.save(output_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return output_path

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_sync).result(timeout=60)


def render_html_and_extract_text(
    html_code: str,
    output_path: str,
    viewport_width: int,
    viewport_height: int,
    device_scale_factor: int = 2,
) -> str:
    """Render HTML to PNG and return the DOM text.

    Renders at `device_scale_factor`x (supersampling) then LANCZOS-downscales to the
    requested canvas size, so text edges stay crisp instead of aliased/low-res.
    """
    import concurrent.futures

    def _sync():
        from playwright.sync_api import sync_playwright

        tmp = os.path.join(tempfile.gettempdir(), f"text_render_html_{int(time.time())}.html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html_code)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=[
                    "--no-sandbox",
                    "--font-render-hinting=none",
                    "--disable-font-subpixel-positioning",
                    "--disable-lcd-text",
                ])
                page = browser.new_page(
                    viewport={"width": viewport_width, "height": viewport_height},
                    device_scale_factor=device_scale_factor,
                )
                page.goto(f"file://{tmp}", wait_until="networkidle")
                page.wait_for_timeout(1000)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=output_path, full_page=False)
                dom_text = page.evaluate(
                    """() => {
                        const text = (document.body && document.body.innerText) || '';
                        return text.replace(/\\s+/g, ' ').trim();
                    }"""
                )
                browser.close()

            if device_scale_factor > 1:
                from PIL import Image as _PILImg
                _img = _PILImg.open(output_path)
                target_size = (viewport_width, viewport_height)
                if _img.size != target_size:
                    _img.resize(target_size, _PILImg.Resampling.LANCZOS).save(output_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return dom_text

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_sync).result(timeout=90)


def inspect_html_geometry(
    html_code: str,
    viewport_width: int,
    viewport_height: int,
    expected_texts: list[str] | None = None,
    render_type: str = "",
    category: str = "",
) -> dict:
    """Return conservative DOM geometry issues for rendered text nodes.

    This is a deterministic companion to the VLM reviewer. It only flags
    objective layout problems that a browser can measure reliably: viewport
    overflow, scroll overflow, clipped text boxes, and substantial text/text
    overlap. It intentionally avoids judging aesthetics or semantic placement.
    """
    import concurrent.futures

    def _sync():
        from playwright.sync_api import sync_playwright

        tmp = os.path.join(tempfile.gettempdir(), f"text_geom_{int(time.time())}.html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html_code)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=[
                    "--no-sandbox",
                    "--font-render-hinting=none",
                    "--disable-font-subpixel-positioning",
                    "--disable-lcd-text",
                ])
                page = browser.new_page(
                    viewport={"width": viewport_width, "height": viewport_height},
                    device_scale_factor=1,
                )
                page.goto(f"file://{tmp}", wait_until="networkidle")
                page.wait_for_timeout(300)
                report = page.evaluate(
                    r"""({vw, vh, expectedTexts, renderType, category}) => {
                        const issues = [];
                        const nodes = [];
                        const rectObj = (r) => ({
                            x: Math.round(r.x * 10) / 10,
                            y: Math.round(r.y * 10) / 10,
                            w: Math.round(r.width * 10) / 10,
                            h: Math.round(r.height * 10) / 10,
                            right: Math.round(r.right * 10) / 10,
                            bottom: Math.round(r.bottom * 10) / 10,
                        });
                        const rangeRectObj = (range) => {
                            const rects = Array.from(range.getClientRects()).filter((r) => r.width > 0 && r.height > 0);
                            if (!rects.length) return rectObj(range.getBoundingClientRect());
                            const left = Math.min(...rects.map((r) => r.left));
                            const top = Math.min(...rects.map((r) => r.top));
                            const right = Math.max(...rects.map((r) => r.right));
                            const bottom = Math.max(...rects.map((r) => r.bottom));
                            return rectObj({x: left, y: top, width: right - left, height: bottom - top, right, bottom});
                        };
                        const area = (r) => Math.max(0, r.w) * Math.max(0, r.h);
                        const intersects = (a, b) => {
                            const w = Math.max(0, Math.min(a.right, b.right) - Math.max(a.x, b.x));
                            const h = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y));
                            return {w, h, area: w * h};
                        };
                        const isVisible = (el) => {
                            const cs = window.getComputedStyle(el);
                            return cs.display !== 'none' && cs.visibility !== 'hidden' && Number(cs.opacity || 1) > 0.02;
                        };
                        const weightedUnits = (text) => {
                            let units = 0;
                            for (const ch of (text || '')) {
                                if (/[\u4e00-\u9fff]/.test(ch)) units += 1.1;
                                else if (/[A-Z]/.test(ch)) units += 0.72;
                                else if (/[a-z]/.test(ch)) units += 0.58;
                                else if (/[0-9]/.test(ch)) units += 0.56;
                                else if (/\s/.test(ch)) units += 0.32;
                                else units += 0.45;
                            }
                            return Math.max(units, text ? 1 : 0);
                        };
                        const pct = (value) => `${Math.round(value * 100)}%`;
                        const walker = document.createTreeWalker(
                            document.body || document.documentElement,
                            NodeFilter.SHOW_TEXT,
                            {
                                acceptNode(node) {
                                    if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
                                    const parent = node.parentElement;
                                    if (!parent || !isVisible(parent)) return NodeFilter.FILTER_REJECT;
                                    const tag = parent.tagName.toLowerCase();
                                    if (['script', 'style', 'noscript', 'template'].includes(tag)) return NodeFilter.FILTER_REJECT;
                                    return NodeFilter.FILTER_ACCEPT;
                                }
                            }
                        );
                        let node;
                        while ((node = walker.nextNode())) {
                            const parent = node.parentElement;
                            const range = document.createRange();
                            range.selectNodeContents(node);
                            const r = rangeRectObj(range);
                            const pr = rectObj(parent.getBoundingClientRect());
                            const text = node.nodeValue.replace(/\\s+/g, ' ').trim();
                            range.detach();
                            if (!text || r.w < 1 || r.h < 1) continue;
                            const cs = window.getComputedStyle(parent);
                            const idx = nodes.length;
                            nodes.push({
                                idx,
                                text,
                                units: weightedUnits(text),
                                rect: r,
                                parentRect: pr,
                                parentTag: parent.tagName.toLowerCase(),
                                fontSize: Number.parseFloat(cs.fontSize || '0') || 0,
                                writingMode: cs.writingMode || 'horizontal-tb',
                            });

                            if (r.x < -1 || r.y < -1 || r.right > vw + 1 || r.bottom > vh + 1) {
                                issues.push({type: 'viewport_overflow', text, rect: r});
                            }
                            const parentClips = /(hidden|clip|auto|scroll)/.test(`${cs.overflow} ${cs.overflowX} ${cs.overflowY}`);
                            if ((parent.scrollWidth > parent.clientWidth + 2 || parent.scrollHeight > parent.clientHeight + 2) && parentClips) {
                                issues.push({
                                    type: 'scroll_overflow',
                                    text,
                                    rect: r,
                                    scroll: {sw: parent.scrollWidth, cw: parent.clientWidth, sh: parent.scrollHeight, ch: parent.clientHeight},
                                });
                            }
                            const parentArea = pr.w * pr.h;
                            const viewportArea = vw * vh;
                            const parentLooksLikeCarrier = parentArea > 100 && parentArea < viewportArea * 0.92;
                            if (parentLooksLikeCarrier && (r.x < pr.x - 2 || r.y < pr.y - 2 || r.right > pr.right + 2 || r.bottom > pr.bottom + 2)) {
                                issues.push({type: 'text_outside_carrier', text, rect: r, parentRect: pr});
                            } else if (parentClips && (r.x < pr.x - 2 || r.y < pr.y - 2 || r.right > pr.right + 2 || r.bottom > pr.bottom + 2)) {
                                issues.push({type: 'parent_clip_risk', text, rect: r, parentRect: pr});
                            }
                        }
                        for (let i = 0; i < nodes.length; i++) {
                            for (let j = i + 1; j < nodes.length; j++) {
                                const a = nodes[i];
                                const b = nodes[j];
                                const hit = intersects(a.rect, b.rect);
                                if (hit.area <= 0) continue;
                                const smaller = Math.min(area(a.rect), area(b.rect));
                                if (smaller > 0 && hit.area / smaller > 0.18) {
                                    issues.push({
                                        type: 'text_overlap',
                                        text: `${a.text.slice(0, 80)} <> ${b.text.slice(0, 80)}`,
                                        rect: {a: a.rect, b: b.rect, overlap: Math.round(hit.area * 10) / 10},
                                    });
                                }
                            }
                        }
                        const layoutUsage = {
                            text_area_bbox: null,
                            body_text_bbox: null,
                            coverage: {width_ratio: 0, height_ratio: 0, area_ratio: 0},
                            font: {min_px: null, median_px: null},
                            body_font: {min_px: null, median_px: null, writing_modes: []},
                            carrier_usage: [],
                            total_text_units: 0,
                            coverage_enforced: false,
                            issues: [],
                        };
                        const expectedUnits = (expectedTexts || []).reduce((sum, text) => sum + weightedUnits(text), 0);
                        const domUnits = nodes.reduce((sum, item) => sum + item.units, 0);
                        const totalUnits = expectedUnits || domUnits;
                        layoutUsage.total_text_units = Math.round(totalUnits * 10) / 10;

                        const significant = nodes.filter((item) => item.units >= 2);
                        if (significant.length) {
                            const left = Math.min(...significant.map((item) => item.rect.x));
                            const top = Math.min(...significant.map((item) => item.rect.y));
                            const right = Math.max(...significant.map((item) => item.rect.right));
                            const bottom = Math.max(...significant.map((item) => item.rect.bottom));
                            const bbox = rectObj({x: left, y: top, width: right - left, height: bottom - top, right, bottom});
                            const widthRatio = bbox.w / Math.max(vw, 1);
                            const heightRatio = bbox.h / Math.max(vh, 1);
                            const areaRatio = (bbox.w * bbox.h) / Math.max(vw * vh, 1);
                            const centerXRatio = (bbox.x + bbox.w / 2) / Math.max(vw, 1);
                            const leftMarginRatio = bbox.x / Math.max(vw, 1);
                            const rightMarginRatio = Math.max(0, vw - bbox.right) / Math.max(vw, 1);
                            const fontSizes = significant
                                .map((item) => item.fontSize)
                                .filter((value) => Number.isFinite(value) && value > 0)
                                .sort((a, b) => a - b);
                            const medianFont = fontSizes.length ? fontSizes[Math.floor(fontSizes.length / 2)] : null;
                            const minFont = fontSizes.length ? fontSizes[0] : null;
                            layoutUsage.text_area_bbox = bbox;
                            layoutUsage.coverage = {
                                width_ratio: Math.round(widthRatio * 1000) / 1000,
                                height_ratio: Math.round(heightRatio * 1000) / 1000,
                                area_ratio: Math.round(areaRatio * 1000) / 1000,
                                center_x_ratio: Math.round(centerXRatio * 1000) / 1000,
                                left_margin_ratio: Math.round(leftMarginRatio * 1000) / 1000,
                                right_margin_ratio: Math.round(rightMarginRatio * 1000) / 1000,
                            };
                            layoutUsage.font = {
                                min_px: minFont === null ? null : Math.round(minFont * 10) / 10,
                                median_px: medianFont === null ? null : Math.round(medianFont * 10) / 10,
                            };
                            const bodyNodes = significant
                                .filter((item) => item.units >= Math.max(12, totalUnits * 0.03))
                                .sort((a, b) => b.units - a.units);
                            const bodyPool = bodyNodes.length ? bodyNodes : [significant.slice().sort((a, b) => b.units - a.units)[0]];
                            const bodyLeft = Math.min(...bodyPool.map((item) => item.rect.x));
                            const bodyTop = Math.min(...bodyPool.map((item) => item.rect.y));
                            const bodyRight = Math.max(...bodyPool.map((item) => item.rect.right));
                            const bodyBottom = Math.max(...bodyPool.map((item) => item.rect.bottom));
                            const bodyBbox = rectObj({
                                x: bodyLeft,
                                y: bodyTop,
                                width: bodyRight - bodyLeft,
                                height: bodyBottom - bodyTop,
                                right: bodyRight,
                                bottom: bodyBottom,
                            });
                            const bodyFontSizes = bodyPool
                                .map((item) => item.fontSize)
                                .filter((value) => Number.isFinite(value) && value > 0)
                                .sort((a, b) => a - b);
                            const bodyMedianFont = bodyFontSizes.length ? bodyFontSizes[Math.floor(bodyFontSizes.length / 2)] : null;
                            const bodyMinFont = bodyFontSizes.length ? bodyFontSizes[0] : null;
                            layoutUsage.body_text_bbox = bodyBbox;
                            layoutUsage.body_font = {
                                min_px: bodyMinFont === null ? null : Math.round(bodyMinFont * 10) / 10,
                                median_px: bodyMedianFont === null ? null : Math.round(bodyMedianFont * 10) / 10,
                                writing_modes: Array.from(new Set(bodyPool.map((item) => item.writingMode).filter(Boolean))),
                            };
                            layoutUsage.carrier_usage = significant.map((item) => {
                                const pr = item.parentRect;
                                const rw = pr.w > 0 ? item.rect.w / pr.w : 0;
                                const rh = pr.h > 0 ? item.rect.h / pr.h : 0;
                                return {
                                    text: item.text.slice(0, 80),
                                    text_rect: item.rect,
                                    carrier_rect: pr,
                                    coverage: {
                                        width_ratio: Math.round(rw * 1000) / 1000,
                                        height_ratio: Math.round(rh * 1000) / 1000,
                                    },
                                };
                            });

                            const longestSide = Math.max(vw, vh);
                            const renderTypeL = String(renderType || '').toLowerCase();
                            const categoryL = String(category || '').toLowerCase();
                            const embeddedLike = renderTypeL === 'embedded' || renderTypeL === 'embedded_svg';
                            const longFormCategory = ['document', 'poster', 'menu', 'scoreboard', 'article', 'poem'].includes(categoryL);
                            const enforceCanvasCoverage = !embeddedLike && (totalUnits >= 80 || (longFormCategory && totalUnits >= 20));
                            layoutUsage.coverage_enforced = enforceCanvasCoverage;
                            let thresholds = {
                                width: 0.20,
                                height: 0.08,
                                area: 0.015,
                                minMedianFont: longestSide >= 1900 ? 14 : 12,
                                label: 'text',
                            };
                            if (totalUnits >= 200) {
                                thresholds = {
                                    width: 0.62,
                                    height: 0.58,
                                    area: 0.28,
                                    minMedianFont: longestSide >= 3800 ? 52 : 40,
                                    label: 'long text',
                                };
                            } else if (totalUnits >= 80) {
                                thresholds = {
                                    width: 0.56,
                                    height: 0.50,
                                    area: 0.20,
                                    minMedianFont: longestSide >= 3800 ? 36 : 28,
                                    label: 'medium-long text',
                                };
                            } else if (totalUnits >= 20) {
                                thresholds = {
                                    width: 0.35,
                                    height: 0.20,
                                    area: 0.07,
                                    minMedianFont: longestSide >= 1900 ? 15 : 12,
                                    label: 'multi-line text',
                                };
                            }
                            const addLayoutIssue = (type, text) => {
                                const issue = {type, text, rect: bbox};
                                layoutUsage.issues.push(issue);
                                issues.push(issue);
                            };
                            if (enforceCanvasCoverage) {
                                if (widthRatio < thresholds.width) {
                                    addLayoutIssue(
                                        'text_underfilled_width',
                                        `${thresholds.label} uses only ${pct(widthRatio)} of canvas width; expand the main text container toward both sides and add/balance columns`
                                    );
                                }
                                if (totalUnits >= 80 && Math.abs(centerXRatio - 0.5) > 0.12) {
                                    addLayoutIssue(
                                        'text_off_center_horizontal',
                                        `${thresholds.label} is horizontally off-center (center at ${pct(centerXRatio)} of canvas; left margin ${pct(leftMarginRatio)}, right margin ${pct(rightMarginRatio)}); center the main text region and use both sides`
                                    );
                                }
                                if (totalUnits >= 80 && Math.abs(leftMarginRatio - rightMarginRatio) > 0.18) {
                                    addLayoutIssue(
                                        'text_unbalanced_side_margins',
                                        `${thresholds.label} side margins are unbalanced (left ${pct(leftMarginRatio)}, right ${pct(rightMarginRatio)}); distribute text across the canvas instead of one side`
                                    );
                                }
                                if (heightRatio < thresholds.height) {
                                    addLayoutIssue(
                                        'text_underfilled_height',
                                        `${thresholds.label} uses only ${pct(heightRatio)} of canvas height; distribute text across more of the canvas`
                                    );
                                }
                                if (areaRatio < thresholds.area) {
                                    addLayoutIssue(
                                        'text_underfilled_area',
                                        `${thresholds.label} occupies only ${pct(areaRatio)} of the canvas area; avoid tiny text blocks and large empty regions`
                                    );
                                }
                            }
                            if (bodyMedianFont !== null && bodyMedianFont < thresholds.minMedianFont) {
                                addLayoutIssue(
                                    'body_text_font_too_small',
                                    `${thresholds.label} body font median is ${Math.round(bodyMedianFont * 10) / 10}px; use at least ${thresholds.minMedianFont}px for the main body text or rebalance columns`
                                );
                            }
                        }
                        return {viewport: {width: vw, height: vh}, text_node_count: nodes.length, layout_usage: layoutUsage, issues};
                    }""",
                    {
                        "vw": viewport_width,
                        "vh": viewport_height,
                        "expectedTexts": expected_texts or [],
                        "renderType": render_type,
                        "category": category,
                    },
                )
                browser.close()
                return report
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_sync).result(timeout=60)

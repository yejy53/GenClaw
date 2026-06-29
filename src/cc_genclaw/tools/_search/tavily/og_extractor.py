"""Extract og:image / twitter:image / the first large img from page_url."""
from __future__ import annotations

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)


_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _DEFAULT_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_META_PATTERNS = [
    re.compile(r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::secure_url)?["\']', re.I),
    re.compile(r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)["\']', re.I),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image(?::src)?["\']', re.I),
    re.compile(r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']', re.I),
]

_IMG_PATTERN = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.I)


def _absolutize(url: str, base: str) -> str:
    """Handle protocol-relative / site-relative paths."""
    from urllib.parse import urljoin
    return urljoin(base, url)


def extract_og_image(
    page_url: str,
    *,
    timeout: int = 15,
    fallback_to_first_img: bool = True,
) -> Optional[str]:
    try:
        resp = requests.get(page_url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        logger.debug("[og_extractor] page fetch failed for %s: %s", page_url, e)
        return None

    html = resp.text
    if not html:
        return None

    for pat in _META_PATTERNS:
        m = pat.search(html)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return _absolutize(candidate, resp.url)

    if fallback_to_first_img:
        m = _IMG_PATTERN.search(html)
        if m:
            candidate = m.group(1).strip()
            if candidate and not candidate.startswith("data:"):
                return _absolutize(candidate, resp.url)

    return None

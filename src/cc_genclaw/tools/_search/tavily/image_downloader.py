"""Image download and local caching."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .og_extractor import extract_og_image

logger = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
)
BING_UA = (
    "Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)"
)
GOOGLEBOT_UA = (
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
)
UA_POOL = [DEFAULT_UA, MOBILE_UA, BING_UA, GOOGLEBOT_UA]

DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_UA,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

_TL = threading.local()


def _get_session() -> requests.Session:
    s = getattr(_TL, "sess", None)
    if s is None:
        s = requests.Session()
        _TL.sess = s
    return s


def _same_host_root(url: str) -> str:
    p = urlparse(url)
    if p.scheme and p.netloc:
        return f"{p.scheme}://{p.netloc}/"
    return "https://www.google.com/"


def _safe_prefix(prefix: str | None) -> str:
    if not prefix:
        return ""
    s = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(prefix).strip())
    return (s + "-") if s else ""


def _is_valid_image_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    if not PIL_AVAILABLE:
        return True
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _convert_to_rgb_for_jpeg(im: "Image.Image") -> "Image.Image":
    mode = (im.mode or "").upper()
    if mode in ("RGBA", "LA") or ("A" in mode):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if "A" in im.getbands():
            alpha = im.getchannel("A")
            bg.paste(im.convert("RGBA"), mask=alpha)
        else:
            bg.paste(im.convert("RGB"))
        return bg
    if mode == "P" or mode != "RGB":
        return im.convert("RGB")
    return im


def download_image(
    img_url: str,
    save_dir: str | Path = "outputs/images",
    timeout: int = 30,
    name_prefix: str | None = None,
    page_url: str | None = None,
    user_agent: str | None = None,
) -> Path:
    """Single download attempt using the given UA. Raises Exception on failure."""
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(img_url.encode("utf-8")).hexdigest()
    prefix = _safe_prefix(name_prefix)
    local_path = save_dir / f"{prefix}{url_hash}.jpg"

    if _is_valid_image_file(local_path):
        return local_path

    headers = dict(DEFAULT_HEADERS)
    if user_agent:
        headers["User-Agent"] = user_agent
    headers["Referer"] = page_url.strip() if (page_url or "").strip() else _same_host_root(img_url)

    sess = _get_session()
    r = sess.get(img_url, timeout=min(timeout, 30), stream=True, headers=headers, allow_redirects=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ct and not ct.startswith("image/"):
        raise ValueError(f"Non-image content-type: {ct} url={img_url}")

    tmp_raw = local_path.with_suffix(".raw.tmp")
    tmp_jpg = local_path.with_suffix(".jpg.tmp")
    bytes_written = 0
    with open(tmp_raw, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    if bytes_written < 1024:
        tmp_raw.unlink(missing_ok=True)
        raise ValueError("Downloaded file too small")

    if PIL_AVAILABLE:
        try:
            with Image.open(tmp_raw) as im:
                im.load()
                rgb = _convert_to_rgb_for_jpeg(im)
                rgb.save(tmp_jpg, format="JPEG", quality=85, optimize=True)
            if not _is_valid_image_file(tmp_jpg):
                raise ValueError("Converted JPG invalid")
            os.replace(tmp_jpg, local_path)
        finally:
            tmp_raw.unlink(missing_ok=True)
            tmp_jpg.unlink(missing_ok=True)
    else:
        os.replace(tmp_raw, local_path)

    return local_path


def download_image_robust(
    img_url: str,
    *,
    save_dir: str | Path = "outputs/images",
    timeout: int = 30,
    name_prefix: str | None = None,
    page_url: str | None = None,
    enable_og_fallback: bool = True,
) -> tuple[Path | None, list[str]]:
    attempts: list[str] = []

    for ua_idx, ua in enumerate(UA_POOL[:3]):
        try:
            p = download_image(
                img_url,
                save_dir=save_dir,
                timeout=timeout,
                name_prefix=name_prefix,
                page_url=page_url,
                user_agent=ua,
            )
            attempts.append(f"L{ua_idx+1}-direct: OK ({ua.split('/')[0]})")
            return p, attempts
        except Exception as e:
            attempts.append(f"L{ua_idx+1}-direct: {type(e).__name__}: {str(e)[:80]}")

    if enable_og_fallback and page_url:
        try:
            og_url = extract_og_image(page_url, timeout=timeout)
        except Exception as e:
            attempts.append(f"L4-og-extract: {type(e).__name__}: {str(e)[:80]}")
            og_url = None

        if og_url and og_url != img_url:
            attempts.append(f"L4-og-found: {og_url[:80]}")
            for ua_idx, ua in enumerate(UA_POOL[:2]):
                try:
                    p = download_image(
                        og_url,
                        save_dir=save_dir,
                        timeout=timeout,
                        name_prefix=name_prefix,
                        page_url=page_url,
                        user_agent=ua,
                    )
                    attempts.append(f"L4.{ua_idx+1}-og-direct: OK")
                    return p, attempts
                except Exception as e:
                    attempts.append(f"L4.{ua_idx+1}-og-direct: {type(e).__name__}: {str(e)[:60]}")
        else:
            attempts.append("L4-og-fallback: no og:image found")
    else:
        attempts.append("L4-og-fallback: skipped (no page_url)")

    return None, attempts


def download_images_parallel(
    items: list[dict[str, Any]],
    save_dir: str | Path = "outputs/images",
    name_prefix: str | None = None,
    max_workers: int = 4,
    timeout: int = 30,
    robust: bool = True,
) -> list[dict[str, Any]]:
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    def _one(item: dict) -> dict:
        url = item.get("url") or ""
        if not url:
            item["download_error"] = "missing url"
            return item
        if robust:
            local, attempts = download_image_robust(
                url,
                save_dir=save_dir,
                timeout=timeout,
                name_prefix=name_prefix,
                page_url=item.get("page_url"),
            )
            item["download_attempts"] = attempts
            if local:
                item["local_path"] = str(local)
            else:
                item["download_error"] = "all fallbacks failed"
            return item
        try:
            local = download_image(
                url,
                save_dir=save_dir,
                timeout=timeout,
                name_prefix=name_prefix,
                page_url=item.get("page_url"),
            )
            item["local_path"] = str(local)
            return item
        except Exception as e:
            item["download_error"] = f"{type(e).__name__}: {e}"
            return item

    results: list[dict] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return results


def download_images_serial_until_n(
    items: list[dict[str, Any]],
    n_needed: int = 1,
    *,
    save_dir: str | Path = "outputs/images",
    name_prefix: str | None = None,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    out: list[dict[str, Any]] = []
    for item in items:
        item = dict(item)
        url = item.get("url") or ""
        if not url:
            item["download_error"] = "missing url"
            out.append(item)
            continue

        local, attempts = download_image_robust(
            url,
            save_dir=save_dir,
            timeout=timeout,
            name_prefix=name_prefix,
            page_url=item.get("page_url"),
        )
        item["download_attempts"] = attempts
        if local:
            item["local_path"] = str(local)
            success_count += 1
        else:
            item["download_error"] = "all fallbacks failed"
        out.append(item)

        if success_count >= n_needed:
            break

    return out

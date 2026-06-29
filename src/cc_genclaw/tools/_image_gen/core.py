"""Image generation core — the single implementation behind t2i and i2i.

This module is the **atomic** image-generation primitive:

    generate_image(prompt, reference_images=None) -> str

No reference images  -> text-to-image.
With reference images -> text+image-to-image.

Everything else (which backend, which endpoint, which model) is decided by
``providers.yaml`` (next to this file), NOT by the caller and NOT by the
agent. The function returns the absolute path of the generated PNG and
raises ``RuntimeError`` on failure.

Design (per docs/IMAGE_GEN_REFACTOR_PLAN.md):
- No size / provider / mode / quality parameters.
- One retry policy at this entry point: 1 original + 3 retries with
  5s / 15s / 45s backoff, all exceptions retried.
- Two protocol handlers (openai / openai_chat). Each is a bare API call
  that raises on failure; none does its own retry, review, fallback, or
  prompt wrapping.
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("cc_genclaw.image_gen")

_MAX_RETRIES = 3
_RETRY_DELAYS = [5, 15, 45]  # seconds, exponential backoff; worst case 65s

_PROVIDERS_PATH = Path(__file__).resolve().parent / "providers.yaml"

_REQUEST_TIMEOUT = 300
_REF_ASPECT_MIN = 0.5
_REF_ASPECT_MAX = 2.0


# ==============================================================================
# Provider config
# ==============================================================================

def _load_providers() -> Dict[str, Any]:
    with open(_PROVIDERS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "providers" not in data or "default" not in data:
        raise RuntimeError(
            f"{_PROVIDERS_PATH} must define both 'default' and 'providers'."
        )
    return data


def _resolve_env_value(value: Any) -> Any:
    """Resolve ${ENV_NAME} placeholders in providers.yaml values."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.startswith("${") and text.endswith("}"):
        key = text[2:-1].strip()
        if key:
            return os.environ.get(key, "")
    return value


def _load_default_provider() -> Dict[str, Any]:
    data = _load_providers()
    name = os.environ.get("CC_GENCLAW_IMAGE_PROVIDER", data["default"])
    providers = data["providers"]
    if name not in providers:
        raise RuntimeError(
            f"default provider '{name}' not found in {_PROVIDERS_PATH}. "
            f"Available: {list(providers)}"
        )
    cfg = {k: _resolve_env_value(v) for k, v in dict(providers[name]).items()}
    cfg["_name"] = name
    return cfg


def _resolve_model(cfg: Dict[str, Any], has_refs: bool) -> str:
    model = cfg.get("i2i_model" if has_refs else "t2i_model") or cfg.get("model")
    if not model:
        raise RuntimeError(
            f"provider '{cfg.get('_name')}' has no usable model "
            f"(need 'model' or '{'i2i_model' if has_refs else 't2i_model'}')."
        )
    return model


# ==============================================================================
# Output helpers
# ==============================================================================

def _output_path() -> str:
    """An absolute target PNG path inside the current session's image_gen dir."""
    from ...runtime.session_dir import tool_output_dir

    out_dir = tool_output_dir("image_gen")
    return str(Path(out_dir) / f"gen_{uuid.uuid4().hex[:8]}.png")


def _encode_image_to_data_uri(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


# ==============================================================================
# Protocol handlers — each is a bare API call that raises on failure.
# ==============================================================================

def _handle_openai(
    prompt: str,
    reference_images: Optional[List[str]],
    model: str,
    cfg: Dict[str, Any],
) -> str:
    """OpenAI-compatible images.generate / images.edit."""
    import requests
    from openai import OpenAI

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    out_path = _output_path()

    if reference_images:
        merged = merge_images_smart(list(reference_images))
        buf = io.BytesIO()
        merged.save(buf, format="PNG")
        merged.close()
        result = client.with_options(timeout=120.0).images.edit(
            model=model, image=buf.getvalue(), prompt=prompt, n=1,
        )
    else:
        result = client.with_options(timeout=120.0).images.generate(
            model=model, prompt=prompt, n=1,
        )

    img_data = result.data[0]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if getattr(img_data, "url", None):
        r = requests.get(img_data.url, timeout=120)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    elif getattr(img_data, "b64_json", None):
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(img_data.b64_json))
    else:
        raise RuntimeError("openai image API returned no url / b64_json")
    return out_path


def _handle_openai_chat(
    prompt: str,
    reference_images: Optional[List[str]],
    model: str,
    cfg: Dict[str, Any],
) -> str:
    """OpenAI /chat/completions with images base64 in-message (e.g. a Gemini image model)."""
    import requests

    base_url = cfg["base_url"].rstrip("/")
    api_key = cfg["api_key"]

    message_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in reference_images or []:
        message_content.append(
            {"type": "image_url", "image_url": {"url": _encode_image_to_data_uri(img)}}
        )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
        "stream": False,
    }
    rsp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=_REQUEST_TIMEOUT,
    )
    result = rsp.json()
    if "error" in result:
        raise RuntimeError(f"openai_chat error: {str(result['error'])[:300]}")

    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("openai_chat returned no choices")
    text = choices[0].get("message", {}).get("content", "") or ""

    matches = re.findall(r"data:image/[a-z]+;base64,([A-Za-z0-9+/=\s]+)", text)
    if not matches:
        raise RuntimeError("openai_chat returned no image in the response content")

    img_bytes = base64.b64decode(matches[0].replace("\n", "").replace(" ", ""))
    out_path = _output_path()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return out_path


_PROTOCOL_HANDLERS = {
    "openai": _handle_openai,
    "openai_chat": _handle_openai_chat,
}


# ==============================================================================
# merge helper
# ==============================================================================

def merge_images_smart(
    image_paths: List[str],
    max_side: int = 2048,
    cell_max_w: int = 768,
    cell_max_h: int = 768,
    background_color: tuple = (255, 255, 255),
):
    """Combine multiple images into an approximately square grid (no stretch)."""
    import math

    from PIL import Image

    images = []
    for path in image_paths:
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
    if not images:
        raise ValueError("No valid images found to merge.")

    count = len(images)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    combined = Image.new("RGB", (cols * cell_max_w, rows * cell_max_h), background_color)

    for index, img in enumerate(images):
        x_off = (index % cols) * cell_max_w
        y_off = (index // cols) * cell_max_h
        img_aspect = img.width / img.height
        cell_aspect = cell_max_w / cell_max_h
        if img_aspect > cell_aspect:
            new_w, new_h = cell_max_w, int(cell_max_w / img_aspect)
        else:
            new_h, new_w = cell_max_h, int(cell_max_h * img_aspect)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        combined.paste(
            resized,
            (x_off + (cell_max_w - new_w) // 2, y_off + (cell_max_h - new_h) // 2),
        )

    if combined.width > max_side or combined.height > max_side:
        combined.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return combined


def _is_search_reference(path: str) -> bool:
    parts = set(Path(path).parts)
    return "search" in parts and "code_scene_draft" not in parts and "code_text_draft" not in parts


def _normalize_search_reference_aspect(path: str) -> str:
    """Pad extreme search references so they do not become accidental canvases.

    Search references often include tall character sheets or wide product
    banners. Image models tend to inherit that aspect ratio even when the image
    is only an identity/detail reference. We only pad search refs and never crop
    or modify the original file.
    """
    if not path or not os.path.isfile(path) or not _is_search_reference(path):
        return path

    try:
        from PIL import Image

        with Image.open(path) as im:
            src = im.convert("RGB")
            width, height = src.size
            if width <= 0 or height <= 0:
                return path
            aspect = width / height
            if _REF_ASPECT_MIN <= aspect <= _REF_ASPECT_MAX:
                return path

            if aspect < _REF_ASPECT_MIN:
                canvas_w = int(round(height * _REF_ASPECT_MIN))
                canvas_h = height
            else:
                canvas_w = width
                canvas_h = int(round(width / _REF_ASPECT_MAX))

            out_dir = Path(path).parent / "_normalized_refs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(path).stem}_aspect_padded.png"
            if out_path.is_file():
                return str(out_path)

            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            canvas.paste(src, ((canvas_w - width) // 2, (canvas_h - height) // 2))
            canvas.save(out_path, "PNG")
            return str(out_path)
    except Exception:
        return path


def _prepare_reference_images(reference_images: List[str]) -> List[str]:
    return [_normalize_search_reference_aspect(str(path)) for path in reference_images]


# ==============================================================================
# Public API
# ==============================================================================

def generate_image(
    prompt: str,
    reference_images: Optional[List[str]] = None,
) -> str:
    """Generate an image. Returns absolute PNG path. Raises RuntimeError on failure.

    No reference_images -> text-to-image.
    With reference_images -> text+image-to-image.

    Backend, endpoint, and model come from providers.yaml (default entry).
    Retries once-original + 3 times with 5/15/45s backoff on any exception.
    """
    if not prompt or not prompt.strip():
        raise RuntimeError("generate_image requires a non-empty prompt")

    refs = _prepare_reference_images([r for r in (reference_images or []) if r])
    cfg = _load_default_provider()
    protocol = cfg.get("protocol")
    handler = _PROTOCOL_HANDLERS.get(protocol)
    if handler is None:
        raise RuntimeError(
            f"provider '{cfg.get('_name')}' has unknown protocol '{protocol}'. "
            f"Known: {list(_PROTOCOL_HANDLERS)}"
        )
    model = _resolve_model(cfg, has_refs=bool(refs))

    last_err: Optional[Exception] = None
    total = _MAX_RETRIES + 1
    for attempt in range(total):
        try:
            t0 = time.time()
            path = handler(prompt, refs, model, cfg)
            logger.info(
                "[image_gen] provider=%s model=%s refs=%d elapsed=%.1fs saved=%s",
                cfg.get("_name"), model, len(refs), time.time() - t0, path,
            )
            try:
                from ...runtime.artifacts import record_artifact

                record_artifact(
                    path,
                    tool_name="image_gen",
                    role="final",
                    kind="image",
                    label="Generated image",
                    metadata={
                        "provider": cfg.get("_name"),
                        "model": model,
                        "reference_count": len(refs),
                    },
                )
            except Exception:
                pass
            return path
        except Exception as e:  # noqa: BLE001 — bare retry, all exceptions
            last_err = e
            logger.warning("[image_gen] attempt %d/%d failed: %s", attempt + 1, total, e)
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[attempt]
                logger.info("[image_gen] retrying in %ds...", delay)
                time.sleep(delay)

    raise RuntimeError(f"image_gen failed after {total} attempts: {last_err}")

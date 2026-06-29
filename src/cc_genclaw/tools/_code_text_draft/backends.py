"""
Text Rendering — Image Generation Backends
Gemini T2I/I2I, Qwen T2I/I2I, and unified wrappers.
"""
from __future__ import annotations

import os
import re
import json
import time
import base64
from pathlib import Path

from .config import GEN_BACKEND
from .utils import _is_transient_error

from openai import OpenAI


# ============================================================
# Gemini helpers
# ============================================================

def _gemini_post_json(base_url: str, api_key: str, payload: dict, timeout: int = 300, retries: int = 3):
    import requests

    delay = 5.0
    last_exc = None
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    for attempt in range(retries):
        try:
            rsp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            result = rsp.json()
            if "error" in result:
                raise RuntimeError(f"Gemini error: {json.dumps(result['error'], ensure_ascii=False)[:300]}")
            return result
        except Exception as exc:
            last_exc = exc
            if attempt == retries - 1 or not _is_transient_error(exc):
                raise
            print(f"  Gemini request transient failure: {exc}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= 2

    raise last_exc


def gemini_i2i(image_path: str, prompt: str, description: str, output_path: str, custom_instruction: str = "") -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("GEMINI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("GEMINI_I2I_MODEL_NAME", "gemini-3.1-flash-image-preview")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    refinement = custom_instruction.strip()
    if not refinement:
        refinement = f"Transform this into a photorealistic image.\nTarget: {prompt}\n"
        if description:
            refinement += f"Layout: {description}\n"
        refinement += "Preserve spatial arrangement. Replace flat shapes with realistic textures/lighting. Do NOT add or remove elements."

    result = _gemini_post_json(
        base_url,
        api_key,
        {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": refinement},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ]}], "stream": False},
        timeout=300,
    )
    content = result["choices"][0]["message"]["content"]
    matches = re.findall(r"data:image/[a-z]+;base64,([A-Za-z0-9+/=\s]+)", content)
    if not matches:
        raise RuntimeError("Gemini returned no image")
    img_bytes = base64.b64decode(matches[0].replace("\n", "").replace(" ", ""))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    return output_path


def gemini_text_to_image(prompt: str, output_path: str) -> str:
    """Generate image from text prompt only (no reference image)."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("GEMINI_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("GEMINI_I2I_MODEL_NAME", "gemini-3.1-flash-image-preview")

    result = _gemini_post_json(
        base_url,
        api_key,
        {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]}], "stream": False},
        timeout=300,
    )
    content = result["choices"][0]["message"]["content"]
    matches = re.findall(r"data:image/[a-z]+;base64,([A-Za-z0-9+/=\s]+)", content)
    if not matches:
        raise RuntimeError("Gemini returned no image")
    img_bytes = base64.b64decode(matches[0].replace("\n", "").replace(" ", ""))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    return output_path


# ============================================================
# Qwen helpers
# ============================================================

def _qwen_text_to_image(prompt: str, output_path: str, model: str = "") -> str:
    """Generate image from text using Qwen image generation API (via boyue)."""
    api_key = os.environ.get("IMAGE_EDIT_API_KEY") or os.environ.get("IMAGE_API_KEY", "")
    base_url = (
        os.environ.get("IMAGE_EDIT_BASE_URL")
        or os.environ.get("IMAGE_BASE_URL")
        or "https://api.boyuerichdata.opensphereai.com/v1"
    )
    if not model:
        model = os.environ.get("IMAGE_GEN_MODEL_NAME", "qwen-image-plus")

    client = OpenAI(api_key=api_key, base_url=base_url)
    result = client.with_options(timeout=300.0).images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    img_data = result.data[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if getattr(img_data, "url", None):
        import requests as _req
        rsp = _req.get(img_data.url, timeout=60)
        rsp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(rsp.content)
    elif getattr(img_data, "b64_json", None):
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(img_data.b64_json))
    else:
        raise RuntimeError("Qwen T2I returned no image data")
    return output_path


def _qwen_edit_from_image(image_path: str, prompt: str, output_path: str) -> str:
    """Edit/refine an existing image using Qwen image-edit API."""
    api_key = os.environ.get("IMAGE_EDIT_API_KEY") or os.environ.get("IMAGE_API_KEY", "")
    base_url = os.environ.get("IMAGE_EDIT_BASE_URL") or os.environ.get("IMAGE_BASE_URL", "")
    model = os.environ.get("IMAGE_EDIT_MODEL_NAME", "qwen-image-edit-plus-2025-12-15")

    client = OpenAI(api_key=api_key, base_url=base_url)
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    result = client.with_options(timeout=300.0).images.edit(
        model=model,
        image=img_bytes,
        prompt=prompt,
        size="1024x1024",
        n=1,
    )
    img_data = result.data[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if getattr(img_data, "url", None):
        import requests as _req
        rsp = _req.get(img_data.url, timeout=60)
        rsp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(rsp.content)
    elif getattr(img_data, "b64_json", None):
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(img_data.b64_json))
    else:
        raise RuntimeError("Qwen Edit returned no image data")
    return output_path


# ============================================================
# Unified T2I / I2I dispatchers
# ============================================================

def t2i_generate(prompt: str, output_path: str) -> str:
    """Dispatch T2I generation based on GEN_BACKEND."""
    if GEN_BACKEND == "qwen":
        return _qwen_text_to_image(prompt, output_path)
    return gemini_text_to_image(prompt, output_path)


def i2i_refine(image_path: str, prompt: str, description: str, output_path: str, custom_instruction: str = "") -> str:
    """Dispatch I2I refinement based on GEN_BACKEND."""
    if GEN_BACKEND == "qwen":
        edit_prompt = custom_instruction.strip() if custom_instruction.strip() else (
            f"Transform this into a photorealistic image. Target: {prompt}. "
            "Preserve spatial arrangement and all text content exactly as shown."
        )
        return _qwen_edit_from_image(image_path, edit_prompt, output_path)
    return gemini_i2i(image_path, prompt, description, output_path, custom_instruction=custom_instruction)

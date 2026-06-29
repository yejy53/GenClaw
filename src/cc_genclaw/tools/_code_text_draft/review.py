"""
Text Rendering — Review & Validation
LLM-based visual and text content review for generated images.
"""
from __future__ import annotations

import json
import base64

from .config import REVIEW_MODEL, load_prompt
from .utils import _openai_chat_completion


def _parse_review_reply(reply: str) -> dict:
    reply = reply.strip()
    if reply.startswith("PASS"):
        return {"passed": True, "feedback": None}
    if "<feedback>" in reply and "</feedback>" in reply:
        feedback = reply.split("<feedback>", 1)[1].split("</feedback>", 1)[0].strip()
    else:
        feedback = reply.replace("FAIL", "").strip()
    return {"passed": False, "feedback": feedback}


def review_text(prompt: str, png_path: str, expected_texts: list, code: str = "") -> dict:
    review_template = load_prompt("text_review")
    review_prompt_text = review_template.format(
        prompt=prompt,
        expected_texts=json.dumps(expected_texts, ensure_ascii=False),
        code=code if code else "(code omitted for review)",
    )
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = _openai_chat_completion(
        timeout=60.0,
        model=REVIEW_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": review_prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        max_tokens=512, temperature=0.1,
    )
    return _parse_review_reply(response.choices[0].message.content)


def review_html_layout(prompt: str, png_path: str, expected_texts: list, html_code: str, dom_text: str) -> dict:
    review_template = load_prompt("html_text_review")
    review_prompt_text = review_template.format(
        prompt=prompt,
        expected_texts=json.dumps(expected_texts, ensure_ascii=False),
        dom_text=dom_text or "(empty DOM text)",
        html_code=html_code,
    )
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = _openai_chat_completion(
        timeout=60.0,
        model=REVIEW_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": review_prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        max_tokens=512,
        temperature=0.1,
    )
    return _parse_review_reply(response.choices[0].message.content)

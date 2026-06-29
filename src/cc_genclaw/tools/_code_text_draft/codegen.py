"""
Text Rendering — Code Generation
LLM-based HTML code generation with legend/code parsing.
"""
from __future__ import annotations

import re

from .config import SVG_MODEL, SVG_DELIMITER, load_prompt
from .utils import _openai_chat_completion


def generate_code(prompt: str, system_prompt_name: str, **template_vars) -> dict:
    system = load_prompt(system_prompt_name)
    for key, value in template_vars.items():
        system = system.replace(f"{{{key}}}", str(value))
    response = _openai_chat_completion(
        timeout=180.0,
        model=SVG_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Generate layout code for:\n{prompt}"},
        ],
        max_tokens=16384,
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_generated_code(raw)


def generate_code_from_content(user_content: str, system_prompt_name: str, **template_vars) -> dict:
    system = load_prompt(system_prompt_name)
    for key, value in template_vars.items():
        system = system.replace(f"{{{key}}}", str(value))
    response = _openai_chat_completion(
        timeout=180.0,
        model=SVG_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=16384,
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    return _parse_generated_code(raw)


def _parse_generated_code(raw: str) -> dict:
    if SVG_DELIMITER in raw:
        desc, code = raw.split(SVG_DELIMITER, 1)
        description = desc.strip()
        code = code.strip()
    else:
        description = ""
        code = raw

    code = re.sub(r"^```(?:html|xml)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code).strip()
    code = code.replace(SVG_DELIMITER, "")
    code = code.replace("===SVG_CODE_END===", "")

    html_start = re.search(r"(?:<\!doctype|<html)[\s>]", code, re.IGNORECASE)
    if html_start and html_start.start() > 0:
        if not description:
            description = code[:html_start.start()].strip()
        code = code[html_start.start():]

    code_type = "html" if ("<html" in code.lower() or "<!doctype" in code.lower()) else "other"
    return {"code": code, "description": description, "code_type": code_type}

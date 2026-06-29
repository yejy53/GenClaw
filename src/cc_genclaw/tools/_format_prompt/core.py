"""Core implementation for the self-contained format_prompt tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .prompts import FORMAT_PROMPT_SYSTEM
from .schema import (
    FormatPromptBrief,
    brief_to_user_text,
    build_fallback_prompt,
    normalize_brief,
)
from .utils import encode_image_to_data_uri, existing_file, parse_json_object, unique_preserve_order


def _path_role(path: str, brief: FormatPromptBrief) -> str:
    if path == brief.text_template_path:
        if (brief.render_type or "").lower() in {"embedded", "embedded_svg"}:
            return (
                "source embedded text reference draft; preserve exact text and "
                "object association, but naturalize carrier scale, materials, "
                "perspective, and scene proportions"
            )
        return "source text-template image; preserve rendered text exactly"
    if path == brief.draft_png_path:
        return "source layout draft; preserve composition, counts, and spatial relations"
    if path == brief.user_image_path:
        return "user-provided source image"
    return "upstream curated visual reference; bind to the matching subject or object"


_EXTREME_ASPECT_MIN = 0.5
_EXTREME_ASPECT_MAX = 2.0


def _image_size(path: str) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return None


def _is_extreme_aspect(path: str) -> bool:
    size = _image_size(path)
    if not size:
        return False
    width, height = size
    if width <= 0 or height <= 0:
        return False
    aspect = width / height
    return aspect < _EXTREME_ASPECT_MIN or aspect > _EXTREME_ASPECT_MAX


def _source_candidates(brief: FormatPromptBrief) -> List[str]:
    return unique_preserve_order(
        [p for p in [brief.source_image_path] if p] + brief.reference_image_paths
    )


def _validated_source_hint(raw_hint: Any, brief: FormatPromptBrief) -> str:
    hint = str(raw_hint or "").strip()
    if not hint:
        return ""
    allowed = set(_source_candidates(brief))
    return hint if hint in allowed else ""


def _source_hint(brief: FormatPromptBrief) -> str:
    if brief.source_image_path:
        return brief.source_image_path
    if brief.reference_image_paths and brief.target_tool != "t2i":
        first = brief.reference_image_paths[0]
        if existing_file(first) and _is_extreme_aspect(first):
            for ref in brief.reference_image_paths[1:]:
                if existing_file(ref) and not _is_extreme_aspect(ref):
                    return ref
        return first
    return ""


def _extra_refs_hint(brief: FormatPromptBrief, source: str) -> List[str]:
    refs = list(brief.reference_image_paths)
    return [r for r in refs if r and r != source]


def _default_bindings(brief: FormatPromptBrief) -> List[Dict[str, str]]:
    paths = unique_preserve_order(
        [p for p in [brief.source_image_path] if p] + brief.reference_image_paths
    )
    return [{"path": p, "role": _path_role(p, brief)} for p in paths]


def _reference_label(index: int) -> str:
    return chr(ord("A") + index) if 0 <= index < 26 else str(index + 1)


def _role_for_path(path: str, bindings: List[Dict[str, str]], brief: FormatPromptBrief) -> str:
    for binding in bindings:
        if binding.get("path") == path and str(binding.get("role") or "").strip():
            return str(binding["role"]).strip()
    return _path_role(path, brief)


def _reference_use_preface(
    source: str,
    extra_refs: List[str],
    bindings: List[Dict[str, str]],
    brief: FormatPromptBrief,
) -> str:
    refs = unique_preserve_order([p for p in [source] if p] + extra_refs)
    if len(refs) < 2:
        return ""

    lines = ["Reference use:"]
    for idx, path in enumerate(refs):
        label = _reference_label(idx)
        role = _role_for_path(path, bindings, brief)
        if idx == 0:
            lines.append(f"- Reference {label}: main canvas / composition anchor. {role}")
        else:
            lines.append(f"- Reference {label}: secondary visual reference. {role}")
    return "\n".join(lines)


def _ensure_reference_preface(
    final_prompt: str,
    source: str,
    extra_refs: List[str],
    bindings: List[Dict[str, str]],
    brief: FormatPromptBrief,
) -> str:
    if "Reference A" in final_prompt or "Reference use:" in final_prompt:
        return final_prompt
    preface = _reference_use_preface(source, extra_refs, bindings, brief)
    if not preface:
        return final_prompt
    return f"{preface}\n\n{final_prompt}".strip()


def _message_content(brief: FormatPromptBrief) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": brief_to_user_text(brief)}]
    attached = unique_preserve_order(
        [p for p in [brief.source_image_path] if p] + brief.reference_image_paths
    )
    for idx, path in enumerate(attached, 1):
        if not existing_file(path):
            continue
        content.append({
            "type": "text",
            "text": f"[Attached Image {idx}] path={path} role={_path_role(path, brief)}",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_to_data_uri(path), "detail": "high"},
        })
    return content


def _call_model(brief: FormatPromptBrief, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(
        base_url=cfg.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=cfg.get("OPENAI_API_KEY", ""),
    )
    response = client.chat.completions.create(
        model=cfg.get("OPENAI_MODEL_NAME", "gpt-5.4"),
        messages=[
            {"role": "system", "content": FORMAT_PROMPT_SYSTEM},
            {"role": "user", "content": _message_content(brief)},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    content = response.choices[0].message.content or ""
    return parse_json_object(content)


def _normalize_output(raw: Dict[str, Any], brief: FormatPromptBrief) -> Dict[str, Any]:
    source = _validated_source_hint(raw.get("i2i_image_path_hint"), brief) or _source_hint(brief)
    extra_refs = _extra_refs_hint(brief, source)

    final_prompt = str(raw.get("final_prompt") or "").strip()
    if not final_prompt:
        final_prompt = build_fallback_prompt(brief)

    raw_bindings = raw.get("reference_bindings")
    bindings: List[Dict[str, str]] = []
    if isinstance(raw_bindings, list):
        allowed = set(unique_preserve_order(
            [p for p in [brief.source_image_path] if p] + brief.reference_image_paths
        ))
        for item in raw_bindings:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if path not in allowed:
                continue
            role = str(item.get("role") or "").strip() or _path_role(path, brief)
            bindings.append({"path": path, "role": role})
    if not bindings:
        bindings = _default_bindings(brief)

    target_tool = str(raw.get("target_tool") or brief.target_tool or "").strip()
    if target_tool not in ("t2i", "i2i"):
        target_tool = "i2i" if source or brief.reference_image_paths else "t2i"
    if target_tool == "i2i":
        final_prompt = _ensure_reference_preface(
            final_prompt, source, extra_refs, bindings, brief
        )

    return {
        "final_prompt": final_prompt,
        # Search/reference paths are an ordered passthrough. Do not use model
        # output to reorder or filter them.
        "reference_paths": list(brief.reference_image_paths),
        "reference_bindings": bindings,
        "target_tool": target_tool,
        "rationale_summary": str(raw.get("rationale_summary") or "").strip(),
        "i2i_image_path_hint": source,
        "i2i_extra_reference_paths_hint": extra_refs,
        "status": "success",
    }


def _fallback_output(brief: FormatPromptBrief, error: Optional[str] = None) -> Dict[str, Any]:
    source = _source_hint(brief)
    extra_refs = _extra_refs_hint(brief, source)
    bindings = _default_bindings(brief)
    final_prompt = build_fallback_prompt(brief)
    if brief.target_tool == "i2i" or source:
        final_prompt = _ensure_reference_preface(
            final_prompt, source, extra_refs, bindings, brief
        )
    payload = {
        "final_prompt": final_prompt,
        "reference_paths": list(brief.reference_image_paths),
        "reference_bindings": bindings,
        "target_tool": brief.target_tool or ("i2i" if source else "t2i"),
        "rationale_summary": "Fallback prompt composed from provided trajectory fields.",
        "i2i_image_path_hint": source,
        "i2i_extra_reference_paths_hint": extra_refs,
        "status": "success",
    }
    if error:
        payload["composer_error"] = error
    return payload


def compose_generation_prompt(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compose a final generation prompt from tool arguments."""
    from cc_genclaw.config import load_genclaw_config

    brief = normalize_brief(args)
    if not brief.original_user_prompt and not brief.image_intent:
        raise ValueError("format_prompt requires original_user_prompt or base_prompt")

    cfg = load_genclaw_config()
    try:
        raw = _call_model(brief, cfg)
        return _normalize_output(raw, brief)
    except Exception as exc:  # noqa: BLE001
        return _fallback_output(brief, f"{type(exc).__name__}: {exc}")


def write_artifacts(out_dir: str, payload: Dict[str, Any], args: Dict[str, Any]) -> None:
    """Best-effort artifact writer used by the thin tool wrapper."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    Path(path, "final_prompt.txt").write_text(
        str(payload.get("final_prompt") or ""), encoding="utf-8"
    )
    Path(path, "meta.json").write_text(
        json.dumps({"args": args, "payload": payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

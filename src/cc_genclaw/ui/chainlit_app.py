"""Chainlit runtime UI for CC-GenClaw.

Run from the repository root:

    chainlit run src/cc_genclaw/ui/chainlit_app.py -w

The UI consumes CC-GenClaw loop events in natural order and renders each tool
call as a Chainlit step. It intentionally avoids raw trace dumps, full
arguments, absolute paths, and full tool payloads in the main view.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chainlit as cl  # noqa: E402
from chainlit.context import local_steps  # noqa: E402

from cc_genclaw.runtime.agent_runner import (  # noqa: E402
    AgentRun,
    initialize_perception,
    persist_event,
    prepare_agent_run,
    run_prompt,
)
from cc_genclaw.types import (  # noqa: E402
    AgentEvent,
    AssistantMessage,
    MessageEndEvent,
    TextContent,
    ToolCallContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)


CONFIG_PATH = Path(__file__).with_name("step_config.yaml")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
FINAL_IMAGE_TOOLS = {"i2i", "t2i", "image_gen", "code_text_draft"}
FIELD_LABELS = {
    "painter_thoughts": "Painter Thoughts",
    "image_intent": "Image Intent",
    "planning_questions": "Questions",
    "notes_for_planner": "Notes For Planner",
    "text_queries": "web-search",
    "image_queries": "image-search",
    "facts": "search-summary",
    "ranking_log": "ranking",
    "downloaded_count": "downloaded references",
    "candidate_count": "rerank candidates",
    "reference_paths": "downloaded references",
    "process_summary": "Process",
    "process_images": "Process images",
}
local_steps.set(None)


class ActiveStep:
    def __init__(
        self,
        *,
        step: cl.Step,
        started_at: float,
        tool_name: str,
        tool_call_id: str,
        args: Dict[str, Any],
        agent_note: str = "",
    ) -> None:
        self.step = step
        self.started_at = started_at
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.args = args
        self.agent_note = agent_note


class UIState:
    def __init__(self, *, run: AgentRun) -> None:
        self.run = run
        self.active_steps: Dict[str, ActiveStep] = {}
        self.pending_notes: Dict[str, str] = {}
        self.todo_message: Optional[cl.Message] = None
        self.last_todos: List[Dict[str, str]] = []
        self.final_reply = ""
        self.final_images: List[str] = []


def load_step_config() -> Dict[str, Any]:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


STEP_CONFIG = load_step_config()


def tool_config(tool_name: str) -> Dict[str, Any]:
    tools = STEP_CONFIG.get("tools", {})
    default = STEP_CONFIG.get("defaults", {})
    if not isinstance(tools, dict):
        tools = {}
    if not isinstance(default, dict):
        default = {}
    cfg = dict(default)
    cfg.update(tools.get(tool_name, {}) or {})
    return cfg


def display_name(tool_name: str) -> str:
    return str(tool_config(tool_name).get("display_name") or tool_name)


def subtitle(tool_name: str) -> str:
    return str(tool_config(tool_name).get("subtitle") or "")


def parse_jsonish(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def sanitize_text(text: Any) -> str:
    raw = json.dumps(text, ensure_ascii=False, indent=2) if isinstance(text, (dict, list)) else str(text)

    def repl(match: re.Match[str]) -> str:
        return Path(match.group(0).rstrip("`),.")).name

    return re.sub(r"/Volumes/[^\s`),.]+", repl, raw)


def field_markdown(name: str, value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            body = "\n".join(f"- {sanitize_text(item)}" for item in value)
        else:
            body = f"```json\n{sanitize_text(value)}\n```"
    elif isinstance(value, dict):
        body = f"```json\n{sanitize_text(value)}\n```"
    else:
        body = f"> {sanitize_text(value).replace(chr(10), chr(10) + '> ')}"
    title = FIELD_LABELS.get(name, name.replace("_", " ").title())
    return f"**{title}**\n\n{body}"


def visible_fields(tool_name: str, result: Dict[str, Any], args: Dict[str, Any]) -> str:
    cfg = tool_config(tool_name)
    fields = cfg.get("fields") or []
    if not isinstance(fields, list):
        return ""
    parts: List[str] = []
    source = dict(args)
    source.update(result)
    for field_name in fields:
        if not isinstance(field_name, str):
            continue
        block = field_markdown(field_name, source.get(field_name))
        if block:
            parts.append(block)
    return "\n\n".join(parts)


def normalize_todos(args: Dict[str, Any]) -> List[Dict[str, str]]:
    todos = args.get("todos")
    if not isinstance(todos, list):
        return []
    normalized: List[Dict[str, str]] = []
    for item in todos:
        if not isinstance(item, dict):
            continue
        content = sanitize_text(item.get("content") or item.get("activeForm") or "")
        if not content:
            continue
        normalized.append({
            "key": sanitize_text(item.get("id") or content),
            "content": content,
            "status": str(item.get("status", "pending")),
        })
    return normalized


def render_todos(args: Dict[str, Any]) -> str:
    todos = normalize_todos(args)
    if not todos:
        return ""
    completed = sum(1 for item in todos if item["status"] == "completed")
    lines = [f"**Task plan {completed}/{len(todos)}**"]
    for idx, item in enumerate(todos, 1):
        content = item["content"]
        status = item["status"]
        if status == "completed":
            lines.append(f"{idx}. ✓ ~~{content}~~")
        elif status == "in_progress":
            lines.append(f"{idx}. ● **{content}**")
        elif status == "cancelled":
            lines.append(f"{idx}. － ~~{content}~~")
        else:
            lines.append(f"{idx}. ○ {content}")
    return "\n".join(lines)


async def update_todo_message(state: UIState, args: Dict[str, Any]) -> None:
    content = render_todos(args)
    if not content:
        return
    todos = normalize_todos(args)
    previous = state.last_todos
    if state.todo_message is None:
        state.todo_message = cl.Message(content=content)
        await state.todo_message.send()
    else:
        state.todo_message.content = content
        await state.todo_message.update()
    await emit_todo_progress(previous, todos)
    state.last_todos = todos


async def emit_todo_progress(previous: List[Dict[str, str]], current: List[Dict[str, str]]) -> None:
    previous_by_key = {item["key"]: item for item in previous}
    first_snapshot = not previous
    messages: List[str] = []
    total = len(current)
    for idx, item in enumerate(current, 1):
        old = previous_by_key.get(item["key"])
        old_status = old["status"] if old else ""
        new_status = item["status"]
        if first_snapshot and new_status == "in_progress":
            messages.append(f"Started to-do {idx}/{total}  ● {item['content']}")
        elif not first_snapshot and not old:
            messages.append(f"Added to-do {idx}/{total}  ○ {item['content']}")
        elif old_status != new_status:
            if new_status == "in_progress":
                messages.append(f"Started to-do {idx}/{total}  ● {item['content']}")
            elif new_status == "completed":
                messages.append(f"Completed to-do {idx}/{total}  ✓ {item['content']}")
            elif new_status == "cancelled":
                messages.append(f"Cancelled to-do {idx}/{total}  － {item['content']}")
    for message in messages:
        await cl.Message(content=message).send()


def review_fallback(result: Dict[str, Any]) -> str:
    if result.get("suggested_fix_prompt"):
        return ""
    if result.get("passed") is True or str(result.get("verdict", "")).upper() == "PASS":
        return "**Suggested Fix Prompt**\n\n> No repair needed."
    return ""


def render_vlm_review_output(result: Dict[str, Any], agent_note: str = "") -> str:
    verdict = sanitize_text(result.get("verdict") or ("PASS" if result.get("passed") is True else ""))
    passed = result.get("passed")
    issues = result.get("issues")
    criteria = result.get("criteria") or []
    suggested = sanitize_text(result.get("suggested_fix_prompt") or "")

    badge_html = ""
    if isinstance(criteria, list) and criteria:
        badge_html = "".join(
            f'<span class="review-badge">{sanitize_text(item)}</span>'
            for item in criteria
            if item
        )

    status_bits = []
    if verdict:
        status_bits.append(f"<strong>{verdict}</strong>")
    if passed is not None:
        status_bits.append("passed" if passed else "needs repair")
    status_line = " · ".join(status_bits) or "Review completed"

    issue_text = ""
    if isinstance(issues, list) and issues:
        issue_text = "<br>".join(f"- {sanitize_text(item)}" for item in issues)
    elif isinstance(issues, str) and issues.strip():
        issue_text = sanitize_text(issues)
    else:
        issue_text = "No issues found."

    fix_text = ""
    if suggested and suggested.lower() not in {"none", "no repair needed."}:
        fix_text = f"<p><strong>Suggested fix</strong><br>{suggested}</p>"
    elif passed is True:
        fix_text = "<p><strong>Suggested fix</strong><br>No repair needed.</p>"

    note = ""
    if agent_note:
        note = f"<p><strong>Agent note</strong><br>{sanitize_text(agent_note)}</p>"

    return (
        f"{note}"
        '<div class="genclaw-vlm-review">'
        '<div class="review-card">'
        "<strong>Review</strong>"
        f"<p>{status_line}</p>"
        f"{fix_text}"
        "</div>"
        '<div class="review-card">'
        "<strong>Criteria</strong>"
        f'<div class="review-badges">{badge_html}</div>'
        "<p><strong>Issues</strong><br>"
        f"{issue_text}</p>"
        "</div>"
        "</div>"
    )


def summarize_expected_texts(texts: Any) -> str:
    if not isinstance(texts, list):
        return ""
    total_chars = sum(len(str(text)) for text in texts)
    preview = ""
    if texts:
        first = str(texts[0]).replace("\n", " ")
        preview = first[:80] + ("..." if len(first) > 80 else "")
    return f"{len(texts)} text block(s), {total_chars} characters" + (f"\n\n> {sanitize_text(preview)}" if preview else "")


def render_search_output(args: Dict[str, Any], result: Dict[str, Any], agent_note: str = "") -> str:
    parts: List[str] = []
    if agent_note:
        parts.append(f"**Agent note**\n\n> {sanitize_text(agent_note).replace(chr(10), chr(10) + '> ')}")
    for field_name in ("text_queries", "image_queries", "facts", "downloaded_count", "ranking_log"):
        block = field_markdown(field_name, result.get(field_name) or args.get(field_name))
        if block:
            parts.append(block)

    image_results = result.get("image_results") or []
    if isinstance(image_results, list) and image_results:
        lines = ["**Ranked references**"]
        for item in image_results[: int(tool_config("search").get("max_images") or 8)]:
            if not isinstance(item, dict):
                continue
            role = "BEST" if item.get("role") == "primary" else "backup"
            rank = item.get("rank", "?")
            query = sanitize_text(item.get("query", ""))
            score = item.get("rerank_score")
            reason = sanitize_text(item.get("rerank_reason", ""))
            score_text = f" score {float(score):.2f}" if isinstance(score, (int, float)) else ""
            line = f"- #{rank} {role}{score_text}"
            if query:
                line += f" · {query}"
            if reason:
                line += f" · {reason[:160]}"
            lines.append(line)
        parts.append("\n".join(lines))
    elif result.get("reference_paths"):
        parts.append(field_markdown("reference_paths", [Path(str(path)).name for path in result.get("reference_paths", [])]))

    candidate_results = result.get("candidate_image_results") or []
    if isinstance(candidate_results, list) and candidate_results:
        lines = ["**Rerank candidate pool**"]
        for item in candidate_results[: int(tool_config("search").get("max_candidate_images") or 8)]:
            if not isinstance(item, dict):
                continue
            rank = item.get("rank", "?")
            score = item.get("rerank_score")
            reason = sanitize_text(item.get("rerank_reason", ""))
            selected = " selected" if item.get("role") == "selected" else ""
            score_text = f" score {float(score):.2f}" if isinstance(score, (int, float)) else ""
            line = f"- candidate #{rank}{selected}{score_text}"
            if reason:
                line += f" · {reason[:160]}"
            lines.append(line)
        parts.append("\n".join(lines))

    return "\n\n".join(parts) if parts else "search completed."


def render_code_text_draft_output(result: Dict[str, Any], agent_note: str = "") -> str:
    parts: List[str] = []
    if agent_note:
        parts.append(f"**Agent note**\n\n> {sanitize_text(agent_note).replace(chr(10), chr(10) + '> ')}")

    render_type = result.get("render_type")
    if render_type:
        parts.append(f"**Render pipeline**\n\n> {sanitize_text(render_type)}")

    expected_summary = summarize_expected_texts(result.get("expected_texts_used"))
    if expected_summary:
        parts.append(f"**Expected text**\n\n{expected_summary}")

    summary = result.get("process_summary") or {}
    if isinstance(summary, dict):
        canvas = summary.get("canvas_decision") or {}
        if isinstance(canvas, dict):
            width = canvas.get("width")
            height = canvas.get("height")
            reason = canvas.get("reason")
            if width and height:
                parts.append(
                    "**Canvas**\n\n"
                    f"> {width} x {height}" + (f" · {sanitize_text(reason)}" if reason else "")
                )
        if render_type == "layered":
            bg_final = summary.get("background_final_size") or {}
            resize = summary.get("background_resize") or {}
            details: List[str] = []
            if isinstance(bg_final, dict) and bg_final.get("width") and bg_final.get("height"):
                details.append(f"final background {bg_final.get('width')} x {bg_final.get('height')}")
            if isinstance(resize, dict) and resize.get("mode") == "scale_up_keep_aspect":
                details.append(f"scaled up {float(resize.get('scale') or 1.0):.2f}x, aspect preserved")
            if details:
                parts.append("**Layered composition**\n\n> " + " · ".join(details))
        elif render_type in {"embedded", "embedded_svg"}:
            surface_count = summary.get("surface_count")
            object_count = summary.get("supporting_object_count")
            details = []
            if surface_count is not None:
                details.append(f"{surface_count} text surface(s)")
            if object_count is not None:
                details.append(f"{object_count} supporting object(s)")
            if details:
                parts.append("**Embedded layout**\n\n> " + " · ".join(details))
            if result.get("requires_i2i") or result.get("output_role") == "text_structure_reference":
                parts.append(
                    "**Next step**\n\n"
                    "> This is a text/structure reference. Use `format_prompt` then `i2i` for the final natural image."
                )

    return "\n\n".join(parts) if parts else "code_text_draft completed."


def is_failed_tool_result(event: ToolExecutionEndEvent, result: Dict[str, Any]) -> bool:
    status = str(result.get("status", "")).strip().lower()
    return bool(event.is_error or status in {"failed", "failure", "error"})


def error_detail_text(result: Dict[str, Any], raw_result: str) -> str:
    for key in ("error", "message", "detail", "details"):
        value = result.get(key)
        if value:
            return sanitize_text(value)
    return sanitize_text(raw_result)


def classify_error(detail: str) -> str:
    text = detail.lower()
    if "timeout" in text or "timed out" in text:
        return "Timeout"
    if (
        "connection" in text
        or "httpconnectionpool" in text
        or "failed to establish" in text
        or "connection refused" in text
        or "name resolution" in text
    ):
        return "Connection failed"
    if "rate limit" in text or "429" in text:
        return "Rate limited"
    if "503" in text or "service unavailable" in text or "upstream" in text or "overloaded" in text:
        return "Upstream unavailable"
    if "model_not_found" in text or "invalid model" in text or "无效的 model" in text:
        return "Model unavailable"
    if "api" in text or "status=" in text:
        return "API error"
    return "Tool error"


def render_agent_failure_text(text: str) -> str:
    cleaned = sanitize_text(text).strip()
    match = re.match(r"^\[LLM error:\s*(.*?)\]$", cleaned, flags=re.DOTALL)
    if not match:
        return cleaned
    detail = match.group(1).strip()
    category = classify_error(detail)
    short_detail = detail if len(detail) <= 900 else detail[:900].rstrip() + "..."
    return (
        f"Main model call failed: {category}\n\n"
        f"> {short_detail.replace(chr(10), chr(10) + '> ')}\n\n"
        "This task has been aborted and no final image was generated. You can retry directly; if this error keeps occurring, it is usually an upstream model gateway connection or availability issue."
    )


def render_tool_error_output(tool_name: str, result: Dict[str, Any], raw_result: str) -> str:
    detail = error_detail_text(result, raw_result).strip()
    if not detail:
        detail = "The tool returned an error without details."
    category = classify_error(detail)
    short_detail = detail if len(detail) <= 900 else detail[:900].rstrip() + "..."
    return (
        f"**{display_name(tool_name)} failed: {category}**\n\n"
        f"> {short_detail.replace(chr(10), chr(10) + '> ')}"
    )


def tool_output_markdown(
    tool_name: str,
    args: Dict[str, Any],
    result: Dict[str, Any],
    raw_result: str,
    *,
    agent_note: str = "",
) -> str:
    parts: List[str] = []
    if agent_note:
        parts.append(f"**Agent note**\n\n> {sanitize_text(agent_note).replace(chr(10), chr(10) + '> ')}")
    if tool_name == "todo_write":
        return ""
    if tool_name == "search":
        return render_search_output(args, result, agent_note)
    elif tool_name == "code_text_draft":
        return render_code_text_draft_output(result, agent_note)
    elif tool_name == "vlm_review":
        return render_vlm_review_output(result, agent_note)
    else:
        fields = visible_fields(tool_name, result, args)
        if fields:
            parts.append(fields)
        if tool_name == "search" and result.get("reference_paths"):
            parts.append(field_markdown("reference_paths", [
                Path(str(path)).name for path in result.get("reference_paths", [])
            ]))
    if not parts:
        parts.append(f"{display_name(tool_name)} completed.")
    return "\n\n".join(parts)


def result_images(tool_name: str, result: Dict[str, Any]) -> List[str]:
    cfg = tool_config(tool_name)
    fields = cfg.get("image_result_fields") or []
    if not isinstance(fields, list):
        return []
    images: List[str] = []
    for field_name in fields:
        value = result.get(field_name)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and Path(item).is_file() and Path(item).suffix.lower() in IMAGE_SUFFIXES:
                images.append(item)
    return images


def manifest_artifacts(run: AgentRun, tool_call_id: str, tool_name: str) -> List[Dict[str, Any]]:
    artifacts_path = Path(run.session.dir) / "artifacts.jsonl"
    if not artifacts_path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in artifacts_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("tool_call_id") != tool_call_id and entry.get("tool_name") != tool_name:
            continue
        artifact = entry.get("artifact")
        if isinstance(artifact, dict):
            out.append(artifact)
    return out


def configured_artifact_images(run: AgentRun, tool_call_id: str, tool_name: str) -> List[str]:
    cfg = tool_config(tool_name)
    names = cfg.get("artifact_names") or []
    roles = cfg.get("artifact_roles") or []
    if not isinstance(names, list):
        names = []
    if not isinstance(roles, list):
        roles = []
    artifacts = manifest_artifacts(run, tool_call_id, tool_name)
    valid_artifacts: List[Dict[str, Any]] = []
    for artifact in artifacts:
        path = artifact.get("path")
        if not isinstance(path, str) or not Path(path).is_file():
            continue
        suffix = Path(path).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        valid_artifacts.append(artifact)

    for name in names:
        matches = [
            artifact["path"]
            for artifact in valid_artifacts
            if Path(str(artifact.get("path"))).name == name
        ]
        if matches:
            return matches

    selected: List[str] = []
    for artifact in valid_artifacts:
        if artifact.get("role") in roles:
            selected.append(str(artifact["path"]))
    return selected


def image_elements(paths: List[str]) -> List[cl.Image]:
    unique: List[str] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    size = "small" if len(unique) > 1 else "medium"
    return [
        cl.Image(path=path, name=Path(path).name, display="inline", size=size)
        for path in unique
    ]


def best_badge_image(path: str, run: AgentRun, tool_call_id: str) -> str:
    """Create a small green-marked copy for the best search reference."""
    try:
        from PIL import Image as PILImage, ImageDraw
    except Exception:
        return path
    try:
        source = Path(path)
        if not source.is_file() or source.suffix.lower() == ".svg":
            return path
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", tool_call_id or "search")
        out_dir = Path(run.session.dir) / "ui_assets"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{safe_id}_{source.stem}_best.png"
        if out_path.is_file():
            return str(out_path)
        with PILImage.open(source) as img:
            canvas = img.convert("RGB")
            border = max(10, min(canvas.size) // 40)
            label_h = max(42, border * 3)
            marked = PILImage.new("RGB", (canvas.width + border * 2, canvas.height + border * 2 + label_h), "#16a34a")
            marked.paste(canvas, (border, border + label_h))
            draw = ImageDraw.Draw(marked)
            draw.text((border * 2, max(8, label_h // 4)), "BEST MATCH", fill="white")
            marked.save(out_path)
        return str(out_path)
    except Exception:
        return path


def search_image_elements(run: AgentRun, tool_call_id: str, result: Dict[str, Any]) -> List[cl.Image]:
    items = result.get("image_results") or []
    if not isinstance(items, list) or not items:
        items = []

    elements: List[cl.Image] = []
    max_images = int(tool_config("search").get("max_images") or 8)
    for item in items[:max_images]:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if not isinstance(path, str) or not Path(path).is_file():
            continue
        is_primary = item.get("role") == "primary"
        display_path = best_badge_image(path, run, tool_call_id) if is_primary else path
        score = item.get("rerank_score")
        score_text = f" · score {float(score):.2f}" if isinstance(score, (int, float)) else ""
        query = sanitize_text(item.get("query", ""))
        name = ("BEST" if is_primary else f"#{item.get('rank', '?')} backup") + score_text
        if query:
            name += f" · {query[:60]}"
        elements.append(cl.Image(path=display_path, name=name, display="inline", size="small"))

    candidate_items = result.get("candidate_image_results") or []
    if not isinstance(candidate_items, list) or not candidate_items:
        candidate_items = []
        for artifact in manifest_artifacts(run, tool_call_id, "search"):
            path = artifact.get("path")
            name = Path(str(path)).name if path else ""
            if (
                isinstance(path, str)
                and Path(path).is_file()
                and Path(path).suffix.lower() in IMAGE_SUFFIXES
                and (artifact.get("role") == "rerank_candidate" or name.startswith("_rerank_"))
            ):
                metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
                candidate_items.append({
                    "path": path,
                    "rank": metadata.get("rank"),
                    "role": metadata.get("role") or "candidate",
                    "rerank_score": metadata.get("rerank_score"),
                    "rerank_reason": metadata.get("rerank_reason"),
                })

    max_candidates = int(tool_config("search").get("max_candidate_images") or 8)
    for item in candidate_items[:max_candidates]:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("local_path")
        if not isinstance(path, str) or not Path(path).is_file():
            continue
        score = item.get("rerank_score")
        score_text = f" · score {float(score):.2f}" if isinstance(score, (int, float)) else ""
        selected = item.get("role") == "selected"
        display_path = best_badge_image(path, run, f"{tool_call_id}_candidate") if selected else path
        name = f"Candidate #{item.get('rank', '?')}" + (" · selected" if selected else "") + score_text
        elements.append(cl.Image(path=display_path, name=name, display="inline", size="small"))

    if not elements:
        return image_elements(limit_images("search", result_images("search", result)))
    return elements


def code_text_process_items(run: AgentRun, tool_call_id: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = result.get("process_images") or []
    valid: List[Dict[str, Any]] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if isinstance(path, str) and Path(path).is_file() and Path(path).suffix.lower() in IMAGE_SUFFIXES:
                valid.append(item)

    if not valid:
        artifacts = manifest_artifacts(run, tool_call_id, "code_text_draft")
        order = {"background": 0, "draft": 1, "final": 2}
        for artifact in sorted(artifacts, key=lambda a: order.get(str(a.get("role")), 99)):
            path = artifact.get("path")
            if isinstance(path, str) and Path(path).is_file() and Path(path).suffix.lower() in IMAGE_SUFFIXES:
                role = str(artifact.get("role") or "artifact")
                if role in {"background", "draft", "final"}:
                    valid.append({
                        "path": path,
                        "role": role,
                        "label": artifact.get("label") or role,
                    })

    final_path = result.get("final_path")
    if isinstance(final_path, str) and Path(final_path).is_file():
        if all(item.get("path") != final_path for item in valid):
            valid.append({"path": final_path, "role": "final", "label": "Final image"})

    return valid


def code_text_image_elements(run: AgentRun, tool_call_id: str, result: Dict[str, Any]) -> List[cl.Image]:
    items = code_text_process_items(run, tool_call_id, result)
    role_names = {
        "background": "Background only",
        "draft": "Text/layout draft",
        "final": "Final",
    }
    elements: List[cl.Image] = []
    for item in items:
        path = str(item.get("path"))
        role = str(item.get("role") or "")
        label = sanitize_text(item.get("label") or role_names.get(role, Path(path).name))
        name = role_names.get(role, label)
        if label and label != name:
            name = f"{name} · {label}"
        elements.append(cl.Image(path=path, name=name, display="inline", size="small" if len(items) > 1 else "medium"))
    return elements


def limit_images(tool_name: str, paths: List[str]) -> List[str]:
    max_images = tool_config(tool_name).get("max_images")
    if not isinstance(max_images, int) or max_images <= 0:
        return paths
    return paths[:max_images]


async def close_active_step(state: UIState, event: ToolExecutionEndEvent) -> None:
    active = state.active_steps.get(event.tool_call_id)
    if active is None:
        return
    raw_result = ""
    result: Dict[str, Any] = {}
    if event.result is not None:
        raw_result = str(event.result.content)
        result = parse_jsonish(event.result.content)
    output = tool_output_markdown(
        event.tool_name,
        active.args,
        result,
        raw_result,
        agent_note=active.agent_note,
    )
    if is_failed_tool_result(event, result):
        error_output = render_tool_error_output(event.tool_name, result, raw_result)
        output = f"{error_output}\n\n{output}" if output else error_output
    images = configured_artifact_images(state.run, event.tool_call_id, event.tool_name)
    images.extend(result_images(event.tool_name, result))
    images = limit_images(event.tool_name, images)
    if event.tool_name in FINAL_IMAGE_TOOLS:
        final_candidates = images
        if event.tool_name == "code_text_draft":
            if result.get("requires_i2i") or result.get("output_role") == "text_structure_reference":
                final_candidates = []
            else:
                final_path = result.get("final_path")
                final_candidates = [final_path] if isinstance(final_path, str) and Path(final_path).is_file() else []
        for image_path in final_candidates:
            if image_path not in state.final_images:
                state.final_images.append(image_path)
    elapsed = time.time() - active.started_at
    active.step.output = f"{output}\n\n_Time: {elapsed:.1f}s_"
    if event.tool_name == "search":
        elements = search_image_elements(state.run, event.tool_call_id, result)
    elif event.tool_name == "code_text_draft":
        elements = code_text_image_elements(state.run, event.tool_call_id, result)
    else:
        elements = image_elements(images)
    if elements:
        active.step.elements = elements
    await active.step.__aexit__(None, None, None)
    state.active_steps.pop(event.tool_call_id, None)


async def handle_event(state: UIState, event: AgentEvent) -> None:
    if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage):
        text_parts = [part.text for part in event.message.content if isinstance(part, TextContent)]
        tool_calls = [part for part in event.message.content if isinstance(part, ToolCallContent)]
        text = "\n\n".join(text_parts).strip()
        if text and tool_calls:
            for call in tool_calls:
                state.pending_notes[call.id] = text
        elif text and not tool_calls:
            state.final_reply = render_agent_failure_text(text)
        return

    if isinstance(event, ToolExecutionStartEvent):
        if event.tool_name == "todo_write":
            await update_todo_message(state, dict(event.args))
            return

        cfg_subtitle = subtitle(event.tool_name)
        step = cl.Step(
            name=display_name(event.tool_name),
            type="tool",
            default_open=True,
            show_input=False,
        )
        step.output = f"{cfg_subtitle}\n\n_Running..._".strip()
        await step.__aenter__()
        active = ActiveStep(
            step=step,
            started_at=time.time(),
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id,
            args=dict(event.args),
            agent_note=state.pending_notes.pop(event.tool_call_id, ""),
        )
        state.active_steps[event.tool_call_id] = active
        return

    if isinstance(event, ToolExecutionEndEvent):
        await close_active_step(state, event)


async def consume_events(state: UIState, queue: "asyncio.Queue[Optional[AgentEvent]]") -> None:
    while True:
        event = await queue.get()
        if event is None:
            break
        await handle_event(state, event)


async def run_with_chainlit_events(user_prompt: str, state: UIState) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[AgentEvent]] = asyncio.Queue()
    consumer = asyncio.create_task(consume_events(state, queue))

    def emit(event: AgentEvent) -> None:
        persist_event(state.run.session, event)
        loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        await asyncio.to_thread(run_prompt, user_prompt, state.run, emit=emit)
    finally:
        await queue.put(None)
        await consumer


def render_perception_snapshot(snapshot: Dict[str, Any]) -> str:
    blocks = []
    painter = snapshot.get("painter_thoughts")
    if painter:
        blocks.append(field_markdown("painter_thoughts", painter))
    intent = snapshot.get("image_intent")
    if intent:
        blocks.append(field_markdown("image_intent", intent))

    return "\n\n".join(blocks) if blocks else "Intent understanding complete."


async def render_perception_step(state: UIState, user_prompt: str, user_image_path: Optional[str]) -> None:
    step = cl.Step(name="Understanding intent", type="llm", default_open=True, show_input=False)
    step.output = "Understanding the image intent, hard constraints, and planning questions..."
    await step.__aenter__()
    payload = await asyncio.to_thread(
        initialize_perception,
        user_prompt,
        user_image_path=user_image_path,
        session=state.run.session,
    )
    snapshot = payload.get("snapshot", {}) if isinstance(payload, dict) else {}
    step.name = "Intent understanding"
    step.output = render_perception_snapshot(snapshot)
    await step.__aexit__(None, None, None)


async def save_upload(message: cl.Message, run: AgentRun) -> Optional[str]:
    if not message.elements:
        return None
    upload_dir = Path(run.session.dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for element in message.elements:
        path = getattr(element, "path", None)
        name = getattr(element, "name", None) or "upload"
        if path and Path(path).is_file():
            target = upload_dir / Path(name).name
            target.write_bytes(Path(path).read_bytes())
            return str(target)
    return None


@cl.on_message
async def on_message(message: cl.Message) -> None:
    user_prompt = message.content.strip()
    if not user_prompt:
        await cl.Message(content="Please provide an image-generation request.").send()
        return

    run = prepare_agent_run()
    state = UIState(run=run)
    cl.user_session.set("cc_genclaw_session_id", run.session.session_id)

    user_image_path = await save_upload(message, run)

    try:
        await render_perception_step(state, user_prompt, user_image_path)
        await run_with_chainlit_events(user_prompt, state)
    except Exception as exc:
        await cl.Message(content=f"Run failed: `{type(exc).__name__}: {sanitize_text(exc)}`").send()
        raise

    elements = image_elements(state.final_images[-1:]) if state.final_images else None
    final = state.final_reply or "Done."
    await cl.Message(content=final, elements=elements).send()

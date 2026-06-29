"""AgentMessage <-> OpenAI protocol conversion.

cc/pi internal AgentMessage representation:
- UserMessage(content=str)
- AssistantMessage(content=[TextContent | ToolCallContent])
- ToolResultMessage(tool_call_id, tool_name, content, is_error)
- SystemMessage(content=str)

OpenAI wire format (per §4.3 protocol diff table):
- {role: "user", content: str}
- {role: "assistant", content: str | None, tool_calls: [...]}
- {role: "tool", tool_call_id, content: str}
- {role: "system", content: str}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from .types import (
    AgentMessage,
    AgentTool,
    AssistantMessage,
    SystemMessage,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
    make_tool_call_id,
)


# ---------------------------------------------------------------------------
# AgentMessage[] -> OpenAI messages[]
# ---------------------------------------------------------------------------


def convert_to_llm(messages: List[AgentMessage]) -> List[Dict[str, Any]]:
    """Convert internal AgentMessage list to OpenAI wire format."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            # system-reminder messages are still user role on the wire,
            # we just wrap content with the marker tag (cc style)
            content = m.content
            if m.is_system_reminder and not content.startswith("<system-reminder>"):
                content = f"<system-reminder>\n{content}\n</system-reminder>"
            out.append({"role": "user", "content": content})

        elif isinstance(m, AssistantMessage):
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in m.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ToolCallContent):
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.arguments, ensure_ascii=False),
                        },
                    })
            entry: Dict[str, Any] = {"role": "assistant"}
            # OpenAI requires `content` to be string-or-null; null is only
            # accepted when `tool_calls` is present. Use empty string as a
            # safe fallback for "no text, no tool_calls" edge cases (some
            # upstream proxies reject null-content + missing tool_calls).
            text = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                entry["content"] = text  # may be null — that's fine here
                entry["tool_calls"] = tool_calls
            else:
                entry["content"] = text if text is not None else ""
            out.append(entry)

        elif isinstance(m, ToolResultMessage):
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })

        elif isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})

        else:  # pragma: no cover
            raise TypeError(f"Unknown AgentMessage type: {type(m).__name__}")

    return out


# ---------------------------------------------------------------------------
# OpenAI response -> AssistantMessage
# ---------------------------------------------------------------------------


def parse_llm_response(response: Dict[str, Any]) -> AssistantMessage:
    """Parse one OpenAI chat completion response dict into an AssistantMessage."""
    if not response.get("choices"):
        return AssistantMessage(
            content=[],
            stop_reason="error",
            error_message="No choices in response",
        )

    choice = response["choices"][0]
    msg = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    blocks: List[Any] = []

    # Text content
    text = msg.get("content")
    if text and isinstance(text, str) and text.strip():
        blocks.append(TextContent(text=text))

    # Tool calls
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        tc_id = tc.get("id") or make_tool_call_id()
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except json.JSONDecodeError:
            # Malformed args from LLM — surface as empty dict, the downstream
            # validator will error out which the loop will catch.
            args = {"__raw_args__": raw_args, "__parse_error__": True}
        if not name:
            name = _infer_tool_name_from_args(args)
        blocks.append(ToolCallContent(id=tc_id, name=name, arguments=args))

    # Map OpenAI finish_reason to our stop_reason
    stop_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "filtered",
    }
    stop_reason = stop_map.get(finish_reason, finish_reason)

    return AssistantMessage(content=blocks, stop_reason=stop_reason)


def _infer_tool_name_from_args(args: Dict[str, Any]) -> str:
    """Recover from providers that occasionally drop function.name.

    The inference is intentionally conservative and only uses argument
    signatures that are unique in this agent's default tool set.
    """
    if not isinstance(args, dict):
        return ""
    keys = set(args)
    if "todos" in keys:
        return "todo_write"
    if {"criteria", "original_user_prompt", "image_path"}.issubset(keys):
        return "vlm_review"
    if "image_path" in keys and "prompt" in keys:
        return "i2i"
    if "draft_png_path" in keys and "svg_description" in keys:
        return "format_prompt"
    if "self_review" in keys and "prompt" in keys:
        return "code_scene_draft"
    if "expected_long_texts" in keys:
        return "code_text_draft"
    if "need_process_problem" in keys:
        return "search"
    return ""


# ---------------------------------------------------------------------------
# Tool list -> OpenAI tools array
# ---------------------------------------------------------------------------


def convert_tools_to_llm(
    tools: List[AgentTool],
    discovered: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert AgentTool list to OpenAI tools array format.

    Phase-6: handles deferred tools. For each tool:
      - if `should_defer == False` → send full description (eager loading)
      - if `should_defer == True` and tool.name in `discovered` → send full
        description (already revealed via tool_search)
      - if `should_defer == True` and tool.name NOT in `discovered` → send
        only `short_description` (or first line of full description as
        fallback). The tool name + parameters_schema are still sent so the
        model knows the tool exists and how to call it once discovered.

    `discovered` defaults to empty set (everything deferred is hidden).
    """
    discovered = discovered or set()
    out: List[Dict[str, Any]] = []
    for t in tools:
        should_defer = bool(getattr(t, "should_defer", False))
        is_revealed = (not should_defer) or (t.name in discovered)

        if is_revealed:
            description = t.description
        else:
            short = getattr(t, "short_description", None)
            if not short:
                # Fallback: first sentence/line of full description
                first_line = (t.description or "").splitlines()[0] if t.description else ""
                short = first_line or f"(deferred tool: {t.name})"
            description = short

        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": description,
                "parameters": t.parameters_schema,
            },
        })
    return out

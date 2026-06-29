"""Simple compaction — truncate old tool_results when context grows large.

Per PROJECT_PLAN §1.2 #4: do the SIMPLEST thing. We rough-estimate tokens by
chars/4, and if we exceed a threshold, we replace older tool_result content
with a placeholder. We always:
- preserve TodoWriteTool tool_results (white-list by tool_name)
- preserve the last N (default 3) tool_results regardless of size
- preserve all assistant and user messages (they're typically much shorter)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .types import AgentMessage, ToolResultMessage

DEFAULT_TOKEN_BUDGET = 60_000  # roughly 240k chars; comfortable for most models
DEFAULT_RECENCY_KEEP = 3
TODO_TOOL_NAMES = {"todo_write"}  # white-list (PROJECT_PLAN explicit requirement)

# D-07: image-generation tool_results carry the canonical `final_path` /
# `draft_png_path` that downstream tools (especially i2i) MUST extract.
# If compaction stubs these out, the LLM loses the ability to chain steps.
# Protected = compaction never replaces their content.
PROTECTED_TOOL_NAMES = TODO_TOOL_NAMES | {
    "t2i",
    "i2i",
    "search",
    "code_scene_draft",
    "code_text_draft",
    "format_prompt",
}


@dataclass
class CompactionConfig:
    token_budget: int = DEFAULT_TOKEN_BUDGET
    recency_keep: int = DEFAULT_RECENCY_KEEP


def estimate_tokens(messages: List[AgentMessage]) -> int:
    """Rough token estimate: total chars / 4."""
    total_chars = 0
    for m in messages:
        if hasattr(m, "content"):
            c = getattr(m, "content")
            if isinstance(c, str):
                total_chars += len(c)
            elif isinstance(c, list):
                for block in c:
                    text = getattr(block, "text", None)
                    if text:
                        total_chars += len(text)
                    name = getattr(block, "name", None)
                    if name:
                        total_chars += len(name) + 20  # tool_call overhead
    return total_chars // 4


def compact(
    messages: List[AgentMessage], config: CompactionConfig = None  # type: ignore[assignment]
) -> List[AgentMessage]:
    """Return a possibly-compacted copy of messages.

    Strategy: while estimate exceeds budget, replace oldest non-protected
    tool_result content with a stub. Returns a NEW list; does not mutate.
    """
    config = config or CompactionConfig()
    if estimate_tokens(messages) <= config.token_budget:
        return list(messages)

    # Identify all tool_result indices and decide which are protected
    n = len(messages)
    out = list(messages)

    # Find indices of tool_results, in order
    tr_indices = [i for i, m in enumerate(out) if isinstance(m, ToolResultMessage)]

    # The most recent `recency_keep` tool_results are always preserved.
    # NB: tr_indices[-0:] == tr_indices (entire list); guard explicitly.
    if config.recency_keep > 0 and tr_indices:
        recent_protected = set(tr_indices[-config.recency_keep:])
    else:
        recent_protected = set()

    for idx in tr_indices:
        if idx in recent_protected:
            continue
        m = out[idx]
        if not isinstance(m, ToolResultMessage):
            continue
        if m.tool_name in PROTECTED_TOOL_NAMES:
            # White-listed: never compact protected tool results (todo + image tools)
            continue
        # Replace with stub if not already
        if m.content.startswith("[Old tool result content cleared"):
            continue
        stub = f"[Old tool result content cleared by compaction; tool={m.tool_name}]"
        # Build a replacement (don't mutate the original)
        out[idx] = ToolResultMessage(
            tool_call_id=m.tool_call_id,
            tool_name=m.tool_name,
            content=stub,
            is_error=m.is_error,
        )

        if estimate_tokens(out) <= config.token_budget:
            break

    return out

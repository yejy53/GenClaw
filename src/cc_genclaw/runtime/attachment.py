"""Attachment mechanism — inject reminders into messages each turn.

This is the heart of cc's "cross-turn anchor": before every LLM call we
inject fresh user-role <system-reminder> messages. These messages are
ephemeral (never persisted to session JSONL); they only exist in the
LLM-facing message stream.

Hooks here:
- make_todo_reminder_hook  — current TodoWrite state
- make_perception_reminder_hook — painter notes from perception init

Both are designed as `transform_context` hooks; multiple can be
registered and they compose (the loop runs them in registration order).
"""

from __future__ import annotations

from typing import List

from .perception_state import PerceptionState, get_perception_state
from .todo_state import TodoState, get_todo_state
from ..types import AgentMessage, UserMessage


def make_todo_reminder_hook(state: TodoState = None):  # type: ignore[assignment]
    """Factory: returns a transform_context hook bound to a TodoState."""
    state = state or get_todo_state()

    def hook(messages: List[AgentMessage]) -> List[AgentMessage]:
        if state.is_empty():
            return messages
        body = state.render_for_reminder()
        # Append at the end so it's the latest user message before the LLM call
        return list(messages) + [UserMessage(content=body, is_system_reminder=True)]

    return hook


def make_perception_reminder_hook(state: PerceptionState = None):  # type: ignore[assignment]
    """Factory: returns a transform_context hook that injects painter
    notes as a <system-reminder> every turn (D-04).

    If perception hasn't been initialized (e.g. test code calls the
    loop directly without going through Agent.prompt), this is a no-op.
    """
    state = state or get_perception_state()

    def hook(messages: List[AgentMessage]) -> List[AgentMessage]:
        if not state.is_initialized():
            return messages
        body = state.render_for_reminder()
        if not body:
            return messages
        return list(messages) + [UserMessage(content=body, is_system_reminder=True)]

    return hook

"""TodoWrite tool — the cross-turn anchor.

Direct port of cc/src/tools/TodoWriteTool/prompt.ts semantics. The 184-line
prompt is mirrored in prompts/tool_cards/todo_write.yaml (Phase 3); this
module is just the storage backend.

Schema (deliberately minimal — does NOT reference any other tool):
- todos: [{ content, status, activeForm? }]
  - content and status are required.
  - activeForm (present continuous) is optional, stored but not yet rendered;
    reserved for a future progress UI (currently reminders show `content`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseTool
from ..runtime.todo_state import TodoItem, get_todo_state
from ..types import ToolResult


class TodoWriteTool(BaseTool):
    name = "todo_write"
    description = (
        "Update the todo list for the current session. Use proactively to "
        "track progress and pending tasks. At most one task may be in_progress "
        "at a time. Each task needs content (imperative) and status; activeForm "
        "(present continuous) is optional and reserved for future progress UI."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Imperative form, e.g. 'Run tests'.",
                        },
                        "activeForm": {
                            "type": "string",
                            "description": (
                                "Optional present continuous form, e.g. 'Running "
                                "tests'. Reserved for a future progress UI; not "
                                "required."
                            ),
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["content", "status"],
                },
                "description": "Full replacement of the todo list.",
            }
        },
        "required": ["todos"],
    }

    def execute(self, args: Dict[str, Any], signal: Optional[Any] = None) -> ToolResult:
        todos_raw = args.get("todos")
        if not isinstance(todos_raw, list):
            return self.err("'todos' must be a list of TodoItem dicts")

        items: list = []
        for i, t in enumerate(todos_raw):
            if not isinstance(t, dict):
                return self.err(f"todos[{i}] is not an object")
            content = t.get("content")
            status = t.get("status")
            # activeForm is optional now (reserved for a future progress UI).
            active_form = t.get("activeForm") or ""
            if not (content and status):
                return self.err(
                    f"todos[{i}] missing required field (content, status)"
                )
            if status not in ("pending", "in_progress", "completed"):
                return self.err(
                    f"todos[{i}] status must be pending|in_progress|completed, got {status!r}"
                )
            items.append(TodoItem(content=content, activeForm=active_form, status=status))

        # cc rule: at most ONE task in_progress at any time
        n_inprog = sum(1 for it in items if it.status == "in_progress")
        if n_inprog > 1:
            return self.err(
                f"At most one task may be in_progress; got {n_inprog}. "
                "Update the list so no more than one is currently in_progress."
            )

        get_todo_state().replace(items)

        # Render a snapshot back to the LLM as confirmation
        if not items:
            return self.ok("Todo list cleared.")
        snapshot_lines = ["Todo list updated:"]
        for i, it in enumerate(items, start=1):
            snapshot_lines.append(f"  {i}. [{it.status}] {it.content}")
        return self.ok("\n".join(snapshot_lines))

"""Completion guard — make sure an image request actually produced an image.

This agent is an image-generation agent: the deliverable of almost every
request is a final image. But the loop ends as soon as the assistant
emits a turn with no tool calls — so the agent can "finish" with only a
text reply (e.g. after `reason` it announces the answer and hands the
task back to the user without ever calling an image tool).

The completion guard closes that gap. It fires at the moment the agent
is about to end (via the loop's `get_followup_messages` hook):
- An `after_tool_call` hook marks `image_produced=True` when a final-image tool
  (t2i / i2i / code_text_draft pure_code/layered) returns success with a
  `final_path`. code_text_draft embedded_svg references that still require i2i
  do not count as final deliverables.
- When the agent tries to end with no image produced, the followup hook
  injects ONE <system-reminder> telling the agent to either finish the
  image or explain why it cannot. It fires at most `max_fires` times to
  avoid an infinite retry loop.

`code_scene_draft` is intentionally NOT counted as a final image: it
only yields a cartoon draft that MUST be upgraded by a following i2i.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from ..types import AgentMessage, UserMessage
from .perception_state import get_perception_state

# Tools whose successful output IS a final deliverable image.
_FINAL_IMAGE_TOOLS = {"t2i", "i2i", "code_text_draft"}


@dataclass
class CompletionState:
    image_produced: bool = False
    latest_final_path: str = ""
    guard_fired: int = 0
    max_fires: int = 1

    def note_tool_result(self, tool_name: str, result_content: str) -> None:
        try:
            data = json.loads(result_content)
        except Exception:  # noqa: BLE001
            return
        if tool_name not in _FINAL_IMAGE_TOOLS:
            return
        if (
            tool_name == "code_text_draft"
            and isinstance(data, dict)
            and (
                data.get("requires_i2i")
                or data.get("output_role") == "text_structure_reference"
                or data.get("render_type") in {"embedded", "embedded_svg"}
            )
        ):
            return
        if isinstance(data, dict) and data.get("status") == "success" and data.get("final_path"):
            self.image_produced = True
            self.latest_final_path = str(data.get("final_path") or "")


_GLOBAL_COMPLETION_STATE = CompletionState()


def get_completion_state() -> CompletionState:
    return _GLOBAL_COMPLETION_STATE


def reset_completion_state() -> None:
    """Test/example helper. Forget per-run completion tracking."""
    global _GLOBAL_COMPLETION_STATE
    _GLOBAL_COMPLETION_STATE = CompletionState()


_REMINDER_BODY = (
    "[completion-check] The user's original request is an image task, but "
    "this run has NOT produced any final image yet (no image tool returned "
    "a successful final_path). Do NOT stop here and do NOT hand the task "
    "back to the user as a follow-up step. Either:\n"
    "  (a) call the appropriate image tool now (t2i / i2i / code_text_draft, "
    "or code_scene_draft -> i2i) to actually produce the image and complete "
    "the request; or\n"
    "  (b) if the request genuinely cannot be fulfilled with an image, state "
    "clearly to the user WHY.\n"
    "Reasoning output alone is not the deliverable — the image is."
)

def make_completion_after_tool_hook(state: CompletionState = None):  # type: ignore[assignment]
    """Factory: after_tool_call hook that records final-image production."""
    state = state or get_completion_state()

    from ..hooks import AfterToolCallContext  # local import to avoid cycle

    def hook(ctx: "AfterToolCallContext"):
        if ctx.result is not None and not ctx.is_error:
            content = ctx.result.content
            state.note_tool_result(ctx.tool_name, content if isinstance(content, str) else str(content))
        return None

    return hook


def make_completion_followup_hook(state: CompletionState = None):  # type: ignore[assignment]
    """Factory: get_followup_messages callable for the loop.

    Fires when the agent is about to end. If perception ran (i.e. this is
    a real image request) and no final image was produced, inject one
    <system-reminder> to push the agent to finish — at most max_fires
    times to prevent an infinite loop.
    """
    state = state or get_completion_state()

    def hook() -> List[AgentMessage]:
        if state.image_produced:
            return []
        # Only guard genuine image requests (perception initialized).
        if not get_perception_state().is_initialized():
            return []
        if state.guard_fired >= state.max_fires:
            return []
        state.guard_fired += 1
        return [UserMessage(content=_REMINDER_BODY, is_system_reminder=True)]

    return hook

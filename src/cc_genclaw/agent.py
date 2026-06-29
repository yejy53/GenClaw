"""Stateful Agent wrapper — direct port of pi `agent.ts::Agent`.

A thin facade over run_agent_loop that keeps state across calls:
- maintains AgentContext (system prompt + messages + tools)
- exposes prompt() / continue_run() / abort()
- subscribers receive AgentEvent stream
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .loop import LoopConfig, run_agent_loop
from .types import (
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentTool,
    UserMessage,
)


@dataclass
class Agent:
    """Stateful agent. Holds messages between prompts."""

    system_prompt: str
    tools: List[AgentTool] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)

    config: Optional[LoopConfig] = None
    _subscribers: List[Callable[[AgentEvent], None]] = field(default_factory=list)

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Register an event subscriber. Returns an unsubscribe callable."""
        self._subscribers.append(fn)
        def _unsub() -> None:
            try:
                self._subscribers.remove(fn)
            except ValueError:
                pass
        return _unsub

    def _emit(self, event: AgentEvent) -> None:
        for s in self._subscribers:
            try:
                s(event)
            except Exception:  # noqa: BLE001
                pass

    def prompt(self, user_input: str, user_image_path: Optional[str] = None) -> List[AgentMessage]:
        """Send a user message and run the loop. Returns new messages produced.

        cc-genclaw extension (D-16, D-01): on first call we run perception
        once (intent_analysis → painter notes) and store the result in
        the global _GLOBAL_PERCEPTION_STATE. The reminder hook
        registered on this Agent's LoopConfig will pick it up and inject
        it as <system-reminder> every turn. Subsequent calls to
        prompt() do NOT re-run perception (PoC: multi-turn keeps the
        original notes). Use cc_genclaw.runtime.perception_state
        .reset_perception_state() between independent tasks.
        """
        if self.config is None:
            raise RuntimeError("Agent.config must be set before calling prompt()")

        # D-16, D-01: perception init at prompt entry
        try:
            from .runtime.perception_init import run_perception_init
            run_perception_init(user_input, user_image_path)
        except Exception as e:  # noqa: BLE001
            # Never let perception failure crash the agent — the
            # in-module fallback (DEFAULT_PERCEPTION sentinel) already
            # handles the normal failure path. This is a last-resort
            # safety net for unexpected import-time issues.
            import sys as _sys
            print(
                f"[Agent.prompt] perception_init crashed unexpectedly: "
                f"{type(e).__name__}: {e}. Proceeding without painter notes.",
                file=_sys.stderr,
            )

        prompt_msg = UserMessage(content=user_input)
        ctx = AgentContext(
            system_prompt=self.system_prompt,
            messages=list(self.messages),
            tools=list(self.tools),
        )
        new_msgs = run_agent_loop([prompt_msg], ctx, self.config, self._emit)
        # Persist new messages onto agent state
        self.messages.extend(new_msgs)
        return new_msgs

    def continue_run(self) -> List[AgentMessage]:
        """Continue from current messages without adding a new prompt.

        The last message must be a user or tool_result. Used after a transient
        error to retry from the current state, OR after an external system
        appended user/tool_result content directly to ``self.messages``.
        """
        if self.config is None:
            raise RuntimeError("Agent.config must be set")
        if not self.messages:
            raise ValueError("Cannot continue: no messages")
        last = self.messages[-1]
        last_role = getattr(last, "role", None)
        if last_role not in ("user", "tool"):
            raise ValueError(
                f"Cannot continue: last message role must be 'user' or 'tool', "
                f"got {last_role!r}"
            )

        ctx = AgentContext(
            system_prompt=self.system_prompt,
            messages=list(self.messages),
            tools=list(self.tools),
        )
        new_msgs = run_agent_loop([], ctx, self.config, self._emit)
        self.messages.extend(new_msgs)
        return new_msgs

    def reset(self) -> None:
        """Clear all messages. Subscribers and config are preserved."""
        self.messages = []

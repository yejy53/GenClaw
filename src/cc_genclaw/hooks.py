"""Hook system — 5 hook points referenced by loop.py.

Hook points:
- transform_context(messages) -> messages   # called before each LLM call (e.g. inject todo reminder)
- before_agent_start(context)               # called once at agent_start
- before_tool_call(ctx) -> {block?, reason?}  # called before each tool execute
- after_tool_call(ctx) -> {content?, terminate?, ...}  # called after each tool execute
- on_agent_end(messages)                     # called when agent_end emits

This is a minimal stub; concrete hook callbacks are registered via Hooks.register().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .types import AgentMessage, AssistantMessage, ToolResult


@dataclass
class BeforeToolCallContext:
    tool_name: str
    args: Dict[str, Any]
    tool_call_id: str
    assistant_message: Optional[AssistantMessage] = None


@dataclass
class AfterToolCallContext:
    tool_name: str
    args: Dict[str, Any]
    tool_call_id: str
    result: Optional[ToolResult] = None
    is_error: bool = False


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: Optional[str] = None
    content: Optional[str] = None
    is_error: bool = True


@dataclass
class AfterToolCallResult:
    content: Optional[str] = None
    is_error: Optional[bool] = None
    terminate: Optional[bool] = None


# Hook callable types (sync for simplicity)
TransformContextFn = Callable[[List[AgentMessage]], List[AgentMessage]]
BeforeAgentStartFn = Callable[[], None]
BeforeToolCallFn = Callable[[BeforeToolCallContext], Optional[BeforeToolCallResult]]
AfterToolCallFn = Callable[[AfterToolCallContext], Optional[AfterToolCallResult]]
OnAgentEndFn = Callable[[List[AgentMessage]], None]


@dataclass
class Hooks:
    transform_context: List[TransformContextFn] = field(default_factory=list)
    before_agent_start: List[BeforeAgentStartFn] = field(default_factory=list)
    before_tool_call: List[BeforeToolCallFn] = field(default_factory=list)
    after_tool_call: List[AfterToolCallFn] = field(default_factory=list)
    on_agent_end: List[OnAgentEndFn] = field(default_factory=list)

    def register_transform_context(self, fn: TransformContextFn) -> None:
        self.transform_context.append(fn)

    def register_before_agent_start(self, fn: BeforeAgentStartFn) -> None:
        self.before_agent_start.append(fn)

    def register_before_tool_call(self, fn: BeforeToolCallFn) -> None:
        self.before_tool_call.append(fn)

    def register_after_tool_call(self, fn: AfterToolCallFn) -> None:
        self.after_tool_call.append(fn)

    def register_on_agent_end(self, fn: OnAgentEndFn) -> None:
        self.on_agent_end.append(fn)

    # ---- runners ----

    def run_transform_context(
        self, messages: List[AgentMessage]
    ) -> List[AgentMessage]:
        for fn in self.transform_context:
            try:
                messages = fn(messages)
            except Exception:  # noqa: BLE001 -- hooks must not break loop
                pass
        return messages

    def run_before_agent_start(self) -> None:
        for fn in self.before_agent_start:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def run_before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> Optional[BeforeToolCallResult]:
        # First hook to return block=True wins
        for fn in self.before_tool_call:
            try:
                r = fn(ctx)
                if r and r.block:
                    return r
            except Exception:  # noqa: BLE001
                pass
        return None

    def run_after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> Optional[AfterToolCallResult]:
        merged = AfterToolCallResult()
        any_set = False
        for fn in self.after_tool_call:
            try:
                r = fn(ctx)
                if r is None:
                    continue
                any_set = True
                if r.content is not None:
                    merged.content = r.content
                if r.is_error is not None:
                    merged.is_error = r.is_error
                if r.terminate is not None:
                    merged.terminate = r.terminate
            except Exception:  # noqa: BLE001
                pass
        return merged if any_set else None

    def run_on_agent_end(self, messages: List[AgentMessage]) -> None:
        for fn in self.on_agent_end:
            try:
                fn(messages)
            except Exception:  # noqa: BLE001
                pass

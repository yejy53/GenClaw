"""Main agent loop — direct port of pi `agent-loop.ts::runLoop`.

Two-level while structure (see PROJECT_PLAN §3.5):
  outer while: handles follow-up message queue
    inner while: drives tool calls + steering messages
      1. inject pending messages
      2. stream assistant response (LLM call)
      3. extract tool calls from assistant message
      4. execute tools (parallel by default)
      5. emit turn_end
      6. check stop conditions, gather next steering messages
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .hooks import (
    AfterToolCallContext,
    BeforeToolCallContext,
    Hooks,
)
from .messages import convert_to_llm, convert_tools_to_llm, parse_llm_response
from .runtime.artifacts import tool_artifact_context
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AssistantMessage,
    EventSink,
    MessageEndEvent,
    MessageStartEvent,
    TextContent,
    ToolCallContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
    UserMessage,
)


# Pluggable LLM driver: takes (messages, tools, system, model) -> raw response dict
LLMDriver = Callable[..., Dict[str, Any]]


@dataclass
class LoopConfig:
    llm_driver: LLMDriver
    """Callable that takes (messages, tools, system, model) and returns OpenAI-style dict."""

    hooks: Hooks = field(default_factory=Hooks)

    max_turns: int = 60
    """Hard upper bound on total turns to prevent runaway loops."""

    model: Optional[str] = None
    """Override model name; if None, llm_driver uses its default."""

    get_steering_messages: Optional[Callable[[], List[AgentMessage]]] = None
    get_followup_messages: Optional[Callable[[], List[AgentMessage]]] = None


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_agent_loop(
    prompt_messages: List[AgentMessage],
    context: AgentContext,
    config: LoopConfig,
    emit: Optional[EventSink] = None,
) -> List[AgentMessage]:
    """Run the agent loop with new prompt messages.

    Returns the list of NEW messages added during this run (does not include
    pre-existing context.messages).
    """
    sink: EventSink = emit or (lambda _e: None)

    new_messages: List[AgentMessage] = list(prompt_messages)
    current_ctx = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompt_messages),
        tools=list(context.tools),
    )

    config.hooks.run_before_agent_start()
    sink(AgentStartEvent())
    sink(TurnStartEvent())
    for p in prompt_messages:
        sink(MessageStartEvent(message=p))
        sink(MessageEndEvent(message=p))

    _run_loop(current_ctx, new_messages, config, sink)

    config.hooks.run_on_agent_end(new_messages)
    return new_messages


def _run_loop(
    ctx: AgentContext,
    new_messages: List[AgentMessage],
    config: LoopConfig,
    emit: EventSink,
) -> None:
    """The dual-loop core."""
    first_turn = True
    pending_messages: List[AgentMessage] = []
    if config.get_steering_messages:
        pending_messages = config.get_steering_messages() or []

    turn_count = 0

    # Outer loop: follow-up queue
    while True:
        has_more_tool_calls = True

        # Inner loop: tool calls + steering
        while has_more_tool_calls or pending_messages:
            if turn_count >= config.max_turns:
                # Surface as soft termination, not exception
                err = AssistantMessage(
                    content=[TextContent(text=f"[max_turns={config.max_turns} reached, stopping]")],
                    stop_reason="max_turns",
                )
                new_messages.append(err)
                ctx.messages.append(err)
                emit(TurnEndEvent(message=err, tool_results=[]))
                emit(AgentEndEvent(messages=new_messages))
                return
            turn_count += 1

            if not first_turn:
                emit(TurnStartEvent())
            else:
                first_turn = False

            # Inject pending steering messages before next assistant response
            if pending_messages:
                for m in pending_messages:
                    emit(MessageStartEvent(message=m))
                    emit(MessageEndEvent(message=m))
                    ctx.messages.append(m)
                    new_messages.append(m)
                pending_messages = []

            # Stream assistant response (one LLM call)
            asst_msg = _stream_assistant_response(ctx, config, emit)
            new_messages.append(asst_msg)
            ctx.messages.append(asst_msg)

            if asst_msg.stop_reason in ("error", "aborted"):
                emit(TurnEndEvent(message=asst_msg, tool_results=[]))
                emit(AgentEndEvent(messages=new_messages))
                return

            # Extract tool calls
            tool_calls = [c for c in asst_msg.content if isinstance(c, ToolCallContent)]
            tool_results: List[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                tool_results = _execute_tool_calls(
                    ctx, asst_msg, tool_calls, config, emit
                )
                # terminate semantics: stop only if every result asks to terminate
                if tool_results and all(
                    _result_terminate(tr) for tr in tool_results
                ):
                    has_more_tool_calls = False
                else:
                    has_more_tool_calls = True

                for tr in tool_results:
                    ctx.messages.append(tr)
                    new_messages.append(tr)
            else:
                has_more_tool_calls = False

            emit(TurnEndEvent(message=asst_msg, tool_results=tool_results))

            # Check steering between turns
            if config.get_steering_messages:
                pending_messages = config.get_steering_messages() or []

        # Inner loop exited — no more tool calls. Check follow-up.
        if config.get_followup_messages:
            followups = config.get_followup_messages() or []
            if followups:
                pending_messages = followups
                continue

        break

    emit(AgentEndEvent(messages=new_messages))


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------


def _stream_assistant_response(
    ctx: AgentContext, config: LoopConfig, emit: EventSink
) -> AssistantMessage:
    """One LLM round-trip. Applies transform_context hook, calls driver, parses."""
    transformed = config.hooks.run_transform_context(ctx.messages)
    llm_messages = convert_to_llm(transformed)
    if ctx.tools:
        # Phase-6: deferred tools — only reveal full description for tools that
        # have been discovered via tool_search. Lazy import to avoid cycle.
        from .runtime.tool_discovery import get_discovered_tools_state
        discovered = get_discovered_tools_state().snapshot()
        llm_tools = convert_tools_to_llm(
            ctx.tools,
            discovered=discovered,
        )
    else:
        llm_tools = None

    try:
        response = config.llm_driver(
            messages=llm_messages,
            tools=llm_tools,
            system=ctx.system_prompt,
            model=config.model,
        )
    except Exception as e:  # noqa: BLE001
        # Encode as error AssistantMessage; do not raise.
        return AssistantMessage(
            content=[TextContent(text=f"[LLM error: {type(e).__name__}: {e}]")],
            stop_reason="error",
            error_message=str(e),
        )

    asst = parse_llm_response(response)
    emit(MessageStartEvent(message=asst))
    emit(MessageEndEvent(message=asst))
    return asst


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool_calls(
    ctx: AgentContext,
    asst_msg: AssistantMessage,
    tool_calls: List[ToolCallContent],
    config: LoopConfig,
    emit: EventSink,
) -> List[ToolResultMessage]:
    """Execute all tool calls in a batch, sequentially.

    PoC choice: sequential execution. Parallel can be added later via
    threading; cc/pi support both but it complicates emission ordering.
    Sequential is the safer default and matches most cc workflows.
    """
    results: List[ToolResultMessage] = []
    tool_map = {t.name: t for t in ctx.tools}

    for tc in tool_calls:
        emit(ToolExecutionStartEvent(
            tool_call_id=tc.id,
            tool_name=tc.name,
            args=dict(tc.arguments),
        ))

        before_ctx = BeforeToolCallContext(
            tool_name=tc.name,
            args=tc.arguments,
            tool_call_id=tc.id,
            assistant_message=asst_msg,
        )
        before = config.hooks.run_before_tool_call(before_ctx)
        if before and before.block:
            reason = before.reason or "blocked by before_tool_call hook"
            content = before.content if before.content is not None else f"[blocked: {reason}]"
            is_error = bool(before.is_error)
            tr = ToolResultMessage(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=content,
                is_error=is_error,
            )
            emit(ToolExecutionEndEvent(
                tool_call_id=tc.id, tool_name=tc.name,
                result=ToolResult(content=tr.content, is_error=is_error),
                is_error=is_error,
            ))
            results.append(tr)
            continue

        # Look up tool
        tool = tool_map.get(tc.name)
        if tool is None:
            tr = ToolResultMessage(
                tool_call_id=tc.id,
                tool_name=tc.name,
                content=f"[tool not found: {tc.name}]",
                is_error=True,
            )
            emit(ToolExecutionEndEvent(
                tool_call_id=tc.id, tool_name=tc.name,
                result=ToolResult(content=tr.content, is_error=True),
                is_error=True,
            ))
            results.append(tr)
            continue

        # Execute (errors -> ToolResult with is_error=True)
        try:
            with tool_artifact_context(tc.name, tc.id):
                result = tool.execute(tc.arguments)
        except Exception as e:  # noqa: BLE001 -- INVARIANT 6: errors are values
            result = ToolResult(
                content=f"[execute error: {type(e).__name__}: {e}]",
                is_error=True,
            )

        # Run after_tool_call hook
        after_ctx = AfterToolCallContext(
            tool_name=tc.name,
            args=tc.arguments,
            tool_call_id=tc.id,
            result=result,
            is_error=result.is_error,
        )
        after = config.hooks.run_after_tool_call(after_ctx)
        if after:
            if after.content is not None:
                result.content = after.content
            if after.is_error is not None:
                result.is_error = after.is_error
            if after.terminate is not None:
                result.terminate = after.terminate

        emit(ToolExecutionEndEvent(
            tool_call_id=tc.id, tool_name=tc.name,
            result=result, is_error=result.is_error,
        ))

        tr = ToolResultMessage(
            tool_call_id=tc.id,
            tool_name=tc.name,
            content=result.content if isinstance(result.content, str) else str(result.content),
            is_error=result.is_error,
        )
        # Stash terminate hint via attribute for _result_terminate to read
        setattr(tr, "_terminate_hint", result.terminate)
        results.append(tr)

    return results


def _result_terminate(tr: ToolResultMessage) -> bool:
    return bool(getattr(tr, "_terminate_hint", False))

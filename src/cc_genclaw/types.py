"""Core types — AgentMessage / AgentTool / AgentEvent / ToolResult.

Direct port from cc/src/Tool.ts + pi/src/types.ts.

Design notes:
- AgentMessage is a tagged union by `role`. We use plain dataclasses with a
  `role` literal field so json serialization is trivial (no Pydantic
  discriminated union complexity needed).
- ToolResult uses Result-mode error handling (errors are values, not
  exceptions). See INVARIANT 6.
- AgentTool is a Protocol (duck typing) so concrete tools can be either
  classes or instances of dataclasses — both work.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Message types — internal AgentMessage union (cc/pi style)
# ---------------------------------------------------------------------------

# OpenAI tool call shape (we keep Anthropic-style "tool_use" name internally
# but the wire-format is OpenAI; conversion happens in messages.py)


@dataclass
class TextContent:
    """A text block in an assistant message."""

    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCallContent:
    """A tool_use block in an assistant message (cc style: type='tool_use')."""

    id: str
    name: str
    arguments: Dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


# An assistant message can interleave text and tool calls
AssistantBlock = Union[TextContent, ToolCallContent]


@dataclass
class UserMessage:
    """A user message — pure text or with attachments."""

    content: str
    role: Literal["user"] = "user"
    timestamp: float = field(default_factory=time.time)
    # Optional system-reminder marker (used by attachment mechanism)
    is_system_reminder: bool = False


@dataclass
class AssistantMessage:
    """An assistant message — may include tool calls."""

    content: List[AssistantBlock]
    role: Literal["assistant"] = "assistant"
    timestamp: float = field(default_factory=time.time)
    # stopReason: "end_turn" | "tool_use" | "error" | "aborted"
    stop_reason: str = "end_turn"
    error_message: Optional[str] = None


@dataclass
class ToolResultMessage:
    """A tool result message — fed back to the model after tool execution.

    On the wire, OpenAI uses role='tool' with a tool_call_id field (see
    messages.py for protocol conversion). Internally we keep this as a
    distinct message type for clarity.
    """

    tool_call_id: str
    tool_name: str
    content: str  # text representation of the result
    role: Literal["tool"] = "tool"
    is_error: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMessage:
    """A system message — typically only the first message in `messages`.

    The main system prompt is passed separately via AgentContext.system_prompt;
    SystemMessage here is for ad-hoc system-role messages if needed.
    """

    content: str
    role: Literal["system"] = "system"
    timestamp: float = field(default_factory=time.time)


# Union of all in-memory agent messages. AgentMessage is what flows through
# the loop; messages.py converts to/from the OpenAI wire format.
AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage, SystemMessage]


# ---------------------------------------------------------------------------
# Tool protocol — direct port of cc/src/Tool.ts + pi AgentTool
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result returned from a tool execution.

    `content` is the text shown to the model.
    `details` holds arbitrary structured data for logs/UI (not shown to model).
    `is_error` flips the wire-format error flag.
    `terminate` hints to the loop that the agent should stop after this batch.
    """

    content: str
    details: Any = None
    is_error: bool = False
    terminate: bool = False


@runtime_checkable
class AgentTool(Protocol):
    """The tool protocol every concrete tool must satisfy.

    Mirrors pi `AgentTool<TParameters, TDetails>` and cc Tool.ts.
    """

    name: str
    """Tool name as the model sees it (e.g., 'read_file', 'bash')."""

    description: str
    """Tool description (rendered from capability.yaml, fed into system prompt)."""

    parameters_schema: Dict[str, Any]
    """JSON-schema for the tool's parameters (OpenAI tools format)."""

    def execute(self, args: Dict[str, Any], signal: Optional[Any] = None) -> ToolResult:
        """Execute the tool. Errors are encoded in ToolResult.is_error, not raised."""
        ...


# ---------------------------------------------------------------------------
# Agent events — emitted by the loop for UI / logging
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """Base event class with discriminator. Subclasses add fields."""

    type: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentStartEvent(AgentEvent):
    type: str = "agent_start"


@dataclass
class AgentEndEvent(AgentEvent):
    messages: List[AgentMessage] = field(default_factory=list)
    type: str = "agent_end"


@dataclass
class TurnStartEvent(AgentEvent):
    type: str = "turn_start"


@dataclass
class TurnEndEvent(AgentEvent):
    message: Optional[AssistantMessage] = None
    tool_results: List[ToolResultMessage] = field(default_factory=list)
    type: str = "turn_end"


@dataclass
class MessageStartEvent(AgentEvent):
    message: Optional[AgentMessage] = None
    type: str = "message_start"


@dataclass
class MessageEndEvent(AgentEvent):
    message: Optional[AgentMessage] = None
    type: str = "message_end"


@dataclass
class ToolExecutionStartEvent(AgentEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    type: str = "tool_execution_start"


@dataclass
class ToolExecutionEndEvent(AgentEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    result: Optional[ToolResult] = None
    is_error: bool = False
    type: str = "tool_execution_end"


# ---------------------------------------------------------------------------
# Loop config & context
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Snapshot passed into the loop."""

    system_prompt: str
    messages: List[AgentMessage] = field(default_factory=list)
    tools: List[AgentTool] = field(default_factory=list)


EventSink = Callable[[AgentEvent], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_call_id() -> str:
    """Generate a stable id for a tool call (used when LLM doesn't provide one)."""
    return f"call_{uuid.uuid4().hex[:24]}"


def msg_to_dict(msg: AgentMessage) -> Dict[str, Any]:
    """Serialize an AgentMessage to a JSONL-friendly dict."""
    d = asdict(msg)
    return d

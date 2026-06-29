"""Session JSONL persistence.

One JSONL file per session under runs/<session_id>/trace.jsonl. Each line
is one AgentMessage (or AgentEvent if include_events=True). Resume rebuilds
messages by reading the JSONL back in order.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from .types import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)


class Session:
    """Append-only JSONL session log + reload helpers."""

    def __init__(self, session_id: Optional[str] = None, root_dir: str = "runs"):
        self.session_id = session_id or self._make_session_id()
        self.root_dir = root_dir
        self.dir = os.path.join(root_dir, self.session_id)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "trace.jsonl")

    @staticmethod
    def _make_session_id() -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"{ts}-{uuid.uuid4().hex[:6]}"

    def append_message(self, msg: AgentMessage) -> None:
        d = self._serialize_message(msg)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def append_event(self, name: str, payload: Dict[str, Any]) -> None:
        d = {"_kind": "event", "name": name, "payload": payload, "ts": time.time()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    @staticmethod
    def _serialize_message(msg: AgentMessage) -> Dict[str, Any]:
        if isinstance(msg, AssistantMessage):
            return {
                "_kind": "message",
                "role": "assistant",
                "content": [_block_to_dict(b) for b in msg.content],
                "stop_reason": msg.stop_reason,
                "error_message": msg.error_message,
                "timestamp": msg.timestamp,
            }
        if is_dataclass(msg):
            d = asdict(msg)
            d["_kind"] = "message"
            return d
        return {"_kind": "message", "raw": str(msg)}

    @classmethod
    def load(cls, session_id: str, root_dir: str = "runs") -> "Session":
        s = cls(session_id=session_id, root_dir=root_dir)
        if not os.path.exists(s.path):
            raise FileNotFoundError(s.path)
        return s

    def read_messages(self) -> List[AgentMessage]:
        """Reconstruct AgentMessage list from JSONL."""
        if not os.path.exists(self.path):
            return []
        out: List[AgentMessage] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("_kind") != "message":
                    continue
                m = _deserialize_message(d)
                if m is not None:
                    out.append(m)
        return out


def _block_to_dict(b: Any) -> Dict[str, Any]:
    if isinstance(b, TextContent):
        return {"type": "text", "text": b.text}
    if isinstance(b, ToolCallContent):
        return {
            "type": "tool_call",
            "id": b.id,
            "name": b.name,
            "arguments": b.arguments,
        }
    if is_dataclass(b):
        return asdict(b)
    return {"raw": str(b)}


def _deserialize_message(d: Dict[str, Any]) -> Optional[AgentMessage]:
    role = d.get("role")
    if role == "user":
        return UserMessage(
            content=d.get("content", ""),
            timestamp=d.get("timestamp", 0.0),
            is_system_reminder=d.get("is_system_reminder", False),
        )
    if role == "assistant":
        blocks: List[Any] = []
        for b in d.get("content", []):
            t = b.get("type")
            if t == "text":
                blocks.append(TextContent(text=b.get("text", "")))
            elif t == "tool_call":
                blocks.append(
                    ToolCallContent(
                        id=b.get("id", ""),
                        name=b.get("name", ""),
                        arguments=b.get("arguments", {}),
                    )
                )
        return AssistantMessage(
            content=blocks,
            stop_reason=d.get("stop_reason", "end_turn"),
            error_message=d.get("error_message"),
            timestamp=d.get("timestamp", 0.0),
        )
    if role == "tool":
        return ToolResultMessage(
            tool_call_id=d.get("tool_call_id", ""),
            tool_name=d.get("tool_name", ""),
            content=d.get("content", ""),
            is_error=d.get("is_error", False),
            timestamp=d.get("timestamp", 0.0),
        )
    if role == "system":
        return SystemMessage(content=d.get("content", ""), timestamp=d.get("timestamp", 0.0))
    return None

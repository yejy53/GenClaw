"""Shared runner wiring for CC-GenClaw entrypoints.

The example scripts and the Chainlit UI both need the same setup:
configuration, LLM client, default tools, runtime state reset, perception
initialization, session JSONL persistence, and a session-scoped artifact dir.
Keeping that wiring here prevents UI-specific drift from the normal examples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..compaction import CompactionConfig, compact
from ..config import get_session_root, load_genclaw_config
from ..llm import LLMClient, LLMConfig
from ..loop import LoopConfig, run_agent_loop
from ..prompts.capability_loader import attach_descriptions_to_tools
from ..prompts.system_loader import render_system_prompt
from ..runtime.attachment import (
    make_perception_reminder_hook,
    make_todo_reminder_hook,
)
from ..runtime.completion_guard import (
    make_completion_after_tool_hook,
    make_completion_followup_hook,
    reset_completion_state,
)
from ..runtime.perception_init import run_perception_init
from ..runtime.perception_state import get_perception_state, reset_perception_state
from ..runtime.session_dir import reset_session_dir
from ..runtime.todo_state import reset_todo_state
from ..session import Session
from ..tools import default_tools
from ..types import AgentContext, AgentEvent, AgentMessage, UserMessage


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AgentRun:
    """Prepared agent state for one user request."""

    context: AgentContext
    config: LoopConfig
    session: Session
    model: str


def prepare_agent_run(
    *,
    enabled_tool_names: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_turns: int = 60,
) -> AgentRun:
    """Create a fresh CC-GenClaw agent run with one shared session dir."""

    reset_todo_state()
    reset_perception_state()
    reset_completion_state()

    try:
        load_genclaw_config()
    except FileNotFoundError:
        # Preserve examples' permissive behavior: explicit env vars can still
        # provide credentials in contexts without a project-local config.yaml.
        pass

    cfg_llm = LLMConfig.from_env()
    if cfg_llm.provider == "openai_chat" and not cfg_llm.api_key:
        cfg_llm.api_key = os.environ.get("OPENAI_API_KEY", "")
    if not cfg_llm.base_url:
        cfg_llm.base_url = (
            os.environ.get("MAIN_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
    if not cfg_llm.model:
        cfg_llm.model = (
            os.environ.get("MAIN_LLM_MODEL_NAME")
            or os.environ.get("OPENAI_MODEL_NAME", "gpt-5.4")
        )
    if model:
        cfg_llm.model = model
    if not cfg_llm.api_key:
        raise RuntimeError("MAIN_LLM_API_KEY / OPENAI_API_KEY not set.")

    session_dir = Path(reset_session_dir(get_session_root())).resolve()
    session = Session(session_id=session_dir.name, root_dir=str(session_dir.parent))

    tools = default_tools()
    if enabled_tool_names is not None:
        tools = [tool for tool in tools if tool.name in enabled_tool_names]
    attach_descriptions_to_tools(
        tools,
        REPO_ROOT / "src" / "cc_genclaw" / "prompts" / "tool_cards",
    )

    context = AgentContext(
        system_prompt=render_system_prompt(model=cfg_llm.model),
        messages=[],
        tools=tools,
    )
    client = LLMClient(cfg_llm)
    config = LoopConfig(
        llm_driver=client.create,
        max_turns=max_turns,
        model=cfg_llm.model,
    )
    config.hooks.register_transform_context(make_perception_reminder_hook())
    config.hooks.register_transform_context(make_todo_reminder_hook())
    config.hooks.register_after_tool_call(make_completion_after_tool_hook())
    config.get_followup_messages = make_completion_followup_hook()

    compact_cfg = CompactionConfig()

    def compact_hook(messages: List[AgentMessage]) -> List[AgentMessage]:
        return compact(messages, compact_cfg)

    config.hooks.register_transform_context(compact_hook)
    return AgentRun(context=context, config=config, session=session, model=cfg_llm.model)


def initialize_perception(
    user_prompt: str,
    *,
    user_image_path: Optional[str],
    session: Session,
) -> Dict:
    """Run perception once and persist its snapshot to trace.jsonl."""

    run_perception_init(user_prompt, user_image_path)
    state = get_perception_state()
    payload = {
        "snapshot": state.snapshot(),
        "reminder": state.render_for_reminder(),
    }
    session.append_event("perception_snapshot", payload)
    return payload


def run_prompt(
    user_prompt: str,
    run: AgentRun,
    *,
    emit: Optional[Callable[[AgentEvent], None]] = None,
) -> List[AgentMessage]:
    """Run the loop and persist the resulting messages."""

    new_messages = run_agent_loop(
        [UserMessage(content=user_prompt)],
        run.context,
        run.config,
        emit=emit,
    )
    for message in new_messages:
        run.session.append_message(message)
    return new_messages


def persist_event(session: Session, event: AgentEvent) -> None:
    """Append a compact event record to the session trace."""

    try:
        session.append_event(event.type, event_payload(event))
    except Exception:
        return


def event_payload(event: AgentEvent) -> Dict:
    """Return a JSONL-safe event payload for trace persistence."""

    out: Dict = {}
    for attr in ("tool_name", "tool_call_id", "is_error", "args"):
        if hasattr(event, attr):
            value = getattr(event, attr)
            if isinstance(value, (str, int, float, bool, dict, list, type(None))):
                out[attr] = value
            else:
                out[attr] = str(value)
    result = getattr(event, "result", None)
    if result is not None:
        out["result_content"] = str(getattr(result, "content", ""))[:1000]
    return out

"""DiscoveredToolsState — global set of tools that have been "discovered" via ToolSearch.

cc background:
  In cc, deferred tools (`shouldDefer: true`) are sent to the LLM with a
  short DESCRIPTION + `defer_loading: true` flag. The LLM calls the
  `ToolSearchTool` to learn the full PROMPT of a tool. cc relies on the
  Anthropic API backend to expand the returned `tool_reference` content
  block into a full schema injection on the next turn.

OpenAI compat:
  OpenAI protocol has no `tool_reference` block, so we model the same
  behaviour client-side: keep a session-scoped set of "discovered" tool
  names, and consult it when serialising tools (`messages.convert_tools_to_llm`)
  to decide whether to send the full description or just the short one.

Lifecycle:
  - At session start, the set is empty → only short descriptions of
    deferred tools are sent.
  - When tool_search executes, it adds matched names to this set.
  - On the next LLM call, those tools' full descriptions are sent.
  - reset_discovered_tools() can be called for tests / fresh sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set


@dataclass
class DiscoveredToolsState:
    """Session-scoped set of tool names whose full description has been
    "discovered" via ToolSearch. NOT thread-safe (single-agent loop).
    """

    discovered: Set[str] = field(default_factory=set)

    def is_discovered(self, name: str) -> bool:
        return name in self.discovered

    def mark_discovered(self, names: Iterable[str]) -> None:
        for n in names:
            self.discovered.add(n)

    def clear(self) -> None:
        self.discovered.clear()

    def snapshot(self) -> Set[str]:
        return set(self.discovered)


# Module-level global. For tests, call reset_discovered_tools() between cases.
_GLOBAL_DISCOVERED = DiscoveredToolsState()


def get_discovered_tools_state() -> DiscoveredToolsState:
    return _GLOBAL_DISCOVERED


def reset_discovered_tools() -> None:
    _GLOBAL_DISCOVERED.clear()

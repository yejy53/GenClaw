"""BaseTool — abstract base implementing the AgentTool Protocol.

Each concrete tool subclasses this and overrides:
- name: str
- description: str (typically rendered from capability.yaml in Phase 3)
- parameters_schema: JSON-schema dict
- execute(args) -> ToolResult

Optional (added in phase-6 for ToolSearch support):
- should_defer: bool — if True, this tool is "deferred":
  the model only sees its short_description until ToolSearch reveals
  the full description. Mirrors cc's `shouldDefer: true`.
- short_description: str — one-liner shown to the model when this tool
  is deferred. Mirrors cc's `DESCRIPTION` constant.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import ToolResult


class BaseTool:
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    # Phase-6 additions for shouldDefer / ToolSearch mechanism
    should_defer: bool = False
    """If True, this tool is hidden behind ToolSearch until discovered.

    When False (default), the full 7-section description is sent on every
    LLM call (eager loading; matches cc's default for most tools).

    When True, the LLM only sees `short_description` initially. To learn
    the full description, the model must call `tool_search`. Once
    discovered (tracked in runtime.tool_discovery._GLOBAL_DISCOVERED),
    subsequent LLM calls receive the full description for this tool.
    """

    short_description: Optional[str] = None
    """One-liner sent to the LLM when this tool is deferred and not yet
    discovered. Mirrors cc's `DESCRIPTION` constant (vs the long `PROMPT`).

    If `should_defer=True` but `short_description=None`, the loader falls
    back to the first sentence of `description`.
    """

    def execute(self, args: Dict[str, Any], signal: Optional[Any] = None) -> ToolResult:
        raise NotImplementedError

    # Convenience helpers ----------------------------------------------------

    @staticmethod
    def ok(content: str, **kwargs: Any) -> ToolResult:
        return ToolResult(content=content, **kwargs)

    @staticmethod
    def err(message: str, **kwargs: Any) -> ToolResult:
        return ToolResult(content=message, is_error=True, **kwargs)

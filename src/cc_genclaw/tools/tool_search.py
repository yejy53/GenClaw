"""ToolSearchTool — discover deferred tools by keyword or exact selection.

This is the lookup mechanism for the shouldDefer pattern. A "deferred"
tool is one whose full 7-section description is hidden until the model
explicitly asks for it. The model asks via this tool.

Behaviour:
  - Input: { query: str }
      • "select:<ToolName>" — exact match (mirrors cc)
      • any other string — case-insensitive substring search across
        deferred tools' (name, short_description, description-first-line)
  - Side effect: matching tool names are added to the global
    DiscoveredToolsState. On the NEXT LLM call,
    messages.convert_tools_to_llm will detect these and send the
    full description for each.
  - Returns: a textual list of matched tools' full descriptions, so
    the model also has the info immediately in the current turn (we
    cannot rely on Anthropic's tool_reference auto-expansion under
    the OpenAI-compat protocol).

ToolSearchTool itself is NEVER deferred — it's the entry-point that
makes deferred tools reachable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseTool
from ..runtime.tool_discovery import get_discovered_tools_state
from ..types import ToolResult


# Module-level "tool registry": ToolSearchTool needs to know about the
# universe of tools to search across. The runner registers all tools here.
_REGISTERED_TOOLS: List[BaseTool] = []


def register_tools(tools: List[BaseTool]) -> None:
    """Register all tools so ToolSearch can search them. Replaces any prior set."""
    global _REGISTERED_TOOLS
    _REGISTERED_TOOLS = list(tools)


def get_registered_tools() -> List[BaseTool]:
    return list(_REGISTERED_TOOLS)


class ToolSearchTool(BaseTool):
    name = "tool_search"
    description = (
        "Search for and load deferred tools by keyword or exact name. "
        "Some tools are kept hidden behind a short summary to save tokens; "
        "use this to discover and learn their full description before calling them."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. Use 'select:<ToolName>' for exact match (e.g. "
                    "'select:bash'), or a keyword like 'shell' or 'git' for "
                    "case-insensitive substring search."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Cap on number of matches returned (default 5).",
            },
        },
        "required": ["query"],
    }

    # tool_search is NEVER deferred itself
    should_defer = False

    def execute(self, args: Dict[str, Any], signal: Optional[Any] = None) -> ToolResult:
        query = (args.get("query") or "").strip()
        if not query:
            return self.err("query is required")

        max_results = int(args.get("max_results") or 5)

        # Search universe = all deferred tools NOT already discovered
        state = get_discovered_tools_state()
        candidates = [
            t for t in _REGISTERED_TOOLS
            if t.should_defer and not state.is_discovered(t.name)
        ]

        if not candidates:
            return self.ok(
                "[no deferred tools available — all are already discovered or "
                "no tool_search-eligible tools registered]"
            )

        matches: List[BaseTool] = []

        # 1. Exact selection: "select:bash"
        if query.lower().startswith("select:"):
            target = query[len("select:"):].strip()
            for t in candidates:
                if t.name == target:
                    matches.append(t)
                    break
            if not matches:
                return self.err(
                    f"No deferred tool named {target!r}. "
                    f"Available deferred tools: {[t.name for t in candidates]}"
                )
        else:
            # 2. Keyword match: case-insensitive substring on name + short_description
            q = query.lower()
            for t in candidates:
                hay = " ".join([
                    t.name.lower(),
                    (t.short_description or "").lower(),
                    # also a small slice of full description for keyword recall
                    (t.description or "").lower()[:500],
                ])
                if q in hay:
                    matches.append(t)
                if len(matches) >= max_results:
                    break

        if not matches:
            return self.ok(
                f"[no deferred tools matched query={query!r}; "
                f"available deferred tools: {[t.name for t in candidates]}]"
            )

        # Mark discovered — next LLM call will see full description
        state.mark_discovered([t.name for t in matches])

        # Return a textual block with the full description of each match
        body_parts = [
            f"Discovered {len(matches)} tool(s). Their full descriptions are now "
            "available, and from the next turn onward the LLM will see the "
            "complete tool schema for each. Inline copy below for immediate use:\n"
        ]
        for t in matches:
            body_parts.append(f"=== Tool: {t.name} ===")
            body_parts.append(t.description)
            body_parts.append("")
        return self.ok("\n".join(body_parts))

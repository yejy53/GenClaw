"""Global painter-notes (perception) state singleton.

Perception is a planning-hints layer. It does not route tools and does not
pre-extract concrete tool arguments. The reminder exposes the painter's
understanding and explicit planning needs so the main LLM can plan with
todo_write and build tool arguments from the original prompt / tool results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PerceptionState:
    painter_thoughts: str = ""
    image_intent: str = ""
    planning_needs: List[Dict[str, Any]] = field(default_factory=list)
    notes_for_planner: List[str] = field(default_factory=list)
    user_image_path: Optional[str] = None
    _initialized: bool = False
    _fallback_used: bool = False

    def is_initialized(self) -> bool:
        return self._initialized

    def render_for_reminder(self) -> str:
        """Produce the <system-reminder> payload injected every turn."""
        if not self._initialized:
            return ""

        lines = ["[PAINTER NOTES — perception output]"]
        if self.painter_thoughts and self.painter_thoughts != "(perception unavailable)":
            lines.append("painter_thoughts:")
            lines.append(f"  {self.painter_thoughts}")

        lines.append(f"image_intent: {self.image_intent or '(use original prompt)'}")
        lines.append("[planning_needs — hints, not routing decisions]")
        if self.planning_needs:
            for idx, need in enumerate(self.planning_needs, 1):
                kind = need.get("need", "?")
                question = need.get("question", "")
                evidence = need.get("evidence", "")
                resolved = need.get("resolved", False)
                lines.append(
                    f"{idx}. {kind} | resolved={resolved} | question: {question or '(none)'}"
                )
                if evidence:
                    lines.append(f"   evidence: {evidence}")
        else:
            lines.append("(none listed; use the original prompt and tool cards directly)")

        if self.notes_for_planner:
            lines.append("notes_for_planner:")
            for note in self.notes_for_planner:
                lines.append(f"  - {note}")
        lines.append(f"user_image_path: {self.user_image_path!r}")

        if self._fallback_used:
            lines.append(
                "perception fallback was used; rely on the original user prompt "
                "and tool cards."
            )

        lines.append("")
        lines.append("[REMINDER]")
        lines.append(
            "These notes are hints, not tool routing rules. Open todo_write by "
            "default. Build concrete tool arguments from the original user "
            "prompt and prior tool_result.content at the moment you call each "
            "tool."
        )
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "painter_thoughts": self.painter_thoughts,
            "image_intent": self.image_intent,
            "planning_needs": list(self.planning_needs),
            "notes_for_planner": list(self.notes_for_planner),
            "user_image_path": self.user_image_path,
            "_initialized": self._initialized,
            "_fallback_used": self._fallback_used,
        }


_GLOBAL_PERCEPTION_STATE = PerceptionState()


def get_perception_state() -> PerceptionState:
    return _GLOBAL_PERCEPTION_STATE


def reset_perception_state() -> None:
    """Test/example helper. Forget initialization so next prompt re-runs."""
    global _GLOBAL_PERCEPTION_STATE
    _GLOBAL_PERCEPTION_STATE = PerceptionState()

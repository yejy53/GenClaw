"""Prompt-facing perception schema.

Perception is deliberately descriptive: it records how the image request was
understood and which planning needs appear to be present. It does NOT choose a
pipeline and does NOT pre-extract concrete tool arguments such as
`expected_long_texts` or object tables. The main LLM builds tool arguments at
call time from the original user prompt and prior tool results.
"""

from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "PlanningNeed",
    "PerceptionNote",
    "PLANNING_NEED_KINDS",
    "DEFAULT_PERCEPTION",
    "parse_perception",
]

PLANNING_NEED_KINDS = (
    "spatial_layout_fidelity",
    "verbatim_text_fidelity",
    "external_text_source",
    "external_visual_reference",
    "reasoning",
    "input_image_edit",
    "direct_generation",
)


class PlanningNeed(BaseModel):
    """One explicit planning need surfaced from the request.

    This is a hint for the planner, not a route. `evidence` should point
    back to the user wording that triggered the need so the main LLM can
    extract concrete tool arguments from the original prompt when needed.
    """

    model_config = ConfigDict(extra="ignore")

    need: str = Field(
        description=(
            "One of PLANNING_NEED_KINDS; unknown values normalize to "
            "direct_generation."
        )
    )
    question: str = Field(
        default="",
        description="Question / preservation requirement the planner should consider.",
    )
    evidence: str = Field(
        default="",
        description=(
            "Short quote or paraphrase from the user request that triggered "
            "this need."
        ),
    )
    resolved: bool = Field(
        default=False,
        description="Whether this need is already resolved from the user's prompt alone.",
    )

    @model_validator(mode="after")
    def _normalize_need(self) -> "PlanningNeed":
        if self.need not in PLANNING_NEED_KINDS:
            self.need = "direct_generation"
        return self


class PerceptionNote(BaseModel):
    """Painter notes: understanding + planning hints only."""

    model_config = ConfigDict(extra="ignore")

    painter_thoughts: str = Field(
        default="",
        description=(
            "2-5 sentence note capturing intent, needed pre-work, and "
            "non-obvious constraints."
        ),
    )
    image_intent: str = Field(
        default="",
        description=(
            "Plain-language visual intent; may be provisional when lookup/"
            "reasoning is needed."
        ),
    )
    planning_needs: List[PlanningNeed] = Field(
        default_factory=list,
        description=(
            "Explicit planning needs present in the request; omitted needs "
            "are not listed."
        ),
    )
    notes_for_planner: List[str] = Field(
        default_factory=list,
        description=(
            "Non-routing reminders for the main LLM. Do not include concrete "
            "tool args here."
        ),
    )


DEFAULT_PERCEPTION = PerceptionNote(
    painter_thoughts="(perception unavailable)",
    image_intent="(perception unavailable)",
    planning_needs=[],
    notes_for_planner=[
        "Perception failed. Use the original user prompt and tool cards directly."
    ],
)

_CORE_KEYS = (
    "painter_thoughts",
    "image_intent",
    "planning_needs",
    "notes_for_planner",
)


def _legacy_to_note(raw: dict[str, Any]) -> dict[str, Any]:
    """Best-effort adapter for old 11-field payloads during transition."""
    if any(k in raw for k in _CORE_KEYS):
        return raw

    needs: list[dict[str, Any]] = []
    for gap in raw.get("knowledge_gaps") or []:
        if isinstance(gap, dict):
            q = str(gap.get("question") or "").strip()
            if q:
                needs.append(
                    {
                        "need": "external_visual_reference",
                        "question": q,
                        "evidence": q,
                        "resolved": False,
                    }
                )
    if raw.get("needs_reasoning"):
        needs.append(
            {
                "need": "reasoning",
                "question": (
                    "Reason through the non-trivial visual/logical requirement "
                    "before generation."
                ),
                "evidence": str(raw.get("main_idea") or ""),
                "resolved": False,
            }
        )
    if raw.get("needs_spatial_control"):
        needs.append(
            {
                "need": "spatial_layout_fidelity",
                "question": (
                    "Preserve exact counts, positions, or layout from the "
                    "original request."
                ),
                "evidence": str(raw.get("main_idea") or ""),
                "resolved": True,
            }
        )
    if raw.get("needs_long_text_rendering"):
        needs.append(
            {
                "need": "verbatim_text_fidelity",
                "question": (
                    "Extract exact user-supplied text from the original prompt "
                    "at tool-call time."
                ),
                "evidence": str(raw.get("main_idea") or ""),
                "resolved": True,
            }
        )
    if not needs and raw.get("is_simple_case"):
        needs.append(
            {
                "need": "direct_generation",
                "question": "Request appears directly paintable from the original prompt.",
                "evidence": str(raw.get("main_idea") or ""),
                "resolved": True,
            }
        )

    return {
        "painter_thoughts": str(raw.get("painter_thoughts") or ""),
        "image_intent": str(raw.get("main_idea") or ""),
        "planning_needs": needs,
        "notes_for_planner": [
            "Legacy perception payload adapted. Build concrete tool arguments "
            "from the original prompt or tool results."
        ],
    }


def parse_perception(raw: Any) -> PerceptionNote:
    """Resilient parser: dict / JSON-string / PerceptionNote -> PerceptionNote."""
    if isinstance(raw, PerceptionNote):
        return raw
    if isinstance(raw, str):
        import json

        try:
            raw = json.loads(raw)
        except Exception:
            return DEFAULT_PERCEPTION.model_copy(deep=True)
    if not isinstance(raw, dict):
        return DEFAULT_PERCEPTION.model_copy(deep=True)
    if not any(k in raw for k in _CORE_KEYS) and not any(
        k in raw for k in ("main_idea", "knowledge_gaps", "needs_spatial_control")
    ):
        return DEFAULT_PERCEPTION.model_copy(deep=True)
    try:
        return PerceptionNote.model_validate(_legacy_to_note(raw))
    except Exception:
        return DEFAULT_PERCEPTION.model_copy(deep=True)

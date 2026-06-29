"""Perception subsystem for painter notes.

This package is intentionally separate from `tools/`: perception is not a
main-LLM callable tool. It is a runtime sub-LLM pass that runs once at
`Agent.prompt()` entry and writes planning hints into the reminder state.
"""

from .core import run_perception
from .schema import DEFAULT_PERCEPTION, PLANNING_NEED_KINDS, PerceptionNote, PlanningNeed, parse_perception

__all__ = [
    "DEFAULT_PERCEPTION",
    "PLANNING_NEED_KINDS",
    "PerceptionNote",
    "PlanningNeed",
    "parse_perception",
    "run_perception",
]

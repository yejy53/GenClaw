"""All reasoning scripts behind the `reason` atomic tool.

This directory holds everything needed to turn a prompt (+ optional
images) into reasoning conclusions: the 4-step CoT system prompt
(``reasoning_prompt.yaml``), image encoding, and robust JSON parsing.
Self-contained: credentials come from ``os.environ`` (populated by
``cc_genclaw.config.load_genclaw_config``).
"""

from .core import run_reasoning

__all__ = ["run_reasoning"]

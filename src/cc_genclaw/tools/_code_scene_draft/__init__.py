"""All scripts behind the code_scene_draft atomic tool.

Self-contained: configured via os.environ (OPENAI_* + optional
SCENE_DRAFT_* model overrides) and these modules only.

- prompts.py  : the 3 LLM prompts (generate / revise / review)
- generate.py : generate / review / revise + parsing/clamp/crop helpers
- render.py   : SVG/HTML -> PNG via playwright (dynamic canvas size)
- core.py     : draft_scene() pipeline (no i2i; i2i is a separate tool)
"""

from .core import draft_scene

__all__ = ["draft_scene"]

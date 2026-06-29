"""Core tools for the image-generation agent.

The default tool set is the image-generation atomic tools plus
``todo_write`` and ``tool_search``:

- ``t2i`` / ``i2i``         : text-to-image / image-to-image
- ``code_scene_draft``      : SVG layout draft for spatial / counting tasks
- ``code_text_draft``       : verbatim text rendering
- ``search`` / ``reason``   : world-knowledge lookup + multimodal reasoning
- ``format_prompt``         : fuse facts + references into one prompt
- ``vlm_review``            : final-image quality review
- ``todo_write`` / ``tool_search`` : planning + routing helpers
"""

from .base import BaseTool
from .code_scene_draft import CodeSceneDraftTool
from .code_text_draft import CodeTextDraftTool
from .format_prompt import FormatPromptTool
from .i2i import I2ITool
from .reason import ReasonTool
from .search import SearchTool
from .t2i import T2ITool
from .todo_write import TodoWriteTool
from .tool_search import ToolSearchTool, register_tools
from .vlm_review import VLMReviewTool

__all__ = [
    "BaseTool",
    "TodoWriteTool", "ToolSearchTool",
    "T2ITool", "I2ITool",
    "SearchTool", "ReasonTool", "FormatPromptTool",
    "CodeSceneDraftTool", "CodeTextDraftTool",
    "VLMReviewTool",
    "register_tools",
    "default_tools",
]


def default_tools() -> list:
    """Return the default tool set for the image-generation agent.

    Image-generation atomic tools + ``todo_write`` + ``tool_search``.
    This is the canonical entry point for callers.
    """
    tools = [
        TodoWriteTool(), ToolSearchTool(),
        T2ITool(), I2ITool(),
        SearchTool(), ReasonTool(), FormatPromptTool(),
        CodeSceneDraftTool(), CodeTextDraftTool(), VLMReviewTool(),
    ]
    register_tools(tools)
    return tools

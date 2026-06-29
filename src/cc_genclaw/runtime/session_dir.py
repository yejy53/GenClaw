"""Session-scoped output directory state.

Each agent run gets a unique session_dir like
    <repo>/runs/<YYYYMMDD-HHMMSS-<hex6>>/

Tools that write artifacts (search refs, draft png/svg, t2i/i2i final png)
read from `_GLOBAL_SESSION_DIR.current()` to know where to put their
outputs. Subdirectory by tool name + per-turn folder is the per-tool
wrapper's job.

Why a global singleton (vs threading the path through args):
- mirrors `runtime/todo_state.py` and `runtime/perception_state.py`
- BaseTool.execute(args) signature is fixed; tools shouldn't have to
  receive session_dir as an arg from the LLM (LLM should never need to
  know the session path)
- ReAct loop is single-threaded; safe.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionDirState:
    """Singleton holding the current session output dir."""

    _path: Optional[str] = None

    def current(self) -> str:
        """Return the current session dir, creating one if not yet set."""
        if self._path is None:
            self._path = self._make_new()
        return self._path

    def reset(self, root: Optional[str] = None) -> str:
        """Force a brand new session dir (used at example start)."""
        self._path = self._make_new(root)
        return self._path

    @staticmethod
    def _make_new(root: Optional[str] = None) -> str:
        if root is None:
            from cc_genclaw.config import get_session_root
            root = get_session_root()
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        out = Path(root) / sid
        out.mkdir(parents=True, exist_ok=True)
        return str(out)


_GLOBAL_SESSION_DIR = SessionDirState()


def get_session_dir() -> str:
    return _GLOBAL_SESSION_DIR.current()


def reset_session_dir(root: Optional[str] = None) -> str:
    return _GLOBAL_SESSION_DIR.reset(root)


def tool_output_dir(tool_name: str, turn_hint: Optional[int] = None) -> str:
    """Compute a per-tool, per-turn output directory under the session.

    Caller does not need to know the turn number; we use a timestamp +
    short uuid as a unique tag instead of a real turn counter (the loop
    doesn't pass turn into tool.execute). This is good enough for the
    artifact archival that the migration plan §4.3.3 outlines.
    """
    base = Path(get_session_dir()) / tool_name
    tag = f"turn-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
    out = base / tag
    out.mkdir(parents=True, exist_ok=True)
    return str(out)

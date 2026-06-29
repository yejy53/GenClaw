"""Artifact manifest helpers for session visualizations.

Tools still return their normal ToolResult payloads. This module only writes a
best-effort sidecar index so offline/real-time UIs can understand which files a
tool produced without guessing from filenames.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


_CURRENT_TOOL_NAME: ContextVar[Optional[str]] = ContextVar("cc_genclaw_tool_name", default=None)
_CURRENT_TOOL_CALL_ID: ContextVar[Optional[str]] = ContextVar("cc_genclaw_tool_call_id", default=None)


def current_tool_name() -> Optional[str]:
    return _CURRENT_TOOL_NAME.get()


def current_tool_call_id() -> Optional[str]:
    return _CURRENT_TOOL_CALL_ID.get()


@contextmanager
def tool_artifact_context(tool_name: str, tool_call_id: str) -> Iterator[None]:
    """Attach the current tool call id while a tool executes."""
    name_token = _CURRENT_TOOL_NAME.set(tool_name)
    id_token = _CURRENT_TOOL_CALL_ID.set(tool_call_id)
    try:
        yield
    finally:
        _CURRENT_TOOL_CALL_ID.reset(id_token)
        _CURRENT_TOOL_NAME.reset(name_token)


def infer_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
        return "image" if suffix != ".svg" else "svg"
    if suffix == ".json":
        return "json"
    if suffix == ".html":
        return "html"
    if suffix in {".txt", ".md"}:
        return "text"
    return "file"


def infer_role(path: Path) -> str:
    name = path.name.lower()
    if name == "manifest.json":
        return "manifest"
    if name == "meta.json":
        return "metadata"
    if name == "log.json":
        return "log"
    if name == "facts.txt":
        return "facts"
    if name == "final_prompt.txt":
        return "prompt"
    if "layout" in name and path.suffix.lower() == ".json":
        return "layout"
    if "dom_text" in name:
        return "dom_text"
    if name in {"final.png"} or name.startswith("gen_") or "direct_i2i" in name or "direct_t2i" in name:
        return "final"
    if "draft" in name or name.startswith("step2_"):
        return "draft"
    if "attempt" in name or "revise" in name or "crop" in name:
        return "attempt"
    if name.startswith("step1"):
        return "initial_source"
    if name.startswith("final.") or name in {"code.html", "draft.svg"}:
        return "final_source"
    if "background" in name:
        return "background"
    return "artifact"


def _turn_dir_for_path(path: Path, out_dir: Optional[str] = None) -> Optional[Path]:
    if out_dir:
        return Path(out_dir).resolve()
    resolved = path.resolve()
    if resolved.parent.name.startswith("turn-"):
        return resolved.parent
    if resolved.parent.parent.name.startswith("turn-"):
        return resolved.parent.parent
    return None


def _session_dir_for_turn(turn_dir: Optional[Path]) -> Path:
    if turn_dir and turn_dir.parent.parent.is_dir():
        return turn_dir.parent.parent
    from .session_dir import get_session_dir

    return Path(get_session_dir()).resolve()


def _read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def record_artifact(
    path: str | Path,
    *,
    tool_name: Optional[str] = None,
    role: Optional[str] = None,
    kind: Optional[str] = None,
    label: Optional[str] = None,
    out_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort artifact registration.

    Writes:
    - `<turn-dir>/manifest.json` for local per-call browsing.
    - `<session-dir>/artifacts.jsonl` for session-level indexing.
    """
    try:
        artifact_path = Path(path).resolve()
        if not artifact_path.is_file():
            return
        turn_dir = _turn_dir_for_path(artifact_path, out_dir)
        session_dir = _session_dir_for_turn(turn_dir)
        tool = tool_name or current_tool_name() or (turn_dir.parent.name if turn_dir else "")
        artifact = {
            "path": str(artifact_path),
            "kind": kind or infer_kind(artifact_path),
            "role": role or infer_role(artifact_path),
            "label": label or artifact_path.name,
            "size_bytes": artifact_path.stat().st_size,
            "metadata": metadata or {},
        }
        entry = {
            "schema_version": 1,
            "ts": time.time(),
            "tool_name": tool,
            "tool_call_id": current_tool_call_id(),
            "out_dir": str(turn_dir) if turn_dir else "",
            "artifact": artifact,
        }

        if turn_dir:
            manifest_path = turn_dir / "manifest.json"
            manifest = _read_json_object(manifest_path)
            manifest.setdefault("schema_version", 1)
            manifest["tool_name"] = manifest.get("tool_name") or tool
            manifest["tool_call_id"] = manifest.get("tool_call_id") or current_tool_call_id()
            manifest["out_dir"] = str(turn_dir)
            manifest["updated_at"] = entry["ts"]
            artifacts = manifest.setdefault("artifacts", [])
            if isinstance(artifacts, list):
                existing_idx = next(
                    (i for i, item in enumerate(artifacts) if item.get("path") == artifact["path"]),
                    None,
                )
                if existing_idx is None:
                    artifacts.append(artifact)
                else:
                    artifacts[existing_idx] = artifact
            _atomic_write_json(manifest_path, manifest)

        _append_jsonl(session_dir / "artifacts.jsonl", entry)
    except Exception:
        return


def record_artifacts_in_dir(
    out_dir: str | Path,
    *,
    tool_name: Optional[str] = None,
    skip_names: Optional[Iterable[str]] = None,
) -> None:
    """Register all regular files directly under a tool turn directory."""
    skip = set(skip_names or {"manifest.json"})
    directory = Path(out_dir)
    if not directory.is_dir():
        return
    for child in sorted(directory.iterdir()):
        if child.is_file() and not child.name.startswith("._") and child.name not in skip:
            record_artifact(child, tool_name=tool_name, out_dir=str(directory))

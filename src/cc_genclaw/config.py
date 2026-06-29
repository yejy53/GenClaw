"""Read API credentials + endpoints from the project-local config.yaml.

Default path: <repo>/config.yaml.

Why we read this yaml instead of .env:
- All modules consume the same uppercase env vars. Reading once + setting
  os.environ keeps downstream modules working without modification.
- The local yaml carries real keys; it is gitignored at repo root. Commit
  config.example.yaml instead.

Usage:
    from cc_genclaw.config import load_genclaw_config
    cfg = load_genclaw_config()
    # cfg["OPENAI_API_KEY"], cfg["IMAGE_API_KEY"], etc.
    # All UPPERCASE string keys are also pushed into os.environ.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "config.yaml")

# Keys we want to be the canonical source for, even when a third-party
# library loads its own .env with override=True at import time. These get
# force-set into os.environ AFTER any competing .env loaders have run.
_FORCE_OVERRIDE_KEYS = {
    "TAVILY_API_KEY",
    "TAVILY_BASE_URL",
    "MAIN_LLM_ACTIVE_PROFILE",
    "MAIN_LLM_API_KEY",
    "MAIN_LLM_BASE_URL",
    "MAIN_LLM_MODEL_NAME",
    "CODE_LLM_ENABLED",
    "CODE_LLM_BASE_URL",
    "CODE_LLM_API_KEY",
    "CODE_LLM_MODEL_NAME",
    "SVG_GEN_MODEL_NAME",
    "SVG_REVIEW_MODEL_NAME",
    "SCENE_DRAFT_GEN_MODEL",
    "SCENE_DRAFT_REVIEW_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL_NAME",
}

_MAIN_LLM_PROFILE_KEYS = {
    "base_url": "MAIN_LLM_BASE_URL",
    "api_key": "MAIN_LLM_API_KEY",
    "model": "MAIN_LLM_MODEL_NAME",
    "model_name": "MAIN_LLM_MODEL_NAME",
}


def _apply_main_llm_profile(cfg: Dict[str, Any]) -> None:
    """Expand a named MAIN_LLM_PROFILES entry into MAIN_LLM_* keys.

    This gives operators a single manual switch while preserving the old
    MAIN_LLM_* direct configuration shape for callers.
    """
    active = (
        os.environ.get("MAIN_LLM_ACTIVE_PROFILE")
        or str(cfg.get("MAIN_LLM_ACTIVE_PROFILE") or "")
    ).strip()
    if not active:
        return

    profiles = cfg.get("MAIN_LLM_PROFILES")
    if not isinstance(profiles, dict):
        raise ValueError("MAIN_LLM_ACTIVE_PROFILE is set but MAIN_LLM_PROFILES is missing or invalid.")

    profile = profiles.get(active)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(k) for k in profiles.keys())) or "(none)"
        raise ValueError(
            f"MAIN_LLM_ACTIVE_PROFILE={active!r} not found in MAIN_LLM_PROFILES. "
            f"Available profiles: {available}."
        )

    cfg["MAIN_LLM_ACTIVE_PROFILE"] = active
    for raw_key, value in profile.items():
        env_key = _MAIN_LLM_PROFILE_KEYS.get(str(raw_key).strip().lower())
        if env_key and value is not None:
            cfg[env_key] = str(value)


def load_genclaw_config(yaml_path: Optional[str] = None) -> Dict[str, Any]:
    """Load API keys + endpoints from yaml, push UPPERCASE strings to env.

    Args:
        yaml_path: optional override; falls back to env GENCLAW_CONFIG_PATH,
            then to DEFAULT_CONFIG_PATH.

    Returns:
        Parsed yaml dict (also has env-var side-effects).

    Precedence (lowest → highest):
      1. yaml file (<repo>/config.yaml)
      2. process env vars set by the user BEFORE invoking the example
         (e.g. `TAVILY_API_KEY=tvly-... python ex5.py`)
    """
    path = (
        yaml_path
        or os.environ.get("GENCLAW_CONFIG_PATH")
        or DEFAULT_CONFIG_PATH
    )
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GenClaw config.yaml not found at {p}. "
            f"Set GENCLAW_CONFIG_PATH env var to override."
        )

    with open(p, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    _apply_main_llm_profile(cfg)

    # Capture user-supplied env overrides BEFORE anything else can stomp
    # on them.
    user_env_snapshot = {
        k: os.environ[k]
        for k in _FORCE_OVERRIDE_KEYS
        if k in os.environ
    }

    # Push every UPPERCASE str key into os.environ so downstream modules
    # (which read os.environ.get / os.environ[...]) see the same values.
    # setdefault honours pre-existing env overrides.
    #
    # Exception: HTTP_PROXY / HTTPS_PROXY are only pushed when
    # cfg.proxy_on is truthy. Otherwise httpx/openai would route every
    # request through a (possibly absent) local proxy and fail with
    # Connection refused. The yaml carries the values for convenience
    # but they should stay opt-in.
    proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"}
    proxy_on = bool(cfg.get("proxy_on", False))
    for k, v in cfg.items():
        if isinstance(k, str) and k.isupper() and isinstance(v, str):
            if k in proxy_keys and not proxy_on:
                continue
            os.environ.setdefault(k, v)

    # Reapply the user-supplied env overrides on top so they win against
    # any .env that loaded with override=True. Also mirror them back into
    # the cfg dict so callers reading cfg["TAVILY_API_KEY"] see the
    # active key, not the yaml default.
    for k, v in user_env_snapshot.items():
        os.environ[k] = v
        cfg[k] = v

    # Single-gateway convenience: the Gemini image backend (providers.yaml +
    # code_text_draft) reuses the main OpenAI-compatible endpoint unless the
    # user explicitly sets a separate GEMINI_BASE_URL / GEMINI_API_KEY.
    main_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("MAIN_LLM_BASE_URL")
    main_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MAIN_LLM_API_KEY")
    if main_base and not os.environ.get("GEMINI_BASE_URL"):
        os.environ["GEMINI_BASE_URL"] = main_base
        cfg["GEMINI_BASE_URL"] = main_base
    if main_key and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = main_key
        cfg["GEMINI_API_KEY"] = main_key

    return cfg


def get_session_root(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Return the root dir for run/session outputs.

    Resolution: env CC_GENCLAW_RUNS_DIR > <repo>/runs.
    """
    env = os.environ.get("CC_GENCLAW_RUNS_DIR")
    if env:
        return env
    return str(Path(__file__).resolve().parents[2] / "runs")

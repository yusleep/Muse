"""Runtime configuration for Muse."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    model_router_config: dict[str, Any]
    runs_dir: str
    semantic_scholar_api_key: str | None
    openalex_email: str | None
    crossref_mailto: str | None
    refs_dir: str | None  # Resolved absolute path to local reference files, or None
    checkpoint_dir: str | None = None
    middleware_retry_max: int = 2
    middleware_retry_delay: float = 5.0
    middleware_compaction_threshold: float = 0.9
    middleware_compaction_recent_tokens: int = 20_000
    middleware_context_window: int = 128_000


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load runtime settings from environment-like mapping."""

    source = dict(os.environ if env is None else env)

    model_router_config = _load_router_config(source)

    llm_api_key = source.get("MUSE_LLM_API_KEY", "").strip()
    llm_model = source.get("MUSE_LLM_MODEL", "").strip()

    # Backward-compatible requirement for legacy single-model mode.
    if not model_router_config:
        if not llm_api_key:
            raise ValueError("MUSE_LLM_API_KEY is required")
        if not llm_model:
            raise ValueError("MUSE_LLM_MODEL is required")
    elif not llm_model:
        llm_model = _infer_default_model(model_router_config)

    llm_base_url = source.get("MUSE_LLM_BASE_URL", "https://api.openai.com/v1").strip()
    runs_dir = source.get("MUSE_RUNS_DIR", "runs").strip() or "runs"
    checkpoint_dir_raw = source.get("MUSE_CHECKPOINT_DIR", "").strip() or None
    checkpoint_dir = os.path.abspath(checkpoint_dir_raw) if checkpoint_dir_raw else None

    # Resolve local refs directory: CLI/env var takes precedence, then auto-detect ./refs/
    refs_dir_raw = source.get("MUSE_REFS_DIR", "").strip() or None
    if refs_dir_raw:
        resolved = os.path.abspath(refs_dir_raw)
        refs_dir: str | None = resolved if os.path.isdir(resolved) else None
    else:
        candidate = os.path.abspath("refs")
        refs_dir = candidate if os.path.isdir(candidate) else None

    middleware_retry_max = int(
        source.get("MUSE_MIDDLEWARE_RETRY_MAX", "2").strip() or "2"
    )
    middleware_retry_delay = float(
        source.get("MUSE_MIDDLEWARE_RETRY_DELAY", "5.0").strip() or "5.0"
    )
    middleware_compaction_threshold = float(
        source.get("MUSE_MIDDLEWARE_COMPACTION_THRESHOLD", "0.9").strip() or "0.9"
    )
    middleware_compaction_recent_tokens = int(
        source.get("MUSE_MIDDLEWARE_COMPACTION_RECENT_TOKENS", "20000").strip()
        or "20000"
    )
    middleware_context_window = int(
        source.get("MUSE_MIDDLEWARE_CONTEXT_WINDOW", "128000").strip() or "128000"
    )

    return Settings(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        model_router_config=model_router_config,
        runs_dir=runs_dir,
        semantic_scholar_api_key=source.get("MUSE_SEMANTIC_SCHOLAR_API_KEY", "").strip() or None,
        openalex_email=source.get("MUSE_OPENALEX_EMAIL", "").strip() or None,
        crossref_mailto=source.get("MUSE_CROSSREF_MAILTO", "").strip() or None,
        refs_dir=refs_dir,
        checkpoint_dir=checkpoint_dir,
        middleware_retry_max=middleware_retry_max,
        middleware_retry_delay=middleware_retry_delay,
        middleware_compaction_threshold=middleware_compaction_threshold,
        middleware_compaction_recent_tokens=middleware_compaction_recent_tokens,
        middleware_context_window=middleware_context_window,
    )


def _load_router_config(source: Mapping[str, str]) -> dict[str, Any]:
    raw_inline = source.get("MUSE_MODEL_ROUTER_JSON", "").strip()
    raw_file = source.get("MUSE_MODEL_ROUTER_PATH", "").strip()

    raw = raw_inline
    if not raw and raw_file:
        with open(raw_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()

    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MUSE_MODEL_ROUTER_JSON invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("MUSE_MODEL_ROUTER_JSON must be a JSON object")
    return parsed


def _infer_default_model(router_config: dict[str, Any]) -> str:
    model_routes = router_config.get("models", {})
    if isinstance(model_routes, dict):
        if isinstance(model_routes.get("default"), dict):
            primary = model_routes["default"].get("primary")
            if isinstance(primary, str) and primary.strip():
                return primary.strip()
        for route in model_routes.values():
            if isinstance(route, dict):
                primary = route.get("primary")
                if isinstance(primary, str) and primary.strip():
                    return primary.strip()
    return "router/default"

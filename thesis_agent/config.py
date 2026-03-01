"""Runtime configuration for thesis agent."""

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


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load runtime settings from environment-like mapping."""

    source = dict(os.environ if env is None else env)

    model_router_config = _load_router_config(source)

    llm_api_key = source.get("THESIS_AGENT_LLM_API_KEY", "").strip()
    llm_model = source.get("THESIS_AGENT_LLM_MODEL", "").strip()

    # Backward-compatible requirement for legacy single-model mode.
    if not model_router_config:
        if not llm_api_key:
            raise ValueError("THESIS_AGENT_LLM_API_KEY is required")
        if not llm_model:
            raise ValueError("THESIS_AGENT_LLM_MODEL is required")
    elif not llm_model:
        llm_model = _infer_default_model(model_router_config)

    llm_base_url = source.get("THESIS_AGENT_LLM_BASE_URL", "https://api.openai.com/v1").strip()
    runs_dir = source.get("THESIS_AGENT_RUNS_DIR", "runs").strip() or "runs"

    # Resolve local refs directory: CLI/env var takes precedence, then auto-detect ./refs/
    refs_dir_raw = source.get("THESIS_AGENT_REFS_DIR", "").strip() or None
    if refs_dir_raw:
        resolved = os.path.abspath(refs_dir_raw)
        refs_dir: str | None = resolved if os.path.isdir(resolved) else None
    else:
        candidate = os.path.abspath("refs")
        refs_dir = candidate if os.path.isdir(candidate) else None

    return Settings(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        model_router_config=model_router_config,
        runs_dir=runs_dir,
        semantic_scholar_api_key=source.get("THESIS_AGENT_SEMANTIC_SCHOLAR_API_KEY", "").strip() or None,
        openalex_email=source.get("THESIS_AGENT_OPENALEX_EMAIL", "").strip() or None,
        crossref_mailto=source.get("THESIS_AGENT_CROSSREF_MAILTO", "").strip() or None,
        refs_dir=refs_dir,
    )


def _load_router_config(source: Mapping[str, str]) -> dict[str, Any]:
    raw_inline = source.get("THESIS_AGENT_MODEL_ROUTER_JSON", "").strip()
    raw_file = source.get("THESIS_AGENT_MODEL_ROUTER_PATH", "").strip()

    raw = raw_inline
    if not raw and raw_file:
        with open(raw_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()

    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"THESIS_AGENT_MODEL_ROUTER_JSON invalid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("THESIS_AGENT_MODEL_ROUTER_JSON must be a JSON object")
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

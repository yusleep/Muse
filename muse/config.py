"""Runtime configuration for Muse."""

from __future__ import annotations

import copy
import json
import os
import re
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
    fetch_full_text: bool = False
    llamaparse_api_key: str = ""
    max_papers_to_index: int = 20
    local_papers_dir: str = ""
    local_priority: bool = True
    review_mode: str = "classic"
    writing_mode: str = "sequential"


# ---------------------------------------------------------------------------
# YAML config support
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

# Snake-case YAML keys → camelCase keys expected by _ModelRouter
_SNAKE_TO_CAMEL: dict[str, str] = {
    "api_key_env": "apiKeyEnv",
    "api_key": "apiKey",
    "base_url": "baseUrl",
    "api_style": "apiStyle",
    "codex_oauth": "codexOAuth",
    "oauth_provider": "oauthProvider",
    "auth_file": "authFile",
    "token_path": "tokenPath",
    "model_aliases": "modelAliases",
}


def _load_dotenv() -> None:
    """Read .env file into os.environ (if present). No python-dotenv needed."""
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _resolve_env_vars(obj: Any, env: Mapping[str, str]) -> Any:
    """Recursively substitute ``${VAR}`` / ``$VAR`` patterns in strings."""
    if isinstance(obj, str):
        return _ENV_VAR_RE.sub(
            lambda m: env.get(m.group(1) or m.group(2), ""), obj
        )
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v, env) for v in obj]
    return obj


def _load_config_yaml(
    config_path: str | None, env: Mapping[str, str]
) -> tuple[dict[str, Any] | None, str | None]:
    """Find, parse, and env-resolve a config.yaml file.

    Returns ``(config, config_dir)`` when a file is found, otherwise ``(None, None)``.
    Explicit config paths must exist; they do not silently fall back.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return None, None

    explicit = config_path or env.get("MUSE_CONFIG", "").strip()
    if explicit:
        resolved = os.path.abspath(explicit)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Config file not found: {explicit}")
        candidates = [resolved]
    else:
        candidates = [os.path.abspath("config.yaml"), os.path.abspath("config.yml")]

    for resolved in candidates:
        if os.path.isfile(resolved):
            with open(resolved, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if isinstance(raw, dict):
                return _resolve_env_vars(raw, env), os.path.dirname(resolved)
            return None, os.path.dirname(resolved)
    return None, None


def _snake_to_camel_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Translate known snake_case keys to camelCase, recursing into dicts."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        new_key = _SNAKE_TO_CAMEL.get(key, key)
        if isinstance(value, dict):
            value = _snake_to_camel_dict(value)
        out[new_key] = value
    return out


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _yaml_to_router_config(yaml_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert YAML auth/providers/routes into _ModelRouter dict format."""
    router: dict[str, Any] = {}

    auth = yaml_cfg.get("auth")
    if isinstance(auth, dict):
        router["auth"] = _snake_to_camel_dict(auth)

    providers = yaml_cfg.get("providers")
    if isinstance(providers, dict):
        router["providers"] = _snake_to_camel_dict(providers)

    # ``routes:`` in YAML maps to ``models`` in the JSON router format
    routes = yaml_cfg.get("routes") or yaml_cfg.get("models")
    if isinstance(routes, dict):
        router["models"] = routes

    aliases = yaml_cfg.get("aliases") or yaml_cfg.get("model_aliases")
    if isinstance(aliases, dict):
        router["modelAliases"] = aliases

    return router


def _yaml_to_settings(
    yaml_cfg: dict[str, Any], env: Mapping[str, str], config_dir: str | None = None
) -> dict[str, Any]:
    """Extract Settings field values from a parsed config.yaml dict.

    Returns a kwargs dict suitable for ``Settings(**kwargs)``.
    """
    kw: dict[str, Any] = {}

    # model router
    router_cfg = _yaml_to_router_config(yaml_cfg)
    kw["model_router_config"] = router_cfg

    # search section
    search = yaml_cfg.get("search", {})
    if isinstance(search, dict):
        kw["semantic_scholar_api_key"] = search.get("semantic_scholar_api_key") or None
        kw["openalex_email"] = search.get("openalex_email") or None
        kw["crossref_mailto"] = search.get("crossref_mailto") or None
        if "fetch_full_text" in search:
            kw["fetch_full_text"] = _coerce_bool(search.get("fetch_full_text"), default=False)
        if "llamaparse_api_key" in search:
            kw["llamaparse_api_key"] = str(search.get("llamaparse_api_key") or "")
        if "max_papers_to_index" in search:
            kw["max_papers_to_index"] = int(search.get("max_papers_to_index") or 20)
        if "local_papers_dir" in search:
            local_papers_dir = str(search.get("local_papers_dir") or "").strip()
            kw["local_papers_dir"] = (
                _resolve_config_path(local_papers_dir, config_dir) if local_papers_dir else ""
            )
        if "local_priority" in search:
            kw["local_priority"] = _coerce_bool(search.get("local_priority"), default=True)

    # middleware section
    mw = yaml_cfg.get("middleware", {})
    if isinstance(mw, dict):
        if "retry_max" in mw:
            kw["middleware_retry_max"] = int(mw["retry_max"])
        if "retry_delay" in mw:
            kw["middleware_retry_delay"] = float(mw["retry_delay"])
        if "compaction_threshold" in mw:
            kw["middleware_compaction_threshold"] = float(mw["compaction_threshold"])
        if "compaction_recent_tokens" in mw:
            kw["middleware_compaction_recent_tokens"] = int(mw["compaction_recent_tokens"])
        if "context_window" in mw:
            kw["middleware_context_window"] = int(mw["context_window"])

    # paths section
    paths = yaml_cfg.get("paths", {})
    if isinstance(paths, dict):
        if paths.get("runs_dir"):
            kw["runs_dir"] = _resolve_config_path(str(paths["runs_dir"]), config_dir)
        if paths.get("checkpoint_dir"):
            kw["checkpoint_dir"] = _resolve_config_path(str(paths["checkpoint_dir"]), config_dir)
        if paths.get("refs_dir"):
            resolved = _resolve_config_path(str(paths["refs_dir"]), config_dir)
            kw["refs_dir"] = resolved if os.path.isdir(resolved) else None

    review = yaml_cfg.get("review", {})
    if isinstance(review, dict) and review.get("mode"):
        kw["review_mode"] = str(review["mode"]).strip().lower()

    writing = yaml_cfg.get("writing", {})
    if isinstance(writing, dict) and writing.get("mode"):
        kw["writing_mode"] = str(writing["mode"]).strip().lower()
    elif yaml_cfg.get("writing_mode"):
        kw["writing_mode"] = str(yaml_cfg["writing_mode"]).strip().lower()

    return kw


def _resolve_config_path(path_value: str, config_dir: str | None) -> str:
    path_value = path_value.strip()
    if os.path.isabs(path_value) or not config_dir:
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(config_dir, path_value))


def _apply_default_route_override(
    router_config: dict[str, Any],
    llm_model: str,
) -> dict[str, Any]:
    if not llm_model.strip():
        return router_config

    updated = copy.deepcopy(router_config)
    routes = updated.get("models")
    if not isinstance(routes, dict):
        routes = {}
        updated["models"] = routes

    default_route = routes.get("default")
    if not isinstance(default_route, dict):
        default_route = {"fallbacks": []}
        routes["default"] = default_route

    default_route["primary"] = llm_model.strip()
    if "fallbacks" not in default_route or not isinstance(default_route["fallbacks"], list):
        default_route["fallbacks"] = []
    return updated


def load_settings(
    env: Mapping[str, str] | None = None,
    config_path: str | None = None,
) -> Settings:
    """Load runtime settings from environment-like mapping.

    Priority: CLI args > env vars > config.yaml > defaults.
    """
    if env is None:
        _load_dotenv()

    source = dict(os.environ if env is None else env)
    if any(key.startswith("THESIS_AGENT_") for key in source):
        raise ValueError("Legacy THESIS_AGENT_* environment variables are not supported; use MUSE_* instead.")

    yaml_kw: dict[str, Any] = {}
    yaml_cfg = None
    config_dir = None

    # When a caller supplies an explicit env mapping, only load YAML when explicitly requested.
    if env is None:
        yaml_cfg, config_dir = _load_config_yaml(config_path, source)
    else:
        explicit_yaml = config_path or source.get("MUSE_CONFIG", "").strip()
        if explicit_yaml:
            yaml_cfg, config_dir = _load_config_yaml(config_path, source)

    if yaml_cfg:
        yaml_kw = _yaml_to_settings(yaml_cfg, source, config_dir)

    env_router_config = _load_router_config(source)
    # YAML-derived router is the base; env-var router (JSON) overrides it
    if env_router_config:
        yaml_kw["model_router_config"] = env_router_config
    elif "model_router_config" not in yaml_kw:
        yaml_kw["model_router_config"] = {}
    model_router_config = yaml_kw["model_router_config"]

    llm_api_key = source.get("MUSE_LLM_API_KEY", "").strip()
    llm_model = source.get("MUSE_LLM_MODEL", "").strip()

    if llm_model and model_router_config and not env_router_config:
        model_router_config = _apply_default_route_override(model_router_config, llm_model)
        yaml_kw["model_router_config"] = model_router_config

    # Backward-compatible requirement for legacy single-model mode.
    if not model_router_config:
        if not llm_api_key:
            raise ValueError("MUSE_LLM_API_KEY is required")
        if not llm_model:
            raise ValueError("MUSE_LLM_MODEL is required")
    elif not llm_model:
        llm_model = _infer_default_model(model_router_config)

    llm_base_url = source.get("MUSE_LLM_BASE_URL", "").strip()
    if not llm_base_url:
        llm_base_url = "https://api.openai.com/v1"
    runs_dir = source.get("MUSE_RUNS_DIR", "").strip() or yaml_kw.get("runs_dir", "runs") or "runs"
    checkpoint_dir_raw = source.get("MUSE_CHECKPOINT_DIR", "").strip() or None
    checkpoint_dir = os.path.abspath(checkpoint_dir_raw) if checkpoint_dir_raw else yaml_kw.get("checkpoint_dir")

    # Resolve local refs directory: CLI/env var takes precedence, then YAML, then auto-detect ./refs/
    refs_dir_raw = source.get("MUSE_REFS_DIR", "").strip() or None
    if refs_dir_raw:
        resolved = os.path.abspath(refs_dir_raw)
        refs_dir: str | None = resolved if os.path.isdir(resolved) else None
    elif "refs_dir" in yaml_kw:
        refs_dir = yaml_kw["refs_dir"]
    else:
        candidate = os.path.abspath("refs")
        refs_dir = candidate if os.path.isdir(candidate) else None

    # Middleware: env vars override YAML, YAML overrides defaults
    def _mw_int(env_key: str, yaml_key: str, default: int) -> int:
        raw = source.get(env_key, "").strip()
        if raw:
            return int(raw)
        return int(yaml_kw.get(yaml_key, default))

    def _mw_float(env_key: str, yaml_key: str, default: float) -> float:
        raw = source.get(env_key, "").strip()
        if raw:
            return float(raw)
        return float(yaml_kw.get(yaml_key, default))

    middleware_retry_max = _mw_int("MUSE_MIDDLEWARE_RETRY_MAX", "middleware_retry_max", 2)
    middleware_retry_delay = _mw_float("MUSE_MIDDLEWARE_RETRY_DELAY", "middleware_retry_delay", 5.0)
    middleware_compaction_threshold = _mw_float(
        "MUSE_MIDDLEWARE_COMPACTION_THRESHOLD", "middleware_compaction_threshold", 0.9
    )
    middleware_compaction_recent_tokens = _mw_int(
        "MUSE_MIDDLEWARE_COMPACTION_RECENT_TOKENS", "middleware_compaction_recent_tokens", 20_000
    )
    middleware_context_window = _mw_int(
        "MUSE_MIDDLEWARE_CONTEXT_WINDOW", "middleware_context_window", 128_000
    )

    # Search: env vars override YAML
    def _search_str(env_key: str, yaml_key: str) -> str | None:
        raw = source.get(env_key, "").strip()
        if raw:
            return raw
        return yaml_kw.get(yaml_key) or None

    review_mode = source.get("MUSE_REVIEW_MODE", "").strip().lower() or str(
        yaml_kw.get("review_mode", "classic")
    ).strip().lower() or "classic"

    return Settings(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        model_router_config=model_router_config,
        runs_dir=runs_dir,
        semantic_scholar_api_key=_search_str("MUSE_SEMANTIC_SCHOLAR_API_KEY", "semantic_scholar_api_key"),
        openalex_email=_search_str("MUSE_OPENALEX_EMAIL", "openalex_email"),
        crossref_mailto=_search_str("MUSE_CROSSREF_MAILTO", "crossref_mailto"),
        refs_dir=refs_dir,
        checkpoint_dir=checkpoint_dir,
        middleware_retry_max=middleware_retry_max,
        middleware_retry_delay=middleware_retry_delay,
        middleware_compaction_threshold=middleware_compaction_threshold,
        middleware_compaction_recent_tokens=middleware_compaction_recent_tokens,
        middleware_context_window=middleware_context_window,
        review_mode=review_mode,
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

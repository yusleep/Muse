"""MCP server configuration loader.

Reads extensions.yaml and builds typed connection parameter dicts
compatible with langchain-mcp-adapters MultiServerMCPClient.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StdioServerConfig:
    """Config for a stdio-transport MCP server."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class SseServerConfig:
    """Config for an SSE-transport MCP server."""

    name: str
    url: str
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class HttpServerConfig:
    """Config for a streamable-HTTP-transport MCP server."""

    name: str
    url: str
    headers: dict[str, str] | None = None


MCPServerConfig = StdioServerConfig | SseServerConfig | HttpServerConfig


def load_extensions_config(config_path: str | Path | None = None) -> list[MCPServerConfig]:
    """Load MCP server configs from extensions.yaml.

    Search order when *config_path* is None:
    1. $MUSE_EXTENSIONS_PATH env var
    2. ./extensions.yaml (cwd)
    3. ~/.muse/extensions.yaml

    Returns an empty list when the file does not exist or contains no
    ``mcp_servers`` key -- MCP is strictly optional and must never block startup.
    """
    import yaml

    resolved = _resolve_config_path(config_path)
    if resolved is None:
        return []

    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except Exception:
        return []

    if not isinstance(raw, dict):
        return []

    servers_raw = raw.get("mcp_servers", {})
    if not isinstance(servers_raw, dict):
        return []

    configs: list[MCPServerConfig] = []
    for name, entry in servers_raw.items():
        if not isinstance(entry, dict):
            continue
        parsed = _parse_server_entry(str(name), entry)
        if parsed is not None:
            configs.append(parsed)
    return configs


def build_multiserver_params(configs: list[MCPServerConfig]) -> dict[str, dict[str, Any]]:
    """Convert typed configs into MultiServerMCPClient-compatible params."""

    params: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        if isinstance(cfg, StdioServerConfig):
            entry: dict[str, Any] = {
                "transport": "stdio",
                "command": cfg.command,
                "args": cfg.args,
            }
            if cfg.env:
                entry["env"] = cfg.env
            params[cfg.name] = entry
        elif isinstance(cfg, SseServerConfig):
            entry = {
                "transport": "sse",
                "url": cfg.url,
            }
            if cfg.headers:
                entry["headers"] = cfg.headers
            params[cfg.name] = entry
        elif isinstance(cfg, HttpServerConfig):
            entry = {
                "transport": "streamable_http",
                "url": cfg.url,
            }
            if cfg.headers:
                entry["headers"] = cfg.headers
            params[cfg.name] = entry
    return params


def _resolve_config_path(explicit: str | Path | None) -> Path | None:
    """Return the first existing extensions.yaml path, or None."""

    if explicit is not None:
        explicit_path = Path(explicit)
        return explicit_path if explicit_path.is_file() else None

    env_path = os.environ.get("MUSE_EXTENSIONS_PATH", "").strip()
    if env_path:
        env_candidate = Path(env_path)
        if env_candidate.is_file():
            return env_candidate

    cwd_candidate = Path.cwd() / "extensions.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate

    home_candidate = Path.home() / ".muse" / "extensions.yaml"
    if home_candidate.is_file():
        return home_candidate

    return None


def _parse_server_entry(name: str, entry: dict[str, Any]) -> MCPServerConfig | None:
    """Parse a single server entry from the yaml."""

    transport = str(entry.get("transport", "")).strip().lower()

    if transport == "stdio":
        command = str(entry.get("command", "")).strip()
        if not command:
            return None
        args_raw = entry.get("args", [])
        args = [str(arg) for arg in args_raw] if isinstance(args_raw, list) else []
        env_raw = entry.get("env")
        env = {str(key): str(value) for key, value in env_raw.items()} if isinstance(env_raw, dict) else None
        return StdioServerConfig(name=name, command=command, args=args, env=env)

    if transport == "sse":
        url = str(entry.get("url", "")).strip()
        if not url:
            return None
        return SseServerConfig(name=name, url=url, headers=_parse_headers(entry))

    if transport in ("http", "streamable_http"):
        url = str(entry.get("url", "")).strip()
        if not url:
            return None
        return HttpServerConfig(name=name, url=url, headers=_parse_headers(entry))

    return None


def _parse_headers(entry: dict[str, Any]) -> dict[str, str] | None:
    raw = entry.get("headers")
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return None

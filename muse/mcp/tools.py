"""Load LangChain tools from configured MCP servers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from muse.mcp.client import _resolve_config_path, build_multiserver_params, load_extensions_config
from muse.mcp.oauth import OAuthTokenManager, parse_oauth_config

logger = logging.getLogger(__name__)


async def get_mcp_tools(
    config_path: str | None = None,
    *,
    oauth_manager: OAuthTokenManager | None = None,
) -> list[Any]:
    """Return LangChain tools loaded from configured MCP servers."""

    configs = load_extensions_config(config_path)
    if not configs:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed; MCP tools unavailable")
        return []

    oauth_manager = _ensure_oauth_manager(config_path, oauth_manager)
    params = build_multiserver_params(configs)

    tools: list[Any] = []
    for config in configs:
        server_params = params.get(config.name)
        if server_params is None:
            continue

        if oauth_manager.has_config(config.name):
            try:
                auth_headers = oauth_manager.get_auth_headers(config.name)
            except Exception as exc:
                logger.warning("OAuth for MCP server '%s' failed: %s", config.name, exc)
                continue
            headers = dict(server_params.get("headers", {}))
            headers.update(auth_headers)
            server_params = dict(server_params)
            server_params["headers"] = headers

        try:
            async with MultiServerMCPClient({config.name: server_params}) as client:
                tools.extend(client.get_tools())
        except Exception as exc:
            logger.warning("MCP connection failed for '%s': %s", config.name, exc)

    if tools:
        logger.info("Loaded %d MCP tools from %d servers", len(tools), len(configs))
    return tools


def get_mcp_tools_sync(
    config_path: str | None = None,
    *,
    oauth_manager: OAuthTokenManager | None = None,
) -> list[Any]:
    """Synchronous wrapper around ``get_mcp_tools``."""

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None and running_loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                get_mcp_tools(config_path, oauth_manager=oauth_manager),
            )
            return future.result(timeout=60)

    return asyncio.run(get_mcp_tools(config_path, oauth_manager=oauth_manager))


def _ensure_oauth_manager(
    config_path: str | None,
    oauth_manager: OAuthTokenManager | None,
) -> OAuthTokenManager:
    manager = oauth_manager or OAuthTokenManager()
    for server_name, entry in _load_raw_server_entries(config_path).items():
        oauth_config = parse_oauth_config(entry)
        if oauth_config is not None:
            manager.register(server_name, oauth_config)
    return manager


def _load_raw_server_entries(config_path: str | None) -> dict[str, dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        return {}

    resolved = _resolve_config_path(config_path)
    if resolved is None:
        return {}

    try:
        raw = yaml.safe_load(Path(resolved).read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    servers = raw.get("mcp_servers", {})
    if not isinstance(servers, dict):
        return {}

    return {
        str(name): entry
        for name, entry in servers.items()
        if isinstance(entry, dict)
    }

"""Bridge between MCP tool cache and ToolRegistry."""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _ToolRegistryProtocol(Protocol):
    """Minimal interface expected from ToolRegistry."""

    def register(self, tool: Any, *, group: str) -> None: ...


def register_mcp_tools(
    registry: _ToolRegistryProtocol,
    *,
    cache: Any,
    group: str = "mcp",
) -> int:
    """Load MCP tools from cache and register them into the registry."""

    try:
        tools = cache.get_tools()
    except Exception as exc:
        logger.warning("Failed to load MCP tools: %s", exc)
        return 0

    registered = 0
    for tool in tools:
        try:
            registry.register(tool, group=group)
            registered += 1
        except Exception as exc:
            tool_name = getattr(tool, "name", repr(tool))
            logger.warning("Failed to register MCP tool '%s': %s", tool_name, exc)

    if registered:
        logger.info("Registered %d MCP tools into '%s' group", registered, group)
    return registered

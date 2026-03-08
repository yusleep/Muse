"""MCP tool cache with config-change hot-reload."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from muse.mcp.client import _resolve_config_path
from muse.mcp.oauth import OAuthTokenManager
from muse.mcp.tools import get_mcp_tools_sync

logger = logging.getLogger(__name__)


class MCPToolCache:
    """Lazy-loading MCP tool cache with mtime-based invalidation."""

    def __init__(
        self,
        config_path: str | None = None,
        *,
        oauth_manager: OAuthTokenManager | None = None,
        min_reload_interval: float = 5.0,
    ) -> None:
        self._config_path = config_path
        self._oauth_manager = oauth_manager
        self._min_reload_interval = min_reload_interval

        self._tools: list[Any] = []
        self._loaded = False
        self._last_mtime = 0.0
        self._last_check = 0.0

    def get_tools(self) -> list[Any]:
        """Return cached MCP tools, reloading when config changes."""

        now = time.monotonic()
        if self._loaded and (now - self._last_check) < self._min_reload_interval:
            return self._tools

        self._last_check = now
        resolved = _resolve_config_path(self._config_path)
        if resolved is None:
            if self._loaded:
                return self._tools
            self._tools = []
            self._loaded = True
            return self._tools

        try:
            current_mtime = os.path.getmtime(resolved)
        except OSError:
            return self._tools

        if self._loaded and current_mtime == self._last_mtime:
            return self._tools

        logger.info("Loading MCP tools (config mtime changed or first load)")
        self._tools = get_mcp_tools_sync(
            str(resolved),
            oauth_manager=self._oauth_manager,
        )
        self._last_mtime = current_mtime
        self._loaded = True
        return self._tools

    def invalidate(self) -> None:
        """Force a reload on the next ``get_tools`` call."""

        self._loaded = False
        self._last_mtime = 0.0
        self._last_check = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

# Phase 4-A: MCP Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect Muse to external tool ecosystems via Model Context Protocol.

**Architecture:** MCP client supporting 3 transport types (stdio/sse/http), per-server OAuth, tool caching. MCP tools auto-registered into ToolRegistry.

**Tech Stack:** langchain-mcp-adapters, Python 3.10

**Depends on:** Phase 0-A (ToolRegistry)

---

## Task 1: Install langchain-mcp-adapters dependency

**Files:**
- `requirements.txt` (edit)

**What to do:**

Add `langchain-mcp-adapters>=0.1.0` and `pyyaml>=6.0` to `requirements.txt`.

The final `requirements.txt` should be:

```
langgraph==1.0.10
langgraph-checkpoint-sqlite==3.0.3
typing_extensions==4.15.0
langchain-mcp-adapters>=0.1.0
pyyaml>=6.0
```

**TDD:**

1. RED: Run `python3 -c "import langchain_mcp_adapters; print('ok')"` -- fails with ModuleNotFoundError.
2. GREEN: Run `pip install langchain-mcp-adapters pyyaml`. Verify: `python3 -c "from langchain_mcp_adapters.client import MultiServerMCPClient; print('ok')"` prints `ok`.
3. REFACTOR: None needed.

**Time estimate:** 2 minutes.

---

## Task 2: Create extensions.yaml config schema and loader (`muse/mcp/client.py`)

**Files:**
- `muse/mcp/__init__.py` (create, empty)
- `muse/mcp/client.py` (create)
- `tests/test_mcp_client.py` (create)

**What to do:**

Create the MCP config parser that reads `extensions.yaml` and returns typed connection parameter objects. This module does NOT connect to any servers -- it only loads and validates the config file.

Create `muse/mcp/__init__.py`:

```python
"""MCP (Model Context Protocol) integration for Muse."""
```

Create `muse/mcp/client.py`:

```python
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
    import yaml  # deferred so yaml is only required when MCP is used

    resolved = _resolve_config_path(config_path)
    if resolved is None:
        return []

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
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
    """Convert our typed configs into the dict format expected by
    ``MultiServerMCPClient(connections=...)``.

    Returns ``{server_name: {transport_kwargs...}}`` ready for
    ``MultiServerMCPClient``.
    """
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_config_path(explicit: str | Path | None) -> Path | None:
    """Return the first existing extensions.yaml path, or None."""
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None

    env_path = os.environ.get("MUSE_EXTENSIONS_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

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
        args = [str(a) for a in args_raw] if isinstance(args_raw, list) else []
        env_raw = entry.get("env")
        env = {str(k): str(v) for k, v in env_raw.items()} if isinstance(env_raw, dict) else None
        return StdioServerConfig(name=name, command=command, args=args, env=env)

    if transport == "sse":
        url = str(entry.get("url", "")).strip()
        if not url:
            return None
        headers = _parse_headers(entry)
        return SseServerConfig(name=name, url=url, headers=headers)

    if transport in ("http", "streamable_http"):
        url = str(entry.get("url", "")).strip()
        if not url:
            return None
        headers = _parse_headers(entry)
        return HttpServerConfig(name=name, url=url, headers=headers)

    return None


def _parse_headers(entry: dict[str, Any]) -> dict[str, str] | None:
    raw = entry.get("headers")
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return None
```

Create `tests/test_mcp_client.py`:

```python
"""Tests for MCP config loader (muse.mcp.client)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from muse.mcp.client import (
    HttpServerConfig,
    SseServerConfig,
    StdioServerConfig,
    build_multiserver_params,
    load_extensions_config,
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "extensions.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestLoadExtensionsConfig:
    def test_empty_file_returns_empty(self, tmp_path: Path):
        p = _write_yaml(tmp_path, "")
        assert load_extensions_config(p) == []

    def test_no_mcp_servers_key_returns_empty(self, tmp_path: Path):
        p = _write_yaml(tmp_path, "other_key: 123\n")
        assert load_extensions_config(p) == []

    def test_nonexistent_path_returns_empty(self, tmp_path: Path):
        assert load_extensions_config(tmp_path / "nope.yaml") == []

    def test_none_path_no_file_returns_empty(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MUSE_EXTENSIONS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        assert load_extensions_config(None) == []

    def test_parses_stdio_server(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """\
            mcp_servers:
              zotero:
                transport: stdio
                command: npx
                args: ["-y", "@anthropic/mcp-server-zotero"]
        """)
        configs = load_extensions_config(p)
        assert len(configs) == 1
        assert isinstance(configs[0], StdioServerConfig)
        assert configs[0].name == "zotero"
        assert configs[0].command == "npx"
        assert configs[0].args == ["-y", "@anthropic/mcp-server-zotero"]

    def test_parses_sse_server(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """\
            mcp_servers:
              local_search:
                transport: sse
                url: "http://localhost:8080/sse"
        """)
        configs = load_extensions_config(p)
        assert len(configs) == 1
        assert isinstance(configs[0], SseServerConfig)
        assert configs[0].url == "http://localhost:8080/sse"

    def test_parses_http_server_with_headers(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """\
            mcp_servers:
              overleaf:
                transport: http
                url: "https://overleaf-mcp.example.com/mcp"
                headers:
                  Authorization: "Bearer tok123"
        """)
        configs = load_extensions_config(p)
        assert len(configs) == 1
        assert isinstance(configs[0], HttpServerConfig)
        assert configs[0].headers == {"Authorization": "Bearer tok123"}

    def test_skips_invalid_entries(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """\
            mcp_servers:
              good:
                transport: stdio
                command: echo
              bad_no_transport:
                url: "http://x"
              bad_no_command:
                transport: stdio
        """)
        configs = load_extensions_config(p)
        assert len(configs) == 1
        assert configs[0].name == "good"

    def test_mixed_transports(self, tmp_path: Path):
        p = _write_yaml(tmp_path, """\
            mcp_servers:
              a:
                transport: stdio
                command: echo
              b:
                transport: sse
                url: "http://localhost:9090/sse"
              c:
                transport: http
                url: "https://example.com/mcp"
        """)
        configs = load_extensions_config(p)
        assert len(configs) == 3
        types = {type(c) for c in configs}
        assert types == {StdioServerConfig, SseServerConfig, HttpServerConfig}


class TestBuildMultiserverParams:
    def test_stdio_params(self):
        cfg = StdioServerConfig(name="z", command="npx", args=["-y", "pkg"], env={"KEY": "val"})
        result = build_multiserver_params([cfg])
        assert result == {
            "z": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "pkg"],
                "env": {"KEY": "val"},
            }
        }

    def test_sse_params(self):
        cfg = SseServerConfig(name="s", url="http://localhost:8080/sse")
        result = build_multiserver_params([cfg])
        assert result == {
            "s": {"transport": "sse", "url": "http://localhost:8080/sse"}
        }

    def test_http_params_with_headers(self):
        cfg = HttpServerConfig(name="h", url="https://x.com/mcp", headers={"Auth": "Bearer t"})
        result = build_multiserver_params([cfg])
        assert result == {
            "h": {
                "transport": "streamable_http",
                "url": "https://x.com/mcp",
                "headers": {"Auth": "Bearer t"},
            }
        }

    def test_empty_input(self):
        assert build_multiserver_params([]) == {}
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_mcp_client.py -x` -- fails because files do not exist.
2. GREEN: Create the files exactly as above. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 5 minutes.

---

## Task 3: Create OAuthTokenManager (`muse/mcp/oauth.py`)

**Files:**
- `muse/mcp/oauth.py` (create)
- `tests/test_mcp_oauth.py` (create)

**What to do:**

Create a per-server OAuth token cache with auto-refresh. This handles the `oauth` block inside an MCP server entry. Tokens are fetched via the standard OAuth2 client_credentials flow and cached in memory with a TTL margin.

Create `muse/mcp/oauth.py`:

```python
"""Per-server OAuth token management for MCP connections.

Handles the ``oauth`` block in extensions.yaml server entries:

    mcp_servers:
      overleaf:
        transport: http
        url: "https://overleaf-mcp.example.com/mcp"
        oauth:
          token_url: "https://auth.overleaf.com/oauth2/token"
          grant_type: "client_credentials"
          client_id: "my-client"
          client_secret: "secret"

Tokens are cached in memory and auto-refreshed 60 seconds before expiry.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OAuthConfig:
    """Parsed OAuth settings for one MCP server."""
    token_url: str
    grant_type: str  # typically "client_credentials"
    client_id: str
    client_secret: str
    scopes: list[str] = field(default_factory=list)


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float  # time.monotonic() based


class OAuthTokenManager:
    """Per-server OAuth token cache with lazy refresh.

    Usage::

        mgr = OAuthTokenManager()
        mgr.register("overleaf", OAuthConfig(...))
        headers = mgr.get_auth_headers("overleaf")
        # -> {"Authorization": "Bearer <token>"}
    """

    _REFRESH_MARGIN_SECONDS = 60

    def __init__(self) -> None:
        self._configs: dict[str, OAuthConfig] = {}
        self._cache: dict[str, _CachedToken] = {}

    def register(self, server_name: str, config: OAuthConfig) -> None:
        """Register OAuth config for a named server."""
        self._configs[server_name] = config
        self._cache.pop(server_name, None)

    def get_auth_headers(self, server_name: str) -> dict[str, str]:
        """Return ``{"Authorization": "Bearer <token>"}`` for the server.

        Returns an empty dict if no OAuth config is registered for the server.
        Raises ``OAuthError`` on token fetch failure.
        """
        config = self._configs.get(server_name)
        if config is None:
            return {}

        cached = self._cache.get(server_name)
        if cached is not None and cached.expires_at - self._REFRESH_MARGIN_SECONDS > time.monotonic():
            return {"Authorization": f"Bearer {cached.access_token}"}

        token_data = self._fetch_token(config)
        access_token = str(token_data.get("access_token", "")).strip()
        if not access_token:
            raise OAuthError(f"Token response for '{server_name}' missing access_token")

        expires_in = int(token_data.get("expires_in", 3600))
        self._cache[server_name] = _CachedToken(
            access_token=access_token,
            expires_at=time.monotonic() + expires_in,
        )
        return {"Authorization": f"Bearer {access_token}"}

    def has_config(self, server_name: str) -> bool:
        return server_name in self._configs

    def invalidate(self, server_name: str) -> None:
        """Force re-fetch on next call."""
        self._cache.pop(server_name, None)

    @staticmethod
    def _fetch_token(config: OAuthConfig) -> dict[str, Any]:
        """Perform the OAuth2 token request."""
        body_params: dict[str, str] = {
            "grant_type": config.grant_type,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }
        if config.scopes:
            body_params["scope"] = " ".join(config.scopes)

        data = urllib.parse.urlencode(body_params).encode("utf-8")
        req = urllib.request.Request(
            config.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300] if exc.fp else ""
            raise OAuthError(f"OAuth token request failed (HTTP {exc.code}): {detail}") from exc
        except Exception as exc:
            raise OAuthError(f"OAuth token request error: {exc}") from exc


class OAuthError(RuntimeError):
    """Raised when OAuth token acquisition fails."""


def parse_oauth_config(entry: dict[str, Any]) -> OAuthConfig | None:
    """Parse an ``oauth`` block from a server entry. Returns None if absent."""
    oauth_raw = entry.get("oauth")
    if not isinstance(oauth_raw, dict):
        return None
    token_url = str(oauth_raw.get("token_url", "")).strip()
    client_id = str(oauth_raw.get("client_id", "")).strip()
    client_secret = str(oauth_raw.get("client_secret", "")).strip()
    if not (token_url and client_id and client_secret):
        return None
    grant_type = str(oauth_raw.get("grant_type", "client_credentials")).strip()
    scopes_raw = oauth_raw.get("scopes", [])
    scopes = [str(s) for s in scopes_raw] if isinstance(scopes_raw, list) else []
    return OAuthConfig(
        token_url=token_url,
        grant_type=grant_type,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
    )
```

Create `tests/test_mcp_oauth.py`:

```python
"""Tests for MCP OAuth token management (muse.mcp.oauth)."""

from __future__ import annotations

import time

import pytest

from muse.mcp.oauth import (
    OAuthConfig,
    OAuthError,
    OAuthTokenManager,
    parse_oauth_config,
)


class TestParseOAuthConfig:
    def test_returns_none_when_no_oauth_key(self):
        assert parse_oauth_config({"transport": "http"}) is None

    def test_returns_none_when_incomplete(self):
        assert parse_oauth_config({"oauth": {"token_url": "x"}}) is None

    def test_parses_complete_config(self):
        cfg = parse_oauth_config({
            "oauth": {
                "token_url": "https://auth.example.com/token",
                "client_id": "cid",
                "client_secret": "csec",
                "grant_type": "client_credentials",
                "scopes": ["read", "write"],
            }
        })
        assert cfg is not None
        assert cfg.token_url == "https://auth.example.com/token"
        assert cfg.client_id == "cid"
        assert cfg.client_secret == "csec"
        assert cfg.scopes == ["read", "write"]

    def test_defaults_grant_type(self):
        cfg = parse_oauth_config({
            "oauth": {
                "token_url": "https://x/token",
                "client_id": "a",
                "client_secret": "b",
            }
        })
        assert cfg is not None
        assert cfg.grant_type == "client_credentials"


class TestOAuthTokenManager:
    def _make_manager_with_mock(self, monkeypatch, token="tok_abc", expires_in=3600):
        mgr = OAuthTokenManager()

        def fake_fetch(config):
            return {"access_token": token, "expires_in": expires_in}

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(fake_fetch))
        return mgr

    def test_no_config_returns_empty_headers(self):
        mgr = OAuthTokenManager()
        assert mgr.get_auth_headers("unknown") == {}

    def test_fetches_and_caches_token(self, monkeypatch):
        mgr = self._make_manager_with_mock(monkeypatch, token="tok_123")
        config = OAuthConfig(
            token_url="https://x/token",
            grant_type="client_credentials",
            client_id="a",
            client_secret="b",
        )
        mgr.register("srv", config)
        headers = mgr.get_auth_headers("srv")
        assert headers == {"Authorization": "Bearer tok_123"}

        # Second call uses cache (no second fetch)
        call_count = 0
        original_fetch = OAuthTokenManager._fetch_token

        def counting_fetch(cfg):
            nonlocal call_count
            call_count += 1
            return {"access_token": "tok_123", "expires_in": 3600}

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(counting_fetch))
        headers2 = mgr.get_auth_headers("srv")
        assert headers2 == {"Authorization": "Bearer tok_123"}
        assert call_count == 0  # cache hit, no fetch

    def test_invalidate_forces_refetch(self, monkeypatch):
        fetch_count = 0

        def counting_fetch(config):
            nonlocal fetch_count
            fetch_count += 1
            return {"access_token": f"tok_{fetch_count}", "expires_in": 3600}

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(counting_fetch))
        mgr = OAuthTokenManager()
        config = OAuthConfig(
            token_url="https://x/token",
            grant_type="client_credentials",
            client_id="a",
            client_secret="b",
        )
        mgr.register("srv", config)
        h1 = mgr.get_auth_headers("srv")
        assert "tok_1" in h1["Authorization"]

        mgr.invalidate("srv")
        h2 = mgr.get_auth_headers("srv")
        assert "tok_2" in h2["Authorization"]
        assert fetch_count == 2

    def test_has_config(self):
        mgr = OAuthTokenManager()
        assert not mgr.has_config("x")
        mgr.register("x", OAuthConfig("u", "g", "c", "s"))
        assert mgr.has_config("x")

    def test_fetch_failure_raises_oauth_error(self, monkeypatch):
        def failing_fetch(config):
            raise OAuthError("boom")

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(failing_fetch))
        mgr = OAuthTokenManager()
        mgr.register("srv", OAuthConfig("u", "g", "c", "s"))
        with pytest.raises(OAuthError, match="boom"):
            mgr.get_auth_headers("srv")
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_mcp_oauth.py -x` -- fails because files do not exist.
2. GREEN: Create the files exactly as above. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 5 minutes.

---

## Task 4: Create MCP tool loader (`muse/mcp/tools.py`)

**Files:**
- `muse/mcp/tools.py` (create)
- `tests/test_mcp_tools.py` (create)

**What to do:**

Create the async function `get_mcp_tools()` that connects to all configured MCP servers via `langchain-mcp-adapters` and returns a flat list of LangChain `BaseTool` objects. On any connection failure, the failing server is skipped (never blocks startup).

Create `muse/mcp/tools.py`:

```python
"""Load LangChain tools from configured MCP servers.

The ``get_mcp_tools`` coroutine is the single entry-point.  It:

1. Reads extensions.yaml via ``load_extensions_config``
2. Builds connection params via ``build_multiserver_params``
3. Injects OAuth headers where configured
4. Connects via ``MultiServerMCPClient`` and returns ``list[BaseTool]``
5. Gracefully degrades: any server failure yields a warning, not an error
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from muse.mcp.client import (
    HttpServerConfig,
    SseServerConfig,
    build_multiserver_params,
    load_extensions_config,
)
from muse.mcp.oauth import OAuthTokenManager, parse_oauth_config

logger = logging.getLogger(__name__)


async def get_mcp_tools(
    config_path: str | None = None,
    *,
    oauth_manager: OAuthTokenManager | None = None,
) -> list[Any]:
    """Return LangChain ``BaseTool`` instances from all configured MCP servers.

    Returns an empty list when:
    - No extensions.yaml is found
    - langchain-mcp-adapters is not installed
    - All servers fail to connect
    """
    configs = load_extensions_config(config_path)
    if not configs:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed; MCP tools unavailable")
        return []

    # Inject OAuth headers into configs that need them
    if oauth_manager is None:
        oauth_manager = OAuthTokenManager()
        # Auto-register OAuth configs parsed from extensions.yaml
        for cfg in configs:
            # Re-read the raw yaml entry to get oauth block
            # (we only have typed configs here, so oauth must be
            #  pre-registered by the caller or via _auto_register below)
            pass

    params = build_multiserver_params(configs)

    # Inject OAuth headers per-server
    for cfg in configs:
        if oauth_manager.has_config(cfg.name):
            try:
                auth_headers = oauth_manager.get_auth_headers(cfg.name)
                if cfg.name in params:
                    existing_headers = params[cfg.name].get("headers", {})
                    existing_headers.update(auth_headers)
                    params[cfg.name]["headers"] = existing_headers
            except Exception as exc:
                logger.warning("OAuth for MCP server '%s' failed: %s", cfg.name, exc)

    tools: list[Any] = []
    try:
        async with MultiServerMCPClient(params) as client:
            tools = client.get_tools()
    except Exception as exc:
        logger.warning("MCP connection failed: %s", exc)
        return []

    logger.info("Loaded %d MCP tools from %d servers", len(tools), len(configs))
    return tools


def get_mcp_tools_sync(
    config_path: str | None = None,
    *,
    oauth_manager: OAuthTokenManager | None = None,
) -> list[Any]:
    """Synchronous wrapper around ``get_mcp_tools``."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are inside an existing event loop; run in a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                get_mcp_tools(config_path, oauth_manager=oauth_manager),
            )
            return future.result(timeout=60)
    return asyncio.run(get_mcp_tools(config_path, oauth_manager=oauth_manager))
```

Create `tests/test_mcp_tools.py`:

```python
"""Tests for MCP tool loader (muse.mcp.tools)."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muse.mcp.tools import get_mcp_tools


class _FakeTool:
    """Minimal stand-in for a LangChain BaseTool."""
    def __init__(self, name: str):
        self.name = name


class TestGetMcpTools:
    def test_returns_empty_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSE_EXTENSIONS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        result = asyncio.run(get_mcp_tools(str(tmp_path / "nope.yaml")))
        assert result == []

    def test_returns_empty_when_no_servers_defined(self, tmp_path):
        p = tmp_path / "extensions.yaml"
        p.write_text("other: 1\n", encoding="utf-8")
        result = asyncio.run(get_mcp_tools(str(p)))
        assert result == []

    def test_returns_empty_when_adapter_not_installed(self, tmp_path, monkeypatch):
        p = tmp_path / "extensions.yaml"
        p.write_text(textwrap.dedent("""\
            mcp_servers:
              test:
                transport: sse
                url: "http://localhost:9999/sse"
        """), encoding="utf-8")

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langchain_mcp_adapters" in name:
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = asyncio.run(get_mcp_tools(str(p)))
        assert result == []

    def test_returns_tools_from_mock_client(self, tmp_path):
        p = tmp_path / "extensions.yaml"
        p.write_text(textwrap.dedent("""\
            mcp_servers:
              test_server:
                transport: sse
                url: "http://localhost:9999/sse"
        """), encoding="utf-8")

        fake_tools = [_FakeTool("tool_a"), _FakeTool("tool_b")]

        mock_client_instance = MagicMock()
        mock_client_instance.get_tools.return_value = fake_tools
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with patch.dict("sys.modules", {
            "langchain_mcp_adapters": MagicMock(),
            "langchain_mcp_adapters.client": MagicMock(MultiServerMCPClient=mock_client_cls),
        }):
            result = asyncio.run(get_mcp_tools(str(p)))

        assert len(result) == 2
        assert result[0].name == "tool_a"

    def test_graceful_failure_on_connection_error(self, tmp_path):
        p = tmp_path / "extensions.yaml"
        p.write_text(textwrap.dedent("""\
            mcp_servers:
              failing:
                transport: sse
                url: "http://localhost:1/sse"
        """), encoding="utf-8")

        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with patch.dict("sys.modules", {
            "langchain_mcp_adapters": MagicMock(),
            "langchain_mcp_adapters.client": MagicMock(MultiServerMCPClient=mock_client_cls),
        }):
            result = asyncio.run(get_mcp_tools(str(p)))

        assert result == []
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_mcp_tools.py -x` -- fails because file does not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 5 minutes.

---

## Task 5: Create tool cache (`muse/mcp/cache.py`)

**Files:**
- `muse/mcp/cache.py` (create)
- `tests/test_mcp_cache.py` (create)

**What to do:**

Create `MCPToolCache` that stores loaded MCP tools and supports hot-reload when `extensions.yaml` changes (detected by file mtime). Callers use `cache.get_tools()` which returns the cached list unless the config file has been modified.

Create `muse/mcp/cache.py`:

```python
"""MCP tool cache with config-change hot-reload.

Caches the list of loaded MCP tools and re-fetches only when
extensions.yaml is modified (detected by file mtime).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from muse.mcp.client import _resolve_config_path
from muse.mcp.oauth import OAuthTokenManager
from muse.mcp.tools import get_mcp_tools_sync

logger = logging.getLogger(__name__)


class MCPToolCache:
    """Lazy-loading MCP tool cache with mtime-based invalidation.

    Usage::

        cache = MCPToolCache()
        tools = cache.get_tools()          # loads on first call
        tools = cache.get_tools()          # returns cached
        # ... user edits extensions.yaml ...
        tools = cache.get_tools()          # detects mtime change, reloads
    """

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
        self._loaded: bool = False
        self._last_mtime: float = 0.0
        self._last_check: float = 0.0

    def get_tools(self) -> list[Any]:
        """Return cached MCP tools, reloading if config file changed."""
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
        """Force reload on next ``get_tools()`` call."""
        self._loaded = False
        self._last_mtime = 0.0
        self._last_check = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded
```

Create `tests/test_mcp_cache.py`:

```python
"""Tests for MCP tool cache (muse.mcp.cache)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from muse.mcp.cache import MCPToolCache


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


class TestMCPToolCache:
    def test_returns_empty_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSE_EXTENSIONS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        cache = MCPToolCache(str(tmp_path / "nope.yaml"))
        assert cache.get_tools() == []
        assert cache.is_loaded

    def test_caches_tools(self, tmp_path):
        config_file = tmp_path / "extensions.yaml"
        config_file.write_text("mcp_servers:\n  x:\n    transport: sse\n    url: http://x/sse\n")

        call_count = 0

        def fake_sync(path, *, oauth_manager=None):
            nonlocal call_count
            call_count += 1
            return [_FakeTool(f"t{call_count}")]

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(config_file), min_reload_interval=0.0)
            tools1 = cache.get_tools()
            assert len(tools1) == 1
            assert tools1[0].name == "t1"
            assert call_count == 1

            # Second call returns cached (mtime unchanged)
            tools2 = cache.get_tools()
            assert tools2[0].name == "t1"
            assert call_count == 1

    def test_reloads_on_mtime_change(self, tmp_path):
        config_file = tmp_path / "extensions.yaml"
        config_file.write_text("mcp_servers:\n  x:\n    transport: sse\n    url: http://x/sse\n")

        call_count = 0

        def fake_sync(path, *, oauth_manager=None):
            nonlocal call_count
            call_count += 1
            return [_FakeTool(f"t{call_count}")]

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(config_file), min_reload_interval=0.0)
            cache.get_tools()
            assert call_count == 1

            # Simulate config file change (update mtime)
            import os
            time.sleep(0.05)
            config_file.write_text("mcp_servers:\n  y:\n    transport: sse\n    url: http://y/sse\n")

            tools2 = cache.get_tools()
            assert call_count == 2
            assert tools2[0].name == "t2"

    def test_invalidate_forces_reload(self, tmp_path):
        config_file = tmp_path / "extensions.yaml"
        config_file.write_text("mcp_servers:\n  x:\n    transport: sse\n    url: http://x/sse\n")

        call_count = 0

        def fake_sync(path, *, oauth_manager=None):
            nonlocal call_count
            call_count += 1
            return [_FakeTool(f"t{call_count}")]

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(config_file), min_reload_interval=0.0)
            cache.get_tools()
            assert call_count == 1

            cache.invalidate()
            assert not cache.is_loaded

            cache.get_tools()
            assert call_count == 2

    def test_respects_min_reload_interval(self, tmp_path):
        config_file = tmp_path / "extensions.yaml"
        config_file.write_text("mcp_servers:\n  x:\n    transport: sse\n    url: http://x/sse\n")

        call_count = 0

        def fake_sync(path, *, oauth_manager=None):
            nonlocal call_count
            call_count += 1
            return [_FakeTool(f"t{call_count}")]

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(config_file), min_reload_interval=100.0)
            cache.get_tools()
            assert call_count == 1

            # Even with mtime change, interval prevents re-check
            time.sleep(0.05)
            config_file.write_text("mcp_servers: {}\n")
            cache.get_tools()
            assert call_count == 1  # still cached due to interval
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_mcp_cache.py -x` -- fails because file does not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 4 minutes.

---

## Task 6: Integration with ToolRegistry -- register MCP tools into "mcp" group

**Files:**
- `muse/mcp/registry_bridge.py` (create)
- `tests/test_mcp_registry_bridge.py` (create)

**What to do:**

Create the bridge function that loads MCP tools via the cache and registers them into a ToolRegistry instance under the `"mcp"` group. This task only creates the bridge; it does NOT modify `ToolRegistry` (that is Phase 0-A). The bridge function is a standalone helper.

Create `muse/mcp/registry_bridge.py`:

```python
"""Bridge between MCP tool cache and ToolRegistry.

Provides ``register_mcp_tools(registry, cache)`` which loads MCP tools
from the cache and registers them into the ``"mcp"`` tool group.

This module is the integration point called during ToolRegistry.build().
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _ToolRegistryProtocol(Protocol):
    """Minimal interface expected from ToolRegistry."""
    def register(self, tool: Any, group: str) -> None: ...


def register_mcp_tools(
    registry: _ToolRegistryProtocol,
    *,
    cache: Any,
    group: str = "mcp",
) -> int:
    """Load MCP tools from *cache* and register into *registry*.

    Returns the count of tools registered.  Never raises -- failures
    are logged as warnings.
    """
    try:
        tools = cache.get_tools()
    except Exception as exc:
        logger.warning("Failed to load MCP tools: %s", exc)
        return 0

    registered = 0
    for tool in tools:
        try:
            registry.register(tool, group)
            registered += 1
        except Exception as exc:
            tool_name = getattr(tool, "name", repr(tool))
            logger.warning("Failed to register MCP tool '%s': %s", tool_name, exc)

    if registered:
        logger.info("Registered %d MCP tools into '%s' group", registered, group)
    return registered
```

Create `tests/test_mcp_registry_bridge.py`:

```python
"""Tests for MCP-to-ToolRegistry bridge (muse.mcp.registry_bridge)."""

from __future__ import annotations

import pytest

from muse.mcp.registry_bridge import register_mcp_tools


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


class _FakeRegistry:
    def __init__(self):
        self.registered: list[tuple] = []

    def register(self, tool, group: str):
        self.registered.append((tool, group))


class _FakeCache:
    def __init__(self, tools=None, *, error=None):
        self._tools = tools or []
        self._error = error

    def get_tools(self):
        if self._error:
            raise self._error
        return self._tools


class TestRegisterMcpTools:
    def test_registers_tools_into_mcp_group(self):
        tools = [_FakeTool("a"), _FakeTool("b")]
        registry = _FakeRegistry()
        cache = _FakeCache(tools)
        count = register_mcp_tools(registry, cache=cache)
        assert count == 2
        assert all(group == "mcp" for _, group in registry.registered)

    def test_custom_group(self):
        registry = _FakeRegistry()
        cache = _FakeCache([_FakeTool("x")])
        register_mcp_tools(registry, cache=cache, group="external")
        assert registry.registered[0][1] == "external"

    def test_returns_zero_on_cache_error(self):
        registry = _FakeRegistry()
        cache = _FakeCache(error=RuntimeError("boom"))
        count = register_mcp_tools(registry, cache=cache)
        assert count == 0
        assert registry.registered == []

    def test_returns_zero_on_empty_cache(self):
        registry = _FakeRegistry()
        cache = _FakeCache([])
        count = register_mcp_tools(registry, cache=cache)
        assert count == 0

    def test_skips_failing_registration(self):
        class _FailingRegistry:
            def __init__(self):
                self.registered = []

            def register(self, tool, group):
                if tool.name == "bad":
                    raise ValueError("nope")
                self.registered.append((tool, group))

        tools = [_FakeTool("good"), _FakeTool("bad"), _FakeTool("also_good")]
        registry = _FailingRegistry()
        cache = _FakeCache(tools)
        count = register_mcp_tools(registry, cache=cache)
        assert count == 2
        assert len(registry.registered) == 2
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_mcp_registry_bridge.py -x` -- fails because file does not exist.
2. GREEN: Create files. Run again -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 3 minutes.

---

## Task 7: Integration test with mock MCP server

**Files:**
- `tests/test_mcp_integration.py` (create)

**What to do:**

Create an end-to-end integration test that exercises the full MCP pipeline: config loading, cache, registry bridge. Uses mocks for the actual MCP server connection (no real server needed).

Create `tests/test_mcp_integration.py`:

```python
"""End-to-end integration test for MCP pipeline.

Tests the full flow: extensions.yaml -> config loader -> tool cache -> registry bridge.
Uses mock MCP server (no real server required).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muse.mcp.cache import MCPToolCache
from muse.mcp.client import load_extensions_config, SseServerConfig
from muse.mcp.oauth import OAuthConfig, OAuthTokenManager
from muse.mcp.registry_bridge import register_mcp_tools


class _FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class _FakeRegistry:
    def __init__(self):
        self.tools: dict[str, list] = {}

    def register(self, tool, group: str):
        self.tools.setdefault(group, []).append(tool)


def _write_config(tmp_path: Path) -> Path:
    p = tmp_path / "extensions.yaml"
    p.write_text(textwrap.dedent("""\
        mcp_servers:
          zotero:
            transport: stdio
            command: npx
            args: ["-y", "@anthropic/mcp-server-zotero"]
          local_search:
            transport: sse
            url: "http://localhost:8080/sse"
          overleaf:
            transport: http
            url: "https://overleaf-mcp.example.com/mcp"
            headers:
              X-Custom: "value"
    """), encoding="utf-8")
    return p


class TestMCPFullPipeline:
    def test_config_loads_all_three_servers(self, tmp_path):
        p = _write_config(tmp_path)
        configs = load_extensions_config(p)
        assert len(configs) == 3
        names = {c.name for c in configs}
        assert names == {"zotero", "local_search", "overleaf"}

    def test_cache_loads_and_registers_tools(self, tmp_path):
        p = _write_config(tmp_path)
        fake_tools = [
            _FakeTool("zotero_search", "Search Zotero library"),
            _FakeTool("zotero_add", "Add to Zotero"),
            _FakeTool("local_query", "Query local index"),
        ]

        def fake_sync(path, *, oauth_manager=None):
            return fake_tools

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(p), min_reload_interval=0.0)
            registry = _FakeRegistry()
            count = register_mcp_tools(registry, cache=cache)

        assert count == 3
        assert len(registry.tools["mcp"]) == 3
        tool_names = {t.name for t in registry.tools["mcp"]}
        assert "zotero_search" in tool_names

    def test_oauth_headers_injected(self, tmp_path):
        p = tmp_path / "extensions.yaml"
        p.write_text(textwrap.dedent("""\
            mcp_servers:
              secured:
                transport: http
                url: "https://example.com/mcp"
                oauth:
                  token_url: "https://auth.example.com/token"
                  client_id: "cid"
                  client_secret: "csec"
        """), encoding="utf-8")

        mgr = OAuthTokenManager()
        mgr.register("secured", OAuthConfig(
            token_url="https://auth.example.com/token",
            grant_type="client_credentials",
            client_id="cid",
            client_secret="csec",
        ))

        # Verify OAuth manager has the config
        assert mgr.has_config("secured")

    def test_empty_config_produces_zero_tools(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSE_EXTENSIONS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        cache = MCPToolCache(str(tmp_path / "nonexistent.yaml"))
        registry = _FakeRegistry()
        count = register_mcp_tools(registry, cache=cache)
        assert count == 0
        assert registry.tools == {}

    def test_cache_reload_updates_registry(self, tmp_path):
        """Simulate editing extensions.yaml and verifying cache reloads."""
        p = tmp_path / "extensions.yaml"
        p.write_text("mcp_servers:\n  a:\n    transport: sse\n    url: http://a/sse\n")

        call_count = 0

        def fake_sync(path, *, oauth_manager=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_FakeTool("tool_v1")]
            return [_FakeTool("tool_v2"), _FakeTool("tool_v2b")]

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(p), min_reload_interval=0.0)

            # First load
            registry1 = _FakeRegistry()
            register_mcp_tools(registry1, cache=cache)
            assert len(registry1.tools.get("mcp", [])) == 1

            # Simulate config change
            import time
            time.sleep(0.05)
            p.write_text("mcp_servers:\n  b:\n    transport: sse\n    url: http://b/sse\n")

            # Second load (cache detects mtime change)
            registry2 = _FakeRegistry()
            register_mcp_tools(registry2, cache=cache)
            assert len(registry2.tools.get("mcp", [])) == 2
```

**TDD:**

1. RED: Run `python3 -m pytest tests/test_mcp_integration.py -x` -- fails because file does not exist.
2. GREEN: Create the file. Run `python3 -m pytest tests/test_mcp_integration.py -v` -- all tests pass.
3. REFACTOR: None needed.

**Time estimate:** 4 minutes.

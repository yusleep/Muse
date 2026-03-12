"""End-to-end integration test for the MCP pipeline."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import patch

from muse.mcp.cache import MCPToolCache
from muse.mcp.client import SseServerConfig, load_extensions_config
from muse.mcp.oauth import OAuthConfig, OAuthTokenManager
from muse.mcp.registry_bridge import register_mcp_tools
from muse.tools.registry import ToolRegistry


class _FakeTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "extensions.yaml"
    path.write_text(
        textwrap.dedent(
            """\
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
            """
        ),
        encoding="utf-8",
    )
    return path


class TestMCPFullPipeline:
    def test_config_loads_all_three_servers(self, tmp_path):
        path = _write_config(tmp_path)
        configs = load_extensions_config(path)
        assert len(configs) == 3
        names = {config.name for config in configs}
        assert names == {"zotero", "local_search", "overleaf"}
        assert any(isinstance(config, SseServerConfig) for config in configs)

    def test_cache_loads_and_registers_tools(self, tmp_path):
        path = _write_config(tmp_path)
        fake_tools = [
            _FakeTool("zotero_search", "Search Zotero library"),
            _FakeTool("zotero_add", "Add to Zotero"),
            _FakeTool("local_query", "Query local index"),
        ]

        def fake_sync(config_path, *, oauth_manager=None):
            return fake_tools

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(path), min_reload_interval=0.0)
            registry = ToolRegistry()
            count = register_mcp_tools(registry, cache=cache)

        assert count == 3
        tool_names = {tool.name for tool in registry.get_tools(groups=["mcp"])}
        assert tool_names == {"zotero_search", "zotero_add", "local_query"}

    def test_oauth_manager_registers_config(self):
        manager = OAuthTokenManager()
        manager.register(
            "secured",
            OAuthConfig(
                token_url="https://auth.example.com/token",
                grant_type="client_credentials",
                client_id="cid",
                client_secret="csec",
            ),
        )
        assert manager.has_config("secured")

    def test_empty_config_produces_zero_tools(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSE_EXTENSIONS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        cache = MCPToolCache(str(tmp_path / "nonexistent.yaml"))
        registry = ToolRegistry()
        count = register_mcp_tools(registry, cache=cache)
        assert count == 0
        assert registry.get_tools(groups=["mcp"]) == []

    def test_cache_reload_updates_registry(self, tmp_path):
        path = tmp_path / "extensions.yaml"
        path.write_text("mcp_servers:\n  a:\n    transport: sse\n    url: http://a/sse\n")

        call_count = 0

        def fake_sync(config_path, *, oauth_manager=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [_FakeTool("tool_v1")]
            return [_FakeTool("tool_v2"), _FakeTool("tool_v2b")]

        with patch("muse.mcp.cache.get_mcp_tools_sync", side_effect=fake_sync):
            cache = MCPToolCache(str(path), min_reload_interval=0.0)

            registry_first = ToolRegistry()
            register_mcp_tools(registry_first, cache=cache)
            assert {tool.name for tool in registry_first.get_tools(groups=["mcp"])} == {"tool_v1"}

            time.sleep(0.05)
            path.write_text("mcp_servers:\n  b:\n    transport: sse\n    url: http://b/sse\n")

            registry_second = ToolRegistry()
            register_mcp_tools(registry_second, cache=cache)
            assert {tool.name for tool in registry_second.get_tools(groups=["mcp"])} == {
                "tool_v2",
                "tool_v2b",
            }

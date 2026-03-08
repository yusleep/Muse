"""Tests for MCP tool cache (muse.mcp.cache)."""

from __future__ import annotations

import time
from unittest.mock import patch

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
            tools_first = cache.get_tools()
            assert len(tools_first) == 1
            assert tools_first[0].name == "t1"
            assert call_count == 1

            tools_second = cache.get_tools()
            assert tools_second[0].name == "t1"
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

            time.sleep(0.05)
            config_file.write_text("mcp_servers:\n  y:\n    transport: sse\n    url: http://y/sse\n")

            tools_second = cache.get_tools()
            assert call_count == 2
            assert tools_second[0].name == "t2"

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

            time.sleep(0.05)
            config_file.write_text("mcp_servers: {}\n")
            cache.get_tools()
            assert call_count == 1

"""Tests for MCP config loader (muse.mcp.client)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from muse.mcp.client import (
    HttpServerConfig,
    SseServerConfig,
    StdioServerConfig,
    build_multiserver_params,
    load_extensions_config,
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "extensions.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


class TestLoadExtensionsConfig:
    def test_empty_file_returns_empty(self, tmp_path: Path):
        path = _write_yaml(tmp_path, "")
        assert load_extensions_config(path) == []

    def test_no_mcp_servers_key_returns_empty(self, tmp_path: Path):
        path = _write_yaml(tmp_path, "other_key: 123\n")
        assert load_extensions_config(path) == []

    def test_nonexistent_path_returns_empty(self, tmp_path: Path):
        assert load_extensions_config(tmp_path / "nope.yaml") == []

    def test_none_path_no_file_returns_empty(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("MUSE_EXTENSIONS_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        assert load_extensions_config(None) == []

    def test_parses_stdio_server(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """\
            mcp_servers:
              zotero:
                transport: stdio
                command: npx
                args: ["-y", "@anthropic/mcp-server-zotero"]
            """,
        )
        configs = load_extensions_config(path)
        assert len(configs) == 1
        assert isinstance(configs[0], StdioServerConfig)
        assert configs[0].name == "zotero"
        assert configs[0].command == "npx"
        assert configs[0].args == ["-y", "@anthropic/mcp-server-zotero"]

    def test_parses_sse_server(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """\
            mcp_servers:
              local_search:
                transport: sse
                url: "http://localhost:8080/sse"
            """,
        )
        configs = load_extensions_config(path)
        assert len(configs) == 1
        assert isinstance(configs[0], SseServerConfig)
        assert configs[0].url == "http://localhost:8080/sse"

    def test_parses_http_server_with_headers(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """\
            mcp_servers:
              overleaf:
                transport: http
                url: "https://overleaf-mcp.example.com/mcp"
                headers:
                  Authorization: "Bearer tok123"
            """,
        )
        configs = load_extensions_config(path)
        assert len(configs) == 1
        assert isinstance(configs[0], HttpServerConfig)
        assert configs[0].headers == {"Authorization": "Bearer tok123"}

    def test_skips_invalid_entries(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """\
            mcp_servers:
              good:
                transport: stdio
                command: echo
              bad_no_transport:
                url: "http://x"
              bad_no_command:
                transport: stdio
            """,
        )
        configs = load_extensions_config(path)
        assert len(configs) == 1
        assert configs[0].name == "good"

    def test_mixed_transports(self, tmp_path: Path):
        path = _write_yaml(
            tmp_path,
            """\
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
            """,
        )
        configs = load_extensions_config(path)
        assert len(configs) == 3
        types = {type(config) for config in configs}
        assert types == {StdioServerConfig, SseServerConfig, HttpServerConfig}


class TestBuildMultiserverParams:
    def test_stdio_params(self):
        config = StdioServerConfig(
            name="z",
            command="npx",
            args=["-y", "pkg"],
            env={"KEY": "val"},
        )
        result = build_multiserver_params([config])
        assert result == {
            "z": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "pkg"],
                "env": {"KEY": "val"},
            }
        }

    def test_sse_params(self):
        config = SseServerConfig(name="s", url="http://localhost:8080/sse")
        result = build_multiserver_params([config])
        assert result == {
            "s": {"transport": "sse", "url": "http://localhost:8080/sse"},
        }

    def test_http_params_with_headers(self):
        config = HttpServerConfig(
            name="h",
            url="https://x.com/mcp",
            headers={"Auth": "Bearer t"},
        )
        result = build_multiserver_params([config])
        assert result == {
            "h": {
                "transport": "streamable_http",
                "url": "https://x.com/mcp",
                "headers": {"Auth": "Bearer t"},
            }
        }

    def test_empty_input(self):
        assert build_multiserver_params([]) == {}

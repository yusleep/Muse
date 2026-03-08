"""Tests for MCP tool loader (muse.mcp.tools)."""

from __future__ import annotations

import asyncio
import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

from muse.mcp.oauth import OAuthTokenManager
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
        path = tmp_path / "extensions.yaml"
        path.write_text("other: 1\n", encoding="utf-8")
        result = asyncio.run(get_mcp_tools(str(path)))
        assert result == []

    def test_returns_empty_when_adapter_not_installed(self, tmp_path, monkeypatch):
        path = tmp_path / "extensions.yaml"
        path.write_text(
            textwrap.dedent(
                """\
                mcp_servers:
                  test:
                    transport: sse
                    url: "http://localhost:9999/sse"
                """
            ),
            encoding="utf-8",
        )

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langchain_mcp_adapters" in name:
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = asyncio.run(get_mcp_tools(str(path)))
        assert result == []

    def test_returns_tools_from_mock_client(self, tmp_path):
        path = tmp_path / "extensions.yaml"
        path.write_text(
            textwrap.dedent(
                """\
                mcp_servers:
                  test_server:
                    transport: sse
                    url: "http://localhost:9999/sse"
                """
            ),
            encoding="utf-8",
        )

        fake_tools = [_FakeTool("tool_a"), _FakeTool("tool_b")]
        mock_client_instance = MagicMock()
        mock_client_instance.get_tools.return_value = fake_tools
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with patch.dict(
            "sys.modules",
            {
                "langchain_mcp_adapters": MagicMock(),
                "langchain_mcp_adapters.client": MagicMock(MultiServerMCPClient=mock_client_cls),
            },
        ):
            result = asyncio.run(get_mcp_tools(str(path)))

        assert len(result) == 2
        assert result[0].name == "tool_a"

    def test_graceful_failure_on_connection_error(self, tmp_path):
        path = tmp_path / "extensions.yaml"
        path.write_text(
            textwrap.dedent(
                """\
                mcp_servers:
                  failing:
                    transport: sse
                    url: "http://localhost:1/sse"
                """
            ),
            encoding="utf-8",
        )

        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with patch.dict(
            "sys.modules",
            {
                "langchain_mcp_adapters": MagicMock(),
                "langchain_mcp_adapters.client": MagicMock(MultiServerMCPClient=mock_client_cls),
            },
        ):
            result = asyncio.run(get_mcp_tools(str(path)))

        assert result == []

    def test_injects_oauth_headers(self, tmp_path, monkeypatch):
        path = tmp_path / "extensions.yaml"
        path.write_text(
            textwrap.dedent(
                """\
                mcp_servers:
                  secured:
                    transport: http
                    url: "https://example.com/mcp"
                """
            ),
            encoding="utf-8",
        )

        manager = OAuthTokenManager()
        monkeypatch.setattr(
            manager,
            "has_config",
            lambda name: name == "secured",
        )
        monkeypatch.setattr(
            manager,
            "get_auth_headers",
            lambda name: {"Authorization": "Bearer tok"},
        )

        captured_params: list[dict[str, dict[str, str]]] = []
        mock_client_instance = MagicMock()
        mock_client_instance.get_tools.return_value = []
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)

        def make_client(params):
            captured_params.append(params)
            return mock_client_instance

        with patch.dict(
            "sys.modules",
            {
                "langchain_mcp_adapters": MagicMock(),
                "langchain_mcp_adapters.client": MagicMock(MultiServerMCPClient=make_client),
            },
        ):
            asyncio.run(get_mcp_tools(str(path), oauth_manager=manager))

        assert captured_params[0]["secured"]["headers"]["Authorization"] == "Bearer tok"

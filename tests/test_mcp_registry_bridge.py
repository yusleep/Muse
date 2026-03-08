"""Tests for MCP-to-ToolRegistry bridge (muse.mcp.registry_bridge)."""

from __future__ import annotations

from muse.mcp.registry_bridge import register_mcp_tools


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


class _FakeRegistry:
    def __init__(self):
        self.registered: list[tuple] = []

    def register(self, tool, *, group: str):
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

            def register(self, tool, *, group: str):
                if tool.name == "bad":
                    raise ValueError("nope")
                self.registered.append((tool, group))

        tools = [_FakeTool("good"), _FakeTool("bad"), _FakeTool("also_good")]
        registry = _FailingRegistry()
        cache = _FakeCache(tools)
        count = register_mcp_tools(registry, cache=cache)
        assert count == 2
        assert len(registry.registered) == 2

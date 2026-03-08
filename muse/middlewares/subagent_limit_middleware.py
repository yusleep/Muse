"""Middleware that truncates excess ``spawn_subagent`` tool calls."""

from __future__ import annotations

from typing import Any

from muse.tools.orchestration import get_subagent_limit, set_subagent_limit


_SPAWN_TOOL_NAME = "spawn_subagent"


class SubagentLimitMiddleware:
    """Hard-truncate spawn tool calls that exceed the concurrency limit."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._max_concurrent = max_concurrent

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    async def before_invoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        """Protocol no-op; runtime behavior is provided by ``wrap_node``."""

        del config
        return state

    async def after_invoke(
        self,
        state: dict[str, Any],
        result: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Protocol no-op; runtime behavior is provided by ``wrap_node``."""

        del state, config
        return result

    def wrap_node(self, node_fn):
        """Install the active sub-agent limit for the duration of a node call."""

        def wrapped(*args, **kwargs):
            previous_limit = get_subagent_limit()
            set_subagent_limit(self._max_concurrent)
            try:
                return node_fn(*args, **kwargs)
            finally:
                set_subagent_limit(previous_limit)

        return wrapped

    def filter_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return tool calls with excess spawn calls removed."""

        result: list[dict[str, Any]] = []
        spawn_count = 0
        for tool_call in tool_calls:
            if tool_call.get("name") == _SPAWN_TOOL_NAME:
                if spawn_count < self._max_concurrent:
                    result.append(tool_call)
                    spawn_count += 1
                continue
            result.append(tool_call)
        return result

    def dropped_count(self, tool_calls: list[dict[str, Any]]) -> int:
        """Return how many spawn calls would be dropped."""

        spawn_total = sum(1 for tool_call in tool_calls if tool_call.get("name") == _SPAWN_TOOL_NAME)
        return max(0, spawn_total - self._max_concurrent)

"""Middleware that truncates excess ``spawn_subagent`` tool calls."""

from __future__ import annotations

from typing import Any


_SPAWN_TOOL_NAME = "spawn_subagent"


class SubagentLimitMiddleware:
    """Hard-truncate spawn tool calls that exceed the concurrency limit."""

    def __init__(self, max_concurrent: int = 3) -> None:
        self._max_concurrent = max_concurrent

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

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

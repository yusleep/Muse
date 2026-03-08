"""JSONL logging middleware for graph node execution tracing."""

from __future__ import annotations

import json
import os
import time
from typing import Any


class LoggingMiddleware:
    """Record node execution timing and usage metadata to a JSONL log."""

    def __init__(self, log_path: str | None = None, node_name: str = "unknown") -> None:
        self._log_path = log_path
        self._node_name = node_name
        self._start_times: dict[str, float] = {}

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        self._start_times[_context_key(config)] = time.monotonic()
        return state

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        key = _context_key(config)
        start_time = self._start_times.pop(key, None)
        latency_ms = 0.0
        if start_time is not None:
            latency_ms = (time.monotonic() - start_time) * 1000.0

        if self._log_path is not None:
            configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
            entry: dict[str, Any] = {
                "timestamp": time.time(),
                "node": configurable.get("node_name", self._node_name),
                "latency_ms": round(latency_ms, 2),
                "result_keys": sorted(result) if isinstance(result, dict) else [],
            }
            thread_id = configurable.get("thread_id")
            if thread_id:
                entry["thread_id"] = thread_id
            usage = result.get("_usage") if isinstance(result, dict) else None
            if isinstance(usage, dict):
                entry["usage"] = usage
            _append_jsonl(self._log_path, entry)

        return result


def _context_key(config: dict[str, Any]) -> str:
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    thread_id = configurable.get("thread_id")
    if isinstance(thread_id, str) and thread_id:
        return thread_id
    return "__default__"


def _append_jsonl(path: str, entry: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

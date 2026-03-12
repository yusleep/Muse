"""Retry middleware for transient provider and network errors."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from muse.services.http import ProviderError

_TRANSIENT_KEYWORDS = (
    "timed out",
    "timeout",
    "network",
    "connection",
    "502",
    "503",
    "429",
    "invalid json response",
    "rate limit",
)


def is_transient_error(exc: BaseException) -> bool:
    """Return ``True`` for retryable provider/network failures."""

    if not isinstance(exc, (ProviderError, ConnectionError, TimeoutError, OSError)):
        return False
    message = str(exc).lower()
    return any(keyword in message for keyword in _TRANSIENT_KEYWORDS)


class RetryMiddleware:
    """Retry wrapped node execution on transient failures with linear backoff."""

    def __init__(self, max_retries: int = 2, base_delay: float = 5.0) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return state

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        return result

    def wrap_node(self, node_fn: Any) -> Any:
        max_retries = self.max_retries
        base_delay = self.base_delay

        if inspect.iscoroutinefunction(node_fn):
            async def retrying_node(state: dict[str, Any]) -> dict[str, Any]:
                for attempt in range(max_retries + 1):
                    try:
                        return await node_fn(state)
                    except Exception as exc:  # noqa: BLE001
                        if not is_transient_error(exc) or attempt >= max_retries:
                            raise
                        await asyncio.sleep(base_delay * (attempt + 1))
                raise RuntimeError("unreachable")

            return retrying_node

        def retrying_node(state: dict[str, Any]) -> dict[str, Any]:
            for attempt in range(max_retries + 1):
                try:
                    return node_fn(state)
                except Exception as exc:  # noqa: BLE001
                    if not is_transient_error(exc) or attempt >= max_retries:
                        raise
                    time.sleep(base_delay * (attempt + 1))
            raise RuntimeError("unreachable")

        return retrying_node

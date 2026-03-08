"""Middleware protocol and chain for wrapping LangGraph node functions."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
from typing import Any, Callable, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig


@runtime_checkable
class Middleware(Protocol):
    """Hook interface for graph-node cross-cutting concerns."""

    async def before_invoke(
        self, state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def after_invoke(
        self, state: dict[str, Any], result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]: ...


class MiddlewareChain:
    """Ordered middleware stack that wraps a LangGraph node function."""

    def __init__(self, middlewares: list[Middleware]) -> None:
        self._middlewares = list(middlewares)

    def wrap(self, node_fn: Callable[..., Any]) -> Callable[..., dict[str, Any]]:
        middlewares = list(self._middlewares)
        effective_fn = node_fn
        for middleware in reversed(middlewares):
            wrapper = getattr(middleware, "wrap_node", None)
            if callable(wrapper):
                effective_fn = wrapper(effective_fn)

        def wrapped(
            state: dict[str, Any], config: RunnableConfig | None = None
        ) -> dict[str, Any]:
            return _run_sync(middlewares, effective_fn, state, config or {})

        return wrapped


async def _run(
    middlewares: list[Middleware],
    node_fn: Callable[..., Any],
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    current_state = state
    for middleware in middlewares:
        current_state = await middleware.before_invoke(current_state, config)

    result = node_fn(current_state)
    if inspect.isawaitable(result):
        result = await result

    current_result = result
    for middleware in reversed(middlewares):
        current_result = await middleware.after_invoke(current_state, current_result, config)

    return current_result


def _run_sync(
    middlewares: list[Middleware],
    node_fn: Callable[..., Any],
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run(middlewares, node_fn, state, config))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_new_loop, middlewares, node_fn, state, config)
        return future.result()


def _run_in_new_loop(
    middlewares: list[Middleware],
    node_fn: Callable[..., Any],
    state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return asyncio.run(_run(middlewares, node_fn, state, config))

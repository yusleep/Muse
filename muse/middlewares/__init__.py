"""Middleware framework for Muse graph nodes."""

from __future__ import annotations

import os
from typing import Any

from .base import Middleware, MiddlewareChain
from .clarification_middleware import ClarificationMiddleware
from .dangling_tool_call import DanglingToolCallMiddleware
from .logging_middleware import LoggingMiddleware
from muse.memory.middleware import MemoryMiddleware
from .retry_middleware import RetryMiddleware
from .subagent_limit_middleware import SubagentLimitMiddleware
from .summarization_middleware import SummarizationMiddleware

__all__ = [
    "ClarificationMiddleware",
    "DanglingToolCallMiddleware",
    "LoggingMiddleware",
    "MemoryMiddleware",
    "Middleware",
    "MiddlewareChain",
    "RetryMiddleware",
    "SubagentLimitMiddleware",
    "SummarizationMiddleware",
    "build_default_chain",
]


def build_default_chain(
    *,
    log_dir: str | None = None,
    node_name: str = "unknown",
    llm: Any = None,
    context_window: int = 128_000,
    compaction_threshold: float = 0.9,
    compaction_recent_tokens: int = 20_000,
    max_retries: int = 2,
    retry_base_delay: float = 5.0,
    memory_store: Any = None,
    memory_token_budget: int = 2000,
    subagent_max_concurrent: int | None = None,
) -> MiddlewareChain:
    """Build the standard middleware chain for graph nodes."""

    middlewares: list[Middleware] = []
    log_path = os.path.join(log_dir, "nodes.jsonl") if log_dir else None
    middlewares.append(LoggingMiddleware(log_path=log_path, node_name=node_name))
    middlewares.append(RetryMiddleware(max_retries=max_retries, base_delay=retry_base_delay))
    if subagent_max_concurrent is not None:
        middlewares.append(SubagentLimitMiddleware(max_concurrent=subagent_max_concurrent))
    if memory_store is not None:
        middlewares.append(
            MemoryMiddleware(
                memory_store,
                token_budget=memory_token_budget,
            )
        )
    middlewares.append(DanglingToolCallMiddleware())
    middlewares.append(ClarificationMiddleware())
    return MiddlewareChain(middlewares)

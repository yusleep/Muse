"""Middleware framework for Muse graph nodes."""

from __future__ import annotations

import os
from typing import Any

from .base import Middleware, MiddlewareChain
from .clarification_middleware import ClarificationMiddleware
from .dangling_tool_call import DanglingToolCallMiddleware
from .logging_middleware import LoggingMiddleware
from .retry_middleware import RetryMiddleware
from .subagent_limit_middleware import SubagentLimitMiddleware
from .summarization_middleware import SummarizationMiddleware

__all__ = [
    "ClarificationMiddleware",
    "DanglingToolCallMiddleware",
    "LoggingMiddleware",
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
) -> MiddlewareChain:
    """Build the standard middleware chain for graph nodes."""

    middlewares: list[Middleware] = []
    log_path = os.path.join(log_dir, "nodes.jsonl") if log_dir else None
    middlewares.append(LoggingMiddleware(log_path=log_path, node_name=node_name))
    middlewares.append(RetryMiddleware(max_retries=max_retries, base_delay=retry_base_delay))
    if llm is not None:
        middlewares.append(
            SummarizationMiddleware(
                llm=llm,
                context_window=context_window,
                threshold_ratio=compaction_threshold,
                recent_tokens=compaction_recent_tokens,
            )
        )
    middlewares.append(DanglingToolCallMiddleware())
    return MiddlewareChain(middlewares)

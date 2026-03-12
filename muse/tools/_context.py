"""Thread-local service context for tool functions."""

from __future__ import annotations

import threading
from typing import Any


_local = threading.local()


class _NullServices:
    llm = None
    search = None
    metadata = None
    rag_index = None
    local_refs: list = []


def set_services(services: Any) -> None:
    """Set the services object for the current thread."""

    _local.services = services


def get_services() -> Any:
    """Get the services object for the current thread."""

    return getattr(_local, "services", _NullServices())


def set_state(state: Any) -> None:
    """Set the active graph/agent state for tool calls on the current thread."""

    _local.state = state


def get_state(default: Any = None) -> Any:
    """Get the active graph/agent state for tool calls on the current thread."""

    if default is None:
        default = {}
    return getattr(_local, "state", default)


def clear_state() -> None:
    """Clear any active graph/agent state for tool calls on the current thread."""

    if hasattr(_local, "state"):
        delattr(_local, "state")

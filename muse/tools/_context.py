"""Thread-local service context for tool functions."""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, ConfigDict


_local = threading.local()


class AgentRuntimeContext(BaseModel):
    """Typed runtime context passed into ReAct agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    services: Any = None


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


def build_runtime_context(services: Any) -> AgentRuntimeContext:
    """Build a typed runtime context for LangGraph agent invocation."""

    return AgentRuntimeContext(services=services)


def services_from_runtime(runtime: Any) -> Any | None:
    """Extract services from a tool runtime, supporting dict and object contexts."""

    if runtime is None:
        return None

    context = getattr(runtime, "context", None)
    if isinstance(context, dict) and context.get("services") is not None:
        return context["services"]

    services = getattr(context, "services", None)
    if services is not None:
        return services

    return None


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

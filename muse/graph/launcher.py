"""Graph compilation and invocation helpers."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from muse.config import Settings


_LANGGRAPH_INSTALL_HINT = (
    "LangGraph runtime dependencies are not installed. "
    "Install them with `pip install -r requirements.txt` or "
    "`pip install langgraph langgraph-checkpoint-sqlite`."
)


def _is_missing_langgraph(exc: ModuleNotFoundError) -> bool:
    missing = str(exc.name or "")
    message = str(exc)
    return "langgraph" in missing or "langgraph" in message


def _load_langgraph_runtime():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command
        from muse.graph.main_graph import build_graph as build_main_graph
    except ModuleNotFoundError as exc:
        if _is_missing_langgraph(exc):
            raise RuntimeError(_LANGGRAPH_INSTALL_HINT) from exc
        raise
    return SqliteSaver, Command, build_main_graph


def _checkpoint_db_path(settings: Settings, thread_id: str) -> Path:
    if settings.checkpoint_dir:
        root = Path(settings.checkpoint_dir)
    else:
        root = Path(settings.runs_dir) / thread_id / "graph"
    root.mkdir(parents=True, exist_ok=True)
    return root / "checkpoints.sqlite"


def build_graph(
    settings: Settings,
    *,
    services: Any | None = None,
    thread_id: str = "default",
    auto_approve: bool = True,
):
    SqliteSaver, _, build_main_graph = _load_langgraph_runtime()
    db_path = _checkpoint_db_path(settings, thread_id)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(connection)
    graph = build_main_graph(
        settings,
        services=services,
        checkpointer=saver,
        auto_approve=auto_approve,
    )
    setattr(graph, "_muse_checkpoint_db", str(db_path))
    setattr(graph, "_muse_checkpoint_conn", connection)
    return graph


def invoke(graph: Any, state: dict[str, Any] | None, *, thread_id: str, resume: Any | None = None) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    if resume is not None:
        _, Command, _ = _load_langgraph_runtime()
        return graph.invoke(Command(resume=resume), config=config)
    return graph.invoke(state or {}, config=config)

"""Orchestration tools for ReAct agent control flow."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)
_local = threading.local()


def get_submitted_result() -> dict[str, Any] | None:
    """Retrieve the most recent submit_result payload."""

    return getattr(_local, "submitted_result", None)


def clear_submitted_result() -> None:
    """Clear the stored submit_result payload."""

    _local.submitted_result = None


@tool
def submit_result(result_json: str, summary: str) -> str:
    """Submit the final result of this agent and signal completion."""

    try:
        payload = json.loads(result_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"[submit_result] Error: invalid JSON — {exc}. Fix your JSON and try again."

    _local.submitted_result = {"payload": payload, "summary": summary}
    logger.info("submit_result: %s", summary)
    return f"SUBMITTED. Summary: {summary}"


@tool
def update_plan(
    status: str,
    progress_pct: int,
    current_step: str,
    notes: str = "",
) -> str:
    """Report incremental progress without terminating the agent."""

    logger.info(
        "update_plan: [%d%%] %s — %s %s",
        progress_pct,
        status,
        current_step,
        notes,
    )
    return f"Plan updated: {status} ({progress_pct}%) — {current_step}"

"""Orchestration tools for ReAct agent control flow."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
_local = threading.local()
_subagent_executor: Any = None


class ClarificationOption(BaseModel):
    """A selectable option presented to the human reviewer."""

    label: str = Field(description="Short label for the option")
    description: str = Field(description="Longer explanation of what this option entails")


class AskClarificationInput(BaseModel):
    """Input schema for structured clarification requests."""

    question: str = Field(description="The question to ask the human reviewer")
    clarification_type: Literal[
        "missing_info",
        "ambiguous_requirement",
        "approach_choice",
        "risk_confirmation",
        "suggestion",
    ] = Field(description="Category of clarification needed")
    context: str | None = Field(
        default=None,
        description="Background context to help the reviewer understand the question",
    )
    options: list[ClarificationOption] | None = Field(
        default=None,
        description="Selectable options; omit for free-text response",
    )


class SpawnSubagentInput(BaseModel):
    """Input schema for the sub-agent spawn tool."""

    message: str = Field(description="Task description for the sub-agent")
    agent_type: Literal["research", "writing", "bash"] = Field(
        description="Type of sub-agent to spawn"
    )
    wait: bool = Field(
        default=True,
        description="If True, wait for completion. If False, return task_id immediately.",
    )


def get_submitted_result() -> dict[str, Any] | None:
    """Retrieve the most recent submit_result payload."""

    return getattr(_local, "submitted_result", None)


def clear_submitted_result() -> None:
    """Clear the stored submit_result payload."""

    _local.submitted_result = None


def set_subagent_executor(executor: Any) -> None:
    """Inject the executor used by ``spawn_subagent``."""

    global _subagent_executor
    _subagent_executor = executor


def get_subagent_executor() -> Any:
    """Return the configured sub-agent executor, if any."""

    return _subagent_executor


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


@tool(args_schema=AskClarificationInput)
def ask_clarification(
    question: str,
    clarification_type: str,
    context: str | None = None,
    options: list[dict[str, str]] | None = None,
) -> str:
    """Ask the human reviewer for structured clarification."""

    del context, options
    return (
        f"[CLARIFICATION PENDING] {clarification_type}: {question} "
        "(This response means the middleware did not intercept the call.)"
    )


def _get_builtin_registry() -> dict[str, Any]:
    """Return built-in sub-agent factories."""

    try:
        from muse.agents.builtins import BUILTIN_AGENT_FACTORIES
    except ImportError:
        return {}
    return BUILTIN_AGENT_FACTORIES


@tool(args_schema=SpawnSubagentInput)
def spawn_subagent(
    message: str,
    agent_type: str,
    wait: bool = True,
) -> str:
    """Spawn a specialized sub-agent for an independent subtask."""

    executor = get_subagent_executor()
    if executor is None:
        return "[SUBAGENT ERROR] No SubagentExecutor configured. Cannot spawn sub-agent."

    builtin_registry = _get_builtin_registry()
    agent_factory = builtin_registry.get(agent_type)
    if agent_factory is None:
        return f"[SUBAGENT ERROR] Unknown agent type: {agent_type}"

    task_fn = agent_factory(message)
    task_id = executor.submit(agent_fn=task_fn)

    if not wait:
        return f"Sub-agent spawned: task_id={task_id}, type={agent_type}"

    result = executor.get_result(task_id)
    if result is None:
        return f"[SUBAGENT ERROR] No result for task {task_id}"
    return result.summary()

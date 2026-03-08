"""Built-in sub-agent type configurations and factories."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

from muse.agents.result import SubagentResult
from muse.sandbox.local import LocalSandbox
from muse.sandbox.tools import shell as shell_tool
from muse.tools._context import get_services


_RESEARCH_SYSTEM = """\
You are a research sub-agent. Your task is to find and analyze academic literature
relevant to the given query. Use available search tools to find papers, read PDFs,
and synthesize findings. Return key findings and citation information.
"""

_WRITING_SYSTEM = """\
You are a writing sub-agent. Your task is to draft, revise, or edit academic text
for a thesis section. Use available writing and review tools and return a concise result.
"""

_BASH_SYSTEM = """\
You are a command execution sub-agent. Your task is to run shell commands or other
execution-oriented subtasks and report results and files created.
"""

AGENT_TOOL_PROFILES: dict[str, list[str]] = {
    "research": ["research", "file_read", "rag"],
    "writing": ["writing", "review_self", "file"],
    "bash": ["sandbox", "file"],
}

AGENT_MAX_TURNS: dict[str, int] = {
    "research": 15,
    "writing": 25,
    "bash": 15,
}

AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "research": _RESEARCH_SYSTEM,
    "writing": _WRITING_SYSTEM,
    "bash": _BASH_SYSTEM,
}

BLOCKED_TOOLS: set[str] = {"spawn_subagent", "ask_clarification"}


def _runs_dir_for(services) -> Path:
    settings = getattr(services, "settings", None)
    runs_dir = getattr(settings, "runs_dir", "runs")
    return Path(runs_dir)


def _coerce_results(results) -> list[dict]:
    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]
    return []


def build_research_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for research sub-agent tasks."""

    services = get_services()
    search_client = getattr(services, "search", None)

    def run() -> SubagentResult:
        issues: list[str] = []
        results: list[dict] = []
        if search_client is None:
            issues.append("No search client configured for research sub-agent")
        else:
            try:
                try:
                    found, _queries = search_client.search_multi_source(topic=message, discipline="")
                except TypeError:
                    found = search_client.search_multi_source(message)
                results = _coerce_results(found)
            except Exception as exc:  # noqa: BLE001
                issues.append(f"Research search failed: {exc}")

        citations = [
            {
                "ref_id": item.get("ref_id", ""),
                "title": item.get("title", ""),
                "year": item.get("year"),
            }
            for item in results[:5]
        ]
        key_findings = [str(item.get("title", "")).strip() for item in results[:3] if item.get("title")]
        return SubagentResult(
            status="completed",
            accomplishments=[f"research agent processed: {message[:100]}"],
            key_findings=key_findings,
            files_created=[],
            issues=issues,
            citations=citations,
        )

    return run


def build_writing_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for writing sub-agent tasks."""

    services = get_services()
    llm = getattr(services, "llm", None)

    def run() -> SubagentResult:
        issues: list[str] = []
        draft = ""
        if llm is None or not hasattr(llm, "text"):
            issues.append("No LLM configured for writing sub-agent")
        else:
            try:
                draft = llm.text(
                    system=_WRITING_SYSTEM,
                    user=message,
                    route="writing",
                    max_tokens=1200,
                )
            except Exception as exc:  # noqa: BLE001
                issues.append(f"Writing generation failed: {exc}")

        return SubagentResult(
            status="completed",
            accomplishments=[f"writing agent processed: {message[:100]}"],
            key_findings=[draft] if draft else [],
            files_created=[],
            issues=issues,
            citations=[],
        )

    return run


def build_bash_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for bash sub-agent tasks."""

    services = get_services()
    workspace = _runs_dir_for(services) / "_subagents" / "bash"

    def run() -> SubagentResult:
        sandbox = getattr(services, "sandbox", None)
        if sandbox is None:
            sandbox = LocalSandbox(workspace)

        summary = asyncio.run(shell_tool(sandbox, message, timeout=60))
        lowered = summary.lower()
        status = "completed"
        issues: list[str] = []
        if lowered.startswith("[timed out]"):
            status = "timed_out"
            issues.append(summary)
        elif lowered.startswith("[failed]"):
            status = "failed"
            issues.append(summary)

        return SubagentResult(
            status=status,
            accomplishments=[f"bash agent processed: {message[:100]}"],
            key_findings=[summary],
            files_created=[],
            issues=issues,
            citations=[],
        )

    return run


BUILTIN_AGENT_FACTORIES: dict[str, Callable[[str], Callable[[], SubagentResult]]] = {
    "research": build_research_agent,
    "writing": build_writing_agent,
    "bash": build_bash_agent,
}

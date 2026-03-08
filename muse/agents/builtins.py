"""Built-in sub-agent type configurations and factories."""

from __future__ import annotations

from typing import Callable

from muse.agents.result import SubagentResult


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


def _make_stub_agent_fn(agent_type: str, message: str) -> Callable[[], SubagentResult]:
    """Create a stub agent function until full agent execution is wired."""

    def run() -> SubagentResult:
        return SubagentResult(
            status="completed",
            accomplishments=[f"[stub] {agent_type} agent processed: {message[:100]}"],
            key_findings=[],
            files_created=[],
            issues=["Running in stub mode -- direct sub-agent execution not wired yet"],
            citations=[],
        )

    return run


def build_research_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for research sub-agent tasks."""

    return _make_stub_agent_fn("research", message)


def build_writing_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for writing sub-agent tasks."""

    return _make_stub_agent_fn("writing", message)


def build_bash_agent(message: str) -> Callable[[], SubagentResult]:
    """Factory for bash sub-agent tasks."""

    return _make_stub_agent_fn("bash", message)


BUILTIN_AGENT_FACTORIES: dict[str, Callable[[str], Callable[[], SubagentResult]]] = {
    "research": build_research_agent,
    "writing": build_writing_agent,
    "bash": build_bash_agent,
}

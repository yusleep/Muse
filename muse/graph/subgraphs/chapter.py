"""Chapter-level LangGraph subgraph with ReAct dual-mode support."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from muse.graph.helpers.review_state import should_iterate
from muse.graph.nodes.draft import build_chapter_draft_node
from muse.graph.nodes.review import build_chapter_review_node
from muse.models.adapter import MuseChatModel
from muse.services.providers import LLMClient


class ChapterState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    chapter_plan: dict[str, Any]
    references: list[dict[str, Any]]
    topic: str
    discipline: str
    language: str
    subtask_results: list[dict[str, Any]]
    merged_text: str
    quality_scores: dict[str, int]
    review_notes: list[dict[str, Any]]
    revision_instructions: dict[str, str]
    iteration: int
    max_iterations: int
    citation_uses: list[dict[str, Any]]
    claim_text_by_id: dict[str, str]


def _chapter_route(state: ChapterState) -> Literal["revise", "done"]:
    route = should_iterate(
        {
            "quality_scores": state.get("quality_scores", {}),
            "current_iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 3),
        },
        threshold=4,
    )
    return "revise" if route == "revise" else "done"


def _chapter_revise(_: ChapterState) -> dict[str, Any]:
    return {}


def build_chapter_graph(*, services: Any):
    """Build the fixed-flow chapter graph used as the safe fallback path."""

    builder = StateGraph(ChapterState)
    builder.add_node("chapter_draft", build_chapter_draft_node(services))
    builder.add_node("chapter_review", build_chapter_review_node(services))
    builder.add_node("chapter_revise", _chapter_revise)
    builder.add_edge(START, "chapter_draft")
    builder.add_edge("chapter_draft", "chapter_review")
    builder.add_conditional_edges(
        "chapter_review",
        _chapter_route,
        {"revise": "chapter_revise", "done": END},
    )
    builder.add_edge("chapter_revise", "chapter_draft")
    return builder.compile()


def _references_summary(references: list[dict[str, Any]]) -> str:
    if not references:
        return "0 references available."
    top_refs = ", ".join(
        str(reference.get("ref_id", "?"))
        for reference in references[:10]
        if isinstance(reference, dict)
    )
    return f"{len(references)} references available. Top refs: {top_refs}"


def _extract_chapter_result(
    result: dict[str, Any],
    chapter_plan: dict[str, Any],
) -> dict[str, Any]:
    chapter_id = chapter_plan.get("chapter_id", "chapter")
    iterations_used = result.get("iterations_used", result.get("iteration", 0))
    chapter_result = {
        "chapter_id": chapter_id,
        "chapter_title": chapter_plan.get("chapter_title", chapter_id),
        "merged_text": result.get("merged_text", ""),
        "quality_scores": result.get("quality_scores", {}),
        "iterations_used": iterations_used,
        "subtask_results": result.get("subtask_results", []),
        "citation_uses": result.get("citation_uses", []),
        "claim_text_by_id": result.get("claim_text_by_id", {}),
    }
    return {
        "chapters": {chapter_id: chapter_result},
        "claim_text_by_id": result.get("claim_text_by_id", {}),
    }


def _create_react_model(*, services: Any = None, settings: Any = None):
    llm_client = getattr(services, "llm", None) if services is not None else None
    if isinstance(llm_client, LLMClient):
        return MuseChatModel(llm_client=llm_client, route="writing")
    if isinstance(llm_client, BaseChatModel):
        return llm_client
    if llm_client is not None:
        return None

    if settings is None:
        return None

    try:
        from muse.models.factory import create_chat_model
    except ImportError:
        return None

    try:
        return create_chat_model(settings, route="writing")
    except Exception:
        return None


def _build_react_chapter_agent(*, services: Any, settings: Any = None):
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware.types import ModelRequest, dynamic_prompt
    except ImportError:
        return None

    from muse.prompts.chapter_agent import chapter_agent_system_prompt
    from muse.tools.file import edit_file, glob_files, grep, read_file, write_file
    from muse.tools.orchestration import submit_result, update_plan
    from muse.tools.research import (
        academic_search,
        image_search,
        read_pdf,
        retrieve_local_refs,
        web_fetch,
        web_search,
    )
    from muse.tools.review import self_review
    from muse.tools.writing import apply_patch, revise_section, write_section

    tools = [
        write_section,
        revise_section,
        apply_patch,
        self_review,
        academic_search,
        retrieve_local_refs,
        web_search,
        web_fetch,
        read_pdf,
        image_search,
        read_file,
        write_file,
        edit_file,
        glob_files,
        grep,
        submit_result,
        update_plan,
    ]

    model = _create_react_model(settings=settings, services=services)
    if model is None:
        return None

    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        state = request.state
        chapter_plan = state.get("chapter_plan", {})
        return chapter_agent_system_prompt(
            topic=str(state.get("topic", "")),
            language=str(state.get("language", "zh")),
            chapter_title=str(chapter_plan.get("chapter_title", "")),
            chapter_plan=chapter_plan,
            references_summary=_references_summary(state.get("references", [])),
        )

    return create_agent(
        model=model,
        tools=tools,
        middleware=[prompt],
        state_schema=ChapterState,
        name="chapter_react_agent",
    )


def build_chapter_subgraph_node(*, services: Any, settings: Any = None):
    react_agent = _build_react_chapter_agent(services=services, settings=settings)
    fallback_graph = build_chapter_graph(services=services)

    def _fallback(state: dict[str, Any]) -> dict[str, Any]:
        fallback_result = fallback_graph.invoke(state)
        return _extract_chapter_result(fallback_result, state.get("chapter_plan", {}))

    if react_agent is None:
        return _fallback

    def run_react_chapter(state: dict[str, Any]) -> dict[str, Any]:
        from muse.tools._context import clear_state, get_state, set_services, set_state
        from muse.tools.orchestration import (
            clear_submitted_result,
            get_subagent_executor,
            get_submitted_result,
            set_subagent_executor,
        )

        set_services(services)
        previous_state = get_state(default=None)
        clear_submitted_result()
        previous_executor = get_subagent_executor()
        set_subagent_executor(getattr(services, "subagent_executor", None))

        agent_input = dict(state)
        set_state(agent_input)
        agent_input.setdefault(
            "messages",
            [
                {
                    "role": "user",
                    "content": "Draft, review, and submit this thesis chapter.",
                }
            ],
        )

        try:
            react_agent.invoke(
                agent_input,
                {"recursion_limit": 60},
                context={"services": services},
            )
        except Exception:
            clear_submitted_result()
            set_subagent_executor(previous_executor)
            if previous_state is None:
                clear_state()
            else:
                set_state(previous_state)
            return _fallback(state)

        submitted = get_submitted_result()
        clear_submitted_result()
        set_subagent_executor(previous_executor)
        if previous_state is None:
            clear_state()
        else:
            set_state(previous_state)
        if not submitted:
            return _fallback(state)

        payload = submitted.get("payload", {})
        if not isinstance(payload, dict):
            return _fallback(state)
        return _extract_chapter_result(payload, state.get("chapter_plan", {}))

    return run_react_chapter

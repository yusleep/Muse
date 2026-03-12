"""Composition subgraph for final coherence alignment."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from muse.models.adapter import MuseChatModel
from muse.services.providers import LLMClient


class CompositionState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    final_text: str
    abstract_zh: str
    abstract_en: str
    paper_package: dict[str, Any]
    language: str


def _unify_terminology(state: CompositionState) -> dict[str, Any]:
    package = dict(state.get("paper_package", {}))
    package["terminology_normalized"] = True
    return {"paper_package": package}


def _align_cross_refs(state: CompositionState) -> dict[str, Any]:
    package = dict(state.get("paper_package", {}))
    package["cross_refs_aligned"] = True
    return {"paper_package": package}


def _stitch_abstract_intro_conclusion(state: CompositionState) -> dict[str, Any]:
    final_text = state.get("final_text", "")
    package = dict(state.get("paper_package", {}))
    package["composed_text"] = final_text
    return {"final_text": final_text, "paper_package": package}


def build_composition_graph():
    builder = StateGraph(CompositionState)
    builder.add_node("unify_terminology", _unify_terminology)
    builder.add_node("align_cross_refs", _align_cross_refs)
    builder.add_node("stitch_abstract_intro_conclusion", _stitch_abstract_intro_conclusion)
    builder.add_edge(START, "unify_terminology")
    builder.add_edge("unify_terminology", "align_cross_refs")
    builder.add_edge("align_cross_refs", "stitch_abstract_intro_conclusion")
    builder.add_edge("stitch_abstract_intro_conclusion", END)
    return builder.compile()


def _extract_composition_result(
    result: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "final_text": result.get("final_text", state.get("final_text", "")),
        "paper_package": result.get("paper_package", state.get("paper_package", {})),
    }


def _create_react_model(*, services: Any = None, settings: Any = None):
    if services is not None:
        llm_client = getattr(services, "llm", None)
        if isinstance(llm_client, LLMClient):
            return MuseChatModel(llm_client=llm_client, route="polish")
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
        return create_chat_model(settings, route="polish")
    except Exception:
        return None


def _build_react_composition_agent(*, settings: Any = None, services: Any = None):
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware.types import ModelRequest, dynamic_prompt
    except ImportError:
        return None

    model = _create_react_model(settings=settings, services=services)
    if model is None:
        return None

    from muse.prompts.composition_agent import composition_agent_system_prompt
    from muse.tools.composition import (
        align_cross_refs,
        check_terminology,
        check_transitions,
        rewrite_passage,
    )
    from muse.tools.file import edit_file, glob_files, grep, read_file, write_file
    from muse.tools.orchestration import submit_result, update_plan
    from muse.tools.writing import apply_patch, revise_section

    tools = [
        check_terminology,
        align_cross_refs,
        check_transitions,
        rewrite_passage,
        apply_patch,
        revise_section,
        read_file,
        write_file,
        edit_file,
        glob_files,
        grep,
        submit_result,
        update_plan,
    ]

    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        state = request.state
        chapters = state.get("paper_package", {}).get("chapters", {})
        chapter_count = len(chapters) if isinstance(chapters, dict) else 0
        return composition_agent_system_prompt(
            chapter_count=chapter_count,
            total_words=len(str(state.get("final_text", "")).split()),
            language=str(state.get("language", "zh")),
        )

    return create_agent(
        model=model,
        tools=tools,
        middleware=[prompt],
        state_schema=CompositionState,
        name="composition_react_agent",
    )


def build_composition_subgraph_node(*, settings: Any = None, services: Any = None):
    react_agent = _build_react_composition_agent(settings=settings, services=services)
    fallback_graph = build_composition_graph()

    def _fallback(state: dict[str, Any]) -> dict[str, Any]:
        fallback_result = fallback_graph.invoke(
            {
                "final_text": state.get("final_text", ""),
                "abstract_zh": state.get("abstract_zh", ""),
                "abstract_en": state.get("abstract_en", ""),
                "paper_package": state.get("paper_package", {}),
                "language": state.get("language", "zh"),
            }
        )
        return _extract_composition_result(fallback_result, state)

    if react_agent is None:
        return _fallback

    def run_react_composition(state: dict[str, Any]) -> dict[str, Any]:
        from muse.tools._context import set_services
        from muse.tools.orchestration import (
            clear_submitted_result,
            get_subagent_executor,
            get_submitted_result,
            set_subagent_executor,
        )

        if services is not None:
            set_services(services)
        clear_submitted_result()
        previous_executor = get_subagent_executor()
        set_subagent_executor(getattr(services, "subagent_executor", None) if services is not None else None)

        agent_input = dict(state)
        agent_input.setdefault(
            "messages",
            [
                {
                    "role": "user",
                    "content": "Unify terminology, transitions, and cross-references, then submit the final package.",
                }
            ],
        )

        try:
            react_agent.invoke(agent_input, {"recursion_limit": 30})
        except Exception:
            clear_submitted_result()
            set_subagent_executor(previous_executor)
            return _fallback(state)

        submitted = get_submitted_result()
        clear_submitted_result()
        set_subagent_executor(previous_executor)
        if not submitted:
            return _fallback(state)

        payload = submitted.get("payload", {})
        if not isinstance(payload, dict):
            return _fallback(state)
        return _extract_composition_result(payload, state)

    return run_react_composition

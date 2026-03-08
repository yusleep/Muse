"""Chapter-level LangGraph subgraph with revise loop."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from muse.graph.helpers.review_state import should_iterate
from muse.graph.nodes.draft import build_chapter_draft_node
from muse.graph.nodes.review import build_chapter_review_node


class ChapterState(TypedDict, total=False):
    chapter_plan: dict[str, Any]
    references: list[dict[str, Any]]
    topic: str
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


def build_chapter_subgraph_node(*, services: Any):
    chapter_graph = build_chapter_graph(services=services)

    def run_chapter_subgraph(state: dict[str, Any]) -> dict[str, Any]:
        result = chapter_graph.invoke(state)
        chapter_plan = result.get("chapter_plan", {})
        chapter_id = chapter_plan.get("chapter_id", "chapter")
        chapter_result = {
            "chapter_id": chapter_id,
            "chapter_title": chapter_plan.get("chapter_title", chapter_id),
            "merged_text": result.get("merged_text", ""),
            "quality_scores": result.get("quality_scores", {}),
            "iterations_used": result.get("iteration", 0),
            "subtask_results": result.get("subtask_results", []),
            "citation_uses": result.get("citation_uses", []),
            "claim_text_by_id": result.get("claim_text_by_id", {}),
        }
        return {
            "chapters": {chapter_id: chapter_result},
            "claim_text_by_id": result.get("claim_text_by_id", {}),
        }

    return run_chapter_subgraph

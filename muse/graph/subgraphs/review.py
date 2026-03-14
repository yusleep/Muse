"""Layered global review subgraph for merged-thesis revision."""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from muse.graph.nodes.review import (
    build_global_review_node,
    build_global_revise_node,
    build_layered_review_node,
)


class GlobalReviewState(TypedDict, total=False):
    final_text: str
    quality_scores: dict[str, int]
    review_notes: list[dict[str, Any]]
    review_history: Annotated[list[dict[str, Any]], operator.add]
    review_iteration: int
    structural_iterations: int
    content_iterations: int
    line_iterations: int
    review_layer: str


_LAYER_CONFIG = {
    "structural": {
        "threshold": 3,
        "max_iterations": 2,
        "score_keys": ("logic", "structure", "balance"),
    },
    "content": {
        "threshold": 3,
        "max_iterations": 2,
        "score_keys": ("citation", "coverage", "depth"),
    },
    "line": {
        "threshold": 4,
        "max_iterations": 1,
        "score_keys": ("style", "term_consistency", "redundancy"),
    },
}
_LAYER_REVIEW_ROUTES = {
    "structural": "review_structural",
    "content": "review",
    "line": "review_line",
}


def _layer_route(layer: str):
    config = _LAYER_CONFIG[layer]
    iteration_key = f"{layer}_iterations"

    def route(state: GlobalReviewState) -> str:
        scores = state.get("quality_scores", {})
        relevant_scores = [
            int(scores[key])
            for key in config["score_keys"]
            if isinstance(scores, dict) and isinstance(scores.get(key), (int, float))
        ]
        min_score = min(relevant_scores) if relevant_scores else 0
        iterations = state.get(iteration_key, 0)
        try:
            layer_iterations = max(int(iterations), 0)
        except (TypeError, ValueError):
            layer_iterations = 0

        if min_score >= config["threshold"]:
            return "done" if layer == "line" else "next_layer"
        if layer_iterations >= config["max_iterations"]:
            return "done" if layer == "line" else "next_layer"
        return "revise"

    return route


def build_global_review_graph(*, services: Any):
    builder = StateGraph(GlobalReviewState)
    builder.add_node(
        "structural_review",
        build_layered_review_node(
            services,
            layer="structural",
            route=_LAYER_REVIEW_ROUTES["structural"],
        ),
    )
    builder.add_node(
        "structural_revise",
        build_global_revise_node(services, layer="structural", route="writing_revision"),
    )
    builder.add_node(
        "content_review",
        build_layered_review_node(
            services,
            layer="content",
            route=_LAYER_REVIEW_ROUTES["content"],
        ),
    )
    builder.add_node(
        "content_revise",
        build_global_revise_node(services, layer="content", route="writing_revision"),
    )
    builder.add_node(
        "line_review",
        build_layered_review_node(
            services,
            layer="line",
            route=_LAYER_REVIEW_ROUTES["line"],
        ),
    )
    builder.add_node(
        "line_revise",
        build_global_revise_node(services, layer="line", route="writing_revision"),
    )

    builder.add_edge(START, "structural_review")
    builder.add_conditional_edges(
        "structural_review",
        _layer_route("structural"),
        {"revise": "structural_revise", "next_layer": "content_review"},
    )
    builder.add_edge("structural_revise", "structural_review")

    builder.add_conditional_edges(
        "content_review",
        _layer_route("content"),
        {"revise": "content_revise", "next_layer": "line_review"},
    )
    builder.add_edge("content_revise", "content_review")

    builder.add_conditional_edges(
        "line_review",
        _layer_route("line"),
        {"revise": "line_revise", "done": END},
    )
    builder.add_edge("line_revise", "line_review")
    return builder.compile()


def build_global_review_subgraph_node(*, settings: Any = None, services: Any = None):
    review_mode = str(getattr(settings, "review_mode", "classic") or "classic").strip().lower()
    if review_mode == "persona":
        return build_global_review_node(services, mode="persona")
    if review_mode != "layered":
        return lambda state: {}

    graph = build_global_review_graph(services=services)

    def run_layered_review(state: dict[str, Any]) -> dict[str, Any]:
        initial_state: GlobalReviewState = {
            "final_text": str(state.get("final_text", "") or ""),
            "quality_scores": state.get("quality_scores", {}) if isinstance(state.get("quality_scores"), dict) else {},
            "review_notes": state.get("review_notes", []) if isinstance(state.get("review_notes"), list) else [],
            "review_history": state.get("review_history", []) if isinstance(state.get("review_history"), list) else [],
            "review_iteration": int(state.get("review_iteration", 1) or 1),
            "structural_iterations": int(state.get("structural_iterations", 0) or 0),
            "content_iterations": int(state.get("content_iterations", 0) or 0),
            "line_iterations": int(state.get("line_iterations", 0) or 0),
            "review_layer": str(state.get("review_layer", "") or ""),
        }
        result = graph.invoke(initial_state)
        return {
            "final_text": result.get("final_text", initial_state["final_text"]),
            "quality_scores": result.get("quality_scores", initial_state["quality_scores"]),
            "review_notes": result.get("review_notes", initial_state["review_notes"]),
            "review_history": result.get("review_history", initial_state["review_history"]),
            "review_iteration": result.get("review_iteration", initial_state["review_iteration"]),
            "structural_iterations": result.get(
                "structural_iterations",
                initial_state["structural_iterations"],
            ),
            "content_iterations": result.get("content_iterations", initial_state["content_iterations"]),
            "line_iterations": result.get("line_iterations", initial_state["line_iterations"]),
            "review_layer": result.get("review_layer", initial_state["review_layer"]),
        }

    return run_layered_review

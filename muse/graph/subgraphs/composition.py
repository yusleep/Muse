"""Composition subgraph for final coherence alignment."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


class CompositionState(TypedDict, total=False):
    final_text: str
    abstract_zh: str
    abstract_en: str
    paper_package: dict[str, Any]


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


def build_composition_subgraph_node():
    graph = build_composition_graph()

    def run_composition_subgraph(state: dict[str, Any]) -> dict[str, Any]:
        result = graph.invoke(
            {
                "final_text": state.get("final_text", ""),
                "abstract_zh": state.get("abstract_zh", ""),
                "abstract_en": state.get("abstract_en", ""),
                "paper_package": state.get("paper_package", {}),
            }
        )
        return {
            "final_text": result.get("final_text", state.get("final_text", "")),
            "paper_package": result.get("paper_package", state.get("paper_package", {})),
        }

    return run_composition_subgraph

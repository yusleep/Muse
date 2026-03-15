"""Top-level Muse LangGraph definition."""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("muse.graph")

from langgraph.graph import END, START, StateGraph

from muse.config import Settings
from muse.graph.nodes import (
    build_citation_repair_node,
    build_export_node,
    build_initialize_node,
    build_interrupt_node,
    build_merge_chapters_node,
    build_outline_node,
    build_perspective_node,
    build_polish_node,
    build_coherence_check_node,
    build_ref_analysis_node,
    build_search_node,
    build_single_pass_node,
)
from muse.graph.nodes.draft import (
    build_prepare_next_chapter_node,
    build_update_cross_chapter_state_node,
    next_chapter_route,
)
from muse.graph.state import MuseState
from muse.graph.subgraphs.chapter import build_chapter_subgraph_node
from muse.graph.subgraphs.citation import build_citation_subgraph_node
from muse.graph.subgraphs.composition import build_composition_subgraph_node
from muse.graph.subgraphs.review import build_global_review_subgraph_node
from muse.middlewares import build_default_chain


def _citation_quality_route(state: dict[str, Any]) -> str:
    flagged = state.get("flagged_citations", [])
    verified = state.get("verified_citations", [])
    if not isinstance(flagged, list):
        flagged = []
    if not isinstance(verified, list):
        verified = []

    total = len(flagged) + len(verified)
    if total == 0:
        return "polish"

    flagged_ratio = len(flagged) / total
    if flagged_ratio > 0.2 and not bool(state.get("citation_repair_attempted", False)):
        return "citation_repair"
    return "polish"


class _NullServices:
    def __init__(self) -> None:
        self.local_refs = []
        self.rag_index = None
        self.search = None
        self.llm = None


def _default_settings() -> Settings:
    return Settings(
        llm_api_key="",
        llm_base_url="https://api.openai.com/v1",
        llm_model="router/default",
        model_router_config={},
        runs_dir="runs",
        semantic_scholar_api_key=None,
        openalex_email=None,
        crossref_mailto=None,
        refs_dir=None,
        checkpoint_dir=None,
    )


def _writing_mode_route(settings: Settings):
    writing_mode = str(getattr(settings, "writing_mode", "sequential") or "sequential").strip().lower()

    def route(state: dict[str, Any]) -> str:
        del state
        if writing_mode == "single_pass":
            return "single_pass_writer"
        return "prepare_next_chapter"

    return route


def _wrap(node_fn, node_name: str, settings: Settings, services: Any):
    """Wrap a node function with the default middleware chain."""

    log_dir = getattr(settings, "runs_dir", None)
    subagent_executor = getattr(services, "subagent_executor", None)
    chain = build_default_chain(
        log_dir=log_dir,
        node_name=node_name,
        llm=getattr(services, "llm", None),
        max_retries=getattr(settings, "middleware_retry_max", 2),
        retry_base_delay=getattr(settings, "middleware_retry_delay", 5.0),
        context_window=getattr(settings, "middleware_context_window", 128_000),
        compaction_threshold=getattr(settings, "middleware_compaction_threshold", 0.9),
        compaction_recent_tokens=getattr(
            settings, "middleware_compaction_recent_tokens", 20_000
        ),
        memory_store=getattr(services, "memory_store", None),
        subagent_max_concurrent=getattr(subagent_executor, "max_concurrent", None),
    )
    inner = chain.wrap(node_fn)

    def _logged_node(state):
        _log.info("[%s] enter", node_name)
        result = inner(state)
        _log.info("[%s] exit", node_name)
        return result

    return _logged_node


def build_graph(
    settings: Settings | None = None,
    *,
    services: Any | None = None,
    checkpointer: Any | None = None,
    auto_approve: bool = True,
):
    settings = settings or _default_settings()
    services = services or _NullServices()
    review_mode = str(getattr(settings, "review_mode", "classic") or "classic").strip().lower()

    builder = StateGraph(MuseState)
    builder.add_node(
        "initialize",
        _wrap(build_initialize_node(settings, services), "initialize", settings, services),
    )
    builder.add_node(
        "search",
        _wrap(build_search_node(settings, services), "search", settings, services),
    )
    builder.add_node("review_refs", build_interrupt_node("research", auto_approve=auto_approve))
    builder.add_node(
        "perspective_discovery",
        _wrap(
            build_perspective_node(services=services),
            "perspective_discovery",
            settings,
            services,
        ),
    )
    builder.add_node(
        "search_perspectives",
        _wrap(
            build_search_node(settings, services, state_query_key="perspective_queries"),
            "search_perspectives",
            settings,
            services,
        ),
    )
    builder.add_node(
        "outline",
        _wrap(build_outline_node(settings, services), "outline", settings, services),
    )
    builder.add_node("approve_outline", build_interrupt_node("outline", auto_approve=auto_approve))
    builder.add_node(
        "ref_analysis",
        _wrap(
            build_ref_analysis_node(services=services),
            "ref_analysis",
            settings,
            services,
        ),
    )
    builder.add_node(
        "single_pass_writer",
        _wrap(
            build_single_pass_node(settings=settings, services=services),
            "single_pass_writer",
            settings,
            services,
        ),
    )
    builder.add_node(
        "chapter_subgraph",
        _wrap(
            build_chapter_subgraph_node(services=services, settings=settings),
            "chapter_subgraph",
            settings,
            services,
        ),
    )
    builder.add_node(
        "prepare_next_chapter",
        _wrap(
            build_prepare_next_chapter_node(),
            "prepare_next_chapter",
            settings,
            services,
        ),
    )
    builder.add_node(
        "update_cross_chapter_state",
        _wrap(
            build_update_cross_chapter_state_node(),
            "update_cross_chapter_state",
            settings,
            services,
        ),
    )
    builder.add_node(
        "merge_chapters",
        _wrap(
            build_merge_chapters_node(settings, services),
            "merge_chapters",
            settings,
            services,
        ),
    )
    builder.add_node(
        "coherence_check",
        _wrap(
            build_coherence_check_node(services=services),
            "coherence_check",
            settings,
            services,
        ),
    )
    if review_mode != "classic":
        builder.add_node(
            "global_review",
            _wrap(
                build_global_review_subgraph_node(settings=settings, services=services),
                "global_review",
                settings,
                services,
            ),
        )
    builder.add_node(
        "citation_subgraph",
        _wrap(
            build_citation_subgraph_node(services=services, settings=settings),
            "citation_subgraph",
            settings,
            services,
        ),
    )
    builder.add_node(
        "citation_repair",
        _wrap(
            build_citation_repair_node(),
            "citation_repair",
            settings,
            services,
        ),
    )
    builder.add_node(
        "polish",
        _wrap(build_polish_node(services), "polish", settings, services),
    )
    builder.add_node(
        "composition_subgraph",
        _wrap(
            build_composition_subgraph_node(settings=settings, services=services),
            "composition_subgraph",
            settings,
            services,
        ),
    )
    builder.add_node("approve_final", build_interrupt_node("final", auto_approve=auto_approve))
    builder.add_node(
        "export",
        _wrap(build_export_node(settings, services=services), "export", settings, services),
    )
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "search")
    builder.add_edge("search", "review_refs")
    builder.add_edge("review_refs", "perspective_discovery")
    builder.add_edge("perspective_discovery", "search_perspectives")
    builder.add_edge("search_perspectives", "outline")
    builder.add_edge("outline", "approve_outline")
    builder.add_edge("approve_outline", "ref_analysis")
    builder.add_conditional_edges(
        "ref_analysis",
        _writing_mode_route(settings),
        {
            "prepare_next_chapter": "prepare_next_chapter",
            "single_pass_writer": "single_pass_writer",
        },
    )
    builder.add_conditional_edges(
        "prepare_next_chapter",
        next_chapter_route,
        {"chapter_subgraph": "chapter_subgraph", "merge_chapters": "merge_chapters"},
    )
    builder.add_edge("chapter_subgraph", "update_cross_chapter_state")
    builder.add_edge("update_cross_chapter_state", "prepare_next_chapter")
    builder.add_edge("single_pass_writer", "merge_chapters")
    if review_mode == "classic":
        builder.add_edge("merge_chapters", "coherence_check")
        builder.add_edge("coherence_check", "citation_subgraph")
    else:
        builder.add_edge("merge_chapters", "coherence_check")
        builder.add_edge("coherence_check", "global_review")
        builder.add_edge("global_review", "citation_subgraph")
    builder.add_conditional_edges(
        "citation_subgraph",
        _citation_quality_route,
        {"citation_repair": "citation_repair", "polish": "polish"},
    )
    builder.add_edge("citation_repair", "citation_subgraph")
    builder.add_edge("polish", "composition_subgraph")
    builder.add_edge("composition_subgraph", "approve_final")
    builder.add_edge("approve_final", "export")
    builder.add_edge("export", END)
    return builder.compile(checkpointer=checkpointer)

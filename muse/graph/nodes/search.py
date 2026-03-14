"""Search node for literature discovery."""

from __future__ import annotations

import logging
from typing import Any

from muse.prompts.search_queries import search_queries_prompt

_log = logging.getLogger("muse.search")

_ONLINE_QUERY_BUDGET = 4
_LOCAL_REFS_QUERY_BUDGET = 2


def _generate_search_queries(
    llm_client: Any,
    topic: str,
    discipline: str,
    count: int = 7,
) -> list[str]:
    system, user = search_queries_prompt(topic, discipline, count)
    try:
        payload = llm_client.structured(system=system, user=user, route="outline", max_tokens=500)
        queries = payload.get("queries", []) if isinstance(payload, dict) else []
        return [str(query).strip() for query in queries if str(query).strip()][:count]
    except Exception:
        return []


def _search_query_budget(*, local_refs_count: int) -> int:
    return _LOCAL_REFS_QUERY_BUDGET if local_refs_count > 0 else _ONLINE_QUERY_BUDGET


def _summarize_references(references: list[dict[str, Any]]) -> str:
    if not references:
        return "No references found."
    top = references[:12]
    return "\n".join(
        f"- {ref.get('title', 'Untitled')} ({ref.get('year', 'n.d.')}) {ref.get('venue', '')}".strip()
        for ref in top
    )


def build_search_node(settings: Any, services: Any):
    def search(state: dict[str, Any]) -> dict[str, Any]:
        topic = state.get("topic", "")
        discipline = state.get("discipline", "")
        llm = getattr(services, "llm", None)
        search_client = getattr(services, "search", None)
        local_refs = list(getattr(services, "local_refs", []) or [])
        query_budget = _search_query_budget(local_refs_count=len(local_refs))

        extra_queries = None
        if llm is not None and query_budget > 0:
            extra_queries = _generate_search_queries(llm, topic, discipline, query_budget) or None

        _log.info(
            "search node start topic=%r discipline=%r local_refs=%d query_budget=%d extra_queries=%d",
            topic,
            discipline,
            len(local_refs),
            query_budget,
            len(extra_queries or []),
        )

        references, queries = search_client.search_multi_source(
            topic=topic,
            discipline=discipline,
            extra_queries=extra_queries,
        )
        local_ref_ids: set[str] = set()

        if local_refs:
            local_ref_ids = {str(ref.get("ref_id", "")).strip() for ref in local_refs}
            references = local_refs + [ref for ref in references if ref.get("ref_id") not in local_ref_ids]

        _log.info(
            "search node end references=%d search_queries=%d local_refs=%d",
            len(references),
            len(queries) if isinstance(queries, list) else 0,
            len(local_refs),
        )

        result = {
            "references": references,
            "search_queries": queries,
            "literature_summary": _summarize_references(references),
            "local_refs_count": len(local_refs),
            "rag_enabled": getattr(services, "rag_index", None) is not None,
        }
        paper_index = getattr(services, "paper_index", None)
        persisted_indexed = {}
        if paper_index is not None and hasattr(paper_index, "indexed_papers"):
            try:
                persisted_indexed = paper_index.indexed_papers()
            except Exception:  # noqa: BLE001
                persisted_indexed = {}
        if persisted_indexed:
            result["indexed_papers"] = persisted_indexed
            result["paper_index_ready"] = True
        if paper_index is not None and bool(getattr(settings, "fetch_full_text", False)):
            http_client = getattr(services, "api_http", None) or getattr(services, "http", None)
            max_papers = int(getattr(settings, "max_papers_to_index", 20) or 20)
            online_candidates = [
                ref
                for ref in references
                if isinstance(ref, dict) and str(ref.get("ref_id", "")).strip() not in local_ref_ids
            ]
            try:
                indexed_papers = paper_index.ingest_online(online_candidates[:max_papers], http_client)
            except Exception:  # noqa: BLE001
                indexed_papers = {}
            if indexed_papers:
                merged_indexed = dict(persisted_indexed)
                merged_indexed.update(indexed_papers)
                result["indexed_papers"] = merged_indexed
                result["paper_index_ready"] = True
        return result

    return search

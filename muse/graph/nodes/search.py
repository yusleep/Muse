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


def _reference_identity(ref: dict[str, Any]) -> tuple[str, str]:
    ref_id = str(ref.get("ref_id", "")).strip()
    if ref_id:
        return ("ref_id", ref_id.casefold())
    doi = str(ref.get("doi", "")).strip()
    if doi:
        return ("doi", doi.casefold())
    title = str(ref.get("title", "")).strip()
    if title:
        return ("title", title.casefold())
    return ("object", repr(sorted(ref.items())))


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw in items:
        value = str(raw).strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def build_search_node(settings: Any, services: Any, *, state_query_key: str | None = None):
    def search(state: dict[str, Any]) -> dict[str, Any]:
        topic = state.get("topic", "")
        discipline = state.get("discipline", "")
        llm = getattr(services, "llm", None)
        search_client = getattr(services, "search", None)
        local_refs = list(getattr(services, "local_refs", []) or [])
        existing_refs = list(state.get("references", []) or [])
        previous_queries = _unique_strings(list(state.get("search_queries", []) or []))
        query_budget = _search_query_budget(local_refs_count=len(local_refs))

        extra_queries = None
        if state_query_key is not None:
            query_override = _unique_strings(list(state.get(state_query_key, []) or []))
            if not query_override:
                return {
                    "references": [],
                    "search_queries": previous_queries,
                    "literature_summary": _summarize_references(existing_refs),
                    "local_refs_count": len(local_refs),
                    "rag_enabled": getattr(services, "rag_index", None) is not None,
                }
            extra_queries = query_override[:query_budget]
        elif llm is not None and query_budget > 0:
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
        seen_identities = {
            _reference_identity(ref)
            for ref in existing_refs
            if isinstance(ref, dict)
        }
        emitted_refs: list[dict[str, Any]] = []

        for local_ref in local_refs:
            if not isinstance(local_ref, dict):
                continue
            identity = _reference_identity(local_ref)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            emitted_refs.append(local_ref)

        for ref in references:
            if not isinstance(ref, dict):
                continue
            identity = _reference_identity(ref)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            emitted_refs.append(ref)

        merged_refs = existing_refs + emitted_refs
        merged_queries = _unique_strings(previous_queries + list(queries or []))

        _log.info(
            "search node end references=%d search_queries=%d local_refs=%d",
            len(merged_refs),
            len(merged_queries),
            len(local_refs),
        )

        result = {
            "references": emitted_refs,
            "search_queries": merged_queries,
            "literature_summary": _summarize_references(merged_refs),
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
                for ref in emitted_refs
                if isinstance(ref, dict) and ref.get("source") != "local"
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

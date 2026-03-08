"""Search node for literature discovery."""

from __future__ import annotations

from typing import Any

from muse.prompts.search_queries import search_queries_prompt


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

        extra_queries = None
        if llm is not None:
            extra_queries = _generate_search_queries(llm, topic, discipline, 7) or None

        references, queries = search_client.search_multi_source(
            topic=topic,
            discipline=discipline,
            extra_queries=extra_queries,
        )

        if local_refs:
            local_ref_ids = {ref.get("ref_id") for ref in local_refs}
            references = local_refs + [ref for ref in references if ref.get("ref_id") not in local_ref_ids]

        return {
            "references": references,
            "search_queries": queries,
            "literature_summary": _summarize_references(references),
            "local_refs_count": len(local_refs),
            "rag_enabled": getattr(services, "rag_index", None) is not None,
        }

    return search

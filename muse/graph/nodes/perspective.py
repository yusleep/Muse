"""Perspective discovery node for Phase 5 exploratory research."""

from __future__ import annotations

from typing import Any

from muse.prompts.perspective import (
    perspective_dialogues_prompt,
    perspective_personas_prompt,
)


def _clean_personas(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    personas = payload.get("personas", [])
    cleaned: list[dict[str, str]] = []
    for item in personas:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        expertise = str(item.get("expertise", "")).strip()
        focus_area = str(item.get("focus_area", "")).strip()
        if not name or not expertise or not focus_area:
            continue
        cleaned.append(
            {
                "name": name,
                "expertise": expertise,
                "focus_area": focus_area,
            }
        )
    return cleaned[:5]


def _clean_queries(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in payload.get("search_queries", []):
        query = str(raw).strip()
        if not query:
            continue
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(query)
    return cleaned[:5]


def build_perspective_node(*, services: Any):
    def perspective_discovery(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        references = state.get("references", [])
        if llm is None or not isinstance(references, list) or not references:
            return {"perspectives": [], "perspective_queries": []}

        topic = str(state.get("topic", "")).strip()
        discipline = str(state.get("discipline", "")).strip()
        refs_snapshot = [
            {
                "ref_id": str(ref.get("ref_id", "")).strip(),
                "title": str(ref.get("title", "")).strip(),
                "abstract": str(ref.get("abstract", "")).strip(),
                "year": ref.get("year"),
            }
            for ref in references
            if isinstance(ref, dict)
        ]

        personas_system, personas_user = perspective_personas_prompt(
            topic,
            discipline,
            refs_snapshot,
        )
        try:
            personas_payload = llm.structured(
                system=personas_system,
                user=personas_user,
                route="outline",
                max_tokens=1200,
            )
        except Exception:
            personas_payload = {}
        personas = _clean_personas(personas_payload)
        if not personas:
            return {"perspectives": [], "perspective_queries": []}

        dialogues_system, dialogues_user = perspective_dialogues_prompt(
            topic,
            discipline,
            personas,
            refs_snapshot,
        )
        try:
            dialogues_payload = llm.structured(
                system=dialogues_system,
                user=dialogues_user,
                route="outline",
                max_tokens=1800,
            )
        except Exception:
            dialogues_payload = {}
        queries = _clean_queries(dialogues_payload)

        return {
            "perspectives": personas,
            "perspective_queries": queries,
        }

    return perspective_discovery

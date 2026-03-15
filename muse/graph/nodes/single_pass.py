"""Single-pass writing node for exploratory sequential drafting."""

from __future__ import annotations

import json
from typing import Any


def _single_pass_system_prompt() -> str:
    return (
        "Write one thesis chapter while preserving continuity with earlier chapters.\n"
        "Return JSON with keys: merged_text, quality_scores, iterations_used, subtask_results.\n"
        "Each subtask_results item must include: subtask_id, title, target_words, output_text, "
        "actual_words, citations_used, key_claims.\n"
        "Use only ref_id values from the provided references."
    )


def _compact_references(references: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for ref in references[:30]:
        if not isinstance(ref, dict):
            continue
        compact.append(
            {
                "ref_id": str(ref.get("ref_id", "")).strip(),
                "title": str(ref.get("title", "")).strip(),
                "year": ref.get("year"),
                "abstract": str(ref.get("abstract", "")).strip(),
            }
        )
    return compact


def _parse_output(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"merged_text": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"merged_text": raw}
    return {}


def _fallback_subtask_results(chapter_plan: dict[str, Any], merged_text: str) -> list[dict[str, Any]]:
    subtask_plan = chapter_plan.get("subtask_plan", [])
    if isinstance(subtask_plan, list) and subtask_plan:
        first = subtask_plan[0] if isinstance(subtask_plan[0], dict) else {}
        return [
            {
                "subtask_id": str(first.get("subtask_id", "single_pass")),
                "title": str(first.get("title", chapter_plan.get("chapter_title", ""))),
                "target_words": int(first.get("target_words", chapter_plan.get("target_words", 1200)) or 1200),
                "output_text": merged_text,
                "actual_words": len(merged_text.split()),
                "citations_used": [],
                "key_claims": [],
            }
        ]
    return [
        {
            "subtask_id": "single_pass",
            "title": str(chapter_plan.get("chapter_title", "")),
            "target_words": int(chapter_plan.get("target_words", 1200) or 1200),
            "output_text": merged_text,
            "actual_words": len(merged_text.split()),
            "citations_used": [],
            "key_claims": [],
        }
    ]


def _normalize_subtask_results(chapter_plan: dict[str, Any], parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = parsed.get("subtask_results", [])
    normalized: list[dict[str, Any]] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            output_text = str(item.get("output_text", "")).strip()
            normalized.append(
                {
                    "subtask_id": str(item.get("subtask_id", "")).strip(),
                    "title": str(item.get("title", "")).strip(),
                    "target_words": int(item.get("target_words", 0) or 0),
                    "output_text": output_text,
                    "actual_words": int(item.get("actual_words", len(output_text.split())) or len(output_text.split())),
                    "citations_used": [
                        str(cite_key).strip()
                        for cite_key in item.get("citations_used", [])
                        if str(cite_key).strip()
                    ]
                    if isinstance(item.get("citations_used", []), list)
                    else [],
                    "key_claims": [
                        str(claim).strip()
                        for claim in item.get("key_claims", [])
                        if str(claim).strip()
                    ]
                    if isinstance(item.get("key_claims", []), list)
                    else [],
                }
            )
    merged_text = str(parsed.get("merged_text", "")).strip()
    return normalized or _fallback_subtask_results(chapter_plan, merged_text)


def _citation_artifacts(chapter_id: str, subtask_results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    citation_uses: list[dict[str, Any]] = []
    claim_text_by_id: dict[str, str] = {}
    for subtask in subtask_results:
        subtask_id = str(subtask.get("subtask_id", "")).strip()
        claims = subtask.get("key_claims", [])
        citations = subtask.get("citations_used", [])
        if not isinstance(claims, list):
            claims = []
        if not isinstance(citations, list):
            citations = []
        for claim_index, claim in enumerate(claims, start=1):
            claim_text = str(claim).strip()
            if not claim_text or not subtask_id:
                continue
            claim_id = f"{chapter_id}_{subtask_id}_c{claim_index:02d}"
            claim_text_by_id[claim_id] = claim_text
            for cite_key in citations:
                cite_key_text = str(cite_key).strip()
                if not cite_key_text:
                    continue
                citation_uses.append(
                    {
                        "cite_key": cite_key_text,
                        "claim_id": claim_id,
                        "chapter_id": chapter_id,
                        "subtask_id": subtask_id,
                    }
                )
    return citation_uses, claim_text_by_id


def build_single_pass_node(*, settings: Any, services: Any):
    del settings

    def single_pass_writer(state: dict[str, Any]) -> dict[str, Any]:
        llm = getattr(services, "llm", None)
        if llm is None or not hasattr(llm, "text"):
            return {"chapters": {}}

        topic = str(state.get("topic", "")).strip()
        discipline = str(state.get("discipline", "")).strip()
        language = str(state.get("language", "zh")).strip() or "zh"
        references = state.get("references", [])
        chapter_plans = state.get("chapter_plans", [])
        if not isinstance(chapter_plans, list) or not chapter_plans:
            return {"chapters": {}}

        chapters: dict[str, Any] = {}
        conversation_history: list[dict[str, Any]] = []
        previous_chapters: list[dict[str, Any]] = []

        for chapter_plan in chapter_plans:
            if not isinstance(chapter_plan, dict):
                continue
            chapter_id = str(chapter_plan.get("chapter_id", "")).strip()
            if not chapter_id:
                continue
            payload = {
                "topic": topic,
                "discipline": discipline,
                "language": language,
                "chapter_plan": chapter_plan,
                "references": _compact_references(references if isinstance(references, list) else []),
                "previous_chapters": previous_chapters,
                "conversation_history": conversation_history[-6:],
            }
            user = json.dumps(payload, ensure_ascii=False)
            raw = llm.text(
                system=_single_pass_system_prompt(),
                user=user,
                route="writing",
                max_tokens=3600,
            )
            parsed = _parse_output(raw)
            merged_text = str(parsed.get("merged_text", "")).strip()
            subtask_results = _normalize_subtask_results(chapter_plan, parsed)
            if not merged_text:
                merged_text = "\n\n".join(
                    str(item.get("output_text", "")).strip()
                    for item in subtask_results
                    if isinstance(item, dict) and str(item.get("output_text", "")).strip()
                )
            citation_uses, claim_text_by_id = _citation_artifacts(chapter_id, subtask_results)
            chapter_result = {
                "chapter_id": chapter_id,
                "chapter_title": str(chapter_plan.get("chapter_title", chapter_id)),
                "merged_text": merged_text,
                "quality_scores": parsed.get("quality_scores", {})
                if isinstance(parsed.get("quality_scores", {}), dict)
                else {},
                "iterations_used": int(parsed.get("iterations_used", 1) or 1),
                "subtask_results": subtask_results,
                "citation_uses": citation_uses,
                "claim_text_by_id": claim_text_by_id,
            }
            chapters[chapter_id] = chapter_result
            conversation_history.append({"role": "user", "chapter_id": chapter_id, "content": payload})
            conversation_history.append({"role": "assistant", "chapter_id": chapter_id, "content": raw})
            previous_chapters.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_title": chapter_result["chapter_title"],
                    "summary": " ".join(merged_text.split()[:80]),
                }
            )

        return {
            "chapters": chapters,
            "current_chapter_index": len(chapters),
        }

    return single_pass_writer

"""Draft support helpers shared by graph nodes and legacy stages."""

from __future__ import annotations

import json
import logging
from typing import Any

from muse.prompts.argument_plan import argument_plan_prompt

_log = logging.getLogger("muse.draft")


_WORD_TO_FLOAT: dict[str, float] = {
    "高": 0.9, "中": 0.6, "低": 0.3,
    "high": 0.9, "medium": 0.6, "low": 0.3,
}


def _safe_float(value: Any, default: float = 0.5) -> float:
    """Convert *value* to float, mapping common CJK/English words to numbers."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    if s in _WORD_TO_FLOAT:
        return _WORD_TO_FLOAT[s]
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _extract_text_from_raw(llm_client: Any, system: str, user: str, *, route: str) -> dict[str, Any]:
    """Fallback: call LLM in plain text mode and wrap as dict with `text`."""

    try:
        out = llm_client.text(
            system="Write the subsection content directly as plain text. Do NOT use JSON formatting.",
            user=user,
            route=route,
            max_tokens=2800,
        )
        return {
            "text": out,
            "citations_used": [],
            "key_claims": [],
            "self_assessment": {
                "confidence": 0.3,
                "weak_spots": ["fallback-mode"],
                "needs_revision": True,
            },
        }
    except Exception:
        return {"text": ""}


def _call_write_llm(
    llm_client: Any,
    system: str,
    user: str,
    *,
    route: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Call the structured writing route with fallback to raw-text mode."""

    if llm_client is None:
        return {"text": ""}

    out = None
    for attempt in range(max_retries + 1):
        try:
            out = llm_client.structured(system=system, user=user, route=route, max_tokens=2800)
            break
        except Exception:
            if attempt < max_retries:
                continue
            out = _extract_text_from_raw(llm_client, system, user, route=route)
            break
    if out is None:
        return {}
    return out


def _build_refs_snapshot(
    *,
    state: dict[str, Any],
    references: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    refs = references if isinstance(references, list) else state.get("references", [])
    if not isinstance(refs, list):
        refs = []
    indexed_papers = state.get("indexed_papers", {})
    if not isinstance(indexed_papers, dict):
        indexed_papers = {}

    snapshot: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get("ref_id"):
            continue
        ref_id = str(ref.get("ref_id", "")).strip()
        indexed_meta = indexed_papers.get(ref_id, {})
        if not isinstance(indexed_meta, dict):
            indexed_meta = {}
        available_sections = indexed_meta.get("available_sections", [])
        if not isinstance(available_sections, list):
            available_sections = []
        snapshot.append(
            {
                "ref_id": ref_id,
                "title": ref.get("title", ""),
                "year": ref.get("year"),
                "abstract": ref.get("abstract") or "",
                "source": str(indexed_meta.get("source", ref.get("source", "")) or ""),
                "indexed": bool(indexed_meta.get("indexed", False)),
                "available_sections": [
                    str(section).strip()
                    for section in available_sections
                    if str(section).strip()
                ],
            }
        )
    return snapshot[:50]


def _consistency_context_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    from muse.graph.helpers.memory_keeper import ConsistencyStore

    store = ConsistencyStore.from_dict(state.get("consistency_data", {}))
    context = store.get_context_for_draft()
    if not any(context.get(key) for key in ("glossary", "citation_counts", "notation", "chapter_summaries")):
        return None
    context["instruction"] = (
        "Keep terminology, notation, and citation choices consistent with earlier chapters."
    )
    return context


def _reflection_tips_from_state(state: dict[str, Any]) -> list[str]:
    from muse.graph.helpers.reflection_bank import ReflectionBank

    bank = ReflectionBank.from_dict(state.get("reflection_data", {}))
    return bank.get_writing_tips(max_tips=3)


def _chapter_reference_context_from_state(
    state: dict[str, Any],
    *,
    chapter_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    briefs = state.get("reference_briefs", {})
    if not isinstance(briefs, dict):
        return [], []
    chapter_key = str(chapter_id).strip()
    chapter_briefs = briefs.get(chapter_key, [])
    evidence_gaps = briefs.get(f"{chapter_key}_gaps", [])
    if not isinstance(chapter_briefs, list):
        chapter_briefs = []
    if not isinstance(evidence_gaps, list):
        evidence_gaps = []
    return (
        [dict(item) for item in chapter_briefs if isinstance(item, dict)],
        [str(item).strip() for item in evidence_gaps if str(item).strip()],
    )


def _sanitize_argument_plan(
    payload: Any,
    *,
    allowed_sources: set[str],
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    evidence_chain = payload.get("evidence_chain", [])
    if not isinstance(evidence_chain, list):
        evidence_chain = []
    sanitized_chain: list[dict[str, Any]] = []
    for item in evidence_chain:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        claim = str(item.get("claim", "")).strip()
        finding = str(item.get("specific_finding", "")).strip()
        if not source or source not in allowed_sources or not claim:
            continue
        sanitized_chain.append(
            {
                "claim": claim,
                "source": source,
                "specific_finding": finding,
            }
        )

    paragraph_count = payload.get("paragraph_count", 0)
    try:
        paragraph_count = max(int(paragraph_count), 0)
    except (TypeError, ValueError):
        paragraph_count = 0

    sanitized = {
        "core_claim": str(payload.get("core_claim", "")).strip(),
        "evidence_chain": sanitized_chain,
        "logical_flow": str(payload.get("logical_flow", "")).strip(),
        "paragraph_count": paragraph_count,
    }
    if not any(
        (
            sanitized["core_claim"],
            sanitized["evidence_chain"],
            sanitized["logical_flow"],
            sanitized["paragraph_count"],
        )
    ):
        return None
    return sanitized


def _argument_plan_from_briefs(
    llm_client: Any,
    *,
    subtask: dict[str, Any],
    chapter_briefs: list[dict[str, Any]],
    language: str,
) -> dict[str, Any] | None:
    if llm_client is None or not chapter_briefs:
        return None

    system, user = argument_plan_prompt(
        str(subtask.get("title", "")),
        str(subtask.get("description", "")),
        chapter_briefs,
        language=language,
    )
    try:
        payload = llm_client.structured(
            system=system,
            user=user,
            route="default",
            max_tokens=800,
        )
    except Exception:
        return None

    allowed_sources = {
        str(item.get("ref_id", "")).strip()
        for item in chapter_briefs
        if isinstance(item, dict) and str(item.get("ref_id", "")).strip()
    }
    return _sanitize_argument_plan(payload, allowed_sources=allowed_sources)


def write_subtasks(
    *,
    llm_client: Any,
    state: dict[str, Any],
    chapter_id: str = "",
    chapter_title: str,
    subtask_plan: list[dict[str, Any]],
    revision_instructions: dict[str, str],
    previous: list[dict[str, Any]],
    rag_index: Any = None,
    route: str = "writing",
) -> list[dict[str, Any]]:
    results = []
    prev_text = ""
    prev_by_id = {item["subtask_id"]: item for item in previous}

    for st_idx, subtask in enumerate(subtask_plan):
        sid = subtask["subtask_id"]
        _log.info("  subtask %d/%d [%s] %s", st_idx + 1, len(subtask_plan), sid, subtask.get("title", "")[:50])
        if sid in prev_by_id and sid not in revision_instructions:
            kept = dict(prev_by_id[sid])
            results.append(kept)
            prev_text = kept.get("output_text", "")
            continue

        refs_snapshot = _build_refs_snapshot(state=state)

        local_context: list[dict[str, Any]] = []
        if rag_index is not None:
            query = f"{chapter_title} {subtask.get('title', '')} {state.get('topic', '')}"
            try:
                local_context = rag_index.retrieve(query, top_k=5)
            except Exception:
                local_context = []

        system = (
            "Write one thesis subsection with citations. "
            "IMPORTANT: for citations_used, use ONLY ref_id values from the available_references list. "
            "Do not invent citation keys not in that list. "
            "SCOPE GUARD: Write ONLY about the topic defined in subtask.title. "
            "Do NOT include content that belongs to other subtasks. "
            "If a related topic is outside this subtask's scope, mention it briefly "
            "and note that it will be covered in a later section. "
            "If an argument_plan is provided, follow its logical_flow and make each paragraph execute one step "
            "of the evidence_chain. "
            "References marked source=local are author-provided core papers and should be prioritized when relevant. "
            "References marked indexed=true include full-text section metadata that the writing agent can drill into. "
            "Include specific technical details, mathematical notation where appropriate, "
            "and reference concrete experimental results. "
            "Return JSON with keys: "
            "text, citations_used (list of ref_id strings), key_claims (list), transition_out, "
            "glossary_additions (object), "
            "self_assessment (object with confidence, weak_spots, needs_revision)."
        )
        user_payload: dict[str, Any] = {
            "topic": state.get("topic", ""),
            "chapter_title": chapter_title,
            "subtask": subtask,
            "language": state.get("language", "zh"),
            "available_references": refs_snapshot,
            "allowed_refs": [ref["ref_id"] for ref in refs_snapshot],
            "previous_subsection": prev_text,
            "revision_instruction": revision_instructions.get(sid),
        }
        consistency_context = _consistency_context_from_state(state)
        if consistency_context is not None:
            user_payload["consistency_context"] = consistency_context
        reflection_tips = _reflection_tips_from_state(state)
        if reflection_tips:
            user_payload["writing_tips_from_experience"] = reflection_tips
        chapter_briefs, evidence_gaps = _chapter_reference_context_from_state(
            state,
            chapter_id=chapter_id,
        )
        if chapter_briefs:
            user_payload["reference_briefs"] = chapter_briefs
        if evidence_gaps:
            user_payload["evidence_gaps"] = evidence_gaps
        argument_plan = _argument_plan_from_briefs(
            llm_client,
            subtask=subtask,
            chapter_briefs=chapter_briefs,
            language=str(state.get("language", "zh")),
        )
        if argument_plan is not None:
            user_payload["argument_plan"] = argument_plan
        if local_context:
            user_payload["local_context"] = local_context
        user = json.dumps(user_payload, ensure_ascii=False)

        out = _call_write_llm(llm_client, system, user, route=route)

        text = str(out.get("text", "")).strip()
        if not text:
            text = f"[{chapter_title}] {subtask['title']}\n\n(LLM returned empty content.)"

        target_words = int(subtask.get("target_words", 1200) or 1200)
        actual_words = len(text.split())
        ratio = actual_words / max(target_words, 1)

        if ratio < 0.7 and llm_client is not None:
            _log.info(
                "  subtask %s word_count_retry ratio=%.2f (target=%d actual=%d)",
                sid,
                ratio,
                target_words,
                actual_words,
            )
            retry_payload = dict(user_payload)
            retry_payload["revision_instruction"] = (
                f"当前字数 {actual_words} 远低于目标 {target_words}（{ratio:.0%}）。"
                "请在保持已有内容的基础上，补充更多技术细节、实验结果分析或文献论证，"
                f"将字数扩展至 {target_words} 左右。"
            )
            retry_out = _call_write_llm(
                llm_client,
                system,
                json.dumps(retry_payload, ensure_ascii=False),
                route=route,
            )
            retry_text = str(retry_out.get("text", "")).strip()
            retry_words = len(retry_text.split())
            if retry_text and retry_words > actual_words:
                out = retry_out
                text = retry_text
                actual_words = retry_words
        elif ratio > 1.5:
            _log.info("  subtask %s word_count_over ratio=%.2f", sid, ratio)

        citations_used = out.get("citations_used", [])
        if not isinstance(citations_used, list):
            citations_used = []
        allowed_set = {ref["ref_id"] for ref in refs_snapshot}
        hallucinated = [str(c).strip() for c in citations_used if str(c).strip() not in allowed_set]
        if hallucinated:
            _log.warning(
                "subtask %s hallucinated_citations=%d filtered: %s",
                sid,
                len(hallucinated),
                hallucinated[:5],
            )
        citations_used = [str(c).strip() for c in citations_used if str(c).strip() in allowed_set]

        key_claims = out.get("key_claims", [])
        if not isinstance(key_claims, list):
            key_claims = []

        assessment = out.get("self_assessment", {})
        if not isinstance(assessment, dict):
            assessment = {}

        results.append(
            {
                "subtask_id": sid,
                "title": subtask.get("title", ""),
                "target_words": target_words,
                "output_text": text,
                "actual_words": actual_words,
                "citations_used": [str(c).strip() for c in citations_used if str(c).strip()],
                "key_claims": [str(c).strip() for c in key_claims if str(c).strip()],
                "transition_out": str(out.get("transition_out", "")),
                "glossary_additions": out.get("glossary_additions", {})
                if isinstance(out.get("glossary_additions", {}), dict)
                else {},
                "confidence": _safe_float(assessment.get("confidence", 0.5)),
                "weak_spots": assessment.get("weak_spots", [])
                if isinstance(assessment.get("weak_spots", []), list)
                else [],
                "needs_revision": bool(assessment.get("needs_revision", False)),
            }
        )
        prev_text = text

    return results

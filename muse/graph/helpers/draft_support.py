"""Draft support helpers shared by graph nodes and legacy stages."""

from __future__ import annotations

import json
import logging
from typing import Any

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


def _extract_text_from_raw(llm_client: Any, system: str, user: str) -> dict[str, Any]:
    """Fallback: call LLM in plain text mode and wrap as dict with `text`."""

    try:
        out = llm_client.text(
            system="Write the subsection content directly as plain text. Do NOT use JSON formatting.",
            user=user,
            route="writing",
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


def _call_write_llm(llm_client: Any, system: str, user: str, max_retries: int = 2) -> dict[str, Any]:
    """Call the structured writing route with fallback to raw-text mode."""

    if llm_client is None:
        return {"text": ""}

    out = None
    for attempt in range(max_retries + 1):
        try:
            out = llm_client.structured(system=system, user=user, route="writing", max_tokens=2800)
            break
        except Exception:
            if attempt < max_retries:
                continue
            out = _extract_text_from_raw(llm_client, system, user)
            break
    if out is None:
        return {}
    return out


def write_subtasks(
    *,
    llm_client: Any,
    state: dict[str, Any],
    chapter_title: str,
    subtask_plan: list[dict[str, Any]],
    revision_instructions: dict[str, str],
    previous: list[dict[str, Any]],
    rag_index: Any = None,
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

        refs_snapshot = [
            {
                "ref_id": ref["ref_id"],
                "title": ref.get("title", ""),
                "year": ref.get("year"),
                "abstract": ref.get("abstract") or "",
            }
            for ref in state.get("references", [])
            if isinstance(ref, dict) and ref.get("ref_id")
        ][:50]

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
        if local_context:
            user_payload["local_context"] = local_context
        user = json.dumps(user_payload, ensure_ascii=False)

        out = _call_write_llm(llm_client, system, user)

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

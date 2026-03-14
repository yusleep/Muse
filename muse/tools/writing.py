"""Writing tools for the chapter ReAct agent."""

import json
import logging
from typing import Annotated
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool

from muse.tools._context import AgentRuntimeContext
from muse.tools.orchestration import append_partial_subtask_result

MuseToolRuntime = ToolRuntime[AgentRuntimeContext, Any]
_log = logging.getLogger("muse.draft")


def _services_from_runtime(runtime: MuseToolRuntime | None) -> Any:
    from muse.tools._context import get_services, services_from_runtime

    services = services_from_runtime(runtime)
    return services if services is not None else get_services()


def _normalized_partial_output(output: Any) -> dict[str, Any] | None:
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return {"text": output, "citations_used": [], "key_claims": []}
        if isinstance(parsed, dict):
            return parsed
        return {"text": output, "citations_used": [], "key_claims": []}
    return None


def _append_partial_result(
    *,
    subtask_id: str,
    subtask_title: str,
    target_words: int,
    output: dict[str, Any],
    allowed_ref_ids: set[str],
) -> None:
    text = str(output.get("text", "")).strip()
    if not text:
        return

    citations_used = output.get("citations_used", [])
    if not isinstance(citations_used, list):
        citations_used = []
    key_claims = output.get("key_claims", [])
    if not isinstance(key_claims, list):
        key_claims = []
    assessment = output.get("self_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}

    append_partial_subtask_result(
        {
            "subtask_id": str(subtask_id),
            "title": str(subtask_title),
            "target_words": int(target_words),
            "output_text": text,
            "actual_words": len(text.split()),
            "citations_used": [
                str(cite_key).strip()
                for cite_key in citations_used
                if str(cite_key).strip() in allowed_ref_ids
            ],
            "key_claims": [str(claim).strip() for claim in key_claims if str(claim).strip()],
            "transition_out": str(output.get("transition_out", "")),
            "glossary_additions": output.get("glossary_additions", {})
            if isinstance(output.get("glossary_additions", {}), dict)
            else {},
            "confidence": 0.3,
            "weak_spots": assessment.get("weak_spots", [])
            if isinstance(assessment.get("weak_spots", []), list)
            else [],
            "needs_revision": True,
        }
    )


@tool
def write_section(
    chapter_title: str,
    subtask_id: str,
    subtask_title: str,
    target_words: int,
    topic: str,
    language: str,
    references_json: str,
    revision_instruction: str = "",
    previous_subsection: str = "",
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Write a thesis subsection from an outline subtask."""

    services = _services_from_runtime(runtime)
    llm = getattr(services, "llm", None)
    if llm is None:
        return json.dumps(
            {
                "text": f"[{chapter_title}] {subtask_title}\n\n(No LLM available.)",
                "citations_used": [],
                "key_claims": [],
            },
            ensure_ascii=False,
        )

    try:
        references = json.loads(references_json)
    except (json.JSONDecodeError, TypeError):
        references = []

    refs_snapshot = [
        {
            "ref_id": ref.get("ref_id", ""),
            "title": ref.get("title", ""),
            "year": ref.get("year"),
            "abstract": ref.get("abstract") or "",
        }
        for ref in references
        if isinstance(ref, dict) and ref.get("ref_id")
    ][:50]

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
        "Return JSON with keys: text, citations_used (list of ref_id strings), key_claims (list), "
        "transition_out, glossary_additions (object), "
        "self_assessment (object with confidence, weak_spots, needs_revision)."
    )
    user_payload: dict[str, Any] = {
        "topic": topic,
        "chapter_title": chapter_title,
        "subtask": {
            "subtask_id": subtask_id,
            "title": subtask_title,
            "target_words": target_words,
        },
        "language": language,
        "available_references": refs_snapshot,
        "allowed_refs": [ref["ref_id"] for ref in refs_snapshot],
        "previous_subsection": previous_subsection,
        "revision_instruction": revision_instruction or None,
    }
    user = json.dumps(user_payload, ensure_ascii=False)

    llm_call_failed = False
    try:
        output = llm.structured(system=system, user=user, route="writing", max_tokens=2800)
    except Exception:  # noqa: BLE001
        llm_call_failed = True
        output = {
            "text": f"[{chapter_title}] {subtask_title}\n\n(LLM call failed.)",
            "citations_used": [],
            "key_claims": [],
        }

    allowed_set = {ref["ref_id"] for ref in refs_snapshot}
    if isinstance(output, str):
        normalized = _normalized_partial_output(output)
        if normalized is not None and not llm_call_failed:
            _append_partial_result(
                subtask_id=subtask_id,
                subtask_title=subtask_title,
                target_words=target_words,
                output=normalized,
                allowed_ref_ids=allowed_set,
            )
        return output

    if isinstance(output, dict):
        citations_used = output.get("citations_used", [])
        if not isinstance(citations_used, list):
            citations_used = []
        hallucinated = [str(c).strip() for c in citations_used if str(c).strip() not in allowed_set]
        if hallucinated:
            _log.warning(
                "write_section subtask=%s hallucinated_citations=%d filtered: %s",
                subtask_id,
                len(hallucinated),
                hallucinated[:5],
            )
        output["citations_used"] = [str(c).strip() for c in citations_used if str(c).strip() in allowed_set]
        if not llm_call_failed:
            _append_partial_result(
                subtask_id=subtask_id,
                subtask_title=subtask_title,
                target_words=target_words,
                output=output,
                allowed_ref_ids=allowed_set,
            )

    return json.dumps(output, ensure_ascii=False)


@tool
def revise_section(
    section_text: str,
    instruction: str,
    chapter_title: str,
    language: str,
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Revise an existing thesis section using a specific instruction."""

    services = _services_from_runtime(runtime)
    llm = getattr(services, "llm", None)
    if llm is None:
        return section_text

    system = (
        f"Revise the following thesis section from chapter '{chapter_title}'. "
        f"Language: {language}. Follow the instruction precisely. "
        "Return ONLY the revised text, no JSON wrapping."
    )
    user = json.dumps({"instruction": instruction, "text": section_text}, ensure_ascii=False)

    try:
        return llm.text(system=system, user=user, route="writing", max_tokens=2800)
    except Exception:  # noqa: BLE001
        return section_text


@tool
def apply_patch(section_text: str, old_string: str, new_string: str) -> str:
    """Apply a targeted string replacement within a section."""

    if old_string not in section_text:
        return "ERROR: old_string not found in section_text. Ensure the old_string matches exactly."
    return section_text.replace(old_string, new_string, 1)

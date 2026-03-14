"""Chapter-level ReAct subgraph with partial-recovery safeguards."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from muse.models.adapter import MuseChatModel
from muse.services.providers import LLMClient
from muse.tools._context import AgentRuntimeContext, build_runtime_context

_log = logging.getLogger("muse.chapter")


class ChapterState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    chapter_plan: dict[str, Any]
    references: list[dict[str, Any]]
    topic: str
    discipline: str
    language: str
    subtask_results: list[dict[str, Any]]
    merged_text: str
    quality_scores: dict[str, int]
    review_notes: list[dict[str, Any]]
    revision_instructions: dict[str, str]
    iteration: int
    max_iterations: int
    citation_uses: list[dict[str, Any]]
    claim_text_by_id: dict[str, str]


class ChapterAgentExecutionError(RuntimeError):
    """Raised when the chapter ReAct path cannot produce a recoverable result."""


def _references_summary(references: list[dict[str, Any]]) -> str:
    if not references:
        return "0 references available."
    valid_refs = [
        reference
        for reference in references
        if isinstance(reference, dict) and reference.get("ref_id")
    ]
    all_ids = ", ".join(str(reference.get("ref_id", "?")) for reference in valid_refs)
    summaries = []
    for reference in valid_refs[:20]:
        title = str(reference.get("title", ""))[:80]
        year = reference.get("year", "")
        summaries.append(f"  - {reference['ref_id']}: {title} ({year})")
    summary_text = "\n".join(summaries)
    return (
        f"{len(valid_refs)} references available.\n"
        f"All ref_ids: {all_ids}\n"
        f"Top {min(20, len(valid_refs))} summaries:\n{summary_text}"
    )


def _extract_chapter_result(
    result: dict[str, Any],
    chapter_plan: dict[str, Any],
) -> dict[str, Any]:
    chapter_id = chapter_plan.get("chapter_id", "chapter")
    iterations_used = result.get("iterations_used", result.get("iteration", 0))
    chapter_result = {
        "chapter_id": chapter_id,
        "chapter_title": chapter_plan.get("chapter_title", chapter_id),
        "merged_text": result.get("merged_text", ""),
        "quality_scores": result.get("quality_scores", {}),
        "iterations_used": iterations_used,
        "subtask_results": result.get("subtask_results", []),
        "citation_uses": result.get("citation_uses", []),
        "claim_text_by_id": result.get("claim_text_by_id", {}),
    }
    return {
        "chapters": {chapter_id: chapter_result},
        "claim_text_by_id": result.get("claim_text_by_id", {}),
    }


def _message_count(messages: Any) -> int:
    return len(messages) if isinstance(messages, list) else 0


def _subtask_plan(chapter_plan: dict[str, Any]) -> list[dict[str, Any]]:
    plan = chapter_plan.get("subtask_plan", [])
    if not isinstance(plan, list):
        return []
    return [item for item in plan if isinstance(item, dict)]


def _chapter_route(state: dict[str, Any]) -> str:
    """Backward-compatible review route shim for legacy tests/helpers."""

    from muse.graph.helpers.review_state import should_iterate

    return should_iterate(state)


def _ordered_subtask_results(
    subtask_results: list[dict[str, Any]],
    chapter_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered_ids = [
        str(subtask.get("subtask_id", ""))
        for subtask in _subtask_plan(chapter_plan)
        if str(subtask.get("subtask_id", ""))
    ]
    by_id: dict[str, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    for result in subtask_results:
        if not isinstance(result, dict):
            continue
        sid = str(result.get("subtask_id", "")).strip()
        if sid:
            by_id[sid] = dict(result)
        else:
            extras.append(dict(result))

    ordered = [by_id.pop(sid) for sid in ordered_ids if sid in by_id]
    ordered.extend(by_id.values())
    ordered.extend(extras)
    return ordered


def _assemble_chapter_result(
    subtask_results: list[dict[str, Any]],
    chapter_plan: dict[str, Any],
) -> dict[str, Any]:
    ordered_results = _ordered_subtask_results(subtask_results, chapter_plan)
    chapter_id = str(chapter_plan.get("chapter_id", "chapter"))
    citation_uses: list[dict[str, Any]] = []
    claim_text_by_id: dict[str, str] = {}

    for subtask in ordered_results:
        subtask_id = str(subtask.get("subtask_id", ""))
        citations = subtask.get("citations_used", [])
        if not isinstance(citations, list):
            citations = []
        key_claims = subtask.get("key_claims", [])
        if not isinstance(key_claims, list):
            key_claims = []
        for claim_index, claim in enumerate(key_claims, start=1):
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

    merged_text = "\n\n".join(
        str(item.get("output_text", "")).strip()
        for item in ordered_results
        if isinstance(item, dict) and str(item.get("output_text", "")).strip()
    )
    return _extract_chapter_result(
        {
            "merged_text": merged_text,
            "quality_scores": {},
            "iterations_used": 0,
            "subtask_results": ordered_results,
            "citation_uses": citation_uses,
            "claim_text_by_id": claim_text_by_id,
        },
        chapter_plan,
    )


def _partial_subtask_result(
    *,
    subtask: dict[str, Any],
    output: dict[str, Any],
    allowed_ref_ids: set[str],
) -> dict[str, Any]:
    text = str(output.get("text", "")).strip()
    if not text:
        text = f"[{subtask.get('title', '')}]\n\n(Recovery produced empty content.)"

    citations_used = output.get("citations_used", [])
    if not isinstance(citations_used, list):
        citations_used = []
    key_claims = output.get("key_claims", [])
    if not isinstance(key_claims, list):
        key_claims = []
    assessment = output.get("self_assessment", {})
    if not isinstance(assessment, dict):
        assessment = {}

    target_words = subtask.get("target_words", 1200)
    try:
        target_words = int(target_words or 1200)
    except (TypeError, ValueError):
        target_words = 1200

    return {
        "subtask_id": str(subtask.get("subtask_id", "")),
        "title": str(subtask.get("title", "")),
        "target_words": target_words,
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


def _call_recovery_writer(
    writer: Any,
    *,
    system: str,
    user: str,
) -> dict[str, Any]:
    if writer is None:
        raise RuntimeError("No structured writing client available for partial recovery.")

    if hasattr(writer, "structured"):
        return writer.structured(
            system=system,
            user=user,
            route="writing",
            max_tokens=2800,
        )

    llm_client = getattr(writer, "llm_client", None)
    if llm_client is not None and hasattr(llm_client, "structured"):
        return llm_client.structured(
            system=system,
            user=user,
            route="writing",
            max_tokens=2800,
        )

    if isinstance(writer, BaseChatModel) or hasattr(writer, "invoke"):
        message = writer.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=user),
            ]
        )
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {"text": content, "citations_used": [], "key_claims": []}
        if isinstance(parsed, dict):
            return parsed
        return {"text": content, "citations_used": [], "key_claims": []}

    raise RuntimeError("No structured writing client available for partial recovery.")


def _resolve_recovery_writer(*, services: Any, settings: Any) -> Any:
    services_llm = getattr(services, "llm", None) if services is not None else None
    if services_llm is not None:
        return services_llm
    return _create_react_model(settings=settings, services=services)


def _write_single_subtask(
    *,
    writer: Any,
    state: dict[str, Any],
    chapter_plan: dict[str, Any],
    subtask: dict[str, Any],
    existing_results: list[dict[str, Any]],
) -> dict[str, Any]:
    from muse.graph.helpers.draft_support import _build_refs_snapshot

    references = state.get("references", [])
    if not isinstance(references, list):
        references = []
    refs_snapshot = _build_refs_snapshot(state=state, references=references)
    allowed_ref_ids = {
        str(ref.get("ref_id", "")).strip()
        for ref in refs_snapshot
        if str(ref.get("ref_id", "")).strip()
    }
    previous_text = "\n\n".join(
        str(result.get("output_text", "")).strip()
        for result in existing_results
        if isinstance(result, dict) and str(result.get("output_text", "")).strip()
    )
    chapter_title = str(chapter_plan.get("chapter_title", chapter_plan.get("chapter_id", "chapter")))
    system = (
        "Write one thesis subsection with citations. "
        "IMPORTANT: for citations_used, use ONLY ref_id values from the available_references list. "
        "Do not invent citation keys not in that list. "
        "SCOPE GUARD: Write ONLY about the topic defined in subtask.title. "
        "Do NOT include content that belongs to other subtasks. "
        "Previous sections have already been drafted; continue by writing only the missing subtask. "
        "References marked source=local are author-provided core papers and should be prioritized when relevant. "
        "For references marked indexed=true, use get_paper_section when you need section-level evidence. "
        "Return JSON with keys: text, citations_used (list of ref_id strings), key_claims (list), "
        "transition_out, glossary_additions (object), "
        "self_assessment (object with confidence, weak_spots, needs_revision)."
    )
    user_payload = {
        "topic": state.get("topic", ""),
        "chapter_title": chapter_title,
        "subtask": {
            "subtask_id": subtask.get("subtask_id", ""),
            "title": subtask.get("title", ""),
            "target_words": subtask.get("target_words", 1200),
        },
        "language": state.get("language", "zh"),
        "available_references": refs_snapshot,
        "allowed_refs": sorted(allowed_ref_ids),
        "previous_subsection": previous_text[-1000:] if previous_text else "",
        "revision_instruction": "Partial recovery mode: complete only this missing subtask.",
    }
    output = _call_recovery_writer(
        writer,
        system=system,
        user=json.dumps(user_payload, ensure_ascii=False),
    )
    if isinstance(output, str):
        output = {"text": output, "citations_used": [], "key_claims": []}
    if not isinstance(output, dict):
        raise RuntimeError("Partial recovery writer returned a non-dict payload.")
    return _partial_subtask_result(
        subtask=subtask,
        output=output,
        allowed_ref_ids=allowed_ref_ids,
    )


def _create_react_model(*, services: Any = None, settings: Any = None):
    llm_client = getattr(services, "llm", None) if services is not None else None
    if isinstance(llm_client, LLMClient):
        return MuseChatModel(llm_client=llm_client, route="writing")
    if isinstance(llm_client, BaseChatModel):
        return llm_client
    if llm_client is not None:
        return None

    if settings is None:
        return None

    try:
        from muse.models.factory import create_chat_model
    except ImportError:
        return None

    try:
        return create_chat_model(settings, route="writing")
    except Exception:
        return None


def _build_react_chapter_agent(*, services: Any, settings: Any = None):
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware.types import ModelRequest, dynamic_prompt
    except ImportError:
        return None

    from muse.prompts.chapter_agent import chapter_agent_system_prompt
    from muse.tools.file import edit_file, glob_files, grep, read_file, write_file
    from muse.tools.orchestration import submit_result, update_plan
    from muse.tools.research import (
        academic_search,
        get_paper_section,
        image_search,
        read_pdf,
        retrieve_local_refs,
        web_fetch,
        web_search,
    )
    from muse.tools.review import self_review
    from muse.tools.writing import apply_patch, revise_section, write_section

    tools = [
        write_section,
        revise_section,
        apply_patch,
        self_review,
        academic_search,
        retrieve_local_refs,
        web_search,
        web_fetch,
        read_pdf,
        image_search,
        read_file,
        write_file,
        edit_file,
        glob_files,
        grep,
        submit_result,
        update_plan,
    ]
    if getattr(services, "paper_index", None) is not None:
        tools.append(get_paper_section)

    model = _create_react_model(settings=settings, services=services)
    if model is None:
        return None

    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        state = request.state
        chapter_plan = state.get("chapter_plan", {})
        return chapter_agent_system_prompt(
            topic=str(state.get("topic", "")),
            language=str(state.get("language", "zh")),
            chapter_title=str(chapter_plan.get("chapter_title", "")),
            chapter_plan=chapter_plan,
            references_summary=_references_summary(state.get("references", [])),
        )

    return create_agent(
        model=model,
        tools=tools,
        middleware=[prompt],
        state_schema=ChapterState,
        context_schema=AgentRuntimeContext,
        name="chapter_react_agent",
    )


def build_chapter_subgraph_node(*, services: Any, settings: Any = None):
    react_agent = _build_react_chapter_agent(services=services, settings=settings)

    def run_react_chapter(state: dict[str, Any]) -> dict[str, Any]:
        from muse.tools._context import clear_state, get_state, set_services, set_state
        from muse.graph.nodes.draft import build_chapter_draft_node
        from muse.tools.orchestration import (
            clear_partial_subtask_results,
            clear_submitted_result,
            get_partial_subtask_results,
            get_subagent_executor,
            get_submitted_result,
            set_subagent_executor,
        )

        chapter_plan = state.get("chapter_plan", {})
        if not isinstance(chapter_plan, dict):
            chapter_plan = {}
        chapter_id = str(chapter_plan.get("chapter_id", "chapter"))
        subtasks = _subtask_plan(chapter_plan)
        if react_agent is None:
            _log.error(
                "chapter react unavailable chapter_id=%s subtasks=%d",
                chapter_id,
                len(subtasks),
            )
            if settings is not None and getattr(services, "llm", None) is not None:
                _log.warning(
                    "chapter react fallback chapter_id=%s strategy=direct_draft",
                    chapter_id,
                )
                direct_result = build_chapter_draft_node(services)(state)
                return _extract_chapter_result(direct_result, chapter_plan)
            raise ChapterAgentExecutionError(
                "Chapter ReAct agent unavailable for non-empty chapter workload."
            )

        def _restore_thread_state(previous_executor: Any, previous_state: Any) -> None:
            set_subagent_executor(previous_executor)
            if previous_state is None:
                clear_state()
            else:
                set_state(previous_state)

        set_services(services)
        previous_state = get_state(default=None)
        previous_executor = get_subagent_executor()
        clear_submitted_result()
        clear_partial_subtask_results()
        set_subagent_executor(getattr(services, "subagent_executor", None))

        _log.info(
            "chapter react start chapter_id=%s references=%d subtasks=%d",
            chapter_id,
            len(state.get("references", [])) if isinstance(state.get("references"), list) else 0,
            len(subtasks),
        )

        agent_input = dict(state)
        set_state(agent_input)
        agent_input.setdefault(
            "messages",
            [
                {
                    "role": "user",
                    "content": "Draft, review, and submit this thesis chapter.",
                }
            ],
        )

        react_result: dict[str, Any] = {}
        invoke_error: Exception | None = None
        try:
            maybe_result = react_agent.invoke(
                agent_input,
                {"recursion_limit": 60},
                context=build_runtime_context(services),
            )
            if isinstance(maybe_result, dict):
                react_result = maybe_result
        except Exception as exc:
            invoke_error = exc
            _log.exception(
                "chapter react error chapter_id=%s error=%s: %s",
                chapter_id,
                type(exc).__name__,
                exc,
            )

        submitted = get_submitted_result()
        partial_results = get_partial_subtask_results()
        _log.info(
            "chapter react invoke_return chapter_id=%s messages=%d submitted=%s",
            chapter_id,
            _message_count(react_result.get("messages")),
            bool(submitted),
        )
        clear_submitted_result()
        clear_partial_subtask_results()
        _restore_thread_state(previous_executor, previous_state)

        payload = submitted.get("payload") if isinstance(submitted, dict) else None
        if isinstance(payload, dict):
            _log.info(
                "chapter react partial_recovery chapter_id=%s strategy=submitted_result",
                chapter_id,
            )
            result = _extract_chapter_result(payload, chapter_plan)
            _log.info(
                "chapter react end chapter_id=%s quality_keys=%s",
                chapter_id,
                sorted(payload.get("quality_scores", {}).keys())
                if isinstance(payload.get("quality_scores", {}), dict)
                else [],
            )
            return result

        if submitted is not None:
            _log.warning(
                "chapter react invalid_submit_payload chapter_id=%s",
                chapter_id,
            )

        recovered_results = [dict(result) for result in partial_results if isinstance(result, dict)]
        if recovered_results:
            completed_ids = {
                str(result.get("subtask_id", "")).strip()
                for result in recovered_results
                if str(result.get("subtask_id", "")).strip()
            }
            missing_subtasks = [
                subtask
                for subtask in subtasks
                if str(subtask.get("subtask_id", "")).strip() not in completed_ids
            ]
            _log.info(
                "chapter react partial_recovery chapter_id=%s strategy=partial_results recovered=%d missing=%d",
                chapter_id,
                len(recovered_results),
                len(missing_subtasks),
            )

            recovery_writer = _resolve_recovery_writer(services=services, settings=settings)
            for subtask in missing_subtasks:
                try:
                    recovered_results.append(
                        _write_single_subtask(
                            writer=recovery_writer,
                            state=state,
                            chapter_plan=chapter_plan,
                            subtask=subtask,
                            existing_results=recovered_results,
                        )
                    )
                except Exception as write_exc:
                    _log.warning(
                        "chapter react partial_recovery_failed chapter_id=%s subtask_id=%s error=%s: %s",
                        chapter_id,
                        str(subtask.get("subtask_id", "")),
                        type(write_exc).__name__,
                        write_exc,
                    )

            result = _assemble_chapter_result(recovered_results, chapter_plan)
            chapter_result = result.get("chapters", {}).get(chapter_id, {})
            _log.info(
                "chapter react end chapter_id=%s recovered_subtasks=%d",
                chapter_id,
                len(chapter_result.get("subtask_results", []))
                if isinstance(chapter_result.get("subtask_results", []), list)
                else 0,
            )
            return result

        if invoke_error is not None:
            _log.error(
                "chapter react unrecoverable chapter_id=%s reason=no_recoverable_progress error=%s",
                chapter_id,
                type(invoke_error).__name__,
            )
            raise ChapterAgentExecutionError(
                f"Chapter ReAct agent failed without recoverable progress: "
                f"{type(invoke_error).__name__}: {invoke_error}."
            ) from invoke_error

        _log.error(
            "chapter react unrecoverable chapter_id=%s reason=missing_submit_result",
            chapter_id,
        )
        raise ChapterAgentExecutionError(
            "Chapter ReAct agent did not submit a result and no partial subtask results were recoverable."
        )

    return run_react_chapter

"""Citation verification subgraph and ledger construction."""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated
from typing import Any, Literal
from typing import Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from muse.models.adapter import MuseChatModel
from muse.services.providers import LLMClient
from muse.tools._context import AgentRuntimeContext, build_runtime_context

_log = logging.getLogger("muse.citation")


class CitationState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    references: list[dict[str, Any]]
    citation_uses: list[dict[str, Any]]
    claim_text_by_id: dict[str, str]
    citation_worklist: list[dict[str, Any]]
    evidence_items: list[dict[str, Any]]
    match_results: list[dict[str, Any]]
    citation_ledger: dict[str, dict[str, Any]]
    verified_citations: list[str]
    flagged_citations: list[dict[str, Any]]


class CitationAgentExecutionError(RuntimeError):
    """Raised when the citation ReAct path cannot produce a valid structured review."""


class _NullMetadata:
    def verify_doi(self, doi: str) -> bool:
        return bool(doi)

    def crosscheck_metadata(self, ref: dict[str, Any]) -> bool:
        return bool(ref.get("title"))


class _NullLLM:
    def entailment(self, *, premise: str, hypothesis: str, route: str = "reasoning") -> str:
        return "neutral"


def _metadata_service(services: Any):
    return getattr(services, "metadata", None) or _NullMetadata()


def _llm_service(services: Any):
    llm = getattr(services, "llm", None)
    if llm is None or not hasattr(llm, "entailment"):
        return _NullLLM()
    return llm


def _state_workload_summary(state: Mapping[str, Any]) -> tuple[int, int, int]:
    citations = state.get("citation_uses", [])
    claims = state.get("claim_text_by_id", {})
    references = state.get("references", [])
    return (
        len(citations) if isinstance(citations, list) else 0,
        len(claims) if isinstance(claims, dict) else 0,
        len(references) if isinstance(references, list) else 0,
    )


def _evidence_size_summary(items: list[dict[str, Any]]) -> tuple[int, int]:
    total_chars = 0
    max_chars = 0
    for item in items:
        evidence = item.get("evidence")
        if not isinstance(evidence, str):
            continue
        size = len(evidence)
        total_chars += size
        max_chars = max(max_chars, size)
    return total_chars, max_chars


def _fuzzy_match_ref(cite_key: str, ref_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    surname_match = re.match(r"^([A-Za-z\u4e00-\u9fff]+)", cite_key.strip())
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", cite_key)
    if not surname_match or not year_match:
        return None
    surname = surname_match.group(1).lower()
    year = year_match.group(1)
    for ref_id, ref in ref_lookup.items():
        if surname in ref_id.lower() and year in ref_id:
            return ref
    return None


def _extract_claims(state: CitationState) -> dict[str, Any]:
    return {}


def _build_evidence_items(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    ref_lookup = {
        ref.get("ref_id"): ref
        for ref in state.get("references", [])
        if isinstance(ref, dict) and ref.get("ref_id")
    }
    evidence_items: list[dict[str, Any]] = []
    for use in state.get("citation_uses", []):
        if not isinstance(use, dict):
            continue
        cite_key = use.get("cite_key")
        claim_id = use.get("claim_id")
        ref = None
        if isinstance(cite_key, str) and cite_key:
            ref = ref_lookup.get(cite_key) or _fuzzy_match_ref(cite_key, ref_lookup)
        claim = state.get("claim_text_by_id", {}).get(claim_id, "") if isinstance(claim_id, str) else ""
        evidence_items.append(
            {
                "cite_key": cite_key,
                "claim_id": claim_id,
                "claim": claim,
                "reference": ref,
                "metadata_status": "not_found" if ref is None else "ok",
                "evidence": (ref or {}).get("abstract") or (ref or {}).get("title", ""),
            }
        )
    return evidence_items


def _lookup_evidence(state: CitationState) -> dict[str, Any]:
    evidence_items = _build_evidence_items(state)
    total_chars, max_chars = _evidence_size_summary(evidence_items)
    _log.info(
        "lookup_evidence built items=%d total_evidence_chars=%d max_evidence_chars=%d",
        len(evidence_items),
        total_chars,
        max_chars,
    )
    if len(evidence_items) > 200 or total_chars > 100_000:
        _log.warning(
            "large citation fallback workload items=%d total_evidence_chars=%d",
            len(evidence_items),
            total_chars,
        )
    return {"evidence_items": evidence_items}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _build_citation_worklist(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    worklist: list[dict[str, Any]] = []
    for item in _build_evidence_items(state):
        reference = item.get("reference")
        if not isinstance(reference, dict):
            reference = {}
        worklist.append(
            {
                "cite_key": str(item.get("cite_key") or ""),
                "claim_id": str(item.get("claim_id") or ""),
                "claim": str(item.get("claim") or ""),
                "evidence": str(item.get("evidence") or ""),
                "metadata_status": str(item.get("metadata_status") or ""),
                "reference": reference,
                "reference_json": json.dumps(reference, ensure_ascii=False, sort_keys=True) if reference else "",
                "title": str(reference.get("title") or ""),
                "authors": ", ".join(_string_list(reference.get("authors"))),
                "year": str(reference.get("year") or ""),
                "doi": str(reference.get("doi") or ""),
            }
        )
    return worklist


def build_match_support_node(services: Any):
    metadata = _metadata_service(services)
    llm = _llm_service(services)

    def match_support(state: CitationState) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        _log.info("match_support processing items=%d", len(state.get("evidence_items", [])))
        for item in state.get("evidence_items", []):
            ref = item.get("reference") or {}
            metadata_status = str(item.get("metadata_status") or "ok")
            doi = ref.get("doi") if isinstance(ref, dict) else None
            if metadata_status == "ok":
                if isinstance(doi, str) and doi and not metadata.verify_doi(doi):
                    metadata_status = "doi_invalid"
                elif not metadata.crosscheck_metadata(ref):
                    metadata_status = "metadata_mismatch"

            metadata_ok = metadata_status == "ok"
            entailment = "skipped"
            if metadata_ok:
                entailment = llm.entailment(
                    premise=str(item.get("evidence", "")),
                    hypothesis=str(item.get("claim", "")),
                    route="reasoning",
                )
            results.append(
                {
                    **item,
                    "metadata_ok": metadata_ok,
                    "metadata_status": metadata_status,
                    "entailment": entailment,
                }
            )
        _log.info("match_support completed items=%d", len(results))
        return {"match_results": results}

    return match_support


def _flag_weak_claims(state: CitationState) -> dict[str, Any]:
    ledger: dict[str, dict[str, Any]] = {}
    verified: list[str] = []
    flagged: list[dict[str, Any]] = []

    for item in state.get("match_results", []):
        claim_id = str(item.get("claim_id") or "")
        cite_key = str(item.get("cite_key") or "")
        evidence = str(item.get("evidence") or "")
        entailment = str(item.get("entailment") or "neutral")
        metadata_ok = bool(item.get("metadata_ok"))
        metadata_status = str(item.get("metadata_status") or "ok")

        support_score = 1.0 if metadata_ok and entailment == "entailment" else 0.3 if metadata_ok else 0.1
        confidence = "high" if support_score >= 0.9 else "medium" if support_score >= 0.3 else "low"
        repair_status = "verified" if support_score >= 0.9 else "flagged"
        ledger[claim_id] = {
            "claim": str(item.get("claim") or ""),
            "cited_source": cite_key,
            "support_score": support_score,
            "evidence_excerpt": evidence[:240],
            "confidence": confidence,
            "repair_status": repair_status,
        }

        if repair_status == "verified":
            if cite_key and cite_key not in verified:
                verified.append(cite_key)
        else:
            reason = "unsupported_claim"
            if metadata_status in {"not_found", "doi_invalid", "metadata_mismatch"}:
                reason = metadata_status
            flagged.append(
                {
                    "cite_key": cite_key,
                    "reason": reason,
                    "claim_id": claim_id or None,
                    "detail": metadata_status if metadata_status != "ok" else entailment,
                }
            )

    _log.info(
        "flag_weak_claims ledger=%d verified=%d flagged=%d",
        len(ledger),
        len(verified),
        len(flagged),
    )
    return {
        "citation_ledger": ledger,
        "verified_citations": verified,
        "flagged_citations": flagged,
    }


def _repair_route(state: CitationState) -> Literal["repair", "done"]:
    return "repair" if any(entry.get("confidence") == "medium" for entry in state.get("citation_ledger", {}).values()) else "done"


def _repair(state: CitationState) -> dict[str, Any]:
    ledger = {key: dict(value) for key, value in state.get("citation_ledger", {}).items()}
    for value in ledger.values():
        if value.get("confidence") == "medium":
            value["repair_status"] = "repaired"
    return {"citation_ledger": ledger}


def build_citation_graph(*, services: Any):
    builder = StateGraph(CitationState)
    builder.add_node("extract_claims", _extract_claims)
    builder.add_node("lookup_evidence", _lookup_evidence)
    builder.add_node("match_support", build_match_support_node(services))
    builder.add_node("flag_weak_claims", _flag_weak_claims)
    builder.add_node("repair", _repair)
    builder.add_edge(START, "extract_claims")
    builder.add_edge("extract_claims", "lookup_evidence")
    builder.add_edge("lookup_evidence", "match_support")
    builder.add_edge("match_support", "flag_weak_claims")
    builder.add_conditional_edges("flag_weak_claims", _repair_route, {"repair": "repair", "done": END})
    builder.add_edge("repair", END)
    return builder.compile()


def _citation_summary(references: list[dict[str, Any]]) -> str:
    if not references:
        return "0 references available."
    return (
        f"{len(references)} references available. Top refs: "
        + ", ".join(
            str(reference.get("ref_id", "?"))
            for reference in references[:10]
            if isinstance(reference, dict)
        )
    )


def _message_payload(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        return message
    if isinstance(message, BaseMessage):
        payload: dict[str, Any] = {
            "type": getattr(message, "type", message.__class__.__name__),
            "content": getattr(message, "content", ""),
            "name": getattr(message, "name", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is not None:
            payload["tool_calls"] = tool_calls
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id is not None:
            payload["tool_call_id"] = tool_call_id
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if isinstance(additional_kwargs, dict) and "tool_calls" in additional_kwargs and "tool_calls" not in payload:
            payload["tool_calls"] = additional_kwargs["tool_calls"]
        return payload
    return None


def _recent_tool_trace(messages: Any, *, limit: int = 6) -> list[str]:
    if not isinstance(messages, list):
        return []

    trace: list[str] = []
    for raw_message in messages[-limit:]:
        message = _message_payload(raw_message)
        if not isinstance(message, dict):
            continue

        tool_calls = message.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                tool_name = str(tool_call.get("name") or "?")
                args = tool_call.get("args")
                arg_parts: list[str] = []
                if isinstance(args, dict):
                    for key in ("file_path", "cite_key", "claim_id", "doi", "query"):
                        value = args.get(key)
                        if value:
                            arg_parts.append(f"{key}={value}")
                trace.append(f"call:{tool_name}" + (f" ({', '.join(arg_parts)})" if arg_parts else ""))

        if str(message.get("type") or "") == "tool" and message.get("name"):
            content = str(message.get("content") or "").strip().replace("\n", " ")
            if len(content) > 120:
                content = content[:117] + "..."
            trace.append(f"result:{message['name']}: {content}")

    return trace[-limit:]


def _extract_citation_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_ledger": result.get("citation_ledger", {}),
        "verified_citations": result.get("verified_citations", []),
        "flagged_citations": result.get("flagged_citations", []),
    }


def _empty_citation_result(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "citation_ledger": dict(state.get("citation_ledger", {})),
        "verified_citations": list(state.get("verified_citations", [])),
        "flagged_citations": list(state.get("flagged_citations", [])),
    }


def _trace_suffix(trace: list[str]) -> str:
    return f" recent_trace={' | '.join(trace)}" if trace else ""


def _validate_finalized_citation_payload(payload: dict[str, Any], worklist: list[dict[str, Any]]) -> None:
    if not isinstance(payload.get("citation_ledger"), dict):
        raise ValueError("citation_ledger missing or invalid")
    if not isinstance(payload.get("verified_citations"), list):
        raise ValueError("verified_citations missing or invalid")
    if not isinstance(payload.get("flagged_citations"), list):
        raise ValueError("flagged_citations missing or invalid")

    assessments = payload.get("assessments")
    if not isinstance(assessments, list):
        raise ValueError("assessments missing or invalid")
    if len(assessments) != len(worklist):
        raise ValueError(
            f"assessment count mismatch: expected {len(worklist)}, got {len(assessments)}"
        )

    recorded_pairs = {
        (str(item.get("cite_key") or ""), str(item.get("claim_id") or ""))
        for item in assessments
        if isinstance(item, dict)
    }
    missing = [
        f"{item.get('cite_key')} / {item.get('claim_id')}"
        for item in worklist
        if (str(item.get("cite_key") or ""), str(item.get("claim_id") or "")) not in recorded_pairs
    ]
    if missing:
        raise ValueError(f"missing assessments for {', '.join(missing)}")

    coverage = payload.get("coverage")
    if isinstance(coverage, dict):
        missing_pairs = coverage.get("missing_pairs")
        if isinstance(missing_pairs, list) and missing_pairs:
            raise ValueError("coverage reported missing_pairs")
        expected_pairs = coverage.get("expected_pairs")
        if isinstance(expected_pairs, int) and expected_pairs != len(worklist):
            raise ValueError(
                f"coverage expected_pairs mismatch: expected {len(worklist)}, got {expected_pairs}"
            )
        recorded_pairs_count = coverage.get("recorded_pairs")
        if isinstance(recorded_pairs_count, int) and recorded_pairs_count != len(assessments):
            raise ValueError(
                "coverage recorded_pairs mismatch: "
                f"expected {len(assessments)}, got {recorded_pairs_count}"
            )


def _create_react_model(*, services: Any = None, settings: Any = None):
    llm_client = getattr(services, "llm", None) if services is not None else None
    if isinstance(llm_client, LLMClient):
        return MuseChatModel(llm_client=llm_client, route="reasoning")
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
        return create_chat_model(settings, route="reasoning")
    except Exception:
        return None


def _build_react_citation_agent(*, services: Any, settings: Any = None):
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware.types import ModelRequest, dynamic_prompt
    except ImportError:
        return None

    from muse.prompts.citation_agent import citation_agent_system_prompt
    from muse.tools.citation import (
        crosscheck_metadata,
        entailment_check,
        finalize_citation_review,
        record_citation_assessment,
        verify_doi,
    )

    tools = [
        verify_doi,
        crosscheck_metadata,
        entailment_check,
        record_citation_assessment,
        finalize_citation_review,
    ]

    model = _create_react_model(settings=settings, services=services)
    if model is None:
        return None

    @dynamic_prompt
    def prompt(request: ModelRequest) -> str:
        state = request.state
        return citation_agent_system_prompt(
            worklist_json=json.dumps(state.get("citation_worklist", []), ensure_ascii=False, sort_keys=True),
            total_citations=len(state.get("citation_uses", [])),
            total_claims=len(state.get("claim_text_by_id", {})),
            references_summary=_citation_summary(state.get("references", [])),
        )

    return create_agent(
        model=model,
        tools=tools,
        middleware=[prompt],
        state_schema=CitationState,
        context_schema=AgentRuntimeContext,
        name="citation_react_agent",
    )


def build_citation_subgraph_node(*, services: Any, settings: Any = None):
    react_agent = _build_react_citation_agent(services=services, settings=settings)

    def run_react_citation(state: dict[str, Any]) -> dict[str, Any]:
        from muse.tools._context import set_services
        from muse.tools.citation import (
            clear_citation_review_session,
            get_finalized_citation_review,
            prepare_citation_review_session,
        )
        from muse.tools.orchestration import get_subagent_executor, set_subagent_executor

        worklist = _build_citation_worklist(state)
        if not worklist:
            _log.info("citation react skipped: no citation worklist items")
            return _empty_citation_result(state)

        if react_agent is None:
            _log.error("citation react unavailable for worklist_items=%d", len(worklist))
            raise CitationAgentExecutionError(
                "Citation ReAct agent unavailable for non-empty citation workload."
            )

        set_services(services)
        previous_executor = get_subagent_executor()
        set_subagent_executor(getattr(services, "subagent_executor", None))
        prepare_citation_review_session(worklist)

        agent_input = dict(state)
        agent_input["citation_worklist"] = worklist
        agent_input.setdefault(
            "messages",
            [
                {
                    "role": "user",
                    "content": (
                        "Review every citation in citation_worklist, record one structured "
                        "assessment per item, and finalize the citation review."
                    ),
                }
            ],
        )

        citations, claims, references = _state_workload_summary(state)
        _log.info(
            "citation react start citations=%d claims=%d references=%d worklist_items=%d",
            citations,
            claims,
            references,
            len(worklist),
        )
        react_result: dict[str, Any] = {}
        try:
            maybe_result = react_agent.invoke(
                agent_input,
                {"recursion_limit": 40},
                context=build_runtime_context(services),
            )
            if isinstance(maybe_result, dict):
                react_result = maybe_result
            recent_trace = _recent_tool_trace(react_result.get("messages", []))
            finalized = get_finalized_citation_review()
            if not isinstance(finalized, dict):
                raise CitationAgentExecutionError(
                    "Citation ReAct agent did not finalize the structured citation review."
                    + _trace_suffix(recent_trace)
                )
            try:
                _validate_finalized_citation_payload(finalized, worklist)
            except ValueError as exc:
                raise CitationAgentExecutionError(
                    f"Invalid structured citation review payload: {exc}."
                    + _trace_suffix(recent_trace)
                ) from exc

            result = _extract_citation_result(finalized)
            _log.info(
                "citation react end verified=%d flagged=%d",
                len(result.get("verified_citations", [])),
                len(result.get("flagged_citations", [])),
            )
            return result
        except CitationAgentExecutionError:
            recent_trace = _recent_tool_trace(react_result.get("messages", []))
            if recent_trace:
                _log.error(
                    "citation react validation failed. recent_trace=%s",
                    " | ".join(recent_trace),
                )
            else:
                _log.error("citation react validation failed.")
            raise
        except Exception as exc:
            recent_trace = _recent_tool_trace(react_result.get("messages", []))
            if recent_trace:
                _log.exception(
                    "citation react path failed (%s: %s). recent_trace=%s",
                    type(exc).__name__,
                    exc,
                    " | ".join(recent_trace),
                )
            else:
                _log.exception(
                    "citation react path failed (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
            raise CitationAgentExecutionError(
                f"Citation ReAct agent failed: {type(exc).__name__}: {exc}."
                + _trace_suffix(recent_trace)
            ) from exc
        finally:
            clear_citation_review_session()
            set_subagent_executor(previous_executor)

    return run_react_citation

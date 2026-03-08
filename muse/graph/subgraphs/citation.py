"""Citation verification subgraph and ledger construction."""

from __future__ import annotations

import re
from typing import Annotated
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CitationState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    references: list[dict[str, Any]]
    citation_uses: list[dict[str, Any]]
    claim_text_by_id: dict[str, str]
    evidence_items: list[dict[str, Any]]
    match_results: list[dict[str, Any]]
    citation_ledger: dict[str, dict[str, Any]]
    verified_citations: list[str]
    flagged_citations: list[dict[str, Any]]


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


def _lookup_evidence(state: CitationState) -> dict[str, Any]:
    ref_lookup = {
        ref.get("ref_id"): ref
        for ref in state.get("references", [])
        if isinstance(ref, dict) and ref.get("ref_id")
    }
    evidence_items = []
    for use in state.get("citation_uses", []):
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
    return {"evidence_items": evidence_items}


def build_match_support_node(services: Any):
    metadata = _metadata_service(services)
    llm = _llm_service(services)

    def match_support(state: CitationState) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
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


def _extract_citation_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_ledger": result.get("citation_ledger", {}),
        "verified_citations": result.get("verified_citations", []),
        "flagged_citations": result.get("flagged_citations", []),
    }


def _build_react_citation_agent(*, services: Any, settings: Any = None):
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError:
        return None

    if settings is None:
        return None

    try:
        from muse.models.factory import create_chat_model
    except ImportError:
        return None

    from muse.prompts.citation_agent import citation_agent_system_prompt
    from muse.tools.citation import (
        crosscheck_metadata,
        entailment_check,
        flag_citation,
        repair_citation,
        verify_doi,
    )
    from muse.tools.file import read_file
    from muse.tools.orchestration import submit_result, update_plan
    from muse.tools.research import academic_search

    tools = [
        verify_doi,
        crosscheck_metadata,
        entailment_check,
        flag_citation,
        repair_citation,
        academic_search,
        read_file,
        submit_result,
        update_plan,
    ]

    try:
        model = create_chat_model(settings, route="reasoning")
    except Exception:
        return None

    def prompt(state: CitationState) -> str:
        return citation_agent_system_prompt(
            total_citations=len(state.get("citation_uses", [])),
            total_claims=len(state.get("claim_text_by_id", {})),
            references_summary=_citation_summary(state.get("references", [])),
        )

    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        state_schema=CitationState,
        name="citation_react_agent",
    )


def build_citation_subgraph_node(*, services: Any, settings: Any = None):
    react_agent = _build_react_citation_agent(services=services, settings=settings)
    fallback_graph = build_citation_graph(services=services)

    def _fallback(state: dict[str, Any]) -> dict[str, Any]:
        fallback_result = fallback_graph.invoke(
            {
                "references": state.get("references", []),
                "citation_uses": state.get("citation_uses", []),
                "claim_text_by_id": state.get("claim_text_by_id", {}),
                "citation_ledger": state.get("citation_ledger", {}),
                "verified_citations": state.get("verified_citations", []),
                "flagged_citations": state.get("flagged_citations", []),
            }
        )
        return _extract_citation_result(fallback_result)

    if react_agent is None:
        return _fallback

    def run_react_citation(state: dict[str, Any]) -> dict[str, Any]:
        from muse.tools._context import set_services
        from muse.tools.orchestration import clear_submitted_result, get_submitted_result

        set_services(services)
        clear_submitted_result()

        agent_input = dict(state)
        agent_input.setdefault(
            "messages",
            [
                {
                    "role": "user",
                    "content": "Verify the thesis citations and submit the ledger.",
                }
            ],
        )

        try:
            react_agent.invoke(agent_input, {"recursion_limit": 40})
        except Exception:
            clear_submitted_result()
            return _fallback(state)

        submitted = get_submitted_result()
        clear_submitted_result()
        if not submitted:
            return _fallback(state)

        payload = submitted.get("payload", {})
        if not isinstance(payload, dict):
            return _fallback(state)
        return _extract_citation_result(payload)

    return run_react_citation

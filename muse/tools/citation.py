"""Citation tools for both LangChain bindings and ReAct agents."""

import json
import logging
import threading
from typing import Annotated
from typing import Any, Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from muse.tools._context import AgentRuntimeContext

MuseToolRuntime = ToolRuntime[AgentRuntimeContext, Any]


_local = threading.local()
_log = logging.getLogger("muse.citation.tools")


class VerifyDoiInput(BaseModel):
    """Input schema for DOI verification."""

    doi: str = Field(description="The DOI string to verify")


class VerifyDoiTool(BaseTool):
    """Check whether a DOI resolves in CrossRef."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "verify_doi"
    description: str = (
        "Check whether a DOI is valid and exists in CrossRef. "
        "Returns whether the DOI is valid or invalid."
    )
    args_schema: type[BaseModel] = VerifyDoiInput
    metadata_client: Any = None

    def _run(self, doi: str) -> str:
        is_valid = self.metadata_client.verify_doi(doi)
        if is_valid:
            return f"DOI '{doi}' is valid and exists in CrossRef."
        return f"DOI '{doi}' is invalid or not found in CrossRef."


def make_verify_doi_tool(metadata_client: Any) -> BaseTool:
    """Create the ``verify_doi`` tool from a metadata client."""

    return VerifyDoiTool(metadata_client=metadata_client)


class CrosscheckMetadataInput(BaseModel):
    """Input schema for citation metadata cross-checking."""

    title: str = Field(description="Paper title")
    authors: str = Field(default="", description="Comma-separated author names")
    year: str = Field(default="", description="Publication year")
    doi: str = Field(default="", description="DOI if available")


class CrosscheckMetadataTool(BaseTool):
    """Verify citation metadata against CrossRef-like records."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "crosscheck_metadata"
    description: str = (
        "Verify a citation's metadata against CrossRef records and return "
        "whether it is verified or has a mismatch."
    )
    args_schema: type[BaseModel] = CrosscheckMetadataInput
    metadata_client: Any = None

    def _run(self, title: str, authors: str = "", year: str = "", doi: str = "") -> str:
        reference = {
            "title": title,
            "authors": [author.strip() for author in authors.split(",") if author.strip()],
            "year": int(year) if year.isdigit() else None,
            "doi": doi or None,
        }
        is_match = self.metadata_client.crosscheck_metadata(reference)
        if is_match:
            return f"Citation metadata verified: '{title}' matches CrossRef records."
        return f"Citation metadata mismatch: '{title}' does not match CrossRef records."


def make_crosscheck_metadata_tool(metadata_client: Any) -> BaseTool:
    """Create the ``crosscheck_metadata`` tool from a metadata client."""

    return CrosscheckMetadataTool(metadata_client=metadata_client)


class RecordCitationAssessmentInput(BaseModel):
    """Structured citation verdict recorded for one cite_key/claim_id pair."""

    cite_key: str = Field(description="Exact citation key from the worklist")
    claim_id: str = Field(description="Exact claim identifier from the worklist")
    verdict: Literal["verified", "flagged", "repaired"] = Field(
        description="Final verdict for this citation/claim pair"
    )
    support_score: float = Field(
        description="Support score between 0.0 and 1.0 after verification"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence level for the verdict"
    )
    reason: str = Field(description="Short reason code such as supported or metadata_mismatch")
    detail: str = Field(default="", description="Human-readable rationale for the verdict")
    evidence_excerpt: str = Field(
        default="",
        description="Short evidence excerpt supporting the verdict",
    )


class FinalizeCitationReviewInput(BaseModel):
    """Input schema for completing the citation review session."""

    summary: str = Field(description="Brief summary of the completed review")


def _active_review_session() -> dict[str, Any] | None:
    session = getattr(_local, "citation_review_session", None)
    return session if isinstance(session, dict) else None


def _pair_key(cite_key: str, claim_id: str) -> tuple[str, str]:
    return (str(cite_key or "").strip(), str(claim_id or "").strip())


def _normalize_work_item(item: dict[str, Any]) -> dict[str, Any] | None:
    cite_key, claim_id = _pair_key(item.get("cite_key", ""), item.get("claim_id", ""))
    if not cite_key or not claim_id:
        return None

    reference = item.get("reference")
    if not isinstance(reference, dict):
        reference = {}

    normalized = {
        "cite_key": cite_key,
        "claim_id": claim_id,
        "claim": str(item.get("claim") or ""),
        "evidence": str(item.get("evidence") or ""),
        "metadata_status": str(item.get("metadata_status") or ""),
        "reference": reference,
    }
    return normalized


def prepare_citation_review_session(worklist: list[dict[str, Any]]) -> None:
    """Initialize thread-local citation review state for one agent run."""

    normalized: list[dict[str, Any]] = []
    worklist_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for item in worklist:
        if not isinstance(item, dict):
            continue
        normalized_item = _normalize_work_item(item)
        if normalized_item is None:
            continue
        normalized.append(normalized_item)
        worklist_lookup[_pair_key(normalized_item["cite_key"], normalized_item["claim_id"])] = normalized_item

    _local.citation_review_session = {
        "worklist": normalized,
        "worklist_lookup": worklist_lookup,
        "records": {},
        "finalized_payload": None,
    }


def clear_citation_review_session() -> None:
    """Clear any active citation review session for the current thread."""

    if hasattr(_local, "citation_review_session"):
        delattr(_local, "citation_review_session")


def get_finalized_citation_review() -> dict[str, Any] | None:
    """Return the finalized citation review payload, if one exists."""

    session = _active_review_session()
    if session is None:
        return None
    payload = session.get("finalized_payload")
    return payload if isinstance(payload, dict) else None


def _coerce_authors(authors: Any) -> list[str]:
    if isinstance(authors, list):
        return [str(author).strip() for author in authors if str(author).strip()]
    if not isinstance(authors, str):
        return []

    stripped = authors.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(author).strip() for author in parsed if str(author).strip()]

    return [part.strip() for part in authors.split(",") if part.strip()]


def _build_reference_payload(
    *,
    reference_json: str,
    title: str,
    authors: Any,
    year: Any,
    doi: str,
    ref_id: str,
) -> dict[str, Any]:
    if reference_json:
        try:
            reference = json.loads(reference_json)
        except (json.JSONDecodeError, TypeError):
            return {}
        return reference if isinstance(reference, dict) else {}

    year_text = str(year or "").strip()
    return {
        "ref_id": ref_id or "unknown",
        "title": title,
        "authors": _coerce_authors(authors),
        "year": int(year_text) if year_text.isdigit() else year_text or None,
        "doi": doi or None,
    }


def _missing_pairs(
    worklist: list[dict[str, Any]],
    records: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for item in worklist:
        pair = _pair_key(item.get("cite_key", ""), item.get("claim_id", ""))
        if pair not in records:
            missing.append({"cite_key": pair[0], "claim_id": pair[1]})
    return missing


def _finalized_citation_payload(session: dict[str, Any], *, summary: str) -> dict[str, Any]:
    worklist = session.get("worklist", [])
    records = session.get("records", {})
    ledger: dict[str, dict[str, Any]] = {}
    verified_citations: list[str] = []
    flagged_citations: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []

    for item in worklist:
        pair = _pair_key(item.get("cite_key", ""), item.get("claim_id", ""))
        record = records.get(pair)
        if not isinstance(record, dict):
            continue

        cite_key = str(item.get("cite_key") or "")
        claim_id = str(item.get("claim_id") or "")
        verdict = str(record.get("verdict") or "flagged")
        evidence_excerpt = str(record.get("evidence_excerpt") or item.get("evidence") or "")[:240]
        reason = str(record.get("reason") or "unsupported_claim")
        detail = str(record.get("detail") or "")

        ledger[claim_id] = {
            "claim": str(item.get("claim") or ""),
            "cited_source": cite_key,
            "support_score": float(record.get("support_score") or 0.0),
            "evidence_excerpt": evidence_excerpt,
            "confidence": str(record.get("confidence") or "low"),
            "repair_status": verdict,
        }

        assessment = {
            "cite_key": cite_key,
            "claim_id": claim_id,
            "verdict": verdict,
            "support_score": float(record.get("support_score") or 0.0),
            "confidence": str(record.get("confidence") or "low"),
            "reason": reason,
            "detail": detail,
            "evidence_excerpt": evidence_excerpt,
        }
        assessments.append(assessment)

        if verdict == "verified":
            if cite_key and cite_key not in verified_citations:
                verified_citations.append(cite_key)
            continue

        flagged_citations.append(
            {
                "cite_key": cite_key,
                "reason": reason,
                "claim_id": claim_id,
                "detail": detail or None,
            }
        )

    return {
        "citation_ledger": ledger,
        "verified_citations": verified_citations,
        "flagged_citations": flagged_citations,
        "assessments": assessments,
        "coverage": {
            "expected_pairs": len(worklist),
            "recorded_pairs": len(assessments),
            "missing_pairs": [],
        },
        "summary": summary,
    }


def _services_from_runtime(runtime: MuseToolRuntime | None) -> Any:
    from muse.tools._context import get_services, services_from_runtime

    services = services_from_runtime(runtime)
    return services if services is not None else get_services()


@tool
def verify_doi(
    doi: str,
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Verify that a DOI resolves to a valid record."""

    services = _services_from_runtime(runtime)
    metadata_client = getattr(services, "metadata", None)
    if metadata_client is None:
        valid = bool(doi and doi.startswith("10."))
        return json.dumps(
            {
                "doi": doi,
                "valid": valid,
                "detail": "format_check_only",
            },
            ensure_ascii=False,
        )

    try:
        valid = bool(metadata_client.verify_doi(doi))
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "doi": doi,
                "valid": False,
                "detail": f"error: {exc}",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "doi": doi,
            "valid": valid,
            "detail": "crossref_verified" if valid else "doi_not_found",
        },
        ensure_ascii=False,
    )


@tool
def crosscheck_metadata(
    reference_json: str = "",
    title: str = "",
    authors: str = "",
    year: str = "",
    doi: str = "",
    ref_id: str = "",
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Cross-check a reference object's metadata for consistency."""

    reference = _build_reference_payload(
        reference_json=reference_json,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        ref_id=ref_id,
    )
    if not reference:
        return json.dumps(
            {
                "ref_id": "unknown",
                "consistent": False,
                "issues": ["invalid JSON input"],
            },
            ensure_ascii=False,
        )

    issues: list[str] = []
    if not reference.get("title"):
        issues.append("missing title")
    if not reference.get("authors"):
        issues.append("missing authors")
    if not reference.get("year"):
        issues.append("missing year")

    services = _services_from_runtime(runtime)
    metadata_client = getattr(services, "metadata", None)
    if metadata_client is not None:
        try:
            if not metadata_client.crosscheck_metadata(reference):
                issues.append("metadata_mismatch")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"crosscheck error: {exc}")

    return json.dumps(
        {
            "ref_id": reference.get("ref_id", "unknown"),
            "consistent": not issues,
            "issues": issues,
        },
        ensure_ascii=False,
    )


@tool(args_schema=RecordCitationAssessmentInput)
def record_citation_assessment(
    cite_key: str,
    claim_id: str,
    verdict: str,
    support_score: float,
    confidence: str,
    reason: str,
    detail: str = "",
    evidence_excerpt: str = "",
) -> str:
    """Record the final verdict for one citation/claim pair in the active review."""

    session = _active_review_session()
    if session is None:
        return "[citation review error] no active citation review session."

    pair = _pair_key(cite_key, claim_id)
    work_item = session.get("worklist_lookup", {}).get(pair)
    if not isinstance(work_item, dict):
        return f"[citation review error] unknown citation pair: {cite_key} / {claim_id}"

    normalized_confidence = str(confidence or "").strip().lower() or "low"
    if normalized_confidence not in {"low", "medium", "high"}:
        return f"[citation review error] invalid confidence: {confidence}"

    normalized_verdict = str(verdict or "").strip().lower() or "flagged"
    if normalized_verdict not in {"verified", "flagged", "repaired"}:
        return f"[citation review error] invalid verdict: {verdict}"

    bounded_score = max(0.0, min(1.0, float(support_score)))
    session["records"][pair] = {
        "cite_key": pair[0],
        "claim_id": pair[1],
        "verdict": normalized_verdict,
        "support_score": bounded_score,
        "confidence": normalized_confidence,
        "reason": str(reason or "unsupported_claim"),
        "detail": str(detail or ""),
        "evidence_excerpt": str(evidence_excerpt or work_item.get("evidence") or ""),
    }
    session["finalized_payload"] = None
    _log.info(
        "record assessment cite_key=%s claim_id=%s verdict=%s confidence=%s support_score=%.2f",
        pair[0],
        pair[1],
        normalized_verdict,
        normalized_confidence,
        bounded_score,
    )
    return f"Recorded citation assessment for {pair[0]} / {pair[1]}."


@tool(args_schema=FinalizeCitationReviewInput)
def finalize_citation_review(summary: str) -> str:
    """Finalize the active citation review after every worklist item is recorded."""

    session = _active_review_session()
    if session is None:
        return "[citation review error] no active citation review session."

    worklist = session.get("worklist", [])
    records = session.get("records", {})
    missing = _missing_pairs(worklist, records)
    if missing:
        session["finalized_payload"] = None
        missing_text = ", ".join(f"{item['cite_key']} / {item['claim_id']}" for item in missing)
        _log.warning(
            "finalize review blocked missing=%d summary=%s",
            len(missing),
            summary,
        )
        return f"[citation review error] missing assessments: {missing_text}"

    session["finalized_payload"] = _finalized_citation_payload(session, summary=summary)
    _log.info(
        "finalize review completed assessments=%d summary=%s",
        len(records),
        summary,
    )
    return f"Citation review completed. Summary: {summary}"


@tool
def entailment_check(
    premise: str,
    hypothesis: str,
    *,
    runtime: Annotated[MuseToolRuntime, InjectedToolArg],
) -> str:
    """Check whether a premise supports a citation claim."""

    services = _services_from_runtime(runtime)
    llm = getattr(services, "llm", None)
    if llm is None or not hasattr(llm, "entailment"):
        return json.dumps(
            {
                "entailment": "skipped",
                "confidence": 0.0,
            },
            ensure_ascii=False,
        )

    try:
        result = llm.entailment(
            premise=premise,
            hypothesis=hypothesis,
            route="reasoning",
        )
    except Exception:  # noqa: BLE001
        return json.dumps(
            {
                "entailment": "skipped",
                "confidence": 0.0,
            },
            ensure_ascii=False,
        )

    confidence = 0.8 if result == "entailment" else 0.3
    return json.dumps(
        {
            "entailment": result,
            "confidence": confidence,
        },
        ensure_ascii=False,
    )


@tool
def flag_citation(
    cite_key: str,
    reason: str,
    claim_id: str = "",
    detail: str = "",
) -> str:
    """Record a problematic citation for later repair."""

    return json.dumps(
        {
            "cite_key": cite_key,
            "reason": reason,
            "claim_id": claim_id or None,
            "detail": detail or None,
            "status": "flagged",
        },
        ensure_ascii=False,
    )


@tool
def repair_citation(
    claim_id: str,
    action: str,
    new_cite_key: str = "",
    justification: str = "",
) -> str:
    """Propose a repair action for a flagged citation."""

    return json.dumps(
        {
            "claim_id": claim_id,
            "action": action,
            "new_cite_key": new_cite_key or None,
            "justification": justification or None,
            "status": "repair_proposed",
        },
        ensure_ascii=False,
    )

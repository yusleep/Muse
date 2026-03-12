"""Citation tools for both LangChain bindings and ReAct agents."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field


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


@tool
def verify_doi(doi: str) -> str:
    """Verify that a DOI resolves to a valid record."""

    from muse.tools._context import get_services

    services = get_services()
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
def crosscheck_metadata(reference_json: str) -> str:
    """Cross-check a reference object's metadata for consistency."""

    from muse.tools._context import get_services

    try:
        reference = json.loads(reference_json)
    except (json.JSONDecodeError, TypeError):
        return json.dumps(
            {
                "ref_id": "unknown",
                "consistent": False,
                "issues": ["invalid JSON input"],
            },
            ensure_ascii=False,
        )

    if not isinstance(reference, dict):
        reference = {}

    issues: list[str] = []
    if not reference.get("title"):
        issues.append("missing title")
    if not reference.get("authors"):
        issues.append("missing authors")
    if not reference.get("year"):
        issues.append("missing year")

    services = get_services()
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


@tool
def entailment_check(premise: str, hypothesis: str) -> str:
    """Check whether a premise supports a citation claim."""

    from muse.tools._context import get_services

    services = get_services()
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

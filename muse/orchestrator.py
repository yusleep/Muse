"""Stage transition guards for the Muse workflow."""

from __future__ import annotations

from typing import Any


def gate_export(state: dict[str, Any]) -> tuple[bool, str]:
    """Block export only when citation entailment explicitly contradicts a claim.

    Soft failures (metadata_mismatch, doi_invalid, not_found) are infrastructure
    limitations rather than citation integrity violations and should not block export.
    """

    flagged = state.get("flagged_citations", [])
    if not isinstance(flagged, list):
        return True, "ok"

    # Only block on explicit contradictions; neutral/missing entailment is a soft
    # warning (the abstract simply lacks coverage, not that it disproves the claim).
    contradictions = [
        f for f in flagged
        if isinstance(f, dict)
        and f.get("reason") == "unsupported_claim"
        and "contradiction" in str(f.get("detail", ""))
    ]
    if contradictions:
        return False, f"{len(contradictions)} claims contradicted by cited sources"

    return True, "ok"


def can_advance_to_stage(state: dict[str, Any], next_stage: int) -> tuple[bool, str]:
    """Validate readiness gates for stage transitions."""

    current_stage = int(state.get("current_stage", 0))
    if next_stage != current_stage + 1:
        return False, f"invalid transition: {current_stage} -> {next_stage}"

    if next_stage == 3:
        outline = state.get("outline_json")
        chapter_plans = state.get("chapter_plans")
        if not isinstance(outline, dict) or not outline:
            return False, "outline not approved or missing"
        if not isinstance(chapter_plans, list) or not chapter_plans:
            return False, "chapter plans missing"

    if next_stage == 6:
        return gate_export(state)

    return True, "ok"

"""Citation verification pipeline helpers."""

from __future__ import annotations

import re
from typing import Callable, Iterable


def _fuzzy_match_ref(cite_key: str, ref_index: dict) -> "dict | None":
    """Fuzzy-match a natural-language cite_key (e.g. 'Smith et al., 2023') to a ref_id.

    Extracts the first author surname and a 4-digit year then looks for a ref_id
    that contains both tokens (case-insensitive).
    """
    surname_m = re.match(r"^([A-Za-z\u4e00-\u9fff]+)", cite_key.strip())
    year_m = re.search(r"\b(20\d{2}|19\d{2})\b", cite_key)
    if not surname_m or not year_m:
        return None
    surname = surname_m.group(1).lower()
    year = year_m.group(1)
    for ref_id, ref in ref_index.items():
        if isinstance(ref_id, str) and surname in ref_id.lower() and year in ref_id:
            return ref
    return None


def verify_all_citations(
    *,
    references: list[dict],
    citation_uses: Iterable[dict],
    claim_text_by_id: dict[str, str],
    verify_doi: Callable[[str], bool],
    crosscheck_metadata: Callable[[dict], bool],
    retrieve_passage: Callable[[dict, str], str],
    check_entailment: Callable[[str, str], str],
) -> tuple[list[str], list[dict]]:
    """Run layered citation checks and classify failures.

    Returns:
        verified citation keys and flagged citation entries.
    """

    ref_index = {ref.get("ref_id"): ref for ref in references if isinstance(ref, dict)}

    verified: list[str] = []
    seen_verified: set[str] = set()
    flagged: list[dict] = []

    # Cache per-ref results that don't depend on the specific claim, so we
    # don't repeat expensive HTTP calls (DOI lookup, metadata crosscheck) for
    # every one of the potentially hundreds of uses of the same cite_key.
    _ref_cache: dict[str, str | None] = {}  # cite_key -> "doi_invalid"|"metadata_mismatch"|"not_found"|None (None = ok so far)

    for use in citation_uses:
        cite_key = use.get("cite_key")
        claim_id = use.get("claim_id")

        if not isinstance(cite_key, str) or not cite_key:
            continue

        claim = claim_text_by_id.get(claim_id, "") if isinstance(claim_id, str) else ""

        # Fast path: we already know this cite_key fails at the ref/doi/metadata level.
        if cite_key in _ref_cache and _ref_cache[cite_key] is not None:
            reason = _ref_cache[cite_key]
            detail_map = {
                "not_found": "citation key not found in references",
                "doi_invalid": "DOI could not be validated",
                "metadata_mismatch": "reference metadata check failed",
            }
            flagged.append({
                "cite_key": cite_key,
                "reason": reason,
                "claim_id": claim_id if isinstance(claim_id, str) else None,
                "detail": detail_map.get(reason, reason),
            })
            continue

        # Fast path: already fully verified — entailment passed for this key before.
        if cite_key in seen_verified:
            continue

        # Fast path: already cleared the ref/doi/metadata checks.
        if cite_key in _ref_cache and _ref_cache[cite_key] is None:
            ref = ref_index.get(cite_key) or _fuzzy_match_ref(cite_key, ref_index)
        else:
            ref = ref_index.get(cite_key) or _fuzzy_match_ref(cite_key, ref_index)

            if not ref:
                _ref_cache[cite_key] = "not_found"
                flagged.append({
                    "cite_key": cite_key,
                    "reason": "not_found",
                    "claim_id": claim_id if isinstance(claim_id, str) else None,
                    "detail": "citation key not found in references",
                })
                continue

            doi = ref.get("doi")
            if isinstance(doi, str) and doi and not verify_doi(doi):
                _ref_cache[cite_key] = "doi_invalid"
                flagged.append({
                    "cite_key": cite_key,
                    "reason": "doi_invalid",
                    "claim_id": claim_id if isinstance(claim_id, str) else None,
                    "detail": f"DOI could not be validated: {doi}",
                })
                continue

            if not crosscheck_metadata(ref):
                _ref_cache[cite_key] = "metadata_mismatch"
                flagged.append({
                    "cite_key": cite_key,
                    "reason": "metadata_mismatch",
                    "claim_id": claim_id if isinstance(claim_id, str) else None,
                    "detail": "reference metadata check failed",
                })
                continue

            # Ref passed structural checks — mark as cleared for future uses.
            _ref_cache[cite_key] = None

        # Entailment is claim-specific so it cannot be cached across uses.
        passage = retrieve_passage(ref, claim)
        entailment = check_entailment(passage, claim)
        if entailment in {"neutral", "contradiction"}:
            flagged.append({
                "cite_key": cite_key,
                "reason": "unsupported_claim",
                "claim_id": claim_id if isinstance(claim_id, str) else None,
                "detail": f"entailment result={entailment}",
            })
            continue

        if cite_key not in seen_verified:
            seen_verified.add(cite_key)
            verified.append(cite_key)

    return verified, flagged

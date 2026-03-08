"""Citation ledger schema types."""

from __future__ import annotations

from typing import TypedDict


class CitationLedgerEntry(TypedDict):
    claim: str
    cited_source: str
    support_score: float
    evidence_excerpt: str
    confidence: str
    repair_status: str

"""Reference and citation schema types."""

from __future__ import annotations

from typing import TypedDict


class ReferenceRecord(TypedDict):
    ref_id: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    venue: str | None
    abstract: str | None
    source: str
    verified_metadata: bool


class CitationUse(TypedDict):
    cite_key: str
    claim_id: str
    chapter_id: str
    subtask_id: str


class FlaggedCitation(TypedDict):
    cite_key: str
    reason: str
    claim_id: str | None
    detail: str | None

"""Academic search service surface."""

from .providers import (
    AcademicSearchClient,
    _dedupe_references,
    _extract_xml_tag,
    _openalex_abstract,
    _reference_id,
)

__all__ = [
    "AcademicSearchClient",
    "_reference_id",
    "_openalex_abstract",
    "_extract_xml_tag",
    "_dedupe_references",
]

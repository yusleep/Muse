"""Public schema package for Muse state and references."""

from .chapter import ChapterResult
from .citation import CitationLedgerEntry
from .reference import CitationUse, FlaggedCitation, ReferenceRecord
from .run import (
    ThesisState,
    _DEFAULT_OPTIONAL_FIELDS,
    _REQUIRED_KEYS,
    hydrate_thesis_state,
    new_thesis_state,
    validate_thesis_state,
)

__all__ = [
    "CitationUse",
    "CitationLedgerEntry",
    "ChapterResult",
    "FlaggedCitation",
    "ReferenceRecord",
    "ThesisState",
    "_DEFAULT_OPTIONAL_FIELDS",
    "_REQUIRED_KEYS",
    "hydrate_thesis_state",
    "new_thesis_state",
    "validate_thesis_state",
]

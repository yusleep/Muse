"""Core package for the Muse runtime."""

from .audit import JsonlAuditSink, build_event
from .config import Settings, load_settings
from .citation import verify_all_citations
from .graph.helpers.review_state import apply_chapter_review, build_revision_instructions, should_iterate
from .planning import plan_subtasks
from .runtime import Runtime
from .schemas import hydrate_thesis_state, new_thesis_state, validate_thesis_state
from .store import RunStore

__all__ = [
    "JsonlAuditSink",
    "build_event",
    "Settings",
    "load_settings",
    "Runtime",
    "RunStore",
    "hydrate_thesis_state",
    "build_revision_instructions",
    "apply_chapter_review",
    "should_iterate",
    "verify_all_citations",
    "plan_subtasks",
    "new_thesis_state",
    "validate_thesis_state",
]

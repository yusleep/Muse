"""Core package for thesis writing agent v3 runtime."""

from .audit import JsonlAuditSink, build_event
from .chapter import apply_chapter_review, build_revision_instructions, should_iterate
from .config import Settings, load_settings
from .citation import verify_all_citations
from .engine import EngineContext, ThesisEngine
from .orchestrator import can_advance_to_stage, gate_export
from .planning import plan_subtasks
from .runtime import Runtime
from .schemas import hydrate_thesis_state, new_thesis_state, validate_thesis_state
from .store import RunStore

__all__ = [
    "JsonlAuditSink",
    "build_event",
    "Settings",
    "load_settings",
    "EngineContext",
    "ThesisEngine",
    "Runtime",
    "RunStore",
    "hydrate_thesis_state",
    "build_revision_instructions",
    "apply_chapter_review",
    "should_iterate",
    "verify_all_citations",
    "can_advance_to_stage",
    "gate_export",
    "plan_subtasks",
    "new_thesis_state",
    "validate_thesis_state",
]

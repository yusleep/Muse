"""Stable service-layer exports for Muse."""

from .audit import JsonlAuditSink, build_event
from .citation import verify_all_citations
from .citation_meta import CitationMetadataClient
from .http import HttpClient, ProviderError
from .latex import export_latex_project
from .paper_index import PaperIndexService
from .planning import plan_subtasks
from .providers import LLMClient
from .search import AcademicSearchClient
from .store import RunStore

__all__ = [
    "AcademicSearchClient",
    "CitationMetadataClient",
    "HttpClient",
    "JsonlAuditSink",
    "LLMClient",
    "PaperIndexService",
    "ProviderError",
    "RunStore",
    "build_event",
    "export_latex_project",
    "plan_subtasks",
    "verify_all_citations",
]

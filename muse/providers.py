"""Backward-compatible provider surface."""

from .services.citation_meta import CitationMetadataClient
from .services.http import HttpClient, ProviderError
from .services.providers import LLMClient
from .services.search import AcademicSearchClient

__all__ = [
    "HttpClient",
    "ProviderError",
    "LLMClient",
    "AcademicSearchClient",
    "CitationMetadataClient",
]

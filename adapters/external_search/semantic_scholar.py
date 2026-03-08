"""Semantic Scholar source adapter."""

from __future__ import annotations


class SemanticScholarSearchAdapter:
    def __init__(self, client) -> None:
        self.client = client

    def search(self, query: str, limit: int = 20) -> list[dict]:
        return self.client.search_semantic_scholar(query, limit=limit)

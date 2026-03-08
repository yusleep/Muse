"""arXiv source adapter."""

from __future__ import annotations


class ArxivSearchAdapter:
    def __init__(self, client) -> None:
        self.client = client

    def search(self, query: str, limit: int = 20) -> list[dict]:
        return self.client.search_arxiv(query, limit=limit)

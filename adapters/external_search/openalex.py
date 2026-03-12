"""OpenAlex source adapter."""

from __future__ import annotations


class OpenAlexSearchAdapter:
    def __init__(self, client) -> None:
        self.client = client

    def search(self, query: str, limit: int = 20) -> list[dict]:
        return self.client.search_openalex(query, limit=limit)

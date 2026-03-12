"""Optional LlamaIndex-backed retrieval adapter with graceful fallback."""

from __future__ import annotations

from typing import Any

from muse.adapters import RetrievalService


class LlamaIndexRetrievalAdapter(RetrievalService):
    def __init__(self, index: Any | None = None, documents: list[dict[str, Any]] | None = None) -> None:
        self.index = index
        self.documents = documents or []

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_lower = query.lower()
        if self.index is not None and hasattr(self.index, "retrieve"):
            results = self.index.retrieve(query, top_k=top_k)
            normalized = []
            for item in results:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    normalized.append({"text": str(item)})
            return normalized[:top_k]

        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in self.documents:
            haystack = " ".join(str(doc.get(key, "")) for key in ("title", "abstract", "full_text"))
            score = haystack.lower().count(query_lower)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

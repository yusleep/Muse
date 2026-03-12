"""Adapter protocols used by graph/runtime integrations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RetrievalService(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        ...


__all__ = ["RetrievalService"]

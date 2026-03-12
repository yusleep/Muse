"""Optional LlamaIndex ingestion adapter built on the existing refs loader."""

from __future__ import annotations

from muse.refs_loader import load_local_refs


class LlamaIndexIngestionAdapter:
    def load_directory(self, path: str) -> list[dict]:
        return load_local_refs(path)

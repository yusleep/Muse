"""RAG index for local reference files.

Builds a per-chunk retrieval index over the full_text of local refs.
Uses sentence-transformers (multilingual) when available, falls back to
BM25 (rank_bm25), or returns empty results if neither is installed.

The index is cached under {refs_dir}/.index/ and only rebuilt when source
file mtimes change.

Public API
----------
RagIndex.build(refs, refs_dir) -> RagIndex
RagIndex.retrieve(query, top_k=5)  -> list[{"ref_id", "text", "score"}]
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 300   # approximate word-count per chunk
_OVERLAP = 50       # words of overlap between consecutive chunks
_INDEX_DIR = ".index"
_EMBEDDINGS_FILE = "embeddings.npy"
_CHUNKS_FILE = "chunks.json"
_META_FILE = "index_meta.json"
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Module-level model cache so the heavy SentenceTransformer is loaded at most once.
_model_cache: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Return a cached SentenceTransformer instance, loading it on first use."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _OVERLAP) -> list[str]:
    """Split text into overlapping word-count-approximate chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# RagIndex
# ---------------------------------------------------------------------------

class RagIndex:
    """Embedding or BM25 retrieval index over chunked local reference texts."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embeddings: Any,        # np.ndarray | None
        use_embedding: bool,
    ) -> None:
        self._chunks = chunks
        self._embeddings = embeddings
        self._use_embedding = use_embedding
        self._bm25: Any = None
        if not use_embedding and chunks:
            try:
                self._bm25 = _build_bm25([c["text"] for c in chunks])
            except ImportError:
                pass  # rank_bm25 not installed; retrieve() returns []

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, refs: list[dict[str, Any]], refs_dir: str) -> "RagIndex":
        """Build or load a cached index from a list of refs.

        Caching: index_meta.json stores the mtime of each source file at
        build time. If all mtimes match the current files, cache is reused.
        """
        index_dir = Path(refs_dir) / _INDEX_DIR
        index_dir.mkdir(parents=True, exist_ok=True)

        meta_path = index_dir / _META_FILE
        chunks_path = index_dir / _CHUNKS_FILE

        if meta_path.exists() and chunks_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if _cache_valid(meta, refs):
                    return _load_cache(index_dir, meta)
            except Exception:  # noqa: BLE001
                pass  # fall through to rebuild

        return _build_fresh(refs, index_dir)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top_k chunks most relevant to query.

        Returns list of {"ref_id": str, "text": str, "score": float}.
        Returns [] if no index is available.
        """
        if not self._chunks:
            return []
        if self._use_embedding and self._embeddings is not None:
            return self._retrieve_embedding(query, top_k)
        if self._bm25 is not None:
            return self._retrieve_bm25(query, top_k)
        return []

    def _retrieve_embedding(self, query: str, top_k: int) -> list[dict[str, Any]]:
        import numpy as np
        model = _get_model(_MODEL_NAME)
        q_emb = model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"ref_id": self._chunks[i]["ref_id"], "text": self._chunks[i]["text"], "score": float(scores[i])}
            for i in top_idx if i < len(self._chunks)
        ]

    def _retrieve_bm25(self, query: str, top_k: int) -> list[dict[str, Any]]:
        import numpy as np
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"ref_id": self._chunks[i]["ref_id"], "text": self._chunks[i]["text"], "score": float(scores[i])}
            for i in top_idx if i < len(self._chunks)
        ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_bm25(texts: list[str]) -> Any:
    from rank_bm25 import BM25Okapi  # type: ignore
    return BM25Okapi([t.lower().split() for t in texts])


def _mtime(filepath: str | None) -> float:
    if not filepath or not os.path.exists(filepath):
        return 0.0
    return os.path.getmtime(filepath)


def _cache_valid(meta: dict[str, Any], refs: list[dict[str, Any]]) -> bool:
    stored: dict[str, float] = meta.get("source_mtimes", {})
    current = {
        (ref.get("filepath") or ref["ref_id"]): _mtime(ref.get("filepath"))
        for ref in refs
    }
    return stored == current


def _build_fresh(refs: list[dict[str, Any]], index_dir: Path) -> RagIndex:
    """Chunk all refs, embed if possible, persist cache, return RagIndex."""
    all_chunks: list[dict[str, Any]] = []
    for ref in refs:
        text = ref.get("full_text") or ref.get("abstract") or ""
        if not text.strip():
            continue
        for idx, chunk in enumerate(_chunk_text(text)):
            all_chunks.append({"ref_id": ref["ref_id"], "text": chunk, "chunk_idx": idx})

    embeddings = None
    use_embedding = False

    try:
        import numpy as np
        model = _get_model(_MODEL_NAME)
        if all_chunks:
            embeddings = model.encode(
                [c["text"] for c in all_chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            np.save(str(index_dir / _EMBEDDINGS_FILE), embeddings)
            use_embedding = True
    except ImportError:
        pass  # fall back to BM25

    (index_dir / _CHUNKS_FILE).write_text(
        json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "built_at": time.time(),
        "use_embedding": use_embedding,
        "source_mtimes": {
            (ref.get("filepath") or ref["ref_id"]): _mtime(ref.get("filepath"))
            for ref in refs
        },
    }
    (index_dir / _META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )

    return RagIndex(chunks=all_chunks, embeddings=embeddings, use_embedding=use_embedding)


def _load_cache(index_dir: Path, meta: dict[str, Any]) -> RagIndex:
    all_chunks = json.loads((index_dir / _CHUNKS_FILE).read_text(encoding="utf-8"))
    use_embedding = bool(meta.get("use_embedding", False))
    embeddings = None

    if use_embedding and (index_dir / _EMBEDDINGS_FILE).exists():
        try:
            import numpy as np
            embeddings = np.load(str(index_dir / _EMBEDDINGS_FILE))
        except Exception:  # noqa: BLE001
            use_embedding = False

    return RagIndex(chunks=all_chunks, embeddings=embeddings, use_embedding=use_embedding)

"""Tests for RagIndex — BM25 and cache paths (no GPU/sentence-transformers required)."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch


class TestRagIndexBm25(unittest.TestCase):
    """Test RagIndex using BM25 fallback (rank_bm25 is optional; skip if absent)."""

    def _make_refs(self, n: int = 3):
        return [
            {
                "ref_id": f"@ref{i}",
                "full_text": f"This is the full text of paper {i} about topic_{i} methodology.",
                "abstract": f"Abstract {i}",
                "filepath": None,
            }
            for i in range(n)
        ]

    def test_build_and_retrieve_returns_list(self):
        from muse.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            refs = self._make_refs()
            idx = RagIndex.build(refs, tmp)
            results = idx.retrieve("topic_1 methodology", top_k=2)
            self.assertIsInstance(results, list)
            # Each result has the expected keys
            for r in results:
                self.assertIn("ref_id", r)
                self.assertIn("text", r)
                self.assertIn("score", r)

    def test_empty_refs_returns_empty_list(self):
        from muse.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            idx = RagIndex.build([], tmp)
            self.assertEqual(idx.retrieve("anything"), [])

    def test_cache_is_reused(self):
        """Second build call must load from cache, not call _build_fresh again."""
        import muse.rag as rag_module
        from muse.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            refs = self._make_refs(2)
            # First build — creates cache files
            RagIndex.build(refs, tmp)
            cache_dir = os.path.join(tmp, ".index")
            self.assertTrue(os.path.exists(os.path.join(cache_dir, "chunks.json")))

            # Second build — _build_fresh must NOT be called (cache valid)
            with patch.object(rag_module, "_build_fresh", wraps=rag_module._build_fresh) as mock_bf:
                RagIndex.build(refs, tmp)
                mock_bf.assert_not_called()

    def test_cache_invalidated_on_content_change(self):
        """If source file mtimes change, cache should be rebuilt."""
        import muse.rag as rag_module
        from muse.rag import RagIndex, _cache_valid

        with tempfile.TemporaryDirectory() as tmp:
            # Write a file so we have a real filepath
            fpath = os.path.join(tmp, "paper.txt")
            with open(fpath, "w") as f:
                f.write("original content")

            refs = [
                {
                    "ref_id": "@ref0",
                    "full_text": "original content",
                    "filepath": fpath,
                }
            ]
            RagIndex.build(refs, tmp)
            meta_path = os.path.join(tmp, ".index", "index_meta.json")
            meta = json.loads(open(meta_path).read())

            # Simulate mtime change by altering stored meta
            meta["source_mtimes"][fpath] = 0.0
            self.assertFalse(_cache_valid(meta, refs))

            # Verify rebuild is triggered when mtime differs
            with patch.object(rag_module, "_build_fresh", wraps=rag_module._build_fresh) as mock_bf:
                # Patch _cache_valid to return False so build path is exercised
                with patch.object(rag_module, "_cache_valid", return_value=False):
                    RagIndex.build(refs, tmp)
                mock_bf.assert_called_once()

    def test_top_k_respected(self):
        from muse.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            refs = self._make_refs(10)
            idx = RagIndex.build(refs, tmp)
            results = idx.retrieve("methodology", top_k=3)
            self.assertLessEqual(len(results), 3)

    def test_use_embedding_false_when_chunks_empty(self):
        """use_embedding should be False when there are no chunks to embed."""
        from muse.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            # Refs with no text → no chunks → use_embedding must be False in meta
            refs = [{"ref_id": "@r0", "full_text": "", "filepath": None}]
            RagIndex.build(refs, tmp)
            meta_path = os.path.join(tmp, ".index", "index_meta.json")
            meta = json.loads(open(meta_path).read())
            self.assertFalse(meta["use_embedding"])


class TestRagIndexEmbedding(unittest.TestCase):
    """Test the embedding retrieval path using a fake model injected into _model_cache."""

    def test_retrieve_embedding_path(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")

        from muse.rag import RagIndex, _model_cache, _MODEL_NAME

        chunks = [
            {"ref_id": "@r0", "text": "deep learning neural network"},
            {"ref_id": "@r1", "text": "Byzantine fault tolerance consensus"},
        ]
        # 2-dim unit embeddings: r0 → [1, 0], r1 → [0, 1]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        class _FakeModel:
            def encode(self, texts, normalize_embeddings=True):
                # Query vector closest to r1
                return np.array([[0.01, 0.99]], dtype=np.float32)

        _model_cache[_MODEL_NAME] = _FakeModel()
        try:
            idx = RagIndex(chunks=chunks, embeddings=embeddings, use_embedding=True)
            results = idx.retrieve("Byzantine consensus", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["ref_id"], "@r1")
            self.assertIn("text", results[0])
            self.assertIn("score", results[0])
            self.assertAlmostEqual(results[0]["score"], 0.99, places=2)
        finally:
            _model_cache.pop(_MODEL_NAME, None)


class TestChunking(unittest.TestCase):
    def test_single_chunk_for_short_text(self):
        from muse.rag import _chunk_text

        chunks = _chunk_text("word " * 100)
        self.assertEqual(len(chunks), 1)

    def test_multiple_chunks_for_long_text(self):
        from muse.rag import _chunk_text

        chunks = _chunk_text("word " * 700, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_overlap_present(self):
        from muse.rag import _chunk_text

        words = [f"w{i}" for i in range(400)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=300, overlap=50)
        # The last words of chunk[0] should appear in the start of chunk[1]
        c0_words = set(chunks[0].split())
        c1_words = set(chunks[1].split())
        overlap_count = len(c0_words & c1_words)
        self.assertGreater(overlap_count, 0)

    def test_empty_text_returns_empty(self):
        from muse.rag import _chunk_text

        self.assertEqual(_chunk_text(""), [])
        self.assertEqual(_chunk_text("   "), [])

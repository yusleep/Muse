"""Tests for RagIndex — BM25 and cache paths (no GPU/sentence-transformers required)."""

import json
import os
import tempfile
import unittest


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
        from thesis_agent.rag import RagIndex

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
        from thesis_agent.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            idx = RagIndex.build([], tmp)
            self.assertEqual(idx.retrieve("anything"), [])

    def test_cache_is_reused(self):
        from thesis_agent.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            refs = self._make_refs(2)
            # First build — creates cache files
            idx1 = RagIndex.build(refs, tmp)
            cache_dir = os.path.join(tmp, ".index")
            self.assertTrue(os.path.exists(os.path.join(cache_dir, "chunks.json")))

            # Second build — should load from cache
            idx2 = RagIndex.build(refs, tmp)
            r1 = idx1.retrieve("topic_0", top_k=1)
            r2 = idx2.retrieve("topic_0", top_k=1)
            # Both should return the same ref_id for the same query
            if r1 and r2:
                self.assertEqual(r1[0]["ref_id"], r2[0]["ref_id"])

    def test_cache_invalidated_on_content_change(self):
        """If source file mtimes change, cache should be rebuilt."""
        from thesis_agent.rag import RagIndex, _cache_valid

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
            idx = RagIndex.build(refs, tmp)
            meta_path = os.path.join(tmp, ".index", "index_meta.json")
            meta = json.loads(open(meta_path).read())

            # Simulate mtime change by altering stored meta
            meta["source_mtimes"][fpath] = 0.0
            self.assertFalse(_cache_valid(meta, refs))

    def test_top_k_respected(self):
        from thesis_agent.rag import RagIndex

        with tempfile.TemporaryDirectory() as tmp:
            refs = self._make_refs(10)
            idx = RagIndex.build(refs, tmp)
            results = idx.retrieve("methodology", top_k=3)
            self.assertLessEqual(len(results), 3)


class TestChunking(unittest.TestCase):
    def test_single_chunk_for_short_text(self):
        from thesis_agent.rag import _chunk_text

        chunks = _chunk_text("word " * 100)
        self.assertEqual(len(chunks), 1)

    def test_multiple_chunks_for_long_text(self):
        from thesis_agent.rag import _chunk_text

        chunks = _chunk_text("word " * 700, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_overlap_present(self):
        from thesis_agent.rag import _chunk_text

        words = [f"w{i}" for i in range(400)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=300, overlap=50)
        # The last words of chunk[0] should appear in the start of chunk[1]
        c0_words = set(chunks[0].split())
        c1_words = set(chunks[1].split())
        overlap_count = len(c0_words & c1_words)
        self.assertGreater(overlap_count, 0)

    def test_empty_text_returns_empty(self):
        from thesis_agent.rag import _chunk_text

        self.assertEqual(_chunk_text(""), [])
        self.assertEqual(_chunk_text("   "), [])

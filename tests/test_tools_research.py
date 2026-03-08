"""Tests for muse/tools/research.py"""

from __future__ import annotations

import json
import unittest


class ResearchToolTests(unittest.TestCase):
    def test_academic_search_returns_json_list(self):
        from muse.tools.research import academic_search

        result_str = academic_search.invoke(
            {
                "query": "graph neural networks",
                "max_results": 3,
            }
        )
        result = json.loads(result_str)
        self.assertIsInstance(result, list)

    def test_retrieve_local_refs_without_index(self):
        from muse.tools.research import retrieve_local_refs

        result_str = retrieve_local_refs.invoke(
            {
                "query": "transformer architecture",
                "top_k": 5,
            }
        )
        result = json.loads(result_str)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_web_search_returns_string(self):
        from muse.tools.research import web_search

        result = web_search.invoke({"query": "LangGraph documentation"})
        self.assertIsInstance(result, str)

    def test_web_fetch_returns_string(self):
        from muse.tools.research import web_fetch

        result = web_fetch.invoke({"url": "https://example.com", "prompt": "summarize"})
        self.assertIsInstance(result, str)

    def test_read_pdf_returns_string(self):
        from muse.tools.research import read_pdf

        result = read_pdf.invoke({"file_path": "/nonexistent/file.pdf", "pages": "1-3"})
        self.assertIn("error", result.lower())

    def test_image_search_returns_string(self):
        from muse.tools.research import image_search

        result = image_search.invoke({"query": "neural network architecture diagram"})
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()

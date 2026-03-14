"""Tests for PaperIndexService."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class PaperIndexServiceTests(unittest.TestCase):
    def test_query_prefers_local_chunks_at_equal_similarity(self):
        from muse.services.paper_index import PaperIndexService

        with tempfile.TemporaryDirectory() as tmp:
            service = PaperIndexService(
                llamaparse_api_key="",
                cache_dir=Path(tmp) / "papers",
                index_dir=Path(tmp) / "index",
            )
            service._register_ingested_paper(
                {
                    "ref_id": "@local",
                    "paper_id": "local",
                    "paper_title": "Local Paper",
                    "source": "local",
                    "source_priority": 1,
                    "available_sections": ["Results"],
                },
                [
                    {
                        "paper_id": "local",
                        "ref_id": "@local",
                        "paper_title": "Local Paper",
                        "section_title": "Results",
                        "page_label": "1",
                        "text": "graph workflow durability results",
                        "source": "local",
                        "source_priority": 1,
                    }
                ],
            )
            service._register_ingested_paper(
                {
                    "ref_id": "@online",
                    "paper_id": "online",
                    "paper_title": "Online Paper",
                    "source": "online",
                    "source_priority": 2,
                    "available_sections": ["Results"],
                },
                [
                    {
                        "paper_id": "online",
                        "ref_id": "@online",
                        "paper_title": "Online Paper",
                        "section_title": "Results",
                        "page_label": "2",
                        "text": "graph workflow durability results",
                        "source": "online",
                        "source_priority": 2,
                    }
                ],
            )

            results = service.query("graph durability results", top_k=2)

        self.assertEqual(results[0]["ref_id"], "@local")
        self.assertEqual(results[1]["ref_id"], "@online")

    def test_get_section_filters_chunks_by_paper_and_section(self):
        from muse.services.paper_index import PaperIndexService

        with tempfile.TemporaryDirectory() as tmp:
            service = PaperIndexService(
                llamaparse_api_key="",
                cache_dir=Path(tmp) / "papers",
                index_dir=Path(tmp) / "index",
            )
            service._register_ingested_paper(
                {
                    "ref_id": "@paper",
                    "paper_id": "paper-1",
                    "paper_title": "Paper One",
                    "source": "local",
                    "source_priority": 1,
                    "available_sections": ["Method", "Results"],
                },
                [
                    {
                        "paper_id": "paper-1",
                        "ref_id": "@paper",
                        "paper_title": "Paper One",
                        "section_title": "Method",
                        "page_label": "3",
                        "text": "method details and setup",
                        "source": "local",
                        "source_priority": 1,
                    },
                    {
                        "paper_id": "paper-1",
                        "ref_id": "@paper",
                        "paper_title": "Paper One",
                        "section_title": "Results",
                        "page_label": "7",
                        "text": "results with durability gains",
                        "source": "local",
                        "source_priority": 1,
                    },
                ],
            )

            results = service.get_section(
                paper_id="paper-1",
                section_title="Results",
                query="durability gains",
                top_k=5,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["section_title"], "Results")


if __name__ == "__main__":
    unittest.main()

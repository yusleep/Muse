import os
import tempfile
import unittest
from unittest.mock import patch

from muse.config import Settings
from muse.graph.nodes.export import build_export_node
from muse.graph.nodes.search import build_search_node


class _FakeSearchClient:
    def search_multi_source(self, topic: str, discipline: str, extra_queries=None):
        return [
            {
                "ref_id": "@paper1",
                "title": "Paper 1",
                "authors": ["A"],
                "year": 2024,
                "doi": "10.1000/1",
                "venue": "Venue",
                "abstract": "Abstract",
                "source": "semantic_scholar",
                "verified_metadata": True,
            }
        ], [f"{topic} {discipline}"]


class _Services:
    def __init__(self):
        self.search = _FakeSearchClient()
        self.llm = None
        self.local_refs = []
        self.rag_index = None


class GraphEntryNodeTests(unittest.TestCase):
    def _make_settings(self, runs_dir: str) -> Settings:
        return Settings(
            llm_api_key="x",
            llm_base_url="http://localhost",
            llm_model="stub",
            model_router_config={},
            runs_dir=runs_dir,
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
            checkpoint_dir=None,
        )

    def test_search_node_populates_references_and_summary(self):
        node = build_search_node(self._make_settings("runs"), _Services())
        result = node({"topic": "topic", "discipline": "cs"})

        self.assertEqual(len(result["references"]), 1)
        self.assertTrue(result["search_queries"])
        self.assertIn("Paper 1", result["literature_summary"])

    def test_export_node_blocks_when_flagged_citations_are_contradictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = build_export_node(self._make_settings(tmp))
            result = node(
                {
                    "project_id": "run-blocked",
                    "paper_package": {"chapter_results": []},
                    "final_text": "text",
                    "flagged_citations": [
                        {
                            "cite_key": "@x",
                            "reason": "unsupported_claim",
                            "detail": "entailment result=contradiction",
                        }
                    ],
                    "references": [],
                    "citation_uses": [],
                    "output_format": "markdown",
                }
            )

            self.assertEqual(result["output_filepath"], "")

    def test_export_node_writes_markdown_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = build_export_node(self._make_settings(tmp))
            result = node(
                {
                    "project_id": "run-markdown",
                    "paper_package": {"chapter_results": []},
                    "final_text": "# Title\n\ncontent",
                    "flagged_citations": [],
                    "references": [],
                    "citation_uses": [],
                    "output_format": "markdown",
                }
            )

            self.assertTrue(os.path.exists(result["output_filepath"]))
            with open(result["output_filepath"], "r", encoding="utf-8") as handle:
                self.assertIn("content", handle.read())

    def test_export_node_writes_latex_project_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = build_export_node(self._make_settings(tmp))
            result = node(
                {
                    "project_id": "run-latex",
                    "paper_package": {"chapter_results": []},
                    "final_text": "# Title\n\ncontent",
                    "flagged_citations": [],
                    "references": [],
                    "citation_uses": [],
                    "output_format": "latex",
                }
            )

            self.assertTrue(os.path.isdir(result["output_filepath"]))
            self.assertTrue(os.path.exists(os.path.join(result["output_filepath"], "main.tex")))
            for dirname in ("Bib", "Chapter", "config", "resources"):
                self.assertTrue(os.path.isdir(os.path.join(result["output_filepath"], dirname)))

    def test_export_node_latex_export_records_archive_and_warning_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = build_export_node(self._make_settings(tmp))
            with patch("muse.latex_export.shutil.which", return_value=None):
                result = node(
                    {
                        "project_id": "run-latex-warning",
                        "paper_package": {"chapter_results": []},
                        "final_text": "# Title\n\ncontent",
                        "flagged_citations": [],
                        "references": [],
                        "citation_uses": [],
                        "output_format": "latex",
                    }
                )

            self.assertTrue(os.path.isdir(result["output_filepath"]))
            self.assertTrue(os.path.isfile(result["export_artifacts"]["latex_zip_path"]))
            self.assertIsNone(result["export_artifacts"]["pdf_path"])
            self.assertTrue(any("latexmk or xelatex" in warning for warning in result["export_warnings"]))

    def test_export_node_rejects_docx_export_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            node = build_export_node(self._make_settings(tmp))
            with patch("muse.graph.nodes.export._pandoc_export") as pandoc_export:
                with self.assertRaises(ValueError):
                    node(
                        {
                            "project_id": "run-docx",
                            "paper_package": {"chapter_results": []},
                            "final_text": "# Title\n\ncontent",
                            "flagged_citations": [],
                            "references": [],
                            "citation_uses": [],
                            "output_format": "docx",
                        }
                    )

            pandoc_export.assert_not_called()


if __name__ == "__main__":
    unittest.main()

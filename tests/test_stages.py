import os
import tempfile
import unittest
from unittest.mock import patch

from muse.schemas import new_thesis_state
from muse.stages import stage1_literature, stage6_export
from muse.store import RunStore


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


class StageTests(unittest.TestCase):
    def test_stage1_populates_references_and_waits_for_hitl(self):
        state = new_thesis_state(
            project_id="r1",
            topic="topic",
            discipline="cs",
            language="zh",
            format_standard="GB/T 7714-2015",
        )
        status = stage1_literature(state, _FakeSearchClient())

        self.assertEqual(status, "hitl")
        self.assertEqual(len(state["references"]), 1)
        self.assertTrue(state["search_queries"])

    def test_stage6_blocks_when_flagged_citations_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["flagged_citations"] = [{"cite_key": "@x", "reason": "unsupported_claim", "detail": "entailment result=contradiction"}]
            state["final_text"] = "text"

            status = stage6_export(state, store, run_id, output_format="markdown")
            self.assertEqual(status, "blocked")

    def test_stage6_writes_markdown_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["final_text"] = "# Title\n\ncontent"
            status = stage6_export(state, store, run_id, output_format="markdown")

            self.assertEqual(status, "done")
            output_path = state["output_filepath"]
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, "r", encoding="utf-8") as f:
                self.assertIn("content", f.read())

    def test_stage6_writes_latex_project_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["final_text"] = "# Title\n\ncontent"

            status = stage6_export(state, store, run_id, output_format="latex")

            self.assertEqual(status, "done")
            self.assertTrue(os.path.isdir(state["output_filepath"]))
            self.assertTrue(os.path.exists(os.path.join(state["output_filepath"], "main.tex")))
            for dirname in ("Bib", "Chapter", "config", "resources"):
                self.assertTrue(os.path.isdir(os.path.join(state["output_filepath"], dirname)))
            markdown_path = os.path.join(tmp, run_id, "output", "thesis.md")
            self.assertTrue(os.path.exists(markdown_path))

    def test_stage6_latex_export_records_archive_and_warning_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["final_text"] = "# Title\n\ncontent"

            with patch("muse.latex_export.shutil.which", return_value=None):
                status = stage6_export(state, store, run_id, output_format="latex")

            self.assertEqual(status, "done")
            self.assertTrue(os.path.isdir(state["output_filepath"]))
            self.assertTrue(os.path.isfile(state["export_artifacts"]["latex_zip_path"]))
            self.assertIsNone(state["export_artifacts"]["pdf_path"])
            self.assertTrue(any("latexmk or xelatex" in warning for warning in state["export_warnings"]))

    def test_stage6_rejects_docx_export_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            state["final_text"] = "# Title\n\ncontent"

            with patch("muse.stages._pandoc_export") as pandoc_export:
                with self.assertRaises(ValueError):
                    stage6_export(state, store, run_id, output_format="docx")

            pandoc_export.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import unittest

from muse.graph.helpers.review_state import build_revision_instructions
from muse.graph.state import MuseState
from muse.schemas import hydrate_thesis_state, new_thesis_state, validate_thesis_state


class ThesisStateSchemaTests(unittest.TestCase):
    def test_muse_state_exposes_global_review_fields(self):
        hints = MuseState.__annotations__

        self.assertIn("quality_scores", hints)
        self.assertIn("review_notes", hints)
        self.assertIn("review_history", hints)
        self.assertIn("review_iteration", hints)
        self.assertIn("structural_iterations", hints)
        self.assertIn("content_iterations", hints)
        self.assertIn("line_iterations", hints)
        self.assertIn("review_layer", hints)
        self.assertIn("revision_instructions", hints)

    def test_new_state_contains_required_keys(self):
        state = new_thesis_state(
            project_id="p1",
            topic="基于多智能体的论文写作",
            discipline="计算机科学",
            language="zh",
            format_standard="GB/T 7714-2015",
        )

        validate_thesis_state(state)

        self.assertEqual(state["project_id"], "p1")
        self.assertIsInstance(state["flagged_citations"], list)

    def test_flagged_citations_must_be_dict_entries(self):
        state = new_thesis_state(
            project_id="p2",
            topic="x",
            discipline="y",
            language="zh",
            format_standard="GB/T 7714-2015",
        )
        state["flagged_citations"] = ["@bad"]

        with self.assertRaises(ValueError):
            validate_thesis_state(state)

    def test_hydrate_backfills_missing_runtime_keys(self):
        partial_state = {
            "project_id": "p_legacy",
            "topic": "legacy",
            "discipline": "cs",
            "language": "zh",
            "format_standard": "GB/T 7714-2015",
            "current_stage": 2,
            "outline_json": {},
            "chapter_plans": [],
            "chapter_results": [],
            "references": [],
        }

        with self.assertRaises(ValueError):
            validate_thesis_state(dict(partial_state))

        hydrated = hydrate_thesis_state(dict(partial_state))

        self.assertIn("flagged_citations", hydrated)
        self.assertIn("stage6_status", hydrated)
        validate_thesis_state(hydrated)

    def test_hydrate_backfills_review_history_defaults(self):
        hydrated = hydrate_thesis_state(
            {
                "project_id": "p_legacy",
                "topic": "legacy",
                "discipline": "cs",
                "language": "zh",
                "format_standard": "GB/T 7714-2015",
                "current_stage": 2,
                "outline_json": {},
                "chapter_plans": [],
                "chapter_results": [],
                "references": [],
            }
        )

        self.assertEqual(hydrated["review_history"], [])
        self.assertEqual(hydrated["review_iteration"], 1)


class ChapterRevisionInstructionTests(unittest.TestCase):
    def test_build_revision_instructions_from_review_notes(self):
        review_notes = [
            {
                "subtask_id": "sub_01",
                "issue": "衔接不足",
                "instruction": "补充过渡段并减少跳跃。",
                "severity": 2,
            },
            {
                "subtask_id": "sub_02",
                "issue": "轻微措辞问题",
                "instruction": "改为更正式学术表达。",
                "severity": 1,
            },
        ]

        revisions = build_revision_instructions(review_notes, min_severity=2)

        self.assertEqual(list(revisions.keys()), ["sub_01"])
        self.assertIn("过渡段", revisions["sub_01"])

    def test_build_revision_instructions_merges_multiple_notes_for_same_subtask(self):
        review_notes = [
            {
                "subtask_id": "sub_01",
                "issue": "衔接不足",
                "instruction": "补充过渡段。",
                "severity": 3,
            },
            {
                "subtask_id": "sub_01",
                "issue": "论证太短",
                "instruction": "补充对比分析。",
                "severity": 2,
            },
        ]

        revisions = build_revision_instructions(review_notes, min_severity=2)

        self.assertIn("- 补充过渡段。", revisions["sub_01"])
        self.assertIn("- 补充对比分析。", revisions["sub_01"])


if __name__ == "__main__":
    unittest.main()

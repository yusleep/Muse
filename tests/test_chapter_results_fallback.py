from __future__ import annotations

import unittest


class ChapterResultsFallbackTests(unittest.TestCase):
    def test_export_chapter_results_falls_back_to_chapters_when_lists_are_empty(self):
        from muse.graph.nodes.export import _chapter_results_from_state

        chapter_1 = {"chapter_id": "ch_01", "chapter_title": "绪论"}
        chapter_2 = {"chapter_id": "ch_02", "chapter_title": "系统设计"}
        state = {
            "chapter_results": [],
            "paper_package": {"chapter_results": []},
            "chapters": {"ch_01": chapter_1, "ch_02": chapter_2},
            "chapter_plans": [{"chapter_id": "ch_01"}, {"chapter_id": "ch_02"}],
        }

        self.assertEqual(_chapter_results_from_state(state), [chapter_1, chapter_2])

    def test_polish_chapter_results_falls_back_to_chapters_when_lists_are_empty(self):
        from muse.graph.nodes.polish import _chapter_results_from_state

        chapter_1 = {"chapter_id": "ch_01", "chapter_title": "绪论"}
        chapter_2 = {"chapter_id": "ch_02", "chapter_title": "系统设计"}
        state = {
            "chapter_results": [],
            "paper_package": {"chapter_results": []},
            "chapters": {"ch_01": chapter_1, "ch_02": chapter_2},
            "chapter_plans": [{"chapter_id": "ch_01"}, {"chapter_id": "ch_02"}],
        }

        self.assertEqual(_chapter_results_from_state(state), [chapter_1, chapter_2])


if __name__ == "__main__":
    unittest.main()

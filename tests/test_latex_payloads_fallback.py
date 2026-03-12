from __future__ import annotations

import unittest


class LatexChapterPayloadFallbackTests(unittest.TestCase):
    def test_chapter_payloads_falls_back_to_chapters_when_chapter_results_missing(self):
        from muse.services.latex import _chapter_payloads

        state = {
            "chapters": {
                "ch_01": {
                    "chapter_id": "ch_01",
                    "chapter_title": "绪论",
                    "merged_text": "绪论内容。",
                }
            },
            "chapter_plans": [{"chapter_id": "ch_01", "chapter_title": "绪论"}],
            "final_text": "",
        }

        self.assertEqual(_chapter_payloads(state), [("绪论", "绪论内容。")])


if __name__ == "__main__":
    unittest.main()

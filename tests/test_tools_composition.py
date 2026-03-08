"""Tests for muse/tools/composition.py."""

from __future__ import annotations

import json
import unittest


class CompositionToolTests(unittest.TestCase):
    def test_check_terminology_returns_json(self):
        from muse.tools.composition import check_terminology

        result_str = check_terminology.invoke(
            {
                "text": "We use deep learning and DL interchangeably. The neural net processes data.",
            }
        )
        result = json.loads(result_str)
        self.assertIn("issues", result)
        self.assertIsInstance(result["issues"], list)

    def test_align_cross_refs_returns_json(self):
        from muse.tools.composition import align_cross_refs

        result_str = align_cross_refs.invoke(
            {
                "text": "As shown in Figure 1 and discussed in Section 2.3, the results in Table 5 confirm our hypothesis.",
            }
        )
        result = json.loads(result_str)
        self.assertIn("cross_refs_found", result)

    def test_check_transitions_returns_json(self):
        from muse.tools.composition import check_transitions

        result_str = check_transitions.invoke(
            {
                "chapter_texts_json": '[{"chapter_id": "ch1", "ending": "In summary, the method works."}, {"chapter_id": "ch2", "opening": "This chapter explores results."}]',
            }
        )
        result = json.loads(result_str)
        self.assertIn("transitions", result)

    def test_rewrite_passage_returns_text(self):
        from muse.tools.composition import rewrite_passage

        result = rewrite_passage.invoke(
            {
                "passage": "The thing works good because of reasons.",
                "instruction": "Improve academic tone and specificity.",
                "context": "Methods section of a CS thesis.",
            }
        )
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()

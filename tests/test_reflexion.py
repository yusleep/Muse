import unittest


class ReflexionTests(unittest.TestCase):
    def test_low_review_score_routes_to_revise_then_done(self):
        from muse.graph.subgraphs.chapter import _chapter_route

        self.assertEqual(_chapter_route({"quality_scores": {"coherence": 3}, "iteration": 1, "max_iterations": 3}), "revise")
        self.assertEqual(_chapter_route({"quality_scores": {"coherence": 4}, "iteration": 1, "max_iterations": 3}), "done")
        self.assertEqual(_chapter_route({"quality_scores": {"coherence": 3}, "iteration": 3, "max_iterations": 3}), "done")


if __name__ == "__main__":
    unittest.main()

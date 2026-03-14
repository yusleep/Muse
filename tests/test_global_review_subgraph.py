from __future__ import annotations

import unittest


class LayeredReviewSubgraphTests(unittest.TestCase):
    def test_layer_route_advances_after_max_iterations(self):
        from muse.graph.subgraphs.review import _layer_route

        route = _layer_route("structural")
        decision = route(
            {
                "quality_scores": {"logic": 2, "structure": 2, "balance": 2},
                "structural_iterations": 2,
            }
        )

        self.assertEqual(decision, "next_layer")

    def test_layered_review_graph_retries_structural_layer_before_advancing(self):
        from muse.graph.subgraphs.review import build_global_review_graph

        class _RetryLLM:
            def __init__(self):
                self.calls = []
                self.structural_reviews = 0

            def structured(self, *, system, user, route="default", max_tokens=2500):
                del user, max_tokens
                self.calls.append((system, route))
                if "Revise the full merged thesis draft for the structural layer" in system:
                    return {"final_text": "Revised structural draft."}
                if "only focus on the structural layer" in system:
                    self.structural_reviews += 1
                    if self.structural_reviews == 1:
                        return {
                            "scores": {"logic": 2, "structure": 2, "balance": 2},
                            "review_notes": [
                                {
                                    "section": "Introduction",
                                    "severity": 4,
                                    "instruction": "Reorder the opening paragraphs.",
                                    "lens": "structural",
                                }
                            ],
                        }
                    return {"scores": {"logic": 4, "structure": 4, "balance": 4}, "review_notes": []}
                if "only focus on the content layer" in system:
                    return {"scores": {"citation": 4, "coverage": 4, "depth": 4}, "review_notes": []}
                if "only focus on the line layer" in system:
                    return {
                        "scores": {"style": 5, "term_consistency": 5, "redundancy": 5},
                        "review_notes": [],
                    }
                raise AssertionError(system)

        services = type("_Services", (), {"llm": _RetryLLM()})()
        graph = build_global_review_graph(services=services)

        result = graph.invoke({"final_text": "Original draft."})

        self.assertEqual(result["final_text"], "Revised structural draft.")
        self.assertEqual(result["structural_iterations"], 2)
        self.assertEqual(result["content_iterations"], 1)
        self.assertEqual(result["line_iterations"], 1)
        self.assertIn("only focus on the structural layer", services.llm.calls[0][0])
        self.assertEqual(services.llm.calls[0][1], "review_structural")
        self.assertIn("Revise the full merged thesis draft for the structural layer", services.llm.calls[1][0])
        self.assertEqual(services.llm.calls[1][1], "writing_revision")
        self.assertIn("only focus on the structural layer", services.llm.calls[2][0])
        self.assertEqual(services.llm.calls[2][1], "review_structural")
        self.assertIn("only focus on the content layer", services.llm.calls[3][0])
        self.assertEqual(services.llm.calls[3][1], "review")

    def test_layered_review_graph_can_pass_all_layers_without_revision(self):
        from muse.graph.subgraphs.review import build_global_review_graph

        class _PassingLLM:
            def __init__(self):
                self.calls = []

            def structured(self, *, system, user, route="default", max_tokens=2500):
                del user, max_tokens
                self.calls.append((system, route))
                if "only focus on the structural layer" in system:
                    return {"scores": {"logic": 4, "structure": 4, "balance": 4}, "review_notes": []}
                if "only focus on the content layer" in system:
                    return {"scores": {"citation": 4, "coverage": 4, "depth": 4}, "review_notes": []}
                if "only focus on the line layer" in system:
                    return {
                        "scores": {"style": 5, "term_consistency": 5, "redundancy": 5},
                        "review_notes": [],
                    }
                raise AssertionError(system)

        services = type("_Services", (), {"llm": _PassingLLM()})()
        graph = build_global_review_graph(services=services)

        result = graph.invoke({"final_text": "Original draft."})

        self.assertEqual(result["final_text"], "Original draft.")
        self.assertEqual(result["structural_iterations"], 1)
        self.assertEqual(result["content_iterations"], 1)
        self.assertEqual(result["line_iterations"], 1)
        self.assertEqual(len(services.llm.calls), 3)
        self.assertEqual(
            [route for _, route in services.llm.calls],
            ["review_structural", "review", "review_line"],
        )


if __name__ == "__main__":
    unittest.main()

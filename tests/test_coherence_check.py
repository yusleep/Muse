import unittest


class CoherenceCheckNodeTests(unittest.TestCase):
    def test_skips_short_draft_without_llm_call(self):
        from muse.graph.nodes.coherence_check import build_coherence_check_node

        class _LLM:
            def structured(self, **kwargs):
                raise AssertionError("LLM should not be called for short drafts")

        services = type("_Services", (), {"llm": _LLM()})()
        node = build_coherence_check_node(services=services)

        result = node({"final_text": "too short"})

        self.assertEqual(result, {})

    def test_low_coherence_score_injects_review_notes(self):
        from muse.graph.nodes.coherence_check import build_coherence_check_node

        class _LLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, max_tokens
                self.route = route
                return {
                    "coherence_score": 2,
                    "issues": [
                        {
                            "location": "ch2.s1->ch2.s2",
                            "type": "missing_transition",
                            "description": "Transition is abrupt.",
                            "fix_suggestion": "Add a bridge paragraph.",
                        }
                    ],
                }

        llm = _LLM()
        services = type("_Services", (), {"llm": llm})()
        node = build_coherence_check_node(services=services)

        result = node({"final_text": "word " * 600})

        self.assertEqual(llm.route, "review")
        self.assertEqual(result["coherence_issues"][0]["type"], "missing_transition")
        self.assertEqual(result["review_notes"][0]["lens"], "coherence")
        self.assertEqual(result["review_notes"][0]["severity"], 4)
        self.assertIn("[连贯性]", result["review_notes"][0]["instruction"])

    def test_non_blocking_score_only_records_issues(self):
        from muse.graph.nodes.coherence_check import build_coherence_check_node

        class _LLM:
            def structured(self, *, system, user, route="default", max_tokens=2500):
                del system, user, route, max_tokens
                return {
                    "coherence_score": 4,
                    "issues": [
                        {
                            "location": "ch3.s2",
                            "type": "unsupported_claim",
                            "description": "The claim lacks evidence.",
                            "fix_suggestion": "Add a citation.",
                        }
                    ],
                }

        services = type("_Services", (), {"llm": _LLM()})()
        node = build_coherence_check_node(services=services)

        result = node({"final_text": "word " * 600})

        self.assertEqual(len(result["coherence_issues"]), 1)
        self.assertNotIn("review_notes", result)


if __name__ == "__main__":
    unittest.main()

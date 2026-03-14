import unittest


class CitationRepairNodeTests(unittest.TestCase):
    def test_remove_citations_handles_bracket_and_latex_formats(self):
        from muse.graph.nodes.citation_repair import _remove_citations

        text = (
            "Supportive claim [@smith2024] remains. "
            "Another sentence cites \\cite{@smith2024,@jones2023}. "
            "Safe cite \\cite{@keep2024} stays."
        )

        cleaned = _remove_citations(text, {"@smith2024", "@jones2023"})

        self.assertNotIn("[@smith2024]", cleaned)
        self.assertNotIn("@smith2024", cleaned)
        self.assertNotIn("@jones2023", cleaned)
        self.assertIn("\\cite{@keep2024}", cleaned)

    def test_citation_repair_removes_flagged_keys_from_text_and_uses(self):
        from muse.graph.nodes.citation_repair import build_citation_repair_node

        node = build_citation_repair_node()
        result = node(
            {
                "flagged_citations": [
                    {"cite_key": "@bad2024", "reason": "metadata_mismatch"},
                ],
                "chapters": {
                    "ch_01": {
                        "chapter_id": "ch_01",
                        "merged_text": "Claim with [@bad2024] and \\cite{@keep2024}.",
                    }
                },
                "final_text": "Combined draft [@bad2024] and \\cite{@keep2024}.",
                "citation_uses": [
                    {"cite_key": "@bad2024", "claim_id": "c1"},
                    {"cite_key": "@keep2024", "claim_id": "c2"},
                ],
            }
        )

        self.assertTrue(result["citation_repair_attempted"])
        self.assertNotIn("@bad2024", result["chapters"]["ch_01"]["merged_text"])
        self.assertNotIn("@bad2024", result["final_text"])
        self.assertEqual(result["citation_uses"], [{"cite_key": "@keep2024", "claim_id": "c2"}])

    def test_citation_repair_marks_attempt_even_when_no_flagged_entries(self):
        from muse.graph.nodes.citation_repair import build_citation_repair_node

        node = build_citation_repair_node()

        result = node({"flagged_citations": []})

        self.assertEqual(result, {"citation_repair_attempted": True})


class CitationQualityRouteTests(unittest.TestCase):
    def test_routes_to_repair_when_flagged_ratio_exceeds_threshold(self):
        from muse.graph.main_graph import _citation_quality_route

        route = _citation_quality_route(
            {
                "verified_citations": ["@a", "@b", "@c"],
                "flagged_citations": [{"cite_key": "@x"}],
                            }
        )

        self.assertEqual(route, "citation_repair")

    def test_routes_to_polish_when_repair_already_attempted(self):
        from muse.graph.main_graph import _citation_quality_route

        route = _citation_quality_route(
            {
                "verified_citations": ["@a"],
                "flagged_citations": [{"cite_key": "@x"}],
                "citation_repair_attempted": True,
            }
        )

        self.assertEqual(route, "polish")

    def test_routes_to_polish_when_no_citations_exist(self):
        from muse.graph.main_graph import _citation_quality_route

        route = _citation_quality_route(
            {
                "verified_citations": [],
                "flagged_citations": [],
            }
        )

        self.assertEqual(route, "polish")


if __name__ == "__main__":
    unittest.main()

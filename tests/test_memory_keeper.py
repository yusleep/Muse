import unittest


class ConsistencyStoreTests(unittest.TestCase):
    def test_update_from_chapter_collects_glossary_counts_and_summary(self):
        from muse.graph.helpers.memory_keeper import ConsistencyStore

        store = ConsistencyStore()
        store.update_from_chapter(
            {
                "chapter_id": "ch_01",
                "merged_text": "This chapter defines the agent architecture in detail.",
                "subtask_results": [
                    {
                        "glossary_additions": {"Agent Runtime": "智能体运行时"},
                        "citations_used": ["@smith2024", "@smith2024", "@jones2023"],
                    }
                ],
            }
        )

        payload = store.get_context_for_draft()

        self.assertEqual(payload["glossary"]["Agent Runtime"], "智能体运行时")
        self.assertEqual(payload["citation_counts"]["@smith2024"], 2)
        self.assertEqual(payload["frequently_cited"][0]["ref_id"], "@smith2024")
        self.assertIn("agent architecture", payload["chapter_summaries"]["ch_01"])

    def test_round_trip_preserves_store_content(self):
        from muse.graph.helpers.memory_keeper import ConsistencyStore

        store = ConsistencyStore()
        store.glossary["Latency"] = "延迟"
        store.citation_counts["@smith2024"] = 3
        store.chapter_summaries["ch_01"] = "Summary"

        restored = ConsistencyStore.from_dict(store.to_dict())

        self.assertEqual(restored.glossary, store.glossary)
        self.assertEqual(restored.citation_counts, store.citation_counts)
        self.assertEqual(restored.chapter_summaries, store.chapter_summaries)


if __name__ == "__main__":
    unittest.main()

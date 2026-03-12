"""Tests for academic multi-source search behavior."""

from __future__ import annotations

import unittest


class AcademicSearchClientTests(unittest.TestCase):
    def test_rate_limited_source_is_circuit_broken_for_remaining_queries(self):
        from muse.services.http import HttpClient, ProviderError
        from muse.services.providers import AcademicSearchClient

        client = AcademicSearchClient(http=HttpClient(timeout_seconds=1))
        semantic_calls: list[str] = []
        openalex_calls: list[str] = []
        arxiv_calls: list[str] = []

        def semantic_scholar(*, query: str, limit: int = 10):
            del limit
            semantic_calls.append(query)
            raise ProviderError("HTTP 429: Too Many Requests")

        def openalex(*, query: str, limit: int = 10):
            del limit
            openalex_calls.append(query)
            return [{"ref_id": f"@oa_{query}", "title": f"OA {query}"}]

        def arxiv(*, query: str, limit: int = 8):
            del limit
            arxiv_calls.append(query)
            return [{"ref_id": f"@ax_{query}", "title": f"AX {query}"}]

        client.search_semantic_scholar = semantic_scholar
        client.search_openalex = openalex
        client.search_arxiv = arxiv

        results, queries = client.search_multi_source(
            topic="topic",
            discipline="discipline",
            extra_queries=["q1", "q2", "q3"],
        )

        self.assertEqual(queries, ["q1", "q2", "q3"])
        self.assertEqual(semantic_calls, ["q1"])
        self.assertEqual(openalex_calls, ["q1", "q2", "q3"])
        self.assertEqual(arxiv_calls, ["q1", "q2", "q3"])
        self.assertEqual(
            {item["ref_id"] for item in results},
            {"@oa_q1", "@oa_q2", "@oa_q3", "@ax_q1", "@ax_q2", "@ax_q3"},
        )

    def test_openalex_tolerates_null_nested_objects(self):
        from muse.services.http import HttpClient
        from muse.services.providers import AcademicSearchClient

        class _StubHttp(HttpClient):
            def get_json(self, url: str, headers=None):
                del url, headers
                return {
                    "results": [
                        {
                            "title": "Null-safe OpenAlex record",
                            "publication_year": 2024,
                            "doi": None,
                            "authorships": [{"author": None}, None],
                            "primary_location": None,
                            "abstract_inverted_index": None,
                        }
                    ]
                }

        client = AcademicSearchClient(http=_StubHttp(timeout_seconds=1))

        results = client.search_openalex("react citation", limit=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Null-safe OpenAlex record")
        self.assertEqual(results[0]["authors"], [])
        self.assertIsNone(results[0]["venue"])


if __name__ == "__main__":
    unittest.main()

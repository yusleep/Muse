from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock


class ConnectivityCheckTests(unittest.TestCase):
    def test_check_skips_semantic_scholar_when_unconfigured(self):
        from muse.config import Settings
        from muse.runtime import Runtime

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                llm_api_key="test-key",
                llm_base_url="https://example.invalid/v1",
                llm_model="test-model",
                model_router_config={},
                runs_dir=tmp,
                semantic_scholar_api_key=None,
                openalex_email=None,
                crossref_mailto=None,
                refs_dir=None,
                checkpoint_dir=None,
            )
            runtime = Runtime(settings)

            runtime.llm.text = Mock(return_value="ok")
            runtime.search.search_semantic_scholar = Mock(side_effect=AssertionError("semantic_scholar should be skipped"))
            runtime.search.search_openalex = Mock(return_value=[])
            runtime.metadata.verify_doi = Mock(return_value=True)

            result = runtime.connectivity_check()

        self.assertTrue(result["llm"])
        self.assertIsNone(result["semantic_scholar"])
        self.assertTrue(result["openalex"])
        self.assertTrue(result["crossref"])
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()

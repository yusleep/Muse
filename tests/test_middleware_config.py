from __future__ import annotations

import unittest


class MiddlewareConfigTests(unittest.TestCase):
    def test_settings_has_middleware_fields(self):
        from muse.config import Settings

        settings = Settings(
            llm_api_key="k",
            llm_base_url="http://localhost",
            llm_model="m",
            model_router_config={},
            runs_dir="runs",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
            checkpoint_dir=None,
            middleware_retry_max=3,
            middleware_retry_delay=2.0,
            middleware_compaction_threshold=0.85,
            middleware_compaction_recent_tokens=15_000,
            middleware_context_window=200_000,
        )
        self.assertEqual(settings.middleware_retry_max, 3)
        self.assertEqual(settings.middleware_retry_delay, 2.0)
        self.assertEqual(settings.middleware_compaction_threshold, 0.85)
        self.assertEqual(settings.middleware_compaction_recent_tokens, 15_000)
        self.assertEqual(settings.middleware_context_window, 200_000)

    def test_settings_middleware_defaults(self):
        from muse.config import Settings

        settings = Settings(
            llm_api_key="k",
            llm_base_url="http://localhost",
            llm_model="m",
            model_router_config={},
            runs_dir="runs",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
        )
        self.assertEqual(settings.middleware_retry_max, 2)
        self.assertEqual(settings.middleware_retry_delay, 5.0)
        self.assertAlmostEqual(settings.middleware_compaction_threshold, 0.9)
        self.assertEqual(settings.middleware_compaction_recent_tokens, 20_000)
        self.assertEqual(settings.middleware_context_window, 128_000)

    def test_load_settings_reads_middleware_env_vars(self):
        from muse.config import load_settings

        env = {
            "MUSE_LLM_API_KEY": "key",
            "MUSE_LLM_MODEL": "gpt-4",
            "MUSE_MIDDLEWARE_RETRY_MAX": "4",
            "MUSE_MIDDLEWARE_RETRY_DELAY": "3.5",
            "MUSE_MIDDLEWARE_COMPACTION_THRESHOLD": "0.8",
            "MUSE_MIDDLEWARE_COMPACTION_RECENT_TOKENS": "10000",
            "MUSE_MIDDLEWARE_CONTEXT_WINDOW": "64000",
        }
        settings = load_settings(env)
        self.assertEqual(settings.middleware_retry_max, 4)
        self.assertAlmostEqual(settings.middleware_retry_delay, 3.5)
        self.assertAlmostEqual(settings.middleware_compaction_threshold, 0.8)
        self.assertEqual(settings.middleware_compaction_recent_tokens, 10_000)
        self.assertEqual(settings.middleware_context_window, 64_000)

    def test_load_settings_middleware_defaults_when_env_absent(self):
        from muse.config import load_settings

        env = {"MUSE_LLM_API_KEY": "key", "MUSE_LLM_MODEL": "gpt-4"}
        settings = load_settings(env)
        self.assertEqual(settings.middleware_retry_max, 2)
        self.assertAlmostEqual(settings.middleware_compaction_threshold, 0.9)

    def test_existing_settings_construction_unchanged(self):
        from muse.config import Settings

        settings = Settings(
            llm_api_key="k",
            llm_base_url="u",
            llm_model="m",
            model_router_config={},
            runs_dir="r",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
            checkpoint_dir=None,
        )
        self.assertEqual(settings.llm_api_key, "k")
        self.assertEqual(settings.middleware_retry_max, 2)


if __name__ == "__main__":
    unittest.main()

"""Tests for the model factory."""

from __future__ import annotations

import unittest

from muse.config import Settings


def _make_settings() -> Settings:
    return Settings(
        llm_api_key="test-key",
        llm_base_url="http://localhost:11434/v1",
        llm_model="test-model",
        model_router_config={},
        runs_dir="runs",
        semantic_scholar_api_key=None,
        openalex_email=None,
        crossref_mailto=None,
        refs_dir=None,
        checkpoint_dir=None,
    )


class ModelFactoryTests(unittest.TestCase):
    def test_create_chat_model_returns_muse_chat_model(self):
        from muse.models.adapter import MuseChatModel
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings(), route="default")
        self.assertIsInstance(model, MuseChatModel)

    def test_create_chat_model_uses_specified_route(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings(), route="writing")
        self.assertEqual(model.route, "writing")

    def test_create_chat_model_default_route(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings())
        self.assertEqual(model.route, "default")

    def test_create_chat_model_custom_temperature(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings(), route="default", temperature=0.7)
        self.assertAlmostEqual(model.temperature, 0.7)

    def test_create_chat_model_custom_max_tokens(self):
        from muse.models.factory import create_chat_model

        model = create_chat_model(_make_settings(), route="default", max_tokens=4000)
        self.assertEqual(model.max_tokens, 4000)

    def test_create_chat_model_with_router_config(self):
        from muse.models.adapter import MuseChatModel
        from muse.models.factory import create_chat_model

        settings = Settings(
            llm_api_key="key",
            llm_base_url="http://localhost/v1",
            llm_model="router/default",
            model_router_config={
                "providers": {
                    "local": {
                        "baseUrl": "http://localhost:11434/v1",
                        "auth": "default",
                        "models": {
                            "local/llama3": {"model": "llama3", "params": {}},
                        },
                    }
                },
                "models": {
                    "default": {"primary": "local/llama3", "fallbacks": []},
                    "writing": {"primary": "local/llama3", "fallbacks": []},
                },
            },
            runs_dir="runs",
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
        )
        model = create_chat_model(settings, route="writing")
        self.assertIsInstance(model, MuseChatModel)
        self.assertEqual(model.route, "writing")


if __name__ == "__main__":
    unittest.main()

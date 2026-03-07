import os
import tempfile
import unittest

from muse.config import Settings, load_settings
from muse.store import RunStore


class SettingsTests(unittest.TestCase):
    def test_load_settings_requires_llm_api_key(self):
        env = {
            "MUSE_LLM_MODEL": "gpt-4.1-mini",
        }
        with self.assertRaises(ValueError):
            load_settings(env)

    def test_load_settings_does_not_accept_legacy_prefix(self):
        env = {
            "THESIS_AGENT_LLM_API_KEY": "legacy-key",
            "THESIS_AGENT_LLM_MODEL": "gpt-4.1-mini",
        }
        with self.assertRaises(ValueError):
            load_settings(env)

    def test_load_settings_success(self):
        env = {
            "MUSE_LLM_API_KEY": "test-key",
            "MUSE_LLM_MODEL": "gpt-4.1-mini",
            "MUSE_RUNS_DIR": "/tmp/runs",
        }

        settings = load_settings(env)

        self.assertIsInstance(settings, Settings)
        self.assertEqual(settings.llm_api_key, "test-key")
        self.assertEqual(settings.llm_model, "gpt-4.1-mini")

    def test_load_settings_with_router_json_without_legacy_model_vars(self):
        env = {
            "MUSE_MODEL_ROUTER_JSON": """
            {
              "auth": {"profiles": {"openai": {"apiKey": "abc"}}},
              "providers": {"openai": {"baseUrl": "https://api.openai.com/v1", "auth": "openai"}},
              "models": {"writing": {"primary": "openai/gpt-4.1-mini", "fallbacks": []}}
            }
            """,
            "MUSE_RUNS_DIR": "/tmp/runs",
        }

        settings = load_settings(env)

        self.assertEqual(settings.llm_model, "openai/gpt-4.1-mini")
        self.assertIn("models", settings.model_router_config)


class RunStoreTests(unittest.TestCase):
    def test_create_run_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="test topic")
            state = {"project_id": run_id, "current_stage": 2, "topic": "test topic"}

            store.save_state(run_id, state)
            loaded = store.load_state(run_id)

            self.assertEqual(loaded["current_stage"], 2)
            self.assertEqual(loaded["topic"], "test topic")

    def test_append_hitl_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="test topic")
            store.append_hitl_feedback(run_id, {"stage": 1, "approved": True})

            feedback = store.load_hitl_feedback(run_id)
            self.assertEqual(len(feedback), 1)
            self.assertTrue(feedback[0]["approved"])


if __name__ == "__main__":
    unittest.main()

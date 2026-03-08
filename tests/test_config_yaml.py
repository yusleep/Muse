"""Tests for YAML config loading in muse.config."""

import os
import tempfile
import textwrap
import unittest

from muse.config import (
    Settings,
    _resolve_env_vars,
    _snake_to_camel_dict,
    _yaml_to_router_config,
    _yaml_to_settings,
    load_settings,
)


class TestResolveEnvVars(unittest.TestCase):
    def test_dollar_brace_syntax(self):
        result = _resolve_env_vars("key=${MY_VAR}", {"MY_VAR": "hello"})
        self.assertEqual(result, "key=hello")

    def test_bare_dollar_syntax(self):
        result = _resolve_env_vars("key=$MY_VAR", {"MY_VAR": "hello"})
        self.assertEqual(result, "key=hello")

    def test_missing_var_resolves_empty(self):
        result = _resolve_env_vars("key=${MISSING}", {})
        self.assertEqual(result, "key=")

    def test_nested_dict_and_list(self):
        obj = {"a": "${X}", "b": ["${Y}", 42]}
        result = _resolve_env_vars(obj, {"X": "1", "Y": "2"})
        self.assertEqual(result, {"a": "1", "b": ["2", 42]})

    def test_non_string_passthrough(self):
        self.assertEqual(_resolve_env_vars(42, {}), 42)
        self.assertTrue(_resolve_env_vars(True, {}))


class TestSnakeToCamel(unittest.TestCase):
    def test_known_keys_translated(self):
        d = {"api_key_env": "FOO", "base_url": "http://x", "unknown_key": 1}
        result = _snake_to_camel_dict(d)
        self.assertEqual(result["apiKeyEnv"], "FOO")
        self.assertEqual(result["baseUrl"], "http://x")
        self.assertEqual(result["unknown_key"], 1)

    def test_camelcase_passthrough(self):
        d = {"apiKeyEnv": "FOO", "baseUrl": "http://x"}
        result = _snake_to_camel_dict(d)
        self.assertEqual(result["apiKeyEnv"], "FOO")
        self.assertEqual(result["baseUrl"], "http://x")

    def test_recursive_translation(self):
        d = {"profiles": {"my_profile": {"api_key_env": "KEY"}}}
        result = _snake_to_camel_dict(d)
        self.assertEqual(result["profiles"]["my_profile"]["apiKeyEnv"], "KEY")


class TestYamlToRouterConfig(unittest.TestCase):
    def test_routes_key_maps_to_models(self):
        yaml_cfg = {
            "auth": {"profiles": {"k": {"api_key_env": "X"}}},
            "providers": {"p": {"base_url": "http://x", "auth": "k"}},
            "routes": {"default": {"primary": "p/m", "fallbacks": []}},
            "aliases": {"a": "p/m"},
        }
        result = _yaml_to_router_config(yaml_cfg)
        self.assertIn("models", result)
        self.assertEqual(result["models"]["default"]["primary"], "p/m")
        self.assertEqual(result["modelAliases"]["a"], "p/m")
        # Auth keys should be camelCase
        self.assertEqual(result["auth"]["profiles"]["k"]["apiKeyEnv"], "X")
        self.assertEqual(result["providers"]["p"]["baseUrl"], "http://x")

    def test_models_key_also_accepted(self):
        yaml_cfg = {
            "models": {"default": {"primary": "x/y", "fallbacks": []}},
        }
        result = _yaml_to_router_config(yaml_cfg)
        self.assertEqual(result["models"]["default"]["primary"], "x/y")

    def test_empty_config(self):
        result = _yaml_to_router_config({})
        self.assertEqual(result, {})


class TestYamlToSettings(unittest.TestCase):
    def test_full_extraction(self):
        yaml_cfg = {
            "auth": {"profiles": {"k": {"api_key_env": "X"}}},
            "providers": {"p": {"base_url": "http://x", "auth": "k", "models": {"p/m": {"model": "m"}}}},
            "routes": {"default": {"primary": "p/m", "fallbacks": []}},
            "search": {
                "semantic_scholar_api_key": "ss-key",
                "openalex_email": "a@b.com",
                "crossref_mailto": "c@d.com",
            },
            "middleware": {
                "retry_max": 3,
                "retry_delay": 10.0,
                "context_window": 64000,
            },
            "paths": {"runs_dir": "my_runs"},
        }
        kw = _yaml_to_settings(yaml_cfg, {})
        self.assertIn("model_router_config", kw)
        self.assertEqual(kw["semantic_scholar_api_key"], "ss-key")
        self.assertEqual(kw["openalex_email"], "a@b.com")
        self.assertEqual(kw["middleware_retry_max"], 3)
        self.assertEqual(kw["middleware_retry_delay"], 10.0)
        self.assertEqual(kw["middleware_context_window"], 64000)
        self.assertEqual(kw["runs_dir"], "my_runs")


class TestLoadFromYamlFile(unittest.TestCase):
    def test_load_from_yaml_file(self):
        yaml_content = textwrap.dedent("""\
            auth:
              profiles:
                test_key:
                  api_key_env: TEST_KEY
            providers:
              test:
                base_url: http://localhost
                api_style: openai
                auth: test_key
                models:
                  test/default: { model: test-model }
            routes:
              default: { primary: test/default, fallbacks: [] }
            search:
              openalex_email: test@example.com
            middleware:
              retry_max: 5
            paths:
              runs_dir: yaml_runs
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            yaml_path = f.name

        try:
            env = {"MUSE_CONFIG": yaml_path}
            settings = load_settings(env=env)
            self.assertIsInstance(settings, Settings)
            self.assertEqual(settings.llm_model, "test/default")
            self.assertIn("models", settings.model_router_config)
            self.assertEqual(settings.openalex_email, "test@example.com")
            self.assertEqual(settings.middleware_retry_max, 5)
            self.assertEqual(settings.runs_dir, "yaml_runs")
        finally:
            os.unlink(yaml_path)

    def test_env_var_resolution_in_yaml(self):
        yaml_content = textwrap.dedent("""\
            auth:
              profiles:
                k: { api_key_env: MY_KEY }
            providers:
              p:
                base_url: http://localhost
                auth: k
                models:
                  p/m: { model: m }
            routes:
              default: { primary: p/m, fallbacks: [] }
            search:
              semantic_scholar_api_key: ${SS_KEY}
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            yaml_path = f.name

        try:
            env = {"MUSE_CONFIG": yaml_path, "SS_KEY": "resolved-key"}
            settings = load_settings(env=env)
            self.assertEqual(settings.semantic_scholar_api_key, "resolved-key")
        finally:
            os.unlink(yaml_path)

    def test_env_override_yaml(self):
        yaml_content = textwrap.dedent("""\
            auth:
              profiles:
                k: { api_key_env: X }
            providers:
              p:
                base_url: http://localhost
                auth: k
                models:
                  p/m: { model: m }
            routes:
              default: { primary: p/m, fallbacks: [] }
            middleware:
              retry_max: 3
            search:
              openalex_email: yaml@example.com
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            yaml_path = f.name

        try:
            env = {
                "MUSE_CONFIG": yaml_path,
                "MUSE_MIDDLEWARE_RETRY_MAX": "10",
                "MUSE_OPENALEX_EMAIL": "env@example.com",
            }
            settings = load_settings(env=env)
            # Env vars win over YAML
            self.assertEqual(settings.middleware_retry_max, 10)
            self.assertEqual(settings.openalex_email, "env@example.com")
        finally:
            os.unlink(yaml_path)

    def test_no_yaml_falls_back(self):
        """When no YAML file exists, env-var-only logic works as before."""
        env = {
            "MUSE_LLM_API_KEY": "test-key",
            "MUSE_LLM_MODEL": "gpt-4.1-mini",
            "MUSE_RUNS_DIR": "/tmp/runs",
        }
        settings = load_settings(env=env)
        self.assertEqual(settings.llm_api_key, "test-key")
        self.assertEqual(settings.llm_model, "gpt-4.1-mini")

    def test_config_path_argument(self):
        yaml_content = textwrap.dedent("""\
            auth:
              profiles:
                k: { api_key_env: X }
            providers:
              p:
                base_url: http://localhost
                auth: k
                models:
                  p/m: { model: m }
            routes:
              default: { primary: p/m, fallbacks: [] }
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            yaml_path = f.name

        try:
            settings = load_settings(env={}, config_path=yaml_path)
            self.assertEqual(settings.llm_model, "p/m")
        finally:
            os.unlink(yaml_path)


if __name__ == "__main__":
    unittest.main()

import json
import os
import tempfile
import unittest
import base64

from thesis_agent.providers import HttpClient, LLMClient, ProviderError


class _FakeHttp(HttpClient):
    def __init__(self):
        super().__init__(timeout_seconds=1)
        self.calls = []

    def post_json(self, url, payload, headers=None):  # noqa: ANN001
        self.calls.append({"url": url, "payload": payload, "headers": headers or {}})
        model = payload.get("model", "")
        auth = (headers or {}).get("Authorization", "")
        if model == "gpt-primary":
            raise ProviderError("primary failed")
        if model == "gpt-auth" and auth.endswith("bad-key"):
            raise ProviderError("auth failed")
        if url.endswith("/responses"):
            return {
                "output_text": "ok-from-fallback",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok-from-fallback"}],
                    }
                ],
                "usage": {"total_tokens": 1},
            }
        return {
            "choices": [{"message": {"content": "ok-from-fallback"}}],
            "usage": {"total_tokens": 1},
        }

    def post_json_sse(self, url, payload, headers=None):  # noqa: ANN001
        return self.post_json(url, payload, headers=headers)


class ModelRoutingTests(unittest.TestCase):
    def test_route_uses_fallback_when_primary_fails(self):
        fake_http = _FakeHttp()
        router_config = {
            "auth": {
                "profiles": {
                    "p1": {"apiKey": "k1"},
                    "p2": {"apiKey": "k2"},
                }
            },
            "providers": {
                "openai": {
                    "baseUrl": "https://openai.local/v1",
                    "auth": "p1",
                    "models": {
                        "openai/gpt-primary": {"model": "gpt-primary"},
                    },
                },
                "openrouter": {
                    "baseUrl": "https://openrouter.local/v1",
                    "auth": "p2",
                    "models": {
                        "openrouter/gpt-fallback": {"model": "gpt-fallback"},
                    },
                },
            },
            "models": {
                "writing": {
                    "primary": "openai/gpt-primary",
                    "fallbacks": ["openrouter/gpt-fallback"],
                }
            },
            "modelAliases": {},
        }

        client = LLMClient(
            api_key="",
            base_url="https://unused.local/v1",
            model="unused",
            http=fake_http,
            model_router_config=router_config,
            env={}
        )

        text = client.text(system="s", user="u", route="writing")

        self.assertEqual(text, "ok-from-fallback")
        self.assertEqual(len(fake_http.calls), 2)
        self.assertEqual(fake_http.calls[0]["payload"]["model"], "gpt-primary")
        self.assertEqual(fake_http.calls[1]["payload"]["model"], "gpt-fallback")

    def test_alias_applies_before_provider_model_lookup(self):
        fake_http = _FakeHttp()
        router_config = {
            "auth": {"profiles": {"p1": {"apiKey": "k1"}}},
            "providers": {
                "openai": {
                    "baseUrl": "https://openai.local/v1",
                    "auth": "p1",
                    "models": {
                        "openai/gpt-aliased": {"model": "gpt-fallback"},
                    },
                }
            },
            "models": {
                "default": {
                    "primary": "openai/gpt-original",
                    "fallbacks": [],
                }
            },
            "modelAliases": {"openai/gpt-original": "openai/gpt-aliased"},
        }

        client = LLMClient(
            api_key="",
            base_url="https://unused.local/v1",
            model="unused",
            http=fake_http,
            model_router_config=router_config,
            env={}
        )

        text = client.text(system="s", user="u", route="default")
        self.assertEqual(text, "ok-from-fallback")
        self.assertEqual(fake_http.calls[0]["payload"]["model"], "gpt-fallback")

    def test_auth_profile_rotation_before_model_fallback(self):
        fake_http = _FakeHttp()
        router_config = {
            "auth": {
                "profiles": {
                    "bad": {"apiKey": "bad-key"},
                    "good": {"apiKey": "good-key"},
                }
            },
            "providers": {
                "openai": {
                    "baseUrl": "https://openai.local/v1",
                    "auth": ["bad", "good"],
                    "models": {
                        "openai/gpt-auth": {"model": "gpt-auth"},
                    },
                }
            },
            "models": {
                "default": {
                    "primary": "openai/gpt-auth",
                    "fallbacks": [],
                }
            },
            "modelAliases": {},
        }

        client = LLMClient(
            api_key="",
            base_url="https://unused.local/v1",
            model="unused",
            http=fake_http,
            model_router_config=router_config,
            env={},
        )

        text = client.text(system="s", user="u", route="default")
        self.assertEqual(text, "ok-from-fallback")
        self.assertEqual(len(fake_http.calls), 2)
        self.assertTrue(fake_http.calls[0]["headers"]["Authorization"].endswith("bad-key"))
        self.assertTrue(fake_http.calls[1]["headers"]["Authorization"].endswith("good-key"))

    def test_oauth_profile_loads_token_from_auth_file(self):
        fake_http = _FakeHttp()
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = os.path.join(tmp, "auth.json")
            with open(auth_file, "w", encoding="utf-8") as f:
                json.dump({"tokens": {"access_token": "oauth-token-123"}}, f)

            router_config = {
                "auth": {
                    "profiles": {
                        "codex_oauth": {
                            "type": "oauth",
                            "oauthProvider": "codex",
                            "authFile": auth_file,
                            "tokenPath": "tokens.access_token",
                        }
                    }
                },
                "providers": {
                    "openai": {
                        "baseUrl": "https://api.openai.com/v1",
                        "auth": "codex_oauth",
                        "models": {
                            "openai/gpt-oauth": {"model": "gpt-fallback"}
                        },
                    }
                },
                "models": {
                    "default": {
                        "primary": "openai/gpt-oauth",
                        "fallbacks": [],
                    }
                },
            }

            client = LLMClient(
                api_key="",
                base_url="https://unused.local/v1",
                model="unused",
                http=fake_http,
                model_router_config=router_config,
                env={},
            )
            text = client.text(system="s", user="u")
            self.assertEqual(text, "ok-from-fallback")
            self.assertTrue(fake_http.calls[0]["headers"]["Authorization"].endswith("oauth-token-123"))
            self.assertEqual(fake_http.calls[0]["url"], "https://chatgpt.com/backend-api/codex/responses")
            self.assertEqual(fake_http.calls[0]["payload"].get("instructions"), "s")

    def test_codex_oauth_adds_chatgpt_headers_and_uses_responses_api(self):
        fake_http = _FakeHttp()
        with tempfile.TemporaryDirectory() as tmp:
            auth_file = os.path.join(tmp, "auth.json")
            payload = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_abc"}}).encode("utf-8")
            payload_b64 = base64.urlsafe_b64encode(payload).decode("utf-8").rstrip("=")
            token = f"header.{payload_b64}.sig"
            with open(auth_file, "w", encoding="utf-8") as f:
                json.dump({"tokens": {"access_token": token}}, f)

            router_config = {
                "auth": {
                    "profiles": {
                        "codex_oauth": {
                            "type": "oauth",
                            "oauthProvider": "codex",
                            "authFile": auth_file,
                            "tokenPath": "tokens.access_token",
                        }
                    }
                },
                "providers": {
                    "openai": {
                        "baseUrl": "https://api.openai.com/v1",
                        "auth": "codex_oauth",
                        "models": {
                            "openai/gpt-oauth": {"model": "gpt-5.1-codex"}
                        },
                    }
                },
                "models": {
                    "default": {
                        "primary": "openai/gpt-oauth",
                        "fallbacks": [],
                    }
                },
            }

            client = LLMClient(
                api_key="",
                base_url="https://unused.local/v1",
                model="unused",
                http=fake_http,
                model_router_config=router_config,
                env={},
            )

            text = client.text(system="s", user="u")
            self.assertEqual(text, "ok-from-fallback")
            call = fake_http.calls[0]
            self.assertEqual(call["url"], "https://chatgpt.com/backend-api/codex/responses")
            self.assertEqual(call["headers"].get("chatgpt-account-id"), "acct_abc")
            self.assertEqual(call["headers"].get("OpenAI-Beta"), "responses=experimental")
            self.assertEqual(call["headers"].get("originator"), "codex_cli_rs")
            self.assertIn("input", call["payload"])
            self.assertNotIn("messages", call["payload"])
            self.assertEqual(call["payload"].get("instructions"), "s")

    def test_debug_probe_reports_failures_per_attempt(self):
        fake_http = _FakeHttp()
        router_config = {
            "auth": {
                "profiles": {
                    "p1": {"apiKey": "k1"},
                    "p2": {"apiKey": "k2"},
                }
            },
            "providers": {
                "openai": {
                    "baseUrl": "https://openai.local/v1",
                    "auth": "p1",
                    "models": {
                        "openai/gpt-primary": {"model": "gpt-primary"},
                    },
                },
                "openrouter": {
                    "baseUrl": "https://openrouter.local/v1",
                    "auth": "p2",
                    "models": {
                        "openrouter/gpt-fallback": {"model": "gpt-primary"},
                    },
                },
            },
            "models": {
                "default": {
                    "primary": "openai/gpt-primary",
                    "fallbacks": ["openrouter/gpt-fallback"],
                }
            },
            "modelAliases": {},
        }

        client = LLMClient(
            api_key="",
            base_url="https://unused.local/v1",
            model="unused",
            http=fake_http,
            model_router_config=router_config,
            env={},
        )

        diag = client.debug_probe(route="default")
        self.assertFalse(diag["success"])
        self.assertEqual(len(diag["attempts"]), 2)
        self.assertIn("primary failed", diag["attempts"][0]["error"])
        self.assertIn("primary failed", diag["attempts"][1]["error"])

    def test_debug_probe_does_not_expose_raw_authorization_token(self):
        fake_http = _FakeHttp()
        router_config = {
            "auth": {"profiles": {"p1": {"apiKey": "secret-token-value"}}},
            "providers": {
                "openai": {
                    "baseUrl": "https://openai.local/v1",
                    "auth": "p1",
                    "models": {"openai/gpt-aliased": {"model": "gpt-fallback"}},
                }
            },
            "models": {"default": {"primary": "openai/gpt-aliased", "fallbacks": []}},
        }

        client = LLMClient(
            api_key="",
            base_url="https://unused.local/v1",
            model="unused",
            http=fake_http,
            model_router_config=router_config,
            env={},
        )

        diag = client.debug_probe(route="default")
        self.assertTrue(diag["success"])
        self.assertTrue(diag["attempts"][0]["authorization_present"])
        self.assertNotIn("secret-token-value", str(diag))


if __name__ == "__main__":
    unittest.main()

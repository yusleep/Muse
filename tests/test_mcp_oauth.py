"""Tests for MCP OAuth token management (muse.mcp.oauth)."""

from __future__ import annotations

import pytest

from muse.mcp.oauth import (
    OAuthConfig,
    OAuthError,
    OAuthTokenManager,
    parse_oauth_config,
)


class TestParseOAuthConfig:
    def test_returns_none_when_no_oauth_key(self):
        assert parse_oauth_config({"transport": "http"}) is None

    def test_returns_none_when_incomplete(self):
        assert parse_oauth_config({"oauth": {"token_url": "x"}}) is None

    def test_parses_complete_config(self):
        config = parse_oauth_config(
            {
                "oauth": {
                    "token_url": "https://auth.example.com/token",
                    "client_id": "cid",
                    "client_secret": "csec",
                    "grant_type": "client_credentials",
                    "scopes": ["read", "write"],
                }
            }
        )
        assert config is not None
        assert config.token_url == "https://auth.example.com/token"
        assert config.client_id == "cid"
        assert config.client_secret == "csec"
        assert config.scopes == ["read", "write"]

    def test_defaults_grant_type(self):
        config = parse_oauth_config(
            {
                "oauth": {
                    "token_url": "https://x/token",
                    "client_id": "a",
                    "client_secret": "b",
                }
            }
        )
        assert config is not None
        assert config.grant_type == "client_credentials"


class TestOAuthTokenManager:
    def _make_manager_with_mock(self, monkeypatch, token: str = "tok_abc", expires_in: int = 3600):
        manager = OAuthTokenManager()

        def fake_fetch(config):
            return {"access_token": token, "expires_in": expires_in}

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(fake_fetch))
        return manager

    def test_no_config_returns_empty_headers(self):
        manager = OAuthTokenManager()
        assert manager.get_auth_headers("unknown") == {}

    def test_fetches_and_caches_token(self, monkeypatch):
        manager = self._make_manager_with_mock(monkeypatch, token="tok_123")
        config = OAuthConfig(
            token_url="https://x/token",
            grant_type="client_credentials",
            client_id="a",
            client_secret="b",
        )
        manager.register("srv", config)
        headers = manager.get_auth_headers("srv")
        assert headers == {"Authorization": "Bearer tok_123"}

        call_count = 0

        def counting_fetch(config):
            nonlocal call_count
            call_count += 1
            return {"access_token": "tok_123", "expires_in": 3600}

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(counting_fetch))
        headers_second = manager.get_auth_headers("srv")
        assert headers_second == {"Authorization": "Bearer tok_123"}
        assert call_count == 0

    def test_invalidate_forces_refetch(self, monkeypatch):
        fetch_count = 0

        def counting_fetch(config):
            nonlocal fetch_count
            fetch_count += 1
            return {"access_token": f"tok_{fetch_count}", "expires_in": 3600}

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(counting_fetch))
        manager = OAuthTokenManager()
        config = OAuthConfig(
            token_url="https://x/token",
            grant_type="client_credentials",
            client_id="a",
            client_secret="b",
        )
        manager.register("srv", config)
        first = manager.get_auth_headers("srv")
        assert "tok_1" in first["Authorization"]

        manager.invalidate("srv")
        second = manager.get_auth_headers("srv")
        assert "tok_2" in second["Authorization"]
        assert fetch_count == 2

    def test_has_config(self):
        manager = OAuthTokenManager()
        assert not manager.has_config("x")
        manager.register("x", OAuthConfig("u", "g", "c", "s"))
        assert manager.has_config("x")

    def test_fetch_failure_raises_oauth_error(self, monkeypatch):
        def failing_fetch(config):
            raise OAuthError("boom")

        monkeypatch.setattr(OAuthTokenManager, "_fetch_token", staticmethod(failing_fetch))
        manager = OAuthTokenManager()
        manager.register("srv", OAuthConfig("u", "g", "c", "s"))
        with pytest.raises(OAuthError, match="boom"):
            manager.get_auth_headers("srv")

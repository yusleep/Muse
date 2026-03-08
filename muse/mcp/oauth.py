"""Per-server OAuth token management for MCP connections.

Handles the ``oauth`` block in extensions.yaml server entries:

    mcp_servers:
      overleaf:
        transport: http
        url: "https://overleaf-mcp.example.com/mcp"
        oauth:
          token_url: "https://auth.overleaf.com/oauth2/token"
          grant_type: "client_credentials"
          client_id: "my-client"
          client_secret: "secret"

Tokens are cached in memory and auto-refreshed 60 seconds before expiry.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OAuthConfig:
    """Parsed OAuth settings for one MCP server."""

    token_url: str
    grant_type: str
    client_id: str
    client_secret: str
    scopes: list[str] = field(default_factory=list)


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float


class OAuthTokenManager:
    """Per-server OAuth token cache with lazy refresh."""

    _REFRESH_MARGIN_SECONDS = 60

    def __init__(self) -> None:
        self._configs: dict[str, OAuthConfig] = {}
        self._cache: dict[str, _CachedToken] = {}

    def register(self, server_name: str, config: OAuthConfig) -> None:
        """Register OAuth config for a named server."""

        self._configs[server_name] = config
        self._cache.pop(server_name, None)

    def get_auth_headers(self, server_name: str) -> dict[str, str]:
        """Return Authorization headers for the configured server."""

        config = self._configs.get(server_name)
        if config is None:
            return {}

        cached = self._cache.get(server_name)
        if cached is not None and cached.expires_at - self._REFRESH_MARGIN_SECONDS > time.monotonic():
            return {"Authorization": f"Bearer {cached.access_token}"}

        token_data = self._fetch_token(config)
        access_token = str(token_data.get("access_token", "")).strip()
        if not access_token:
            raise OAuthError(f"Token response for '{server_name}' missing access_token")

        expires_in = int(token_data.get("expires_in", 3600))
        self._cache[server_name] = _CachedToken(
            access_token=access_token,
            expires_at=time.monotonic() + expires_in,
        )
        return {"Authorization": f"Bearer {access_token}"}

    def has_config(self, server_name: str) -> bool:
        return server_name in self._configs

    def invalidate(self, server_name: str) -> None:
        """Force a fresh token request on next use."""

        self._cache.pop(server_name, None)

    @staticmethod
    def _fetch_token(config: OAuthConfig) -> dict[str, Any]:
        """Perform the OAuth2 token request."""

        body_params: dict[str, str] = {
            "grant_type": config.grant_type,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }
        if config.scopes:
            body_params["scope"] = " ".join(config.scopes)

        data = urllib.parse.urlencode(body_params).encode("utf-8")
        request = urllib.request.Request(
            config.token_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300] if exc.fp else ""
            raise OAuthError(f"OAuth token request failed (HTTP {exc.code}): {detail}") from exc
        except Exception as exc:  # pragma: no cover - network/runtime wrapper
            raise OAuthError(f"OAuth token request error: {exc}") from exc


class OAuthError(RuntimeError):
    """Raised when OAuth token acquisition fails."""


def parse_oauth_config(entry: dict[str, Any]) -> OAuthConfig | None:
    """Parse an ``oauth`` block from a server entry."""

    oauth_raw = entry.get("oauth")
    if not isinstance(oauth_raw, dict):
        return None

    token_url = str(oauth_raw.get("token_url", "")).strip()
    client_id = str(oauth_raw.get("client_id", "")).strip()
    client_secret = str(oauth_raw.get("client_secret", "")).strip()
    if not (token_url and client_id and client_secret):
        return None

    grant_type = str(oauth_raw.get("grant_type", "client_credentials")).strip()
    scopes_raw = oauth_raw.get("scopes", [])
    scopes = [str(scope) for scope in scopes_raw] if isinstance(scopes_raw, list) else []
    return OAuthConfig(
        token_url=token_url,
        grant_type=grant_type,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
    )

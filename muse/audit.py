"""Append-only audit event utilities."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

_REQUIRED_EVENT_FIELDS = {
    "event_id",
    "timestamp",
    "event_type",
    "stage",
    "agent",
    "model",
    "tokens",
    "latency_ms",
    "cost_estimate",
}


def build_event(
    *,
    stage: int,
    agent: str,
    event_type: str,
    model: str,
    tokens: int,
    latency_ms: int,
    cost_estimate: float,
    input_summary: str = "",
    output_summary: str = "",
) -> dict[str, Any]:
    """Build a typed audit event payload."""

    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "stage": int(stage),
        "agent": agent,
        "model": model,
        "tokens": int(tokens),
        "latency_ms": int(latency_ms),
        "cost_estimate": float(cost_estimate),
        "input_summary": input_summary,
        "output_summary": output_summary,
    }


class JsonlAuditSink:
    """Append-only JSONL sink with event_id idempotency."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._seen_ids: set[str] = set()

    def append(self, event: dict[str, Any]) -> None:
        missing = _REQUIRED_EVENT_FIELDS - set(event.keys())
        if missing:
            raise ValueError(f"event missing required keys: {sorted(missing)}")

        event_id = event["event_id"]
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id must be a non-empty string")

        if event_id in self._seen_ids:
            return

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        self._seen_ids.add(event_id)

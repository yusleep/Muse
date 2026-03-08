"""Standardized result type for sub-agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SubagentResult:
    """Uniform output from any sub-agent execution."""

    status: Literal["completed", "failed", "timed_out"] = "completed"
    accomplishments: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    files_created: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into a JSON-compatible dictionary."""

        return {
            "status": self.status,
            "accomplishments": list(self.accomplishments),
            "key_findings": list(self.key_findings),
            "files_created": list(self.files_created),
            "issues": list(self.issues),
            "citations": list(self.citations),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubagentResult":
        """Build a result object from a plain dictionary."""

        return cls(
            status=str(payload.get("status", "completed")),
            accomplishments=list(payload.get("accomplishments", [])),
            key_findings=list(payload.get("key_findings", [])),
            files_created=list(payload.get("files_created", [])),
            issues=list(payload.get("issues", [])),
            citations=list(payload.get("citations", [])),
        )

    def summary(self) -> str:
        """Return a concise human-readable summary."""

        return (
            f"SubagentResult(status={self.status}, "
            f"accomplishments={len(self.accomplishments)}, "
            f"issues={len(self.issues)})"
        )

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

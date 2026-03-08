"""Abstract sandbox interface and execution result.

All sandbox implementations (Docker, Local) conform to this ABC.
The ``ExecResult`` dataclass is the universal return type for any
command execution inside a sandbox.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExecResult:
    """Result of a command execution in a sandbox."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    files_created: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self, max_chars: int = 2000) -> str:
        """Human-readable summary for tool return values."""

        status = "OK" if self.success else f"FAILED (exit={self.exit_code})"
        if self.timed_out:
            status = "TIMED OUT"
        parts = [f"[{status}]"]
        if self.stdout.strip():
            stdout = self.stdout.strip()
            if len(stdout) > max_chars:
                stdout = stdout[:max_chars] + "\n... (truncated)"
            parts.append(f"stdout:\n{stdout}")
        if self.stderr.strip():
            stderr = self.stderr.strip()
            if len(stderr) > max_chars:
                stderr = stderr[:max_chars] + "\n... (truncated)"
            parts.append(f"stderr:\n{stderr}")
        if self.files_created:
            parts.append(f"files: {', '.join(self.files_created)}")
        return "\n".join(parts)


class Sandbox(ABC):
    """Abstract sandbox for isolated command execution."""

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        timeout: int = 60,
        workdir: str | None = None,
    ) -> ExecResult:
        """Execute a shell command and return the result."""

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read a file from the sandbox filesystem."""

    @abstractmethod
    async def write_file(self, path: str, content: bytes) -> None:
        """Write content to a file in the sandbox filesystem."""

    @abstractmethod
    async def list_dir(self, path: str = ".") -> list[str]:
        """List entries in a sandbox directory."""

    async def cleanup(self) -> None:
        """Release resources. Override in implementations that need cleanup."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.cleanup()

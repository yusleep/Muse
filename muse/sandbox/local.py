"""Local subprocess-based sandbox (Docker fallback)."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from muse.sandbox.base import ExecResult, Sandbox


class LocalSandbox(Sandbox):
    """Sandbox backed by local subprocess execution."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        max_output_bytes: int = 256_000,
    ) -> None:
        self._workspace = Path(workspace).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._max_output = max_output_bytes

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def outputs_dir(self) -> Path:
        return self._workspace / "outputs"

    async def exec(
        self,
        command: str,
        *,
        timeout: int = 60,
        workdir: str | None = None,
    ) -> ExecResult:
        cwd = self._workspace
        if workdir:
            cwd = self._workspace / workdir
            cwd.mkdir(parents=True, exist_ok=True)

        process = None
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                start_new_session=True,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            return ExecResult(
                exit_code=process.returncode or 0,
                stdout=self._decode(stdout_bytes),
                stderr=self._decode(stderr_bytes),
                timed_out=False,
            )
        except asyncio.TimeoutError:
            if process is not None:
                os.killpg(process.pid, signal.SIGKILL)
                try:
                    await process.communicate()
                except Exception:
                    pass
            return ExecResult(
                exit_code=137,
                stdout="",
                stderr="Process killed: timeout exceeded",
                timed_out=True,
            )

    async def read_file(self, path: str) -> bytes:
        full_path = self._workspace / path
        if not full_path.is_file():
            raise FileNotFoundError(f"Sandbox file not found: {path}")
        return full_path.read_bytes()

    async def write_file(self, path: str, content: bytes) -> None:
        full_path = self._workspace / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

    async def list_dir(self, path: str = ".") -> list[str]:
        full_path = self._workspace / path
        if not full_path.is_dir():
            return []
        return sorted(entry.name for entry in full_path.iterdir())

    def _decode(self, data: bytes) -> str:
        truncated = data[: self._max_output] if len(data) > self._max_output else data
        return truncated.decode("utf-8", errors="replace")

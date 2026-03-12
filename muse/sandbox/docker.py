"""Docker-based isolated sandbox execution."""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

from muse.sandbox.base import ExecResult, Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "muse-sandbox:latest"


class DockerSandbox(Sandbox):
    """Sandbox backed by a Docker container."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        outputs_dir: str | Path | None = None,
        refs_dir: str | Path | None = None,
        image: str = _DEFAULT_IMAGE,
        max_output_bytes: int = 256_000,
    ) -> None:
        self._workspace = Path(workspace).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

        self._outputs_dir = Path(outputs_dir).resolve() if outputs_dir else self._workspace / "outputs"
        self._outputs_dir.mkdir(parents=True, exist_ok=True)

        self._refs_dir = Path(refs_dir).resolve() if refs_dir else None
        self._image = image
        self._max_output = max_output_bytes
        self._container_id: str | None = None

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def outputs_dir(self) -> Path:
        return self._outputs_dir

    @property
    def container_id(self) -> str | None:
        return self._container_id

    async def _ensure_container(self) -> str:
        """Create and start the sandbox container on first use."""

        if self._container_id is not None:
            return self._container_id

        volumes = [
            f"{self._workspace}:/mnt/workspace:rw",
            f"{self._outputs_dir}:/mnt/outputs:rw",
        ]
        if self._refs_dir and self._refs_dir.is_dir():
            volumes.append(f"{self._refs_dir}:/mnt/refs:ro")

        volume_args: list[str] = []
        for volume in volumes:
            volume_args.extend(["-v", volume])

        create_cmd = [
            "docker",
            "create",
            "--name",
            f"muse-sandbox-{id(self)}",
            "-w",
            "/mnt/workspace",
            *volume_args,
            self._image,
            "sleep",
            "infinity",
        ]
        process = await asyncio.create_subprocess_exec(
            *create_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"Docker create failed (exit {process.returncode}): "
                f"{stderr_bytes.decode('utf-8', errors='replace')[:500]}"
            )

        self._container_id = stdout_bytes.decode("utf-8", errors="replace").strip()
        start_process = await asyncio.create_subprocess_exec(
            "docker",
            "start",
            self._container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await start_process.communicate()
        return self._container_id

    async def exec(
        self,
        command: str,
        *,
        timeout: int = 60,
        workdir: str | None = None,
    ) -> ExecResult:
        container = await self._ensure_container()
        docker_workdir = "/mnt/workspace"
        if workdir:
            docker_workdir = f"/mnt/workspace/{workdir}"

        docker_cmd = [
            "docker",
            "exec",
            "-w",
            docker_workdir,
            container,
            "bash",
            "-c",
            command,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
            kill_process = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                container,
                "bash",
                "-c",
                "kill -9 $(pgrep -f 'bash -c') 2>/dev/null || true",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await kill_process.communicate()
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

    async def cleanup(self) -> None:
        if self._container_id is None:
            return
        for action in ("stop", "rm"):
            process = await asyncio.create_subprocess_exec(
                "docker",
                action,
                self._container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
        self._container_id = None

    def _decode(self, data: bytes) -> str:
        truncated = data[: self._max_output] if len(data) > self._max_output else data
        return truncated.decode("utf-8", errors="replace")


def docker_available() -> bool:
    """Check whether Docker CLI and daemon are available."""

    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0


def create_sandbox(
    workspace: str | Path,
    *,
    refs_dir: str | Path | None = None,
    image: str = _DEFAULT_IMAGE,
    prefer_docker: bool = True,
) -> Sandbox:
    """Factory returning DockerSandbox when available, else LocalSandbox."""

    if prefer_docker and docker_available():
        logger.info("Using Docker sandbox (image=%s)", image)
        return DockerSandbox(workspace, refs_dir=refs_dir, image=image)

    from muse.sandbox.local import LocalSandbox

    logger.info("Docker unavailable; using local subprocess sandbox")
    return LocalSandbox(workspace)

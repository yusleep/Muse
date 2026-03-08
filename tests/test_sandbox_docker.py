"""Tests for DockerSandbox (muse.sandbox.docker)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from muse.sandbox.docker import DockerSandbox, create_sandbox, docker_available


class TestDockerAvailable:
    def test_returns_bool(self):
        result = docker_available()
        assert isinstance(result, bool)


class TestCreateSandbox:
    def test_returns_local_when_docker_unavailable(self, tmp_path):
        with patch("muse.sandbox.docker.docker_available", return_value=False):
            from muse.sandbox.local import LocalSandbox

            sandbox = create_sandbox(tmp_path / "ws", prefer_docker=True)
            assert isinstance(sandbox, LocalSandbox)

    def test_returns_local_when_prefer_false(self, tmp_path):
        from muse.sandbox.local import LocalSandbox

        sandbox = create_sandbox(tmp_path / "ws", prefer_docker=False)
        assert isinstance(sandbox, LocalSandbox)

    def test_returns_docker_when_available(self, tmp_path):
        with patch("muse.sandbox.docker.docker_available", return_value=True):
            sandbox = create_sandbox(tmp_path / "ws")
            assert isinstance(sandbox, DockerSandbox)


class TestDockerSandboxInit:
    def test_creates_workspace(self, tmp_path):
        workspace = tmp_path / "ws"
        assert not workspace.exists()
        DockerSandbox(workspace)
        assert workspace.is_dir()

    def test_creates_outputs_dir(self, tmp_path):
        DockerSandbox(tmp_path / "ws")
        assert (tmp_path / "ws" / "outputs").is_dir()

    def test_custom_outputs_dir(self, tmp_path):
        outputs = tmp_path / "custom_out"
        DockerSandbox(tmp_path / "ws", outputs_dir=outputs)
        assert outputs.is_dir()


class TestDockerSandboxFileOps:
    def test_write_and_read(self, tmp_path):
        sandbox = DockerSandbox(tmp_path / "ws")
        asyncio.run(sandbox.write_file("test.txt", b"hello"))
        data = asyncio.run(sandbox.read_file("test.txt"))
        assert data == b"hello"

    def test_read_missing_raises(self, tmp_path):
        sandbox = DockerSandbox(tmp_path / "ws")
        with pytest.raises(FileNotFoundError):
            asyncio.run(sandbox.read_file("nope.txt"))

    def test_list_dir(self, tmp_path):
        sandbox = DockerSandbox(tmp_path / "ws")
        asyncio.run(sandbox.write_file("a.txt", b""))
        asyncio.run(sandbox.write_file("b.txt", b""))
        entries = asyncio.run(sandbox.list_dir("."))
        assert "a.txt" in entries
        assert "b.txt" in entries


@pytest.mark.docker
class TestDockerSandboxExec:
    @pytest.fixture(autouse=True)
    def _check_docker(self):
        if not docker_available():
            pytest.skip("Docker not available")

    def test_echo(self, tmp_path):
        async def run():
            async with DockerSandbox(tmp_path / "ws") as sandbox:
                result = await sandbox.exec("echo hello")
                assert result.success
                assert "hello" in result.stdout

        asyncio.run(run())

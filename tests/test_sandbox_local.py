"""Tests for LocalSandbox (muse.sandbox.local)."""

from __future__ import annotations

import asyncio

import pytest

from muse.sandbox.local import LocalSandbox


def _run(coro):
    return asyncio.run(coro)


class TestLocalSandboxExec:
    def test_echo_command(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        result = _run(sandbox.exec("echo hello"))
        assert result.success
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_failing_command(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        result = _run(sandbox.exec("exit 42"))
        assert not result.success
        assert result.exit_code == 42

    def test_stderr_captured(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        result = _run(sandbox.exec("echo err >&2"))
        assert "err" in result.stderr

    def test_timeout_kills_process(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        result = _run(sandbox.exec("sleep 60", timeout=1))
        assert result.timed_out
        assert not result.success

    def test_workdir_param(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        result = _run(sandbox.exec("pwd", workdir="sub/dir"))
        assert result.success
        assert "sub/dir" in result.stdout

    def test_workspace_created(self, tmp_path):
        workspace = tmp_path / "new_ws"
        assert not workspace.exists()
        LocalSandbox(workspace)
        assert workspace.is_dir()


class TestLocalSandboxFileOps:
    def test_write_and_read_file(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        _run(sandbox.write_file("test.txt", b"hello world"))
        data = _run(sandbox.read_file("test.txt"))
        assert data == b"hello world"

    def test_write_creates_subdirs(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        _run(sandbox.write_file("a/b/c.txt", b"deep"))
        data = _run(sandbox.read_file("a/b/c.txt"))
        assert data == b"deep"

    def test_read_missing_file_raises(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        with pytest.raises(FileNotFoundError):
            _run(sandbox.read_file("nope.txt"))

    def test_list_dir(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        _run(sandbox.write_file("a.txt", b""))
        _run(sandbox.write_file("b.txt", b""))
        entries = _run(sandbox.list_dir("."))
        assert "a.txt" in entries
        assert "b.txt" in entries
        assert entries == sorted(entries)

    def test_list_dir_empty(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        entries = _run(sandbox.list_dir("."))
        assert entries == []

    def test_list_dir_nonexistent_returns_empty(self, tmp_path):
        sandbox = LocalSandbox(tmp_path / "ws")
        entries = _run(sandbox.list_dir("nope"))
        assert entries == []


class TestLocalSandboxContextManager:
    def test_async_context_manager(self, tmp_path):
        async def run():
            async with LocalSandbox(tmp_path / "ws") as sandbox:
                result = await sandbox.exec("echo hi")
                assert result.success

        asyncio.run(run())

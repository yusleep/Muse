"""Tests for sandbox base types (muse.sandbox.base)."""

from __future__ import annotations

import pytest

from muse.sandbox.base import ExecResult, Sandbox


class TestExecResult:
    def test_success_on_zero_exit(self):
        result = ExecResult(exit_code=0, stdout="ok")
        assert result.success is True

    def test_failure_on_nonzero_exit(self):
        result = ExecResult(exit_code=1, stderr="fail")
        assert result.success is False

    def test_failure_on_timeout(self):
        result = ExecResult(exit_code=0, timed_out=True)
        assert result.success is False

    def test_summary_ok(self):
        result = ExecResult(exit_code=0, stdout="hello")
        summary = result.summary()
        assert "[OK]" in summary
        assert "hello" in summary

    def test_summary_failed(self):
        result = ExecResult(exit_code=2, stderr="bad")
        summary = result.summary()
        assert "FAILED" in summary
        assert "exit=2" in summary
        assert "bad" in summary

    def test_summary_timed_out(self):
        result = ExecResult(exit_code=137, timed_out=True)
        summary = result.summary()
        assert "TIMED OUT" in summary

    def test_summary_truncation(self):
        result = ExecResult(exit_code=0, stdout="x" * 5000)
        summary = result.summary(max_chars=100)
        assert "(truncated)" in summary
        assert len(summary) < 5000

    def test_summary_includes_files(self):
        result = ExecResult(exit_code=0, files_created=["out.pdf", "fig.png"])
        summary = result.summary()
        assert "out.pdf" in summary
        assert "fig.png" in summary

    def test_default_fields(self):
        result = ExecResult(exit_code=0)
        assert result.stdout == ""
        assert result.stderr == ""
        assert result.timed_out is False
        assert result.files_created == []


class TestSandboxABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Sandbox()  # type: ignore[abstract]

    def test_subclass_must_implement_all(self):
        class Incomplete(Sandbox):
            async def exec(self, command, *, timeout=60, workdir=None):
                return None

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self):
        class Complete(Sandbox):
            async def exec(self, command, *, timeout=60, workdir=None):
                return ExecResult(exit_code=0)

            async def read_file(self, path):
                return b""

            async def write_file(self, path, content):
                return None

            async def list_dir(self, path="."):
                return []

        sandbox = Complete()
        assert sandbox is not None

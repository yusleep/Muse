"""Tests for sandbox tool functions (muse.sandbox.tools)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from muse.sandbox.base import ExecResult, Sandbox
from muse.sandbox.tools import latex_compile, present_file, run_python, shell


class MockSandbox(Sandbox):
    """Minimal mock sandbox for tool tests."""

    def __init__(self, outputs_dir: Path | None = None):
        self.exec_results: list[ExecResult] = []
        self.exec_calls: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.outputs_dir = outputs_dir
        self._exec_index = 0

    def set_exec_results(self, *results: ExecResult):
        self.exec_results = list(results)
        self._exec_index = 0

    async def exec(self, command, *, timeout=60, workdir=None):
        self.exec_calls.append({"command": command, "timeout": timeout, "workdir": workdir})
        if self._exec_index < len(self.exec_results):
            result = self.exec_results[self._exec_index]
            self._exec_index += 1
            return result
        return ExecResult(exit_code=0)

    async def read_file(self, path):
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(path)

    async def write_file(self, path, content):
        self.files[path] = content

    async def list_dir(self, path="."):
        return []


def _run(coro):
    return asyncio.run(coro)


class TestShellTool:
    def test_returns_summary(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(ExecResult(exit_code=0, stdout="hello"))
        result = _run(shell(sandbox, "echo hello"))
        assert "[OK]" in result
        assert "hello" in result

    def test_failure_reported(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(ExecResult(exit_code=1, stderr="bad command"))
        result = _run(shell(sandbox, "bad"))
        assert "FAILED" in result

    def test_passes_timeout_and_workdir(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(ExecResult(exit_code=0))
        _run(shell(sandbox, "ls", timeout=30, workdir="sub"))
        assert sandbox.exec_calls[0]["timeout"] == 30
        assert sandbox.exec_calls[0]["workdir"] == "sub"


class TestLatexCompileTool:
    def test_successful_build(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(
            ExecResult(exit_code=0, stdout="pdflatex pass 1"),
            ExecResult(exit_code=0, stdout="bibtex"),
            ExecResult(exit_code=0, stdout="pdflatex pass 2"),
            ExecResult(exit_code=0, stdout="pdflatex pass 3"),
        )
        result = _run(latex_compile(sandbox, "thesis.tex"))
        assert "OK" in result
        assert "thesis.pdf" in result

    def test_build_failure(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(
            ExecResult(exit_code=1, stderr="Undefined control sequence"),
            ExecResult(exit_code=0),
            ExecResult(exit_code=1, stderr="error"),
            ExecResult(exit_code=1, stderr="error"),
        )
        sandbox.files["thesis.log"] = b"! Undefined control sequence.\n"
        result = _run(latex_compile(sandbox, "thesis.tex"))
        assert "FAILED" in result
        assert "Undefined control sequence" in result

    def test_four_exec_calls(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(*(ExecResult(exit_code=0) for _ in range(4)))
        _run(latex_compile(sandbox, "main.tex"))
        assert len(sandbox.exec_calls) == 4
        assert "pdflatex" in sandbox.exec_calls[0]["command"]
        assert "bibtex" in sandbox.exec_calls[1]["command"]


class TestRunPythonTool:
    def test_writes_and_executes_script(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(ExecResult(exit_code=0, stdout="42"))
        result = _run(run_python(sandbox, "print(42)"))
        assert "[OK]" in result
        assert "42" in result
        assert "_muse_script.py" in sandbox.files

    def test_script_failure(self):
        sandbox = MockSandbox()
        sandbox.set_exec_results(ExecResult(exit_code=1, stderr="NameError"))
        result = _run(run_python(sandbox, "undefined_var"))
        assert "FAILED" in result


class TestPresentFileTool:
    def test_copies_file(self, tmp_path):
        sandbox = MockSandbox(outputs_dir=tmp_path / "outputs")
        sandbox.files["thesis.pdf"] = b"PDF content"
        result = _run(present_file(sandbox, "thesis.pdf"))
        assert "OK" in result
        assert "thesis.pdf" in result
        assert (tmp_path / "outputs" / "thesis.pdf").read_bytes() == b"PDF content"

    def test_source_not_found(self):
        sandbox = MockSandbox()
        result = _run(present_file(sandbox, "missing.pdf"))
        assert "FAILED" in result
        assert "not found" in result

    def test_custom_dest_name(self, tmp_path):
        sandbox = MockSandbox(outputs_dir=tmp_path / "outputs")
        sandbox.files["ch1/output.pdf"] = b"PDF"
        result = _run(present_file(sandbox, "ch1/output.pdf", dest_name="chapter1.pdf"))
        assert "chapter1.pdf" in result
        assert (tmp_path / "outputs" / "chapter1.pdf").read_bytes() == b"PDF"

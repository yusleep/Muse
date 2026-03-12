"""Integration tests for sandbox execution pipeline."""

from __future__ import annotations

import asyncio
import json
import shutil

import pytest

from muse.sandbox.local import LocalSandbox
from muse.sandbox.tools import latex_compile, present_file, run_python, shell
from muse.sandbox.vfs import build_vfs

_MINIMAL_TEX = r"""\documentclass{article}
\begin{document}
Hello, Muse sandbox!
\end{document}
"""


class TestLocalSandboxIntegration:
    def test_shell_echo_roundtrip(self, tmp_path):
        async def run():
            async with LocalSandbox(tmp_path / "ws") as sandbox:
                result = await shell(sandbox, "echo 'integration test'")
                assert "integration test" in result
                assert "OK" in result

        asyncio.run(run())

    def test_python_execution(self, tmp_path):
        async def run():
            async with LocalSandbox(tmp_path / "ws") as sandbox:
                result = await run_python(sandbox, "print(2 + 2)")
                assert "OK" in result
                assert "4" in result

        asyncio.run(run())

    def test_python_creates_file(self, tmp_path):
        async def run():
            async with LocalSandbox(tmp_path / "ws") as sandbox:
                script = (
                    "with open('result.txt', 'w') as handle:\n"
                    "    handle.write('computed')\n"
                    "print('done')"
                )
                result = await run_python(sandbox, script)
                assert "OK" in result
                content = await sandbox.read_file("result.txt")
                assert content == b"computed"

        asyncio.run(run())

    @pytest.mark.skipif(
        not shutil.which("pdflatex"),
        reason="pdflatex not installed",
    )
    def test_latex_compile_minimal(self, tmp_path):
        async def run():
            async with LocalSandbox(tmp_path / "ws") as sandbox:
                await sandbox.write_file("test.tex", _MINIMAL_TEX.encode("utf-8"))
                result = await latex_compile(sandbox, "test.tex")
                assert "OK" in result or "test.pdf" in result

        asyncio.run(run())

    def test_present_file_roundtrip(self, tmp_path):
        async def run():
            sandbox = LocalSandbox(tmp_path / "ws")
            await sandbox.write_file("generated.txt", b"output content")
            result = await present_file(sandbox, "generated.txt")
            assert "OK" in result
            assert (sandbox.outputs_dir / "generated.txt").read_bytes() == b"output content"

        asyncio.run(run())


class TestVFSIntegration:
    def test_full_path_mapping(self, tmp_path):
        workspace = str(tmp_path / "workspace")
        outputs = str(tmp_path / "outputs")
        refs = str(tmp_path / "refs")
        vfs = build_vfs(workspace=workspace, outputs=outputs, refs=refs)

        assert vfs.to_sandbox(f"{workspace}/ch1/main.tex") == "/mnt/workspace/ch1/main.tex"
        assert vfs.to_sandbox(f"{outputs}/thesis.pdf") == "/mnt/outputs/thesis.pdf"
        assert vfs.to_sandbox(f"{refs}/paper.pdf") == "/mnt/refs/paper.pdf"

        assert vfs.to_host("/mnt/workspace/ch1/main.tex") == f"{workspace}/ch1/main.tex"
        assert vfs.to_host("/mnt/outputs/thesis.pdf") == f"{outputs}/thesis.pdf"

        assert vfs.is_writable("/mnt/workspace/file.tex") is True
        assert vfs.is_writable("/mnt/refs/readonly.pdf") is False


class TestSandboxFileWorkflow:
    def test_write_compile_read_workflow(self, tmp_path):
        async def run():
            async with LocalSandbox(tmp_path / "ws") as sandbox:
                script = (
                    "import json\n"
                    "data = {'chapters': 5, 'citations': 42}\n"
                    "with open('stats.json', 'w') as handle:\n"
                    "    json.dump(data, handle)\n"
                    "print('Generated stats.json')"
                )
                result = await run_python(sandbox, script)
                assert "OK" in result

                stats = json.loads((await sandbox.read_file("stats.json")).decode("utf-8"))
                assert stats["chapters"] == 5
                assert stats["citations"] == 42

                entries = await sandbox.list_dir(".")
                assert "stats.json" in entries

        asyncio.run(run())

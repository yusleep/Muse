import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_without_langgraph(code: str) -> subprocess.CompletedProcess[str]:
    body = textwrap.dedent(code).strip()
    script = textwrap.dedent(
        f"""
import builtins

_real_import = builtins.__import__

def _blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "langgraph" or name.startswith("langgraph."):
        raise ModuleNotFoundError("No module named 'langgraph'")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _blocking_import

{body}
"""
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
    )


class LangGraphDependencyHandlingTests(unittest.TestCase):
    def test_importing_muse_modules_does_not_require_langgraph_immediately(self):
        result = _run_without_langgraph(
            """
            import muse
            import muse.runtime
            import muse.cli
            print("imports-ok")
            """
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("imports-ok", result.stdout)

    def test_runtime_build_graph_raises_clear_error_when_langgraph_missing(self):
        result = _run_without_langgraph(
            """
            from muse.config import Settings
            from muse.runtime import Runtime

            runtime = Runtime(
                Settings(
                    llm_api_key="x",
                    llm_base_url="http://localhost",
                    llm_model="stub",
                    model_router_config={},
                    runs_dir="runs",
                    semantic_scholar_api_key=None,
                    openalex_email=None,
                    crossref_mailto=None,
                    refs_dir=None,
                    checkpoint_dir=None,
                )
            )

            try:
                runtime.build_graph(thread_id="demo", auto_approve=True)
            except RuntimeError as exc:
                print(exc)
                raise SystemExit(0)

            raise SystemExit(1)
            """
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("langgraph", f"{result.stdout}\n{result.stderr}".lower())


if __name__ == "__main__":
    unittest.main()

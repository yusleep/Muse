from pathlib import Path
import unittest

import muse


class PublicSurfaceTests(unittest.TestCase):
    def test_readme_uses_muse_branding(self):
        readme_text = Path("README.md").read_text(encoding="utf-8")
        self.assertIn("Muse", readme_text)
        self.assertNotIn("Thesis Agent", readme_text)
        self.assertNotIn("python3 -m thesis_agent", readme_text)

    def test_public_docs_describe_latex_overleaf_export_not_docx(self):
        readme_text = Path("README.md").read_text(encoding="utf-8").lower()

        self.assertIn("latex", readme_text)
        self.assertIn("overleaf", readme_text)
        self.assertNotIn("docx", readme_text)

    def test_package_exports_do_not_expose_docx_helpers(self):
        self.assertTrue({"docx_export", "fill_template"}.isdisjoint(set(muse.__all__)))

    def test_package_no_longer_exports_legacy_engine_or_orchestrator_symbols(self):
        legacy = {"EngineContext", "ThesisEngine", "can_advance_to_stage", "gate_export"}
        self.assertTrue(legacy.isdisjoint(set(muse.__all__)))

    def test_docx_export_module_is_removed(self):
        self.assertFalse(Path("muse/docx_export.py").exists())

    def test_legacy_bridge_modules_are_removed(self):
        for path in (
            "muse/engine.py",
            "muse/orchestrator.py",
            "muse/chapter.py",
            "muse/stages.py",
        ):
            self.assertFalse(Path(path).exists(), path)

    def test_gitignore_ignores_ralph_state_directory(self):
        gitignore_text = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn(".ralph-tui/", gitignore_text)

    def test_gitignore_ignores_non_runtime_top_level_docs(self):
        gitignore_text = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn("Research*.md", gitignore_text)
        self.assertIn("muse-plan-v2.md", gitignore_text)
        self.assertIn("CODEX_PROMPTS.md", gitignore_text)
        self.assertIn("docs/plans/", gitignore_text)

    def test_repo_instructions_do_not_depend_on_ignored_plan_directory(self):
        if Path("AGENTS.md").exists():
            agents_text = Path("AGENTS.md").read_text(encoding="utf-8")
            self.assertNotIn("docs/plans/", agents_text)

        if Path("PLANS.md").exists():
            plans_text = Path("PLANS.md").read_text(encoding="utf-8")
            self.assertNotIn("docs/plans/", plans_text)

    def test_requirements_include_langgraph_runtime_dependencies(self):
        requirements = Path("requirements.txt")
        self.assertTrue(requirements.exists())
        text = requirements.read_text(encoding="utf-8")
        self.assertIn("langgraph", text)
        self.assertIn("langgraph-checkpoint-sqlite", text)


if __name__ == "__main__":
    unittest.main()

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
        plan_text = Path("muse-plan-v2.md").read_text(encoding="utf-8").lower()

        self.assertIn("latex", readme_text)
        self.assertIn("overleaf", readme_text)
        self.assertNotIn("docx", readme_text)
        self.assertNotIn("docx", plan_text)

    def test_package_exports_do_not_expose_docx_helpers(self):
        self.assertTrue({"docx_export", "fill_template"}.isdisjoint(set(muse.__all__)))

    def test_docx_export_module_is_removed(self):
        self.assertFalse(Path("muse/docx_export.py").exists())

    def test_gitignore_ignores_ralph_state_directory(self):
        gitignore_text = Path(".gitignore").read_text(encoding="utf-8")
        self.assertIn(".ralph-tui/", gitignore_text)


if __name__ == "__main__":
    unittest.main()

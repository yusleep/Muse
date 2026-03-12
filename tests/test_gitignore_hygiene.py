import pathlib
import unittest


class GitignoreHygieneTests(unittest.TestCase):
    def test_gitignore_ignores_local_config_yaml(self):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        gitignore_path = repo_root / ".gitignore"

        patterns: list[str] = []
        for raw_line in gitignore_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)

        self.assertTrue(
            "config.yaml" in patterns or "/config.yaml" in patterns,
            "Expected .gitignore to ignore config.yaml (prefer /config.yaml).",
        )


if __name__ == "__main__":
    unittest.main()

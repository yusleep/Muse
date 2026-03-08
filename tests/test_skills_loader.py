from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from muse.skills.loader import Skill, SkillLoader, _parse_skill_md


class ParseSkillMdTests(unittest.TestCase):
    def test_valid_skill_file(self):
        text = textwrap.dedent(
            """\
            ---
            name: test-skill
            description: A test skill
            applies_to:
              stages: [writing, polish]
              disciplines: ["*"]
              languages: [zh]
            priority: 10
            ---
            # Test Skill

            Write clearly and concisely.
        """
        )
        skill = _parse_skill_md(text, source_path="/tmp/SKILL.md")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "test-skill")
        self.assertEqual(skill.description, "A test skill")
        self.assertEqual(skill.stages, ["writing", "polish"])
        self.assertEqual(skill.disciplines, ["*"])
        self.assertEqual(skill.languages, ["zh"])
        self.assertEqual(skill.priority, 10)
        self.assertIn("Write clearly", skill.body)
        self.assertEqual(skill.source_path, "/tmp/SKILL.md")

    def test_missing_name_returns_none(self):
        text = "---\ndescription: no name\n---\nbody"
        self.assertIsNone(_parse_skill_md(text))

    def test_no_front_matter_returns_none(self):
        self.assertIsNone(_parse_skill_md("just plain markdown"))

    def test_invalid_yaml_returns_none(self):
        text = "---\n: [invalid yaml\n---\nbody"
        self.assertIsNone(_parse_skill_md(text))

    def test_defaults_applied(self):
        text = "---\nname: minimal\n---\nbody text"
        skill = _parse_skill_md(text)
        self.assertIsNotNone(skill)
        self.assertEqual(skill.stages, ["*"])
        self.assertEqual(skill.disciplines, ["*"])
        self.assertEqual(skill.languages, ["*"])
        self.assertEqual(skill.priority, 50)

    def test_token_estimate(self):
        text = "---\nname: t\n---\n" + ("x" * 400)
        skill = _parse_skill_md(text)
        self.assertEqual(skill.token_estimate, 100)

    def test_string_applies_to_normalized_to_list(self):
        text = textwrap.dedent(
            """\
            ---
            name: single
            applies_to:
              stages: writing
              disciplines: cs
              languages: en
            ---
            body
        """
        )
        skill = _parse_skill_md(text)
        self.assertEqual(skill.stages, ["writing"])
        self.assertEqual(skill.disciplines, ["cs"])
        self.assertEqual(skill.languages, ["en"])


class SkillLoaderTests(unittest.TestCase):
    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_dir = Path(tmp) / "public"
            skill_dir = public_dir / "my-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: my-skill\ndescription: test\npriority: 20\n---\nBody.",
                encoding="utf-8",
            )

            loader = SkillLoader(dirs=[public_dir])
            skills = loader.load_all()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "my-skill")

    def test_custom_overrides_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_dir = Path(tmp) / "public" / "sk"
            public_dir.mkdir(parents=True)
            (public_dir / "SKILL.md").write_text(
                "---\nname: sk\ndescription: public version\npriority: 10\n---\nPublic body.",
                encoding="utf-8",
            )

            custom_dir = Path(tmp) / "custom" / "sk"
            custom_dir.mkdir(parents=True)
            (custom_dir / "SKILL.md").write_text(
                "---\nname: sk\ndescription: custom version\npriority: 90\n---\nCustom body.",
                encoding="utf-8",
            )

            loader = SkillLoader(dirs=[Path(tmp) / "public", Path(tmp) / "custom"])
            skills = loader.load_all()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].description, "custom version")

    def test_nonexistent_directory_ignored(self):
        loader = SkillLoader(dirs=[Path("/nonexistent/path")])
        self.assertEqual(loader.load_all(), [])

    def test_sorted_by_priority_descending(self):
        with tempfile.TemporaryDirectory() as tmp:
            public_dir = Path(tmp) / "public"
            for name, priority in [("low", 5), ("high", 90), ("mid", 50)]:
                skill_dir = public_dir / name
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(
                    f"---\nname: {name}\npriority: {priority}\n---\nbody",
                    encoding="utf-8",
                )

            loader = SkillLoader(dirs=[public_dir])
            skills = loader.load_all()
            self.assertEqual([skill.name for skill in skills], ["high", "mid", "low"])


if __name__ == "__main__":
    unittest.main()

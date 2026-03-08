from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from muse.skills.loader import Skill, SkillLoader, _parse_skill_md
from muse.skills.registry import SkillRegistry, _matches


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


class MatchesTests(unittest.TestCase):
    def test_wildcard_matches_anything(self):
        self.assertTrue(_matches(["*"], "anything"))

    def test_exact_match(self):
        self.assertTrue(_matches(["writing"], "writing"))

    def test_substring_match(self):
        self.assertTrue(_matches(["cs"], "Computer Science"))

    def test_reverse_substring_match(self):
        self.assertTrue(_matches(["Computer Science"], "cs"))

    def test_no_match(self):
        self.assertFalse(_matches(["physics"], "Computer Science"))


class SkillRegistryTests(unittest.TestCase):
    def _make_skill(
        self,
        name,
        stages=None,
        disciplines=None,
        languages=None,
        priority=50,
        body="Body.",
    ):
        return Skill(
            name=name,
            description=f"Desc for {name}",
            body=body,
            stages=stages or ["*"],
            disciplines=disciplines or ["*"],
            languages=languages or ["*"],
            priority=priority,
        )

    def test_get_for_context_filters_by_stage(self):
        s1 = self._make_skill("a", stages=["writing"])
        s2 = self._make_skill("b", stages=["search"])
        registry = SkillRegistry(skills=[s1, s2])
        matched = registry.get_for_context(stage="writing")
        self.assertEqual([skill.name for skill in matched], ["a"])

    def test_get_for_context_wildcard_stage(self):
        s1 = self._make_skill("a", stages=["*"])
        registry = SkillRegistry(skills=[s1])
        matched = registry.get_for_context(stage="polish")
        self.assertEqual(len(matched), 1)

    def test_get_for_context_filters_by_language(self):
        s1 = self._make_skill("zh-skill", languages=["zh"])
        s2 = self._make_skill("en-skill", languages=["en"])
        registry = SkillRegistry(skills=[s1, s2])
        matched = registry.get_for_context(language="zh")
        self.assertEqual([skill.name for skill in matched], ["zh-skill"])

    def test_render_for_prompt_empty_when_no_match(self):
        s1 = self._make_skill("a", stages=["search"])
        registry = SkillRegistry(skills=[s1])
        result = registry.render_for_prompt(stage="polish")
        self.assertEqual(result, "")

    def test_render_for_prompt_includes_body(self):
        s1 = self._make_skill("a", body="Use clear language.")
        registry = SkillRegistry(skills=[s1])
        result = registry.render_for_prompt()
        self.assertIn("Use clear language.", result)
        self.assertIn("### a", result)
        self.assertIn("DOMAIN KNOWLEDGE", result)

    def test_token_budget_enforced(self):
        big_body = "x" * 16400
        s1 = self._make_skill("big", body=big_body, priority=10)
        s2 = self._make_skill("small", body="Small.", priority=90)
        registry = SkillRegistry(skills=[s2, s1], token_budget=4000)
        result = registry.render_for_prompt()
        self.assertIn("Small.", result)
        self.assertNotIn("x" * 16400, result)

    def test_inject_into_prompt_appends(self):
        s1 = self._make_skill("a", body="Injected content.")
        registry = SkillRegistry(skills=[s1])
        original = "You are a thesis writer."
        result = registry.inject_into_prompt(original, stage="writing")
        self.assertTrue(result.startswith("You are a thesis writer."))
        self.assertIn("Injected content.", result)

    def test_inject_into_prompt_noop_when_no_match(self):
        s1 = self._make_skill("a", stages=["search"])
        registry = SkillRegistry(skills=[s1])
        original = "You are a thesis writer."
        result = registry.inject_into_prompt(original, stage="polish")
        self.assertEqual(result, original)

    def test_from_loader_with_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "pub" / "sk"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: sk\n---\nbody", encoding="utf-8")
            registry = SkillRegistry.from_loader(SkillLoader(dirs=[Path(tmp) / "pub"]))
            self.assertEqual(len(registry.all_skills), 1)

    def test_priority_ordering_in_render(self):
        s1 = self._make_skill("low", priority=10, body="Low.")
        s2 = self._make_skill("high", priority=90, body="High.")
        registry = SkillRegistry(skills=[s1, s2])
        result = registry.render_for_prompt()
        high_pos = result.index("High.")
        low_pos = result.index("Low.")
        self.assertLess(high_pos, low_pos)


if __name__ == "__main__":
    unittest.main()

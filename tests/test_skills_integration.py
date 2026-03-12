"""Integration tests for the Skills system (loader + registry + real skill files)."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from muse.skills.loader import Skill, SkillLoader
from muse.skills.registry import SkillRegistry


class SkillsIntegrationTests(unittest.TestCase):
    """Tests using the real skills/public/ directory."""

    def setUp(self):
        project_root = Path(__file__).resolve().parent.parent
        self.public_dir = project_root / "skills" / "public"
        if not self.public_dir.is_dir():
            self.skipTest("skills/public/ not found")

    def test_all_public_skills_load_without_error(self):
        loader = SkillLoader(dirs=[self.public_dir])
        skills = loader.load_all()
        self.assertGreater(len(skills), 0, "No skills loaded from skills/public/")
        for skill in skills:
            self.assertTrue(skill.name, "Skill has empty name")
            self.assertTrue(skill.body.strip(), f"Skill {skill.name} has empty body")
            self.assertGreater(skill.token_estimate, 0, f"Skill {skill.name} has zero tokens")

    def test_all_skills_have_unique_names(self):
        loader = SkillLoader(dirs=[self.public_dir])
        skills = loader.load_all()
        names = [skill.name for skill in skills]
        self.assertEqual(len(names), len(set(names)), f"Duplicate skill names: {names}")

    def test_chinese_writing_context_matches_expected_skills(self):
        registry = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        matched = registry.get_for_context(
            stage="writing", discipline="Computer Science", language="zh"
        )
        names = {skill.name for skill in matched}
        self.assertIn("academic-writing", names)
        self.assertIn("citation-gb-t-7714", names)
        self.assertIn("thesis-structure-zh", names)

    def test_english_writing_excludes_zh_only_skills(self):
        registry = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        matched = registry.get_for_context(stage="writing", language="en")
        names = {skill.name for skill in matched}
        self.assertNotIn("citation-gb-t-7714", names)
        self.assertNotIn("thesis-structure-zh", names)
        self.assertIn("academic-writing", names)

    def test_search_stage_matches_deep_research(self):
        registry = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        matched = registry.get_for_context(stage="search")
        names = {skill.name for skill in matched}
        self.assertIn("deep-research", names)

    def test_render_for_prompt_produces_non_empty_output(self):
        registry = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        result = registry.render_for_prompt(stage="writing", language="zh")
        self.assertIn("DOMAIN KNOWLEDGE", result)
        self.assertIn("END DOMAIN KNOWLEDGE", result)
        self.assertGreater(len(result), 100)

    def test_inject_into_prompt_preserves_original(self):
        registry = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        original = "Write one thesis subsection with citations."
        result = registry.inject_into_prompt(original, stage="writing", language="zh")
        self.assertTrue(result.startswith(original))
        self.assertGreater(len(result), len(original))

    def test_token_budget_caps_total_injection(self):
        registry = SkillRegistry.from_loader(
            SkillLoader(dirs=[self.public_dir]),
            token_budget=500,
        )
        result = registry.render_for_prompt(stage="writing", language="zh")
        if result:
            content_tokens = len(result.encode("utf-8")) // 4
            self.assertLess(content_tokens, 700, "Token budget not enforced")

    def test_custom_dir_overrides_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom" / "academic-writing"
            custom_dir.mkdir(parents=True)
            (custom_dir / "SKILL.md").write_text(
                textwrap.dedent(
                    """\
                    ---
                    name: academic-writing
                    description: Custom override
                    applies_to:
                      stages: ["*"]
                      disciplines: ["*"]
                      languages: ["*"]
                    priority: 99
                    ---
                    Custom academic writing rules for my university.
                """
                ),
                encoding="utf-8",
            )

            loader = SkillLoader(dirs=[self.public_dir, Path(tmp) / "custom"])
            registry = SkillRegistry.from_loader(loader)
            matched = registry.get_for_context(stage="writing")
            academic_writing = [skill for skill in matched if skill.name == "academic-writing"]
            self.assertEqual(len(academic_writing), 1)
            self.assertEqual(academic_writing[0].description, "Custom override")
            self.assertIn("Custom academic writing rules", academic_writing[0].body)

    def test_full_pipeline_simulate_outline_prompt(self):
        registry = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        base_system = (
            "Generate a thesis outline as JSON with keys: chapters (list). Each chapter must include "
            "chapter_id, chapter_title, target_words, complexity, subsections."
        )
        enhanced = registry.inject_into_prompt(
            base_system,
            stage="outline",
            discipline="Computer Science",
            language="zh",
        )
        self.assertTrue(enhanced.startswith(base_system))
        self.assertIn("DOMAIN KNOWLEDGE", enhanced)


class SkillsTokenBudgetEdgeCases(unittest.TestCase):
    """Edge cases for token budget enforcement."""

    def test_single_skill_exceeding_budget_gets_truncated(self):
        big = Skill(
            name="huge",
            description="Big skill",
            body="word " * 5000,
            priority=90,
        )
        registry = SkillRegistry(skills=[big], token_budget=500)
        result = registry.render_for_prompt()
        self.assertIn("[truncated]", result)

    def test_zero_budget_returns_empty(self):
        skill = Skill(name="s", description="d", body="content")
        registry = SkillRegistry(skills=[skill], token_budget=0)
        result = registry.render_for_prompt()
        self.assertEqual(result, "")

    def test_exact_budget_fit(self):
        body = "x" * 400
        skill = Skill(name="fit", description="d", body=body, priority=50)
        registry = SkillRegistry(skills=[skill], token_budget=100)
        result = registry.render_for_prompt()
        self.assertIn("fit", result)
        self.assertNotIn("[truncated]", result)

    def test_multiple_skills_partial_fit(self):
        s1 = Skill(name="first", description="d", body="a" * 800, priority=90)
        s2 = Skill(name="second", description="d", body="b" * 800, priority=50)
        s3 = Skill(name="third", description="d", body="c" * 800, priority=10)
        registry = SkillRegistry(skills=[s1, s2, s3], token_budget=450)
        result = registry.render_for_prompt()
        self.assertIn("first", result)
        self.assertIn("second", result)


if __name__ == "__main__":
    unittest.main()

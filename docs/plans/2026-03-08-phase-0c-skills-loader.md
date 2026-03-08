# Phase 0-C: Skills Loader

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Load domain knowledge from SKILL.md files and inject into LLM prompts by stage/discipline/language.

**Architecture:** SkillLoader scans public/ and custom/ directories, parses YAML front-matter + Markdown body. SkillRegistry matches skills by context and injects into system prompts with a 4000-token budget.

**Tech Stack:** Python 3.10, PyYAML (already installed: 6.0.3)

**Depends on:** Nothing (foundation, parallel with Phase 0-A/B)

---

## Task 1: Create Skill dataclass and SkillLoader

**Files:**
- `muse/skills/__init__.py` (new)
- `muse/skills/loader.py` (new)

**Why:** Foundation for the skills system. SkillLoader scans directories for SKILL.md files, parses the YAML front-matter (between `---` fences) and the Markdown body into a `Skill` dataclass.

### 1a. Create `muse/skills/__init__.py`

```python
"""Skills system for injecting domain knowledge into LLM prompts."""

from .loader import Skill, SkillLoader

__all__ = ["Skill", "SkillLoader"]
```

### 1b. Create `muse/skills/loader.py`

```python
"""Parse SKILL.md files into Skill dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Skill:
    """A single loaded skill with metadata and body text."""

    name: str
    description: str
    body: str
    stages: list[str] = field(default_factory=lambda: ["*"])
    disciplines: list[str] = field(default_factory=lambda: ["*"])
    languages: list[str] = field(default_factory=lambda: ["*"])
    priority: int = 50
    source_path: str = ""

    @property
    def token_estimate(self) -> int:
        """Estimate token count using the 4-bytes-per-token heuristic."""
        return len(self.body.encode("utf-8")) // 4


def _parse_skill_md(text: str, source_path: str = "") -> Skill | None:
    """Parse a SKILL.md file with YAML front-matter + Markdown body.

    Returns None if the file cannot be parsed or is missing required fields.
    """
    text = text.strip()
    if not text.startswith("---"):
        return None

    # Find closing front-matter fence
    end = text.find("---", 3)
    if end == -1:
        return None

    front_matter_raw = text[3:end].strip()
    body = text[end + 3:].strip()

    try:
        meta: dict[str, Any] = yaml.safe_load(front_matter_raw) or {}
    except yaml.YAMLError:
        return None

    if not isinstance(meta, dict):
        return None

    name = meta.get("name", "").strip()
    if not name:
        return None

    applies_to = meta.get("applies_to", {})
    if not isinstance(applies_to, dict):
        applies_to = {}

    stages = applies_to.get("stages", ["*"])
    disciplines = applies_to.get("disciplines", ["*"])
    languages = applies_to.get("languages", ["*"])

    # Normalize to lists
    if isinstance(stages, str):
        stages = [stages]
    if isinstance(disciplines, str):
        disciplines = [disciplines]
    if isinstance(languages, str):
        languages = [languages]

    priority = int(meta.get("priority", 50))

    return Skill(
        name=name,
        description=meta.get("description", ""),
        body=body,
        stages=stages,
        disciplines=disciplines,
        languages=languages,
        priority=priority,
        source_path=source_path,
    )


class SkillLoader:
    """Scan directories for SKILL.md files and load them."""

    def __init__(self, dirs: list[str | Path] | None = None) -> None:
        if dirs is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            dirs = [
                project_root / "skills" / "public",
                project_root / "skills" / "custom",
            ]
        self._dirs = [Path(d) for d in dirs]

    def load_all(self) -> list[Skill]:
        """Scan all directories and return loaded skills, sorted by priority (higher first)."""
        skills: list[Skill] = []
        seen_names: set[str] = set()

        # Process custom/ dirs last so they override public/ by name
        for scan_dir in self._dirs:
            if not scan_dir.is_dir():
                continue
            for skill_dir in sorted(scan_dir.iterdir()):
                skill_file = skill_dir / "SKILL.md"
                if not skill_file.is_file():
                    continue
                try:
                    text = skill_file.read_text(encoding="utf-8")
                except OSError:
                    continue
                skill = _parse_skill_md(text, source_path=str(skill_file))
                if skill is None:
                    continue
                if skill.name in seen_names:
                    # Later dirs (custom) override earlier (public)
                    skills = [s for s in skills if s.name != skill.name]
                seen_names.add(skill.name)
                skills.append(skill)

        # Sort by priority descending (higher priority first)
        skills.sort(key=lambda s: s.priority, reverse=True)
        return skills
```

### TDD

Create `tests/test_skills_loader.py`:

```python
import tempfile
import textwrap
import unittest
from pathlib import Path

from muse.skills.loader import Skill, SkillLoader, _parse_skill_md


class ParseSkillMdTests(unittest.TestCase):
    def test_valid_skill_file(self):
        text = textwrap.dedent("""\
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
        """)
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
        text = "just plain markdown"
        self.assertIsNone(_parse_skill_md(text))

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
        text = textwrap.dedent("""\
            ---
            name: single
            applies_to:
              stages: writing
              disciplines: cs
              languages: en
            ---
            body
        """)
        skill = _parse_skill_md(text)
        self.assertEqual(skill.stages, ["writing"])
        self.assertEqual(skill.disciplines, ["cs"])
        self.assertEqual(skill.languages, ["en"])


class SkillLoaderTests(unittest.TestCase):
    def test_load_from_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            pub = Path(tmp) / "public"
            skill_dir = pub / "my-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: my-skill\ndescription: test\npriority: 20\n---\nBody."
            )

            loader = SkillLoader(dirs=[pub])
            skills = loader.load_all()
            self.assertEqual(len(skills), 1)
            self.assertEqual(skills[0].name, "my-skill")

    def test_custom_overrides_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            pub = Path(tmp) / "public" / "sk"
            pub.mkdir(parents=True)
            (pub / "SKILL.md").write_text(
                "---\nname: sk\ndescription: public version\npriority: 10\n---\nPublic body."
            )

            custom = Path(tmp) / "custom" / "sk"
            custom.mkdir(parents=True)
            (custom / "SKILL.md").write_text(
                "---\nname: sk\ndescription: custom version\npriority: 90\n---\nCustom body."
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
            pub = Path(tmp) / "public"
            for name, prio in [("low", 5), ("high", 90), ("mid", 50)]:
                d = pub / name
                d.mkdir(parents=True)
                (d / "SKILL.md").write_text(
                    f"---\nname: {name}\npriority: {prio}\n---\nbody"
                )

            loader = SkillLoader(dirs=[pub])
            skills = loader.load_all()
            self.assertEqual([s.name for s in skills], ["high", "mid", "low"])


if __name__ == "__main__":
    unittest.main()
```

**Verify:**
```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_skills_loader.py -v
```

---

## Task 2: Create SkillRegistry

**Files:**
- `muse/skills/registry.py` (new)
- Update `muse/skills/__init__.py`

**Why:** SkillRegistry takes loaded skills, matches them against the current context (stage, discipline, language), and produces a formatted text block to inject into system prompts. Enforces a 4000-token budget by priority ordering.

### 2a. Create `muse/skills/registry.py`

```python
"""Match skills to context and inject into system prompts."""

from __future__ import annotations

from muse.skills.loader import Skill, SkillLoader


_DEFAULT_TOKEN_BUDGET = 4000

_INJECTION_HEADER = (
    "\n\n--- DOMAIN KNOWLEDGE (from skills) ---\n"
)
_INJECTION_FOOTER = (
    "\n--- END DOMAIN KNOWLEDGE ---\n"
)


def _matches(skill_values: list[str], target: str) -> bool:
    """Check if a skill's filter list matches a target value.

    A list containing "*" matches everything. Otherwise, case-insensitive
    substring matching is used so that discipline="Computer Science" matches
    a skill with disciplines=["cs"].
    """
    if "*" in skill_values:
        return True
    target_lower = target.lower()
    for val in skill_values:
        val_lower = val.lower()
        if val_lower in target_lower or target_lower in val_lower:
            return True
    return False


class SkillRegistry:
    """Hold loaded skills and resolve them by context."""

    def __init__(
        self,
        skills: list[Skill] | None = None,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._skills = list(skills) if skills else []
        self._token_budget = token_budget

    @classmethod
    def from_loader(
        cls,
        loader: SkillLoader | None = None,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> "SkillRegistry":
        """Build a registry by scanning directories via SkillLoader."""
        loader = loader or SkillLoader()
        return cls(skills=loader.load_all(), token_budget=token_budget)

    @property
    def all_skills(self) -> list[Skill]:
        return list(self._skills)

    def get_for_context(
        self,
        *,
        stage: str = "*",
        discipline: str = "*",
        language: str = "*",
    ) -> list[Skill]:
        """Return skills matching the given context, sorted by priority descending."""
        matched: list[Skill] = []
        for skill in self._skills:
            if (
                _matches(skill.stages, stage)
                and _matches(skill.disciplines, discipline)
                and _matches(skill.languages, language)
            ):
                matched.append(skill)
        # Already sorted by priority from loader, but re-sort to be safe
        matched.sort(key=lambda s: s.priority, reverse=True)
        return matched

    def render_for_prompt(
        self,
        *,
        stage: str = "*",
        discipline: str = "*",
        language: str = "*",
    ) -> str:
        """Render matched skills into a text block for system prompt injection.

        Respects the token budget. Skills are included in priority order until
        the budget is exhausted. Returns empty string if no skills match.
        """
        matched = self.get_for_context(
            stage=stage, discipline=discipline, language=language,
        )
        if not matched:
            return ""

        parts: list[str] = []
        remaining = self._token_budget
        for skill in matched:
            cost = skill.token_estimate
            if cost > remaining:
                # Try to include a truncated version if at least 100 tokens fit
                if remaining >= 100:
                    chars = remaining * 4  # reverse the 4-bytes heuristic
                    truncated_body = skill.body[:chars].rsplit("\n", 1)[0]
                    parts.append(f"### {skill.name}\n{truncated_body}\n[truncated]")
                break
            parts.append(f"### {skill.name}\n{skill.body}")
            remaining -= cost

        if not parts:
            return ""
        return _INJECTION_HEADER + "\n\n".join(parts) + _INJECTION_FOOTER

    def inject_into_prompt(self, system_prompt: str, **context: str) -> str:
        """Append matched skills to an existing system prompt."""
        block = self.render_for_prompt(**context)
        if not block:
            return system_prompt
        return system_prompt + block
```

### 2b. Update `muse/skills/__init__.py`

```python
"""Skills system for injecting domain knowledge into LLM prompts."""

from .loader import Skill, SkillLoader
from .registry import SkillRegistry

__all__ = ["Skill", "SkillLoader", "SkillRegistry"]
```

### TDD

Add to `tests/test_skills_loader.py`:

```python
from muse.skills.registry import SkillRegistry, _matches


class MatchesTests(unittest.TestCase):
    def test_wildcard_matches_anything(self):
        self.assertTrue(_matches(["*"], "anything"))

    def test_exact_match(self):
        self.assertTrue(_matches(["writing"], "writing"))

    def test_substring_match(self):
        self.assertTrue(_matches(["cs"], "Computer Science"))

    def test_reverse_substring_match(self):
        self.assertTrue(_matches(["Computer Science"], "cs"))
        # "cs" is in "Computer Science" (case insensitive) -> True

    def test_no_match(self):
        self.assertFalse(_matches(["physics"], "Computer Science"))


class SkillRegistryTests(unittest.TestCase):
    def _make_skill(self, name, stages=None, disciplines=None, languages=None, priority=50, body="Body."):
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
        reg = SkillRegistry(skills=[s1, s2])
        matched = reg.get_for_context(stage="writing")
        self.assertEqual([s.name for s in matched], ["a"])

    def test_get_for_context_wildcard_stage(self):
        s1 = self._make_skill("a", stages=["*"])
        reg = SkillRegistry(skills=[s1])
        matched = reg.get_for_context(stage="polish")
        self.assertEqual(len(matched), 1)

    def test_get_for_context_filters_by_language(self):
        s1 = self._make_skill("zh-skill", languages=["zh"])
        s2 = self._make_skill("en-skill", languages=["en"])
        reg = SkillRegistry(skills=[s1, s2])
        matched = reg.get_for_context(language="zh")
        self.assertEqual([s.name for s in matched], ["zh-skill"])

    def test_render_for_prompt_empty_when_no_match(self):
        s1 = self._make_skill("a", stages=["search"])
        reg = SkillRegistry(skills=[s1])
        result = reg.render_for_prompt(stage="polish")
        self.assertEqual(result, "")

    def test_render_for_prompt_includes_body(self):
        s1 = self._make_skill("a", body="Use clear language.")
        reg = SkillRegistry(skills=[s1])
        result = reg.render_for_prompt()
        self.assertIn("Use clear language.", result)
        self.assertIn("### a", result)
        self.assertIn("DOMAIN KNOWLEDGE", result)

    def test_token_budget_enforced(self):
        big_body = "x" * 16400  # ~4100 tokens, exceeds 4000 budget
        s1 = self._make_skill("big", body=big_body, priority=10)
        s2 = self._make_skill("small", body="Small.", priority=90)
        reg = SkillRegistry(skills=[s2, s1], token_budget=4000)
        result = reg.render_for_prompt()
        self.assertIn("Small.", result)
        # big skill should be truncated or excluded
        self.assertNotIn("x" * 16400, result)

    def test_inject_into_prompt_appends(self):
        s1 = self._make_skill("a", body="Injected content.")
        reg = SkillRegistry(skills=[s1])
        original = "You are a thesis writer."
        result = reg.inject_into_prompt(original, stage="writing")
        self.assertTrue(result.startswith("You are a thesis writer."))
        self.assertIn("Injected content.", result)

    def test_inject_into_prompt_noop_when_no_match(self):
        s1 = self._make_skill("a", stages=["search"])
        reg = SkillRegistry(skills=[s1])
        original = "You are a thesis writer."
        result = reg.inject_into_prompt(original, stage="polish")
        self.assertEqual(result, original)

    def test_from_loader_with_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "pub" / "sk"
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text("---\nname: sk\n---\nbody")
            reg = SkillRegistry.from_loader(SkillLoader(dirs=[Path(tmp) / "pub"]))
            self.assertEqual(len(reg.all_skills), 1)

    def test_priority_ordering_in_render(self):
        s1 = self._make_skill("low", priority=10, body="Low.")
        s2 = self._make_skill("high", priority=90, body="High.")
        reg = SkillRegistry(skills=[s1, s2])
        result = reg.render_for_prompt()
        high_pos = result.index("High.")
        low_pos = result.index("Low.")
        self.assertLess(high_pos, low_pos)
```

**Verify:**
```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_skills_loader.py -v
```

---

## Task 3: Create academic-writing skill

**File:** `skills/public/academic-writing/SKILL.md` (new)

**Why:** Baseline writing-quality skill applied to all writing and polish stages. Covers academic register, paragraph structure, hedging language, and common pitfalls.

```markdown
---
name: academic-writing
description: Academic writing conventions and quality standards
applies_to:
  stages: [writing, polish, outline]
  disciplines: ["*"]
  languages: ["*"]
priority: 40
---
# Academic Writing Standards

## Paragraph Structure
- Each paragraph must have ONE clear topic sentence (first or second sentence).
- Support with evidence, examples, or reasoning (2-4 sentences).
- End with a transition or implication sentence linking to the next paragraph.
- Target: 150-250 words per paragraph. Split if longer.

## Register and Tone
- Use third person or passive voice for objectivity ("this study examines", "it was observed that").
- Avoid colloquial language, contractions, and first-person pronouns unless discipline conventions allow.
- Use hedging for claims not directly proven: "suggests", "indicates", "appears to", "may".
- Use assertive language for well-established facts: "demonstrates", "confirms", "establishes".

## Logical Flow
- Every claim must be either (a) supported by a citation, (b) derived from presented data, or (c) explicitly flagged as the author's contribution.
- Use signposting phrases: "First,...", "In contrast,...", "Building on this,...", "However,...".
- Each chapter must open with a brief overview paragraph and close with a summary linking to the next chapter.

## Common Pitfalls to Avoid
- Do not repeat the same information in multiple sections.
- Do not introduce new concepts in a conclusion section.
- Do not use vague quantifiers ("many", "some") without specifying context.
- Do not leave acronyms undefined on first use.
- Do not mix tenses: use past tense for completed work, present for general truths.

## Citation Integration
- Integrate citations into the narrative; do not dump a list of references.
- Prefer "Author (year) found that..." or "...has been demonstrated (Author, year)".
- When citing multiple sources for one claim, order chronologically or by relevance.
```

**Verify:**
```bash
cd /home/planck/gradute/Muse && python -c "
from muse.skills.loader import SkillLoader
loader = SkillLoader()
skills = loader.load_all()
names = [s.name for s in skills]
assert 'academic-writing' in names, f'Not found in {names}'
print('OK: academic-writing loaded')
"
```

---

## Task 4: Create citation-gb-t-7714 skill

**File:** `skills/public/citation-gb-t-7714/SKILL.md` (new)

**Why:** Chinese thesis standard (GB/T 7714-2015) has specific citation formatting rules. This skill injects those rules into writing and citation stages when language is Chinese.

```markdown
---
name: citation-gb-t-7714
description: "GB/T 7714-2015 citation format rules for Chinese academic writing"
applies_to:
  stages: [writing, citation, polish]
  disciplines: ["*"]
  languages: [zh]
priority: 60
---
# GB/T 7714-2015 Citation Format

## In-text Citation Rules
- Use sequential numbering in square brackets: [1], [2], [3].
- Multiple citations in one position: [1,3,5] or [1-5] for consecutive ranges.
- Place the citation number BEFORE the period when citing the whole sentence.
- Place it AFTER the relevant phrase when citing a specific claim.
- Superscript format is acceptable: ^\[1\]^.

## Reference List Format
- Number each reference sequentially as [1], [2], etc. in order of first citation.
- Author format: 姓在前名在后, use comma between authors, "et al." or "等" after 3 authors.

### By Document Type

**Journal article [J]:**
[序号] 作者. 题名[J]. 刊名, 出版年, 卷(期): 起止页码.
Example: [1] 王明, 李华. 区块链分片技术综述[J]. 计算机学报, 2024, 47(3): 512-530.

**Conference paper [C]:**
[序号] 作者. 题名[C]// 会议名. 出版地: 出版者, 出版年: 起止页码.

**Book [M]:**
[序号] 作者. 书名[M]. 版本. 出版地: 出版者, 出版年: 页码.

**Dissertation [D]:**
[序号] 作者. 题名[D]. 保存地: 保存单位, 年份.

**Online resource [EB/OL]:**
[序号] 作者. 题名[EB/OL]. (发布日期)[引用日期]. URL.

## Consistency Rules
- Every citation number in the text MUST have a corresponding entry in the reference list.
- No orphan references: every reference list entry MUST be cited in the text.
- Use Chinese punctuation in Chinese text (。，：；).
- Maintain a consistent citation style throughout the entire thesis.
```

**Verify:**
```bash
cd /home/planck/gradute/Muse && python -c "
from muse.skills.loader import SkillLoader
from muse.skills.registry import SkillRegistry
reg = SkillRegistry.from_loader()
matched = reg.get_for_context(stage='writing', language='zh')
names = [s.name for s in matched]
assert 'citation-gb-t-7714' in names, f'Not found in {names}'
print('OK: citation-gb-t-7714 matched for writing+zh')
# Should NOT match for English
matched_en = reg.get_for_context(stage='writing', language='en')
en_names = [s.name for s in matched_en]
assert 'citation-gb-t-7714' not in en_names, f'Should not match en: {en_names}'
print('OK: citation-gb-t-7714 correctly excluded for en')
"
```

---

## Task 5: Create thesis-structure-zh skill

**File:** `skills/public/thesis-structure-zh/SKILL.md` (new)

**Why:** Chinese universities (especially BUPT) have specific thesis structure requirements. This skill guides outline generation and writing stages for Chinese-language theses.

```markdown
---
name: thesis-structure-zh
description: Chinese thesis structure conventions (BUPT and general)
applies_to:
  stages: [outline, writing, polish]
  disciplines: ["*"]
  languages: [zh]
priority: 55
---
# Chinese Thesis Structure Conventions

## Standard Chapter Structure (理工科)
A Chinese master's thesis typically follows this structure:

1. **第一章 绪论** (Introduction, ~3000-5000 words)
   - 研究背景与意义 (Background and significance)
   - 国内外研究现状 (Domestic and international research status)
   - 研究内容与章节安排 (Research content and chapter organization)

2. **第二章 相关技术/理论基础** (Related work / Theoretical foundation, ~4000-6000 words)
   - Present key theories, algorithms, or frameworks used
   - Each subsection covers one technology with formal definitions

3. **第三章-第四章 系统设计与实现** (Design and implementation, ~8000-12000 words)
   - System architecture with diagrams
   - Key module design with pseudocode or algorithms
   - Implementation details

4. **第五章 实验与分析** (Experiments and analysis, ~4000-6000 words)
   - Experimental setup (datasets, metrics, baselines)
   - Results with tables and figures
   - Analysis and discussion

5. **第六章 总结与展望** (Conclusion and future work, ~1500-2500 words)
   - Summarize contributions (match what was promised in Chapter 1)
   - Limitations
   - Future directions

## Formatting Requirements
- Chapter titles: 第X章 + title (e.g., 第三章 系统设计)
- Section numbering: X.Y format (e.g., 3.1, 3.2)
- Subsection numbering: X.Y.Z format (e.g., 3.1.1)
- Figures and tables: 图X-Y / 表X-Y (Figure/Table chapter-sequence)
- All figures and tables MUST be referenced in the text before they appear.

## Language Requirements
- Use formal written Chinese (书面语), avoid spoken Chinese (口语).
- Technical terms: provide English in parentheses on first use, e.g., 区块链(Blockchain).
- Mathematical notation: use standard LaTeX-compatible notation.
- Consistent terminology throughout: create a glossary and stick to it.

## Word Count Guidelines (Master's Thesis)
- Total: 30,000-50,000 Chinese characters (excluding references and appendices).
- Abstract (Chinese): 300-500 characters.
- Abstract (English): 200-300 words.
- Keywords: 3-5, in both Chinese and English.
```

**Verify:**
```bash
cd /home/planck/gradute/Muse && python -c "
from muse.skills.registry import SkillRegistry
reg = SkillRegistry.from_loader()
matched = reg.get_for_context(stage='outline', language='zh')
names = [s.name for s in matched]
assert 'thesis-structure-zh' in names, f'Not found in {names}'
print('OK: thesis-structure-zh matched for outline+zh')
"
```

---

## Task 6: Create deep-research skill

**File:** `skills/public/deep-research/SKILL.md` (new)

**Why:** Inspired by DeerFlow's deep-research skill. Guides the search and writing stages on how to conduct thorough academic literature research with systematic methodology.

```markdown
---
name: deep-research
description: Systematic deep research methodology for academic literature discovery
applies_to:
  stages: [search, outline, writing]
  disciplines: ["*"]
  languages: ["*"]
priority: 35
---
# Deep Research Methodology

## Research Strategy

### Phase 1: Broad Survey
- Start with the core topic and generate 5-7 diverse search queries covering:
  - The exact research topic
  - Key methodology terms
  - Related subfields and alternative terminology
  - Seminal/foundational works ("survey", "tutorial", "comprehensive review")
  - Contrasting or competing approaches
- Prioritize recent survey papers (last 3 years) for initial orientation.

### Phase 2: Citation Chain Exploration
- From the top 5-10 most relevant papers, follow both:
  - **Backward references**: papers cited by these works (foundational literature).
  - **Forward citations**: papers that cite these works (recent developments).
- Look for papers appearing in multiple citation chains (high-impact nodes).

### Phase 3: Gap Identification
- Compare the set of found papers against the research questions.
- Identify areas where coverage is thin or contradictory.
- Generate targeted follow-up queries for under-covered aspects.

## Source Evaluation Criteria
- **Relevance**: Does the paper directly address the research question?
- **Recency**: Prefer papers from the last 5 years unless citing foundational work.
- **Venue quality**: Prefer top-tier conferences (e.g., SIGMOD, VLDB, OSDI, NeurIPS) and high-impact journals.
- **Citation count**: Higher citations indicate broader impact, but do not dismiss new papers.
- **Methodology rigor**: Does the paper provide reproducible experiments with baselines?

## Literature Summary Guidelines
- Group references by theme, NOT by author or chronological order.
- For each theme, present: (a) the problem, (b) key approaches, (c) limitations, (d) how it relates to this thesis.
- Identify the research gap that this thesis fills.
- Use a comparison table for closely related works (columns: method, dataset, metric, result).

## When Writing Literature Review
- Do NOT simply list papers. Synthesize and compare.
- Use transition phrases to connect different research threads.
- Build a narrative that leads from the general field to the specific gap this thesis addresses.
- End the literature review with a clear statement of what is missing and how this thesis contributes.
```

**Verify:**
```bash
cd /home/planck/gradute/Muse && python -c "
from muse.skills.registry import SkillRegistry
reg = SkillRegistry.from_loader()
matched = reg.get_for_context(stage='search')
names = [s.name for s in matched]
assert 'deep-research' in names, f'Not found in {names}'
print('OK: deep-research matched for search stage')
# Should also match writing
matched_w = reg.get_for_context(stage='writing')
assert 'deep-research' in [s.name for s in matched_w]
print('OK: deep-research also matched for writing stage')
"
```

---

## Task 7: Integration test -- SkillLoader + SkillRegistry end-to-end

**File:** `tests/test_skills_integration.py` (new)

**Why:** End-to-end test that loads real skill files from `skills/public/`, verifies the full matching and prompt injection pipeline, and confirms token budget enforcement works with real content.

### Create `tests/test_skills_integration.py`

```python
"""Integration tests for the Skills system (loader + registry + real skill files)."""

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
        names = [s.name for s in skills]
        self.assertEqual(len(names), len(set(names)), f"Duplicate skill names: {names}")

    def test_chinese_writing_context_matches_expected_skills(self):
        reg = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        matched = reg.get_for_context(stage="writing", discipline="Computer Science", language="zh")
        names = {s.name for s in matched}
        # These skills should match Chinese CS writing context
        self.assertIn("academic-writing", names)
        self.assertIn("citation-gb-t-7714", names)
        self.assertIn("thesis-structure-zh", names)

    def test_english_writing_excludes_zh_only_skills(self):
        reg = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        matched = reg.get_for_context(stage="writing", language="en")
        names = {s.name for s in matched}
        self.assertNotIn("citation-gb-t-7714", names)
        self.assertNotIn("thesis-structure-zh", names)
        # academic-writing has languages=["*"], so it should match
        self.assertIn("academic-writing", names)

    def test_search_stage_matches_deep_research(self):
        reg = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        matched = reg.get_for_context(stage="search")
        names = {s.name for s in matched}
        self.assertIn("deep-research", names)

    def test_render_for_prompt_produces_non_empty_output(self):
        reg = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        result = reg.render_for_prompt(stage="writing", language="zh")
        self.assertIn("DOMAIN KNOWLEDGE", result)
        self.assertIn("END DOMAIN KNOWLEDGE", result)
        # Should contain actual content
        self.assertGreater(len(result), 100)

    def test_inject_into_prompt_preserves_original(self):
        reg = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        original = "Write one thesis subsection with citations."
        result = reg.inject_into_prompt(original, stage="writing", language="zh")
        self.assertTrue(result.startswith(original))
        self.assertGreater(len(result), len(original))

    def test_token_budget_caps_total_injection(self):
        reg = SkillRegistry.from_loader(
            SkillLoader(dirs=[self.public_dir]),
            token_budget=500,
        )
        result = reg.render_for_prompt(stage="writing", language="zh")
        if result:
            # Estimate tokens of the rendered content (excluding header/footer)
            content_bytes = len(result.encode("utf-8"))
            content_tokens = content_bytes // 4
            # Should be roughly within budget (allow some overhead for headers)
            self.assertLess(content_tokens, 700, "Token budget not enforced")

    def test_custom_dir_overrides_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_dir = Path(tmp) / "custom" / "academic-writing"
            custom_dir.mkdir(parents=True)
            (custom_dir / "SKILL.md").write_text(textwrap.dedent("""\
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
            """))

            loader = SkillLoader(dirs=[self.public_dir, Path(tmp) / "custom"])
            reg = SkillRegistry.from_loader(loader)
            matched = reg.get_for_context(stage="writing")
            aw = [s for s in matched if s.name == "academic-writing"]
            self.assertEqual(len(aw), 1)
            self.assertEqual(aw[0].description, "Custom override")
            self.assertIn("Custom academic writing rules", aw[0].body)

    def test_full_pipeline_simulate_outline_prompt(self):
        """Simulate what prompt injection looks like for the outline stage."""
        reg = SkillRegistry.from_loader(SkillLoader(dirs=[self.public_dir]))
        base_system = (
            "Generate a thesis outline as JSON with keys: chapters (list). Each chapter must include "
            "chapter_id, chapter_title, target_words, complexity, subsections."
        )
        enhanced = reg.inject_into_prompt(
            base_system,
            stage="outline",
            discipline="Computer Science",
            language="zh",
        )
        # Should have the original prompt plus skill content
        self.assertTrue(enhanced.startswith(base_system))
        # Should include relevant skills (thesis-structure-zh, academic-writing match outline+zh)
        self.assertIn("DOMAIN KNOWLEDGE", enhanced)


class SkillsTokenBudgetEdgeCases(unittest.TestCase):
    """Edge cases for token budget enforcement."""

    def test_single_skill_exceeding_budget_gets_truncated(self):
        big = Skill(
            name="huge",
            description="Big skill",
            body="word " * 5000,  # ~5000 tokens
            priority=90,
        )
        reg = SkillRegistry(skills=[big], token_budget=500)
        result = reg.render_for_prompt()
        self.assertIn("[truncated]", result)

    def test_zero_budget_returns_empty(self):
        s = Skill(name="s", description="d", body="content")
        reg = SkillRegistry(skills=[s], token_budget=0)
        result = reg.render_for_prompt()
        self.assertEqual(result, "")

    def test_exact_budget_fit(self):
        body = "x" * 400  # exactly 100 tokens
        s = Skill(name="fit", description="d", body=body, priority=50)
        reg = SkillRegistry(skills=[s], token_budget=100)
        result = reg.render_for_prompt()
        self.assertIn("fit", result)
        self.assertNotIn("[truncated]", result)

    def test_multiple_skills_partial_fit(self):
        s1 = Skill(name="first", description="d", body="a" * 800, priority=90)   # 200 tokens
        s2 = Skill(name="second", description="d", body="b" * 800, priority=50)  # 200 tokens
        s3 = Skill(name="third", description="d", body="c" * 800, priority=10)   # 200 tokens
        reg = SkillRegistry(skills=[s1, s2, s3], token_budget=450)
        result = reg.render_for_prompt()
        self.assertIn("first", result)
        self.assertIn("second", result)
        # third might be truncated or excluded depending on overhead
        # but first two should definitely be there


if __name__ == "__main__":
    unittest.main()
```

**Verify:**
```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_skills_integration.py -v
```

**Full suite:**
```bash
cd /home/planck/gradute/Muse && python -m pytest tests/test_skills_loader.py tests/test_skills_integration.py -v
```

---

## Summary of Files Created

| File | Purpose |
|------|---------|
| `muse/skills/__init__.py` | Package init, exports Skill, SkillLoader, SkillRegistry |
| `muse/skills/loader.py` | Skill dataclass + SkillLoader (parse SKILL.md, scan dirs) |
| `muse/skills/registry.py` | SkillRegistry (match context, render, inject into prompt) |
| `skills/public/academic-writing/SKILL.md` | General academic writing conventions |
| `skills/public/citation-gb-t-7714/SKILL.md` | GB/T 7714-2015 citation rules (Chinese) |
| `skills/public/thesis-structure-zh/SKILL.md` | Chinese thesis structure conventions |
| `skills/public/deep-research/SKILL.md` | Deep research methodology |
| `tests/test_skills_loader.py` | Unit tests for loader + registry |
| `tests/test_skills_integration.py` | Integration tests with real skill files |

## Integration Points (for later phases)

After this phase, skills are injected into prompts by calling:

```python
# In a prompt function or node:
registry = SkillRegistry.from_loader()
system_prompt = registry.inject_into_prompt(
    base_system_prompt,
    stage="writing",
    discipline=state["discipline"],
    language=state["language"],
)
```

Or in Phase 0-A's `MuseChatModel._generate()`:

```python
if self._skill_registry:
    system = self._skill_registry.inject_into_prompt(
        system, stage=config.get("stage", "*"),
        discipline=config.get("discipline", "*"),
        language=config.get("language", "*"),
    )
```

No existing files are modified in this phase. Skills are purely additive.

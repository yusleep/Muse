# Rename Project To Muse Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rename the project package, CLI, tests, templates, and primary docs from `Thesis Agent` / `thesis_agent` / `thesis-agent` to `Muse` / `muse` with no compatibility alias layer.

**Architecture:** Treat the rename as a pure naming refactor layered on top of the current working tree. Rename the Python package directory to `muse`, update all imports and command examples, then sweep docs/tests/templates for the old name and validate with full-text search plus the full test suite.

**Tech Stack:** Python 3, `unittest`, `rg`, existing package/test layout, isolated git worktree.

---

### Task 1: Rename the Python package directory and internal imports

**Files:**
- Move: `thesis_agent` → `muse`
- Modify: `muse/__init__.py`
- Modify: `muse/__main__.py`
- Modify: `muse/*.py`
- Test: `tests/*.py`

**Step 1: Write the failing test**

Add or update a CLI/import regression test that imports `muse` and executes `python3 -m muse --help`.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest discover -s tests -p 'test_cli.py' -v`

Expected: FAIL because the `muse` module/package does not exist yet.

**Step 3: Write minimal implementation**

- Rename `thesis_agent/` to `muse/`
- Replace internal imports from `thesis_agent` to `muse`
- Update package exports and module entrypoints

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest discover -s tests -p 'test_cli.py' -v`

Expected: PASS with the new package name.

**Step 5: Commit**

Do not commit unless the user explicitly requests it.

### Task 2: Rename the CLI surface and command examples

**Files:**
- Modify: `muse/cli.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing test**

Update CLI tests to assert the argparse program name and example command usage match `muse`.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest discover -s tests -p 'test_cli.py' -v`

Expected: FAIL because `prog="muse"` and old command examples still remain.

**Step 3: Write minimal implementation**

- Change `argparse.ArgumentParser(prog="thesis-agent")` to `prog="muse"`
- Update README command examples to `python3 -m muse ...`

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest discover -s tests -p 'test_cli.py' -v`

Expected: PASS.

**Step 5: Commit**

Do not commit unless the user explicitly requests it.

### Task 3: Sweep tests and docs for old project names

**Files:**
- Modify: `tests/*.py`
- Modify: `README.md`
- Modify: `muse-plan-v2.md`
- Modify: `docs/plans/*.md`
- Modify: `tasks/*.md`
- Modify: `tasks/*.json`

**Step 1: Write the failing test**

Add/extend a public-surface regression that fails if the live project docs still advertise `Muse`, `muse`, or `muse`.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest discover -s tests -p 'test_public_surface.py' -v`

Expected: FAIL while docs still contain old names.

**Step 3: Write minimal implementation**

- Update the current project’s docs and task files to `Muse` / `muse`
- Keep archive/history paths untouched

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest discover -s tests -p 'test_public_surface.py' -v`

Expected: PASS.

**Step 5: Commit**

Do not commit unless the user explicitly requests it.

### Task 4: Rename template/project placeholder strings

**Files:**
- Modify: `muse/templates/**/*`
- Test: `tests/test_latex_export.py`

**Step 1: Write the failing test**

Add a LaTeX export assertion for default placeholder/project-name text that should now say `Muse` instead of `Thesis Agent`.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest discover -s tests -p 'test_latex_export.py' -v`

Expected: FAIL because template placeholder strings still mention `Thesis Agent`.

**Step 3: Write minimal implementation**

Update template placeholder/project-default text that refers to the project itself.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest discover -s tests -p 'test_latex_export.py' -v`

Expected: PASS.

**Step 5: Commit**

Do not commit unless the user explicitly requests it.

### Task 5: Run the final verification sweep

**Files:**
- Read: `README.md`
- Read: `muse/`
- Read: `tests/`

**Step 1: Run targeted grep verification**

Run: `rg -n "thesis_agent|thesis-agent|Thesis Agent" README.md muse-plan-v2.md muse tests docs tasks`

Expected: No matches in live code/docs except intentionally preserved historical mentions if any remain in archived design docs.

**Step 2: Run the full suite**

Run: `python3 -m unittest discover -s tests -v`

Expected: PASS.

**Step 3: Smoke-test module entrypoint**

Run: `python3 -m muse --help`

Expected: help output renders successfully and uses the `muse` program name.

**Step 4: Commit**

Do not commit unless the user explicitly requests it.

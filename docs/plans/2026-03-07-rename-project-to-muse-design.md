# Rename Thesis Agent to Muse Design

**Date:** 2026-03-07

## Goal

Rename the project from `Thesis Agent` / `thesis_agent` / `thesis-agent` to `Muse` / `muse` across code, CLI, tests, templates, and primary documentation, with no backward-compatibility alias layer.

## Approved Scope

The user explicitly approved:

- full code/package rename,
- full CLI rename,
- no compatibility shims for the old package or command names,
- documentation and test updates to match the new name.

## Rename Targets

### Code

- Package directory: `thesis_agent/` → `muse/`
- Python module execution: `python3 -m thesis_agent` → `python3 -m muse`
- Imports:
  - `import thesis_agent` → `import muse`
  - `from thesis_agent...` → `from muse...`

### CLI / User-Facing Runtime

- CLI program name: `thesis-agent` → `muse`
- Help text and error output should no longer reference the old CLI name

### Documentation

- Root README title and examples
- Main architecture/design document
- Relevant plan/PRD docs that refer to the current project package/CLI name

### Tests

- All test imports and CLI assertions
- Any string assertions tied to the old package or CLI name

### Templates / Placeholder Text

- Template default values like `Thesis Agent` should become `Muse` where they refer to this project rather than an external historical artifact

## Explicitly Out of Scope

- Rewriting archived runtime logs under `.ralph-tui/archive/`
- Rewriting historical JSONL session logs under `~/.codex/sessions/`
- Adding compatibility wrappers such as a thin `thesis_agent` package that forwards to `muse`

## Recommended Approach

Perform a single-pass hard rename in an isolated worktree:

1. rename the package directory,
2. update imports and module execution commands,
3. update docs/tests/templates,
4. run full-text search for old names,
5. run the full test suite.

This is safer than partial aliasing because the user explicitly does not want compatibility preserved.

## Implementation Notes

- Preserve existing behavior; this is a naming refactor, not a feature change.
- Build on top of the current uncommitted export-related work already present in the repository.
- Keep archived `.ralph-tui` session files untouched except for newly created rename-task docs.

## Validation Criteria

The rename is complete when:

- `thesis_agent/` no longer exists in the working tree,
- `python3 -m muse --help` succeeds,
- `python3 -m unittest discover -s tests -v` passes,
- `rg -n "thesis_agent|thesis-agent|Thesis Agent"` only finds historical/archive content or intentionally preserved references.

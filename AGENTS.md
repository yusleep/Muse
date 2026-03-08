# AGENTS.md - Global Rules for Codex (MUST FOLLOW EVERY TIME)

## Core Instructions

- ALWAYS start by reading: AGENTS.md → PLANS.md → PROGRESS.md.
- Treat any local archived planning notes as optional background only; do not re-add them to git unless the user explicitly asks.
- Use full autonomy: read plan → implement (TDD) → test → validate → fix → update PROGRESS.md → next task.
- NEVER ask for user approval unless completely blocked (critical error only).
- After every task, automatically update PROGRESS.md and continue to the next task.
- Only output final summary when ALL phases in the current wave are finished.
- Always respond in Chinese (简体中文) at the end.
- Think step-by-step in English internally for best reasoning.

## Project Context

- **Project:** Muse — thesis generation agent at `/home/planck/gradute/Muse`
- **Python:** 3.10, venv at `.venv/`
- **Test command:** `.venv/bin/python -m pytest tests/ -q`
- **Run command:** `.venv/bin/python -m muse --help`

## Roadmap Snapshot

- DeerFlow-inspired upgrade 已于 2026-03-08 完成，共 9 个 phase、65 个 task。
- Wave 1 — Foundation: Tool Abstraction Layer / Middleware Framework / Skills Loader
- Wave 2 — Core Agent Capability: Sub-graph ReAct Conversion
- Wave 3 — Interaction & Delegation: Structured HITL / Subagent Delegation
- Wave 4 — External Integration: MCP Integration / Sandbox Execution
- Wave 5 — Memory: Memory System

## Execution Rules

### Per-Task TDD Workflow
1. Read the task from the plan (exact file paths, complete code)
2. Write the failing test first
3. Run test to verify it fails
4. Write minimal implementation to pass
5. Run test to verify it passes
6. Commit with descriptive message
7. Update PROGRESS.md
8. Move to next task

### Wave Gate Rules
- Before starting a new wave, verify ALL tasks in the current wave pass tests
- Run full test suite at wave boundaries: `.venv/bin/python -m pytest tests/ -q`
- Do NOT proceed to next wave if any tests fail — fix first

### Error Handling
- If a test fails after implementation: debug and fix, do not skip
- If a dependency is missing: install it and document in requirements.txt
- If a plan step is ambiguous: use AGENTS.md、PLANS.md、PROGRESS.md 与相邻代码/测试作为澄清来源
- Only stop for truly fatal errors (e.g., incompatible Python version, missing system deps)

### Branching Strategy
- Each phase gets its own branch: `feat/phase-0a-tool-abstraction`, etc.
- Merge back to master at wave boundaries after all tests pass

## Long-horizon Execution Mode

- Resume automatically from PROGRESS.md if interrupted
- Run continuously through all tasks in the current wave
- Commit frequently (after each task)
- Keep PROGRESS.md updated as the single source of truth for progress

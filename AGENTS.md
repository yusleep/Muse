# AGENTS.md - Global Rules for Codex (MUST FOLLOW EVERY TIME)

## Core Instructions

- ALWAYS start by reading: AGENTS.md → PLANS.md → PROGRESS.md → the current wave's plan files.
- Execute plans in **wave order** (Wave 1 → 2 → 3 → 4 → 5). Within a wave, phases can run in any order.
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
- **Design doc:** `docs/plans/2026-03-08-deerflow-inspired-upgrade-design.md`
- **Master plan:** `docs/plans/2026-03-08-deerflow-upgrade-master-plan.md`

## Plan Files (9 phases, 5 waves)

### Wave 1 — Foundation (all 3 parallel, no deps)
- `docs/plans/2026-03-08-phase-0a-tool-abstraction.md` — MuseChatModel, ToolRegistry, LangChain bridge
- `docs/plans/2026-03-08-phase-0b-middleware-framework.md` — Middleware chain, logging, retry, compaction
- `docs/plans/2026-03-08-phase-0c-skills-loader.md` — SKILL.md parsing, SkillRegistry, built-in skills

### Wave 2 — Core Agent Capability (depends on Wave 1)
- `docs/plans/2026-03-08-phase-1-react-subgraphs.md` — Chapter/Citation/Composition ReAct agents + tools

### Wave 3 — Interaction & Delegation (depends on Wave 2)
- `docs/plans/2026-03-08-phase-2-structured-hitl.md` — ask_clarification tool, ClarificationMiddleware
- `docs/plans/2026-03-08-phase-3-subagent-delegation.md` — SubagentExecutor, spawn_subagent

### Wave 4 — External Integration (depends on Wave 1, parallel with Wave 2-3)
- `docs/plans/2026-03-08-phase-4a-mcp-integration.md` — MCP client, OAuth, tool caching
- `docs/plans/2026-03-08-phase-4b-sandbox-execution.md` — Docker sandbox, LaTeX/Python execution

### Wave 5 — Memory (depends on Wave 1)
- `docs/plans/2026-03-08-phase-5-memory-system.md` — SQLite memory, MemoryMiddleware

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
- If a plan step is ambiguous: use the design doc for clarification
- Only stop for truly fatal errors (e.g., incompatible Python version, missing system deps)

### Branching Strategy
- Each phase gets its own branch: `feat/phase-0a-tool-abstraction`, etc.
- Merge back to master at wave boundaries after all tests pass

## Long-horizon Execution Mode

- Resume automatically from PROGRESS.md if interrupted
- Run continuously through all tasks in the current wave
- Commit frequently (after each task)
- Keep PROGRESS.md updated as the single source of truth for progress

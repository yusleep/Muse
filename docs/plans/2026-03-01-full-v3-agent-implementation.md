# Full v3 Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a full usable v3 thesis agent runnable from CLI with real integrations, checkpoints, audit logs, HITL, and exports.

**Architecture:** Build a Python package with stage modules, provider adapters, persistent run store, and orchestrator-driven pipeline. Expose commands via CLI for run/resume/review/check/export. Keep existing tested contracts stable while extending to full workflow.

**Tech Stack:** Python 3.10+, standard library HTTP (`urllib`), JSONL persistence, unittest.

---

### Task 1: Runtime foundations
- Add config/env validation and run storage.
- Add tests for required keys, settings load, state checkpoint persistence.

### Task 2: Provider clients
- Implement real LLM and academic search/verification clients.
- Add parsing, retries, and error normalization.

### Task 3: Stage implementation
- Implement Stage 1-6 modules with input/output contracts.
- Preserve HITL gates and stage transition validation.

### Task 4: Orchestrator + CLI
- Add resumable run engine and CLI commands (`run`, `resume`, `review`, `check`, `export`).
- Add tests for HITL pause/resume and auto-approve flow.

### Task 5: Verification + docs
- Run full tests.
- Update README with full setup, required env vars, and executable examples.

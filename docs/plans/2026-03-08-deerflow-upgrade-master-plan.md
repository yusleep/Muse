# Muse DeerFlow-Inspired Upgrade — Master Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> each phase plan linked below, in dependency order.

**Goal:** Transform Muse from a fixed-pipeline thesis agent into a hybrid system
with deterministic top-level pipeline + flexible ReAct sub-agents, tool-calling,
middleware, skills, MCP, sandbox, and memory.

**Source Design:** `docs/plans/2026-03-08-deerflow-inspired-upgrade-design.md`

---

## Phase Dependency Graph

```
Phase 0-A (Tool Abstraction)  ──┐
Phase 0-B (Middleware)         ──┼── Foundation (no deps, all parallel)
Phase 0-C (Skills Loader)     ──┘
        │
        ├── Phase 1 (ReAct Sub-graphs)      ← needs 0-A + 0-C
        │         │
        │         └── Phase 2 (Structured HITL) ← needs 0-B + 1
        │                   │
        │                   └── Phase 3 (Subagent Delegation) ← needs 1 + 2
        │
        ├── Phase 4-A (MCP Integration)     ← needs 0-A
        │
        ├── Phase 4-B (Sandbox Execution)   ← needs 0-A
        │
        └── Phase 5 (Memory System)         ← needs 0-B
```

## Execution Order

### Wave 1 — Foundation (parallel)

| Phase | Plan | Est. Tasks |
|-------|------|------------|
| 0-A | [phase-0a-tool-abstraction.md](2026-03-08-phase-0a-tool-abstraction.md) | 8 |
| 0-B | [phase-0b-middleware-framework.md](2026-03-08-phase-0b-middleware-framework.md) | 7 |
| 0-C | [phase-0c-skills-loader.md](2026-03-08-phase-0c-skills-loader.md) | 7 |

**Gate:** All three complete. Run full test suite before proceeding.

### Wave 2 — Core Agent Capability

| Phase | Plan | Est. Tasks |
|-------|------|------------|
| 1 | [phase-1-react-subgraphs.md](2026-03-08-phase-1-react-subgraphs.md) | 12 |

**Gate:** Chapter/Citation/Composition sub-graphs work as ReAct agents. Existing
e2e tests still pass (with adapted assertions).

### Wave 3 — Interaction & Delegation (parallel where possible)

| Phase | Plan | Est. Tasks |
|-------|------|------------|
| 2 | [phase-2-structured-hitl.md](2026-03-08-phase-2-structured-hitl.md) | 5 |
| 3 | [phase-3-subagent-delegation.md](2026-03-08-phase-3-subagent-delegation.md) | 6 |

Phase 2 must complete before Phase 3 (ClarificationMiddleware needed).

**Gate:** Subagent spawning works. HITL provides structured options.

### Wave 4 — External Integration (parallel)

| Phase | Plan | Est. Tasks |
|-------|------|------------|
| 4-A | [phase-4a-mcp-integration.md](2026-03-08-phase-4a-mcp-integration.md) | 7 |
| 4-B | [phase-4b-sandbox-execution.md](2026-03-08-phase-4b-sandbox-execution.md) | 7 |

4-A and 4-B are independent and can be developed in parallel.

**Gate:** MCP tools load from config. Sandbox executes LaTeX. Dynamic tool
assembly merges all 4 sources.

### Wave 5 — Memory

| Phase | Plan | Est. Tasks |
|-------|------|------------|
| 5 | [phase-5-memory-system.md](2026-03-08-phase-5-memory-system.md) | 6 |

**Gate:** Memory persists across runs. Middleware injects relevant memories.

---

## Final Middleware Chain (after all phases)

```
1. LoggingMiddleware              (Phase 0-B)
2. RetryMiddleware                (Phase 0-B)
3. SummarizationMiddleware        (Phase 0-B)
4. SubagentLimitMiddleware        (Phase 3)
5. MemoryMiddleware               (Phase 5)
6. DanglingToolCallMiddleware     (Phase 0-B)
7. ClarificationMiddleware        (Phase 2)
```

## New Directory Structure (after all phases)

```
muse/
├── models/
│   ├── adapter.py          (Phase 0-A)
│   └── factory.py          (Phase 0-A)
├── tools/
│   ├── registry.py         (Phase 0-A)
│   ├── academic_search.py  (Phase 0-A)
│   ├── citation.py         (Phase 0-A)
│   ├── research.py         (Phase 1)
│   ├── writing.py          (Phase 1)
│   ├── review.py           (Phase 1)
│   ├── file.py             (Phase 1)
│   └── orchestration.py    (Phase 1/2/3)
├── middlewares/
│   ├── base.py             (Phase 0-B)
│   ├── logging_middleware.py       (Phase 0-B)
│   ├── retry_middleware.py         (Phase 0-B)
│   ├── summarization_middleware.py (Phase 0-B)
│   ├── dangling_tool_call.py       (Phase 0-B)
│   ├── clarification_middleware.py (Phase 2)
│   └── subagent_limit_middleware.py (Phase 3)
├── skills/
│   ├── loader.py           (Phase 0-C)
│   └── registry.py         (Phase 0-C)
├── agents/
│   ├── executor.py         (Phase 3)
│   ├── result.py           (Phase 3)
│   └── builtins.py         (Phase 3)
├── mcp/
│   ├── client.py           (Phase 4-A)
│   ├── tools.py            (Phase 4-A)
│   ├── oauth.py            (Phase 4-A)
│   └── cache.py            (Phase 4-A)
├── sandbox/
│   ├── base.py             (Phase 4-B)
│   ├── docker.py           (Phase 4-B)
│   ├── local.py            (Phase 4-B)
│   ├── tools.py            (Phase 4-B)
│   └── vfs.py              (Phase 4-B)
├── memory/
│   ├── store.py            (Phase 5)
│   ├── prompt.py           (Phase 5)
│   └── middleware.py       (Phase 5)
├── graph/                  (existing, modified)
├── services/               (existing, extended)
├── prompts/                (existing, preserved)
├── schemas/                (existing)
└── ...

skills/                     (project root)
├── public/
│   ├── academic-writing/SKILL.md       (Phase 0-C)
│   ├── citation-gb-t-7714/SKILL.md    (Phase 0-C)
│   ├── thesis-structure-zh/SKILL.md   (Phase 0-C)
│   └── deep-research/SKILL.md         (Phase 0-C)
└── custom/                             (Phase 0-C)

docker/
└── Dockerfile.sandbox      (Phase 4-B)
```

## Config Structure (after all phases)

```yaml
# config.yaml
tools:
  research: [{name: web_search, use: "muse.tools.research:web_search_tool"}, ...]
  file: [{name: read_file, use: "muse.tools.file:read_file_tool"}, ...]
  writing: [...]
  review: [...]
  sandbox: [{name: shell, use: "muse.sandbox.tools:shell_tool", enabled: true}, ...]

profiles:
  chapter: [research, file, writing, sandbox]
  citation: [review, file]
  composition: [review, file, writing]

sandbox:
  image: "muse-sandbox:latest"
  fallback: local

memory:
  db_path: "~/.muse/memory.sqlite"
  token_budget: 2000

summarization:
  threshold_ratio: 0.9
  recent_tokens: 20000

subagents:
  max_concurrent: 3
  types:
    research: {max_turns: 15}
    writing: {max_turns: 25}
    bash: {max_turns: 15}

skills:
  dirs: ["skills/public", "skills/custom"]
  token_budget: 4000
```

## Branching Strategy

Each phase gets its own feature branch:
```
master
  └── feat/phase-0a-tool-abstraction
  └── feat/phase-0b-middleware
  └── feat/phase-0c-skills
  └── feat/phase-1-react-subgraphs
  └── feat/phase-2-hitl
  └── feat/phase-3-subagent
  └── feat/phase-4a-mcp
  └── feat/phase-4b-sandbox
  └── feat/phase-5-memory
```

Merge in wave order. Run full test suite at each gate.

# Muse DeerFlow-Inspired Upgrade Design

Date: 2026-03-08
Status: Approved

## Overview

Integrate 9 capabilities inspired by DeerFlow, Claude Code, and Codex CLI into Muse,
transforming it from a fixed-pipeline thesis agent into a hybrid system with a deterministic
top-level pipeline and flexible ReAct sub-agents with tool-calling.

### Key Decisions

| Decision | Choice |
|----------|--------|
| Architecture | Hybrid: top-level pipeline (unchanged) + sub-graph ReAct agents |
| LLM backend | Mixed/router-based (OpenAI, Anthropic, Codex all supported) |
| Deployment | Local + Docker |
| Tool-calling | LangChain (@tool + bind_tools + create_react_agent) |
| Skills format | SKILL.md (YAML front-matter + Markdown body) |
| Context compaction | Codex CLI local compaction mode (LLM summarization) |

### Phase Overview

```
Phase 0 (Tool + Middleware + Skills)      ← foundation
  ├── Phase 1 (Sub-graph ReAct)           ← depends on Tool layer
  │     └── Phase 2 (Structured HITL)     ← depends on Middleware
  │           └── Phase 3 (Subagent)      ← depends on ReAct + HITL
  └── Phase 4 (MCP + Sandbox)             ← depends on Tool assembly
        └── Phase 5 (Memory)              ← depends on Middleware
```

Phase 1-2 and Phase 4 can be developed in parallel.

---

## Phase 0-A: Tool Abstraction Layer

### Problem

Muse's `LLMClient` is a custom HTTP wrapper (OpenAI/Anthropic/Codex). LangChain's
`create_react_agent` requires `BaseChatModel`. The two must be bridged.

### New Files

```
muse/tools/
├── __init__.py
├── registry.py              # ToolRegistry: dynamic tool assembly
├── academic_search.py       # @tool: search_semantic_scholar, search_openalex, search_arxiv
├── citation.py              # @tool: verify_doi, crosscheck_metadata, entailment_check
├── writing.py               # @tool: write_section, revise_section
├── review.py                # @tool: self_review (multi-lens)
├── rag.py                   # @tool: retrieve_local_refs

muse/models/
├── __init__.py
├── adapter.py               # MuseChatModel(BaseChatModel): wraps LLMClient
├── factory.py               # create_chat_model(settings, route) -> BaseChatModel
```

### MuseChatModel

Wraps existing `LLMClient` as a LangChain `BaseChatModel`:

- `_generate()`: Converts LangChain Messages to system+user strings, constructs
  function-calling payload if `tools` present, calls `llm_client._chat_completion`,
  converts response to LangChain `AIMessage` (with `tool_calls`).
- `bind_tools()`: Converts tool schemas to OpenAI function schema or Anthropic tool_use
  schema, auto-adapting by `api_style`.

### _build_request_payload Extension

Extend existing `providers.py` to support `tools` parameter:

- OpenAI chat_completions: add `tools` + `tool_choice` fields
- Anthropic: add `tools` field (Anthropic tool_use format)
- Codex responses: add `tools` field if supported; degrade to structured prompt if not

### ToolRegistry

```python
class ToolRegistry:
    _tools: dict[str, BaseTool]
    _groups: dict[str, list[str]]
    _subgraph_profiles: dict[str, list[str]]

    def get_tools_for_profile(self, profile: str) -> list[BaseTool]: ...
    def register(self, tool: BaseTool, group: str): ...
    def register_mcp_tools(self, mcp_tools: list[BaseTool]): ...
```

### What Stays

`LLMClient`, `_ModelRouter`, `HttpClient`, `post_json_sse` all preserved.
`MuseChatModel` is a thin wrapper, not a replacement.

---

## Phase 0-B: Middleware Framework

### New Files

```
muse/middlewares/
├── __init__.py
├── base.py                    # Middleware protocol + MiddlewareChain
├── logging_middleware.py      # Token usage, latency, call tracing
├── summarization_middleware.py # Context compaction (Codex CLI local mode)
├── retry_middleware.py        # Unified retry (replaces scattered try/except)
├── dangling_tool_call.py      # Fix incomplete tool_calls (ReAct safety net)
```

### Middleware Protocol

```python
class Middleware(Protocol):
    async def before_invoke(self, state: dict, config: dict) -> dict: ...
    async def after_invoke(self, state: dict, result: dict, config: dict) -> dict: ...

class MiddlewareChain:
    def __init__(self, middlewares: list[Middleware]): ...
    async def wrap(self, node_fn, state, config): ...
```

Integration: wraps node functions in `main_graph.py`:
```python
chain = MiddlewareChain([LoggingMiddleware(), RetryMiddleware(), ...])
builder.add_node("search", chain.wrap(build_search_node(settings, services)))
```

### SummarizationMiddleware (Codex CLI Local Compaction)

Trigger: `context_window * 0.9` threshold (configurable).
Token estimation: `len(json.dumps(messages).encode()) // 4` (Codex 4 bytes/token heuristic).

Flow:
1. Send full history + compaction prompt to LLM (route="default")
2. LLM returns summary
3. Rebuild messages: summary (with prefix) + recent user messages (up to 20k tokens)

Compaction prompt (from Codex CLI):
```
You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary
for another LLM that will resume the task. Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
```

Summary prefix:
```
Another language model started to solve this problem and produced a summary
of its thinking process. Use this to build on the work that has already been
done and avoid duplicating work.
```

### Execution Order (Phase 0)

```
1. LoggingMiddleware
2. RetryMiddleware
3. SummarizationMiddleware     (ReAct sub-graphs only)
4. DanglingToolCallMiddleware   (ReAct sub-graphs only)
```

---

## Phase 0-C: Skills Loader

### New Files

```
muse/skills/
├── __init__.py
├── loader.py              # SkillLoader: scan dirs, parse SKILL.md
├── registry.py            # SkillRegistry: match by discipline/stage/language

skills/                    # project root, user-editable
├── public/
│   ├── academic-writing/SKILL.md
│   ├── citation-gb-t-7714/SKILL.md
│   ├── citation-apa/SKILL.md
│   ├── computer-science/SKILL.md
│   ├── deep-research/SKILL.md
│   └── thesis-structure-zh/SKILL.md
├── custom/                # user-defined, auto-discovered
```

### SKILL.md Format

```yaml
---
name: citation-gb-t-7714
description: GB/T 7714-2015 citation format
applies_to:
  stages: [writing, citation, polish]
  disciplines: ["*"]
  languages: ["zh"]
priority: 10
---
(Markdown body injected into system prompt)
```

### SkillRegistry

```python
class SkillRegistry:
    def get_for_context(self, *, stage, discipline, language) -> list[Skill]: ...
    def inject_into_prompt(self, system_prompt, skills) -> str: ...
```

Injection point: `MuseChatModel._generate()` calls registry before LLM invocation.
Token budget: 4000 tokens max for skills; excess truncated by priority.

### Relationship to Existing Prompts

- Existing `muse/prompts/*.py` functions preserved as "core prompts"
  (structured JSON output requirements etc.)
- Skills are supplementary knowledge appended to system prompt

---

## Phase 1: Sub-graph ReAct Conversion

### Complete Tool Inventory (6 groups)

#### 1. Research Tools (research)

| Tool | Source | Description |
|------|--------|-------------|
| `web_search` | Claude Code / Codex | Web search (general knowledge) |
| `web_fetch` | Claude Code / DeerFlow | Fetch web page as markdown |
| `academic_search` | Muse original | Semantic Scholar/OpenAlex/arXiv |
| `retrieve_local_refs` | Muse RAG | Local reference RAG retrieval |
| `read_pdf` | Claude Code Read (PDF pages) | Read PDF pages, extract content |
| `image_search` | DeerFlow | Search images for figures |

#### 2. File Tools (file)

| Tool | Source | Description |
|------|--------|-------------|
| `read_file` | Claude Code / Codex | Read file with offset/limit |
| `write_file` | Claude Code / DeerFlow | Write/create file |
| `edit_file` | Claude Code / DeerFlow str_replace | Exact string replacement |
| `glob` | Claude Code / Codex grep_files | Find files by pattern |
| `grep` | Claude Code | Regex content search |

#### 3. Writing Tools (writing)

| Tool | Source | Description |
|------|--------|-------------|
| `write_section` | Muse original | Write a subsection by outline |
| `revise_section` | Muse original | Revise per instructions |
| `apply_patch` | Codex (LLM-friendly patch format) | Partial text modification |

#### 4. Review Tools (review)

| Tool | Source | Description |
|------|--------|-------------|
| `self_review` | Muse multi-lens | Multi-dimension review |
| `verify_doi` | Muse original | Check DOI via CrossRef |
| `crosscheck_metadata` | Muse original | Verify citation metadata |
| `entailment_check` | Muse NLI | NLI claim-reference check |
| `check_terminology` | New | Scan terminology consistency |
| `check_transitions` | New | Check inter-chapter transitions |

#### 5. Sandbox Tools (sandbox)

| Tool | Source | Description |
|------|--------|-------------|
| `shell` | Claude Code / Codex / DeerFlow | Execute in Docker sandbox |
| `latex_compile` | New (shell wrapper) | Compile LaTeX, return errors |
| `run_python` | Codex js_repl (Python equivalent) | Execute Python (plots/analysis) |
| `present_file` | DeerFlow | Present generated files to user |

#### 6. Orchestration Tools (orchestration)

| Tool | Source | Description |
|------|--------|-------------|
| `update_plan` | Codex | Update chapter progress (UI signal) |
| `ask_clarification` | DeerFlow / Claude Code | Structured HITL (Phase 2) |
| `spawn_subagent` | Codex / DeerFlow / Claude Code | Delegate subtask (Phase 3) |
| `submit_result` | New | Termination tool, submit stage result |

### Sub-graph Tool Profiles

| Sub-graph | Available Groups | Blocked |
|-----------|-----------------|---------|
| Chapter Agent | research, file, writing, review(self_review), sandbox, update_plan, submit_result | spawn_subagent, ask_clarification |
| Citation Agent | review(verify/crosscheck/entailment), file(read), research(academic_search), submit_result | writing, sandbox, spawn_subagent |
| Composition Agent | review(terminology/transitions), file, writing(apply_patch/revise), submit_result | research, sandbox, spawn_subagent |
| Top-level nodes | All | - |

### Phase 1-A: Chapter Subgraph

Replace fixed `chapter_draft -> chapter_review -> (revise?)` with:

```python
agent = create_react_agent(
    model=create_chat_model(settings, route="writing"),
    tools=tool_registry.get_tools_for_profile("chapter"),
    state_schema=ChapterAgentState,
)
```

Max turns: 30. Exit: `submit_chapter` tool or turn limit.

System prompt guides workflow (search -> write -> review -> revise -> verify -> submit)
but agent decides autonomously.

### Phase 1-B: Citation Subgraph

ReAct agent with review tools. Max turns: 20. Agent decides verification depth
per citation (full 3-layer for critical claims, DOI-only for minor).

### Phase 1-C: Composition Subgraph

ReAct agent with terminology/transitions/rewrite tools. Max turns: 15.
Replaces current placeholder logic.

### Top-level Pipeline

Unchanged. `main_graph.py` nodes and edges preserved. Only the internal
implementation of `build_*_subgraph_node()` changes from StateGraph to
`create_react_agent`. Interface stays: accept MuseState, return state update dict.

---

## Phase 2: Structured HITL

### ask_clarification Tool

```python
@tool
def ask_clarification(
    question: str,
    clarification_type: Literal[
        "missing_info", "ambiguous_requirement", "approach_choice",
        "risk_confirmation", "suggestion",
    ],
    context: str | None = None,
    options: list[dict] | None = None,   # [{label, description}]
) -> str: ...
```

### ClarificationMiddleware

Intercepts `ask_clarification` tool calls before execution.
Extracts question/type/context/options, calls `interrupt(payload)` to halt graph.
User response injected as ToolMessage to continue ReAct loop.
Must be LAST in middleware chain.

### Top-level Interrupt Upgrade (backward-compatible)

Existing interrupt nodes gain `type`, `question`, `options`, `context` fields.
Old `approved: true/false` still works.

| Stage | type | Example options |
|-------|------|-----------------|
| review_refs | risk_confirmation | [continue, add keywords, add manually] |
| approve_outline | approach_choice | [plan A: 5 chapters, plan B: 6, custom] |
| review_draft | suggestion | [auto-fix, guide revision, skip] |
| approve_final | risk_confirmation | [accept, review details, remove weak] |

### CLI Integration

`review` command upgraded to display options and collect structured feedback.

---

## Phase 3: Subagent Delegation

### SubagentExecutor

```python
class SubagentExecutor:
    def __init__(self, settings, services, max_concurrent=3): ...
    async def execute_async(self, *, agent_type, message, parent_context,
                            tools, max_turns=20) -> SubagentResult: ...
    def get_status(self, task_id) -> SubagentStatus: ...
    def get_result(self, task_id) -> SubagentResult | None: ...
```

### spawn_subagent Tool

```python
@tool
def spawn_subagent(
    message: str,
    agent_type: Literal["research", "writing", "bash"],
    wait: bool = True,
) -> str: ...
```

### Built-in Agent Types

| Type | Tools | Max Turns | Purpose |
|------|-------|-----------|---------|
| research | web_search, web_fetch, academic_search, read_pdf, retrieve_local_refs | 15 | Deep literature research |
| writing | write_section, revise_section, apply_patch, read_file, edit_file, self_review | 25 | Independent writing |
| bash | shell, read_file, write_file, latex_compile, run_python, present_file | 15 | Command execution |

Blocked tools for all sub-agents: `spawn_subagent` (no nesting), `ask_clarification` (no direct user interaction).

### SubagentLimitMiddleware

Truncates excess `spawn_subagent` tool calls per LLM response (max 3 concurrent).
More reliable than prompt-based limits (DeerFlow experience).

### SubagentResult Protocol

```python
@dataclass
class SubagentResult:
    status: Literal["completed", "failed", "timed_out"]
    accomplishments: list[str]
    key_findings: list[str]
    files_created: list[str]
    issues: list[str]
    citations: list[dict]
```

---

## Phase 4-A: MCP Integration

### New Files

```
muse/mcp/
├── __init__.py
├── client.py          # Build connection params
├── tools.py           # MCP tools -> LangChain BaseTool
├── oauth.py           # Per-server OAuth token management
├── cache.py           # Tool list cache (hot-reload)
```

### Configuration (extensions.yaml)

```yaml
mcp_servers:
  zotero:
    transport: stdio
    command: "npx"
    args: ["-y", "@anthropic/mcp-server-zotero"]
  overleaf:
    transport: http
    url: "https://overleaf-mcp.example.com/mcp"
    oauth: {token_url, grant_type, client_id, client_secret}
  local_search:
    transport: sse
    url: "http://localhost:8080/sse"
```

Three transport types: stdio, sse, http. Per-server OAuth with auto-refresh.
MCP tools auto-registered into `mcp` group in ToolRegistry.
Graceful failure: empty list on error, never blocks startup.

---

## Phase 4-B: Sandbox Execution

### New Files

```
muse/sandbox/
├── __init__.py
├── base.py            # Sandbox ABC + ExecResult
├── docker.py          # DockerSandbox (texlive, python3, matplotlib)
├── local.py           # LocalSandbox (subprocess fallback)
├── tools.py           # shell, latex_compile, run_python, present_file
├── vfs.py             # Virtual filesystem path mapping
```

### Sandbox Abstraction

```python
class Sandbox(ABC):
    async def exec(self, command, timeout=60) -> ExecResult: ...
    async def read_file(self, path) -> bytes: ...
    async def write_file(self, path, content): ...
    async def list_dir(self, path) -> list[str]: ...
```

Docker mounts:
- `/mnt/workspace` -> `{runs_dir}/{project_id}/workspace` (rw)
- `/mnt/outputs` -> `{runs_dir}/{project_id}/outputs` (rw)
- `/mnt/refs` -> `{refs_dir}` (ro)

Auto-fallback to LocalSandbox when Docker unavailable.

### High-level Sandbox Tools

- `latex_compile`: pdflatex + bibtex + pdflatex x2, returns {success, output_path, errors}
- `run_python`: Execute Python in sandbox (matplotlib/numpy/pandas pre-installed)
- `present_file`: Copy generated files to user-accessible location

---

## Phase 4-C: Dynamic Tool Assembly

### ToolRegistry.build()

```python
@classmethod
async def build(cls, settings, services) -> "ToolRegistry":
    # 1. Config tools (from config.yaml, dynamic import via resolve_tool)
    # 2. Built-in tools (always registered)
    # 3. MCP tools (dynamic, from extensions.yaml)
    # 4. Sandbox tools (conditional on Docker availability)
```

`resolve_tool("muse.tools.research:web_search_tool")` dynamically imports by dotted path
(same pattern as DeerFlow `resolve_variable`).

---

## Phase 5: Memory System

### New Files

```
muse/memory/
├── __init__.py
├── store.py           # MemoryStore (SQLite)
├── prompt.py          # Format memory for system prompt injection
├── middleware.py      # MemoryMiddleware
```

### MemoryEntry

```python
@dataclass
class MemoryEntry:
    key: str
    category: str       # user_pref | writing_style | citation | feedback_pattern | fact
    content: str
    confidence: float   # 0.0-1.0
    source_run: str | None
    created_at: datetime
    updated_at: datetime
```

### MemoryStore

SQLite at `~/.muse/memory.sqlite`. CRUD operations with category/confidence filtering.

### MemoryMiddleware

- `before_invoke`: Query relevant memories, format, truncate to 2000-token budget,
  inject into config for MuseChatModel.
- `after_invoke`: At specific nodes (HITL feedback, citation results, review),
  use LLM to extract memorable facts and upsert.

### Confidence Lifecycle

- Each LLM extraction confirmation: `confidence += 0.1` (capped at 1.0)
- User explicit denial: `confidence = 0` -> auto-delete
- 90 days unused: `confidence *= 0.9`

### Extraction Triggers

| Trigger | Categories Extracted |
|---------|---------------------|
| After HITL feedback | user_pref, writing_style |
| After citation subgraph | citation (verified DOIs) |
| After review node | feedback_pattern |
| At initialize | fact (topic/discipline) |

---

## Final Middleware Chain (All Phases)

```
1. LoggingMiddleware              (Phase 0)
2. RetryMiddleware                (Phase 0)
3. SummarizationMiddleware        (Phase 0, ReAct sub-graphs only)
4. SubagentLimitMiddleware        (Phase 3)
5. MemoryMiddleware               (Phase 5)
6. DanglingToolCallMiddleware     (Phase 0, ReAct sub-graphs only)
7. ClarificationMiddleware        (Phase 2, always last)
```

---

## Config Structure (Final)

```yaml
# config.yaml
tools:
  research:
    - {name: web_search, use: "muse.tools.research:web_search_tool"}
    - {name: academic_search, use: "muse.tools.research:academic_search_tool"}
  file:
    - {name: read_file, use: "muse.tools.file:read_file_tool"}
    - {name: write_file, use: "muse.tools.file:write_file_tool"}
    - {name: edit_file, use: "muse.tools.file:edit_file_tool"}
  # ... other groups

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
```

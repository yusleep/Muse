# Muse v3 (LangGraph Runtime)

This repository now contains a LangGraph-native Muse pipeline with resumable checkpoints, chapter subgraphs, citation ledger verification, and CLI-driven HITL review.

Legacy `engine/orchestrator/stages/chapter` bridge modules have been removed; the supported runtime surface is now the LangGraph path under `muse/graph/` plus the CLI/runtime wrappers that call it.

## What is implemented

- LangGraph main flow:
  - `initialize → search → review_refs → outline → approve_outline`
  - `fan_out_chapters → chapter_subgraph × N → merge_chapters → review_draft`
  - `citation_subgraph → polish → composition_subgraph → approve_final → export`
- Chapter subgraph with Reflexion-style revise loop and parallel fan-out.
- SQLite checkpoint persistence under `runs/<run_id>/graph/checkpoints.sqlite`.
- Citation ledger with verified/flagged outcomes and export gating.
- `latex` export generates a vendored BUPT thesis project, an Overleaf-ready `.zip`, and an optional locally compiled PDF when `latexmk` or `xelatex` is available.
- HITL checkpoints with named stages: `research`, `outline`, `draft`, `final`.
- Persistent run directories under `runs/<run_id>/`.
- CLI commands: `check`, `debug-llm`, `run`, `resume`, `review`, `export`.
- Optional adapter scaffolding for `LlamaIndex` retrieval/ingestion and per-source external search wrappers.

## Local setup

Create the local virtual environment used by the current test/runtime flow:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

## Environment variables (required)

- `MUSE_LLM_API_KEY`
- `MUSE_LLM_MODEL`

## Environment variables (optional)

- `MUSE_LLM_BASE_URL` (default: `https://api.openai.com/v1`)
- `MUSE_RUNS_DIR` (default: `runs`)
- `MUSE_CHECKPOINT_DIR` (optional override for LangGraph checkpoint directory)
- `MUSE_SEMANTIC_SCHOLAR_API_KEY`
- `MUSE_OPENALEX_EMAIL`
- `MUSE_CROSSREF_MAILTO`
- `MUSE_MODEL_ROUTER_JSON` (OpenClaw-style multi-model router config)
- `MUSE_MODEL_ROUTER_PATH` (path to JSON config file, alternative to inline JSON)

## OpenClaw-style multi-model routing

The runtime now supports OpenClaw-like model routing with:

- `auth.profiles` for API keys / key-env bindings
- `providers` with `baseUrl`, `auth`, optional `headers`, optional `apiStyle` (`chat_completions` or `responses`), optional `codexOAuth`, per-model `params`
- `models` routes with `primary` + `fallbacks`
- `modelAliases` for model id aliasing
- `auth` can be a list to rotate profiles before switching fallback models

Minimal example:

```bash
export OPENAI_API_KEY="<openai-key>"
export OPENROUTER_API_KEY="<openrouter-key>"
export MUSE_MODEL_ROUTER_JSON='{
  "auth": {
    "profiles": {
      "openai": { "apiKeyEnv": "OPENAI_API_KEY" },
      "openrouter": { "apiKeyEnv": "OPENROUTER_API_KEY" }
    }
  },
  "providers": {
    "openai": {
      "baseUrl": "https://api.openai.com/v1",
      "auth": "openai",
      "models": {
        "openai/gpt-4.1": { "model": "gpt-4.1" },
        "openai/gpt-4.1-mini": { "model": "gpt-4.1-mini" }
      }
    },
    "openrouter": {
      "baseUrl": "https://openrouter.ai/api/v1",
      "auth": "openrouter",
      "models": {
        "openrouter/anthropic/claude-sonnet-4": {
          "model": "anthropic/claude-sonnet-4"
        }
      }
    }
  },
  "models": {
    "default": {
      "primary": "openai/gpt-4.1-mini",
      "fallbacks": ["openrouter/anthropic/claude-sonnet-4"]
    },
    "outline": {
      "primary": "openai/gpt-4.1",
      "fallbacks": ["openai/gpt-4.1-mini"]
    },
    "writing": {
      "primary": "openrouter/anthropic/claude-sonnet-4",
      "fallbacks": ["openai/gpt-4.1-mini"]
    },
    "review": {
      "primary": "openai/gpt-4.1",
      "fallbacks": ["openrouter/anthropic/claude-sonnet-4"]
    },
    "reasoning": {
      "primary": "openai/gpt-4.1",
      "fallbacks": ["openai/gpt-4.1-mini"]
    },
    "polish": {
      "primary": "openrouter/anthropic/claude-sonnet-4",
      "fallbacks": ["openai/gpt-4.1-mini"]
    }
  },
  "modelAliases": {
    "openai/gpt-4.1-latest": "openai/gpt-4.1"
  }
}'
```

When `MUSE_MODEL_ROUTER_JSON` is set, the runtime automatically routes by task (`outline`, `writing`, `review`, `reasoning`, `polish`) and falls back when primary fails.

### Quick config for `https://api.123nhh.me/v1`

Template file is provided at:

- `model-router.123nhh.example.json`

Use it directly:

```bash
export NHH_API_KEY="<fill-your-key>"
export MUSE_MODEL_ROUTER_PATH="./model-router.123nhh.example.json"
```

You can keep legacy vars as fallback compatibility:

```bash
export MUSE_LLM_API_KEY="$NHH_API_KEY"
export MUSE_LLM_MODEL="gpt-4.1-mini"
```

### Quick config for local Codex Plus OAuth

Template file:

- `model-router.codex-plus-oauth.example.json`

This reads token from local Codex login file (`~/.codex/auth.json`, path `tokens.access_token`) and sends requests to ChatGPT Codex backend (`https://chatgpt.com/backend-api/codex/responses`) using OAuth-compatible headers (same transport style used by opencode Codex OAuth plugin).

```bash
export MUSE_MODEL_ROUTER_PATH="./model-router.codex-plus-oauth.example.json"
.venv/bin/python -m muse check
```

If not logged in locally, run `codex login` first.

## Quick start

1. Export keys:

```bash
export MUSE_LLM_API_KEY="<your-key>"
export MUSE_LLM_MODEL="gpt-4.1-mini"
```

2. Validate connectivity:

```bash
.venv/bin/python -m muse check
```

Debug LLM routing failures in detail:

```bash
.venv/bin/python -m muse debug-llm --route default
.venv/bin/python -m muse debug-llm --route reasoning
```

3. Start a run (pause at HITL):

```bash
.venv/bin/python -m muse run \
  --topic "Multi-agent systems for academic writing" \
  --discipline "Computer Science" \
  --language zh \
  --format-standard "GB/T 7714-2015" \
  --output-format markdown
```

4. Resume after review:

```bash
.venv/bin/python -m muse review --run-id <run_id> --stage research --approve --comment "ok"
.venv/bin/python -m muse resume --run-id <run_id>
```

5. One-shot auto-approved full run:

```bash
.venv/bin/python -m muse run \
  --topic "Multi-agent systems for academic writing" \
  --discipline "Computer Science" \
  --auto-approve \
  --output-format markdown
```

6. Export a BUPT LaTeX project for Overleaf:

```bash
.venv/bin/python -m muse export --run-id <run_id> --output-format latex
```

This writes `runs/<run_id>/output/latex_project/`, packages `runs/<run_id>/output/latex_project.zip`, and attempts `runs/<run_id>/output/thesis.pdf` when local TeX tooling is available.

## Notes

- `latex` is the rich thesis export path; it always produces the project directory plus an Overleaf-uploadable `.zip`.
- Local PDF compilation is best-effort and uses `latexmk` first, then `xelatex` when available.
- Real integrations are used; no offline mock mode is included.
- The agent enforces export blocking when `flagged_citations` is non-empty.
- `adapters/llamaindex/` and `adapters/external_search/` are optional extension points; the default runtime still works without `llama_index`.
- The only supported orchestration entrypoints are `Runtime.build_graph()` and the CLI commands built on top of `muse.graph.launcher`.

## Test suite

```bash
.venv/bin/python -m pytest tests/ -q
```

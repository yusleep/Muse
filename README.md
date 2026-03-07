# Muse v3 (Usable Runtime)

This repository now contains the runnable Muse v3 pipeline with real integrations and resumable runs.

## What is implemented

- 6-stage pipeline:
  - Stage 1: multi-source literature search (Semantic Scholar + OpenAlex + arXiv)
  - Stage 2: structured outline generation (LLM JSON output)
  - Stage 3: chapter/subtask writing with review-revise loop
  - Stage 4: citation verification (DOI + metadata + claim support NLI)
  - Stage 5: cross-chapter polish
  - Stage 6: export (markdown / LaTeX project / PDF)
- `latex` export generates a vendored BUPT thesis project, an Overleaf-ready `.zip`, and an optional locally compiled PDF when `latexmk` or `xelatex` is available.
- HITL checkpoints with pause/resume flow.
- Persistent run directories under `runs/<run_id>/`.
- Append-only audit log `runs/<run_id>/audit.jsonl`.
- CLI commands: `check`, `debug-llm`, `run`, `resume`, `review`, `export`.

## Environment variables (required)

- `MUSE_LLM_API_KEY`
- `MUSE_LLM_MODEL`

## Environment variables (optional)

- `MUSE_LLM_BASE_URL` (default: `https://api.openai.com/v1`)
- `MUSE_RUNS_DIR` (default: `runs`)
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
python3 -m muse check
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
python3 -m muse check
```

Debug LLM routing failures in detail:

```bash
python3 -m muse debug-llm --route default
python3 -m muse debug-llm --route reasoning
```

3. Start a run (pause at HITL):

```bash
python3 -m muse run \
  --topic "Multi-agent systems for academic writing" \
  --discipline "Computer Science" \
  --language zh \
  --format-standard "GB/T 7714-2015" \
  --output-format markdown
```

4. Resume after review:

```bash
python3 -m muse review --run-id <run_id> --stage 1 --approve --comment "ok"
python3 -m muse resume --run-id <run_id>
```

5. One-shot auto-approved full run:

```bash
python3 -m muse run \
  --topic "Multi-agent systems for academic writing" \
  --discipline "Computer Science" \
  --auto-approve \
  --output-format markdown
```

6. Export a BUPT LaTeX project for Overleaf:

```bash
python3 -m muse export --run-id <run_id> --output-format latex
```

This writes `runs/<run_id>/output/latex_project/`, packages `runs/<run_id>/output/latex_project.zip`, and attempts `runs/<run_id>/output/thesis.pdf` when local TeX tooling is available.

## Notes

- `latex` is the rich thesis export path; it always produces the project directory plus an Overleaf-uploadable `.zip`.
- Local PDF compilation is best-effort and uses `latexmk` first, then `xelatex` when available.
- Real integrations are used; no offline mock mode is included.
- The agent enforces export blocking when `flagged_citations` is non-empty.

## Test suite

```bash
python3 -m unittest discover -s tests -v
```

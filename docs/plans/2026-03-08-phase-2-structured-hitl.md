# Phase 2: Structured HITL

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace binary interrupt() with structured clarification tool supporting 5 types, options, and context.

**Architecture:** ask_clarification @tool + ClarificationMiddleware (intercepts tool calls, converts to interrupt). Top-level interrupts upgraded with options/context (backward-compatible).

**Tech Stack:** LangChain, LangGraph interrupt(), Python 3.10

**Depends on:** Phase 0-B (Middleware framework), Phase 1 (ReAct sub-graphs)

---

## Task 1: Create `ask_clarification` tool

**File:** `muse/tools/orchestration.py` (extend existing file; create if absent)

**Why:** ReAct sub-agents need a way to pause execution and ask the human a structured
question. This tool is the agent-facing API surface. It never executes directly; the
ClarificationMiddleware intercepts it before execution and converts it into a LangGraph
`interrupt()`.

**TDD steps:**

1. Write `tests/test_ask_clarification_tool.py` with these tests:
   - `test_tool_schema_has_required_fields` -- verify the tool's JSON schema contains
     `question`, `clarification_type`, and optional `context` / `options`.
   - `test_clarification_type_enum` -- verify only the 5 allowed types are accepted
     (`missing_info`, `ambiguous_requirement`, `approach_choice`, `risk_confirmation`,
     `suggestion`).
   - `test_tool_returns_placeholder_when_called_directly` -- calling the function without
     middleware returns a placeholder string (safety net).
   - `test_options_schema_structure` -- verify each item in `options` must have `label`
     (str) and `description` (str).
2. Run tests: all 4 fail.
3. Implement the tool.
4. Run tests: all 4 pass.

**Implementation:**

```python
# muse/tools/orchestration.py
"""Orchestration tools for agent coordination and HITL."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ClarificationOption(BaseModel):
    """A selectable option presented to the human."""

    label: str = Field(description="Short label for the option")
    description: str = Field(description="Longer explanation of what this option entails")


class AskClarificationInput(BaseModel):
    """Input schema for ask_clarification tool."""

    question: str = Field(description="The question to ask the human reviewer")
    clarification_type: Literal[
        "missing_info",
        "ambiguous_requirement",
        "approach_choice",
        "risk_confirmation",
        "suggestion",
    ] = Field(description="Category of clarification needed")
    context: str | None = Field(
        default=None,
        description="Background context to help the reviewer understand the question",
    )
    options: list[ClarificationOption] | None = Field(
        default=None,
        description="Selectable options; omit for free-text response",
    )


@tool(args_schema=AskClarificationInput)
def ask_clarification(
    question: str,
    clarification_type: str,
    context: str | None = None,
    options: list[dict[str, str]] | None = None,
) -> str:
    """Ask the human reviewer for clarification.

    Use this when you need human input to proceed. The question will be
    presented to the reviewer with any provided options and context.
    Do NOT use this for routine progress updates.
    """
    # Direct invocation fallback. In production the ClarificationMiddleware
    # intercepts this tool call before it reaches this function body.
    return (
        f"[CLARIFICATION PENDING] {clarification_type}: {question} "
        "(This response means the middleware did not intercept the call.)"
    )
```

**Test file:**

```python
# tests/test_ask_clarification_tool.py
"""Tests for the ask_clarification orchestration tool."""

import unittest


class AskClarificationToolTests(unittest.TestCase):
    def test_tool_schema_has_required_fields(self):
        from muse.tools.orchestration import ask_clarification

        schema = ask_clarification.args_json_schema
        props = schema.get("properties", {})
        self.assertIn("question", props)
        self.assertIn("clarification_type", props)
        self.assertIn("context", props)
        self.assertIn("options", props)
        required = schema.get("required", [])
        self.assertIn("question", required)
        self.assertIn("clarification_type", required)

    def test_clarification_type_enum(self):
        from muse.tools.orchestration import ask_clarification

        schema = ask_clarification.args_json_schema
        ct = schema["properties"]["clarification_type"]
        allowed = set(ct.get("enum", []))
        expected = {
            "missing_info",
            "ambiguous_requirement",
            "approach_choice",
            "risk_confirmation",
            "suggestion",
        }
        self.assertEqual(allowed, expected)

    def test_tool_returns_placeholder_when_called_directly(self):
        from muse.tools.orchestration import ask_clarification

        result = ask_clarification.invoke(
            {"question": "How many chapters?", "clarification_type": "missing_info"}
        )
        self.assertIn("CLARIFICATION PENDING", result)
        self.assertIn("missing_info", result)

    def test_options_schema_structure(self):
        from muse.tools.orchestration import AskClarificationInput, ClarificationOption

        opt = ClarificationOption(label="Plan A", description="Five chapters")
        inp = AskClarificationInput(
            question="Which plan?",
            clarification_type="approach_choice",
            options=[opt],
        )
        self.assertEqual(inp.options[0].label, "Plan A")
        self.assertEqual(inp.options[0].description, "Five chapters")


if __name__ == "__main__":
    unittest.main()
```

---

## Task 2: Create ClarificationMiddleware

**File:** `muse/middlewares/clarification_middleware.py` (new file)

**Why:** The middleware intercepts `ask_clarification` tool calls emitted by ReAct
sub-agents, converts them into LangGraph `interrupt()` payloads, and injects the
human response back as a `ToolMessage`. This is the bridge between the tool abstraction
and the checkpoint-based HITL mechanism.

**TDD steps:**

1. Write `tests/test_clarification_middleware.py` with these tests:
   - `test_intercepts_ask_clarification_tool_call` -- a fake AIMessage with an
     `ask_clarification` tool_call gets intercepted; the middleware raises/returns
     an interrupt payload containing question, type, options, context.
   - `test_passes_through_non_clarification_tool_calls` -- a tool call to any other
     tool passes through unchanged.
   - `test_interrupt_payload_structure` -- the interrupt payload dict has keys
     `question`, `clarification_type`, `context`, `options`, `tool_call_id`.
   - `test_human_response_converted_to_tool_message` -- given a resume value (the
     human's answer), the middleware produces a `ToolMessage` with the correct
     `tool_call_id` and content.
2. Run tests: all 4 fail.
3. Implement the middleware.
4. Run tests: all 4 pass.

**Implementation:**

```python
# muse/middlewares/clarification_middleware.py
"""Middleware that intercepts ask_clarification tool calls and converts to HITL interrupt."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt


_TOOL_NAME = "ask_clarification"


class ClarificationMiddleware:
    """Intercepts ask_clarification tool calls, fires interrupt(), and injects response.

    Intended to be the LAST middleware in the chain so that all other middleware
    has already processed the state before we potentially halt execution.
    """

    def should_intercept(self, tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return the first ask_clarification tool call, or None."""
        for tc in tool_calls:
            if tc.get("name") == _TOOL_NAME:
                return tc
        return None

    def build_interrupt_payload(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Build the interrupt payload from a tool call dict."""
        args = tool_call.get("args", {})
        return {
            "question": args.get("question", ""),
            "clarification_type": args.get("clarification_type", "missing_info"),
            "context": args.get("context"),
            "options": args.get("options"),
            "tool_call_id": tool_call.get("id", ""),
            "source": "ask_clarification",
        }

    def fire_interrupt(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Build payload and call interrupt(). Returns the human response."""
        payload = self.build_interrupt_payload(tool_call)
        return interrupt(payload)

    def build_tool_message(
        self, *, tool_call_id: str, human_response: str
    ) -> dict[str, Any]:
        """Build a ToolMessage-compatible dict from the human response."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": human_response,
        }
```

**Test file:**

```python
# tests/test_clarification_middleware.py
"""Tests for ClarificationMiddleware."""

import unittest


class ClarificationMiddlewareTests(unittest.TestCase):
    def _make_middleware(self):
        from muse.middlewares.clarification_middleware import ClarificationMiddleware
        return ClarificationMiddleware()

    def test_intercepts_ask_clarification_tool_call(self):
        mw = self._make_middleware()
        tool_calls = [
            {"name": "ask_clarification", "id": "tc_1", "args": {"question": "How many?", "clarification_type": "missing_info"}},
        ]
        matched = mw.should_intercept(tool_calls)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["name"], "ask_clarification")

    def test_passes_through_non_clarification_tool_calls(self):
        mw = self._make_middleware()
        tool_calls = [
            {"name": "write_section", "id": "tc_2", "args": {"text": "hello"}},
            {"name": "self_review", "id": "tc_3", "args": {}},
        ]
        matched = mw.should_intercept(tool_calls)
        self.assertIsNone(matched)

    def test_interrupt_payload_structure(self):
        mw = self._make_middleware()
        tool_call = {
            "name": "ask_clarification",
            "id": "tc_99",
            "args": {
                "question": "Which format?",
                "clarification_type": "approach_choice",
                "context": "Two citation formats are common",
                "options": [
                    {"label": "APA", "description": "American Psychological Association"},
                    {"label": "GB/T", "description": "Chinese national standard"},
                ],
            },
        }
        payload = mw.build_interrupt_payload(tool_call)
        self.assertEqual(payload["question"], "Which format?")
        self.assertEqual(payload["clarification_type"], "approach_choice")
        self.assertEqual(payload["context"], "Two citation formats are common")
        self.assertEqual(len(payload["options"]), 2)
        self.assertEqual(payload["tool_call_id"], "tc_99")
        self.assertEqual(payload["source"], "ask_clarification")

    def test_human_response_converted_to_tool_message(self):
        mw = self._make_middleware()
        msg = mw.build_tool_message(tool_call_id="tc_99", human_response="Use GB/T 7714")
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["tool_call_id"], "tc_99")
        self.assertEqual(msg["content"], "Use GB/T 7714")


if __name__ == "__main__":
    unittest.main()
```

---

## Task 3: Upgrade `build_interrupt_node` with options/context (backward-compatible)

**File:** `muse/graph/nodes/review.py` (modify existing)

**Why:** The four top-level interrupt nodes (`review_refs`, `approve_outline`,
`review_draft`, `approve_final`) currently emit a minimal payload with only `stage`,
`project_id`, `ref_count`, `chapter_count`. We upgrade them to include `question`,
`clarification_type`, `options`, and `context` so the CLI can display richer prompts.
Old resume payloads (`{approved: true}`) continue to work.

**TDD steps:**

1. Write additional tests in `tests/test_hitl_interrupt.py`:
   - `test_interrupt_payload_contains_structured_fields` -- the interrupt value dict
     has `question` (str), `clarification_type` (str), and `options` (list).
   - `test_interrupt_backward_compat_bool_resume` -- resuming with a bare `True`
     still works (produces `{"approved": True}`).
   - `test_interrupt_payload_stage_specific_options` -- each stage has its own
     options list (e.g., `review_refs` has "continue", "add keywords", "add manually").
2. Run tests: new tests fail, existing test still passes.
3. Modify `build_interrupt_node`.
4. Run tests: all pass (old + new).

**Changes to `muse/graph/nodes/review.py`:**

```python
# --- Add after _REVIEW_LENSES ---

# Stage-specific structured interrupt metadata.
_STAGE_INTERRUPT_META: dict[str, dict[str, Any]] = {
    "research": {
        "question": "References collected. How would you like to proceed?",
        "clarification_type": "risk_confirmation",
        "options": [
            {"label": "continue", "description": "Accept references and proceed to outline"},
            {"label": "add_keywords", "description": "Add more search keywords and re-search"},
            {"label": "add_manually", "description": "Provide additional references manually"},
        ],
    },
    "outline": {
        "question": "Outline generated. Choose a plan or customize.",
        "clarification_type": "approach_choice",
        "options": [
            {"label": "approve", "description": "Accept the proposed outline"},
            {"label": "revise", "description": "Request outline revisions with feedback"},
            {"label": "custom", "description": "Provide a custom outline structure"},
        ],
    },
    "draft": {
        "question": "Draft chapters complete. Review quality and decide next step.",
        "clarification_type": "suggestion",
        "options": [
            {"label": "approve", "description": "Accept draft and proceed to citation verification"},
            {"label": "auto_fix", "description": "Auto-fix flagged issues and re-draft"},
            {"label": "guide_revision", "description": "Provide specific revision guidance"},
        ],
    },
    "final": {
        "question": "Final thesis assembled. Confirm before export.",
        "clarification_type": "risk_confirmation",
        "options": [
            {"label": "accept", "description": "Accept and export the thesis"},
            {"label": "review_details", "description": "Show detailed quality report before accepting"},
            {"label": "remove_weak", "description": "Remove weakly-supported citations and re-polish"},
        ],
    },
}


def build_interrupt_node(stage: str, *, auto_approve: bool):
    def interrupt_node(state: dict[str, Any]) -> dict[str, Any]:
        meta = _STAGE_INTERRUPT_META.get(stage, {})
        payload = {
            "stage": stage,
            "project_id": state.get("project_id"),
            "ref_count": len(state.get("references", [])),
            "chapter_count": len(state.get("chapter_plans", [])),
            "question": meta.get("question", f"Stage '{stage}' complete. Approve?"),
            "clarification_type": meta.get("clarification_type", "risk_confirmation"),
            "options": meta.get("options", []),
            "context": _build_stage_context(stage, state),
        }
        if auto_approve:
            feedback = {"stage": stage, "approved": True, "auto_approve": True}
        else:
            feedback = interrupt(payload)
            if not isinstance(feedback, dict):
                feedback = {"stage": stage, "approved": bool(feedback)}
        return {"review_feedback": [feedback]}

    return interrupt_node


def _build_stage_context(stage: str, state: dict[str, Any]) -> str:
    """Build human-readable context string for a stage interrupt."""
    parts: list[str] = []
    if stage == "research":
        refs = state.get("references", [])
        parts.append(f"{len(refs)} reference(s) found.")
        queries = state.get("search_queries", [])
        if queries:
            parts.append(f"Search queries used: {', '.join(queries[:5])}")
    elif stage == "outline":
        plans = state.get("chapter_plans", [])
        titles = [p.get("chapter_title", "?") for p in plans if isinstance(p, dict)]
        parts.append(f"{len(plans)} chapter(s): {', '.join(titles)}")
    elif stage == "draft":
        chapters = state.get("chapters", {})
        scores = []
        for ch_id, ch in chapters.items():
            if isinstance(ch, dict) and ch.get("quality_scores"):
                min_score = min(ch["quality_scores"].values()) if ch["quality_scores"] else 0
                scores.append(f"{ch_id}={min_score}")
        parts.append(f"{len(chapters)} chapter(s) drafted.")
        if scores:
            parts.append(f"Min quality scores: {', '.join(scores)}")
    elif stage == "final":
        flagged = state.get("flagged_citations", [])
        verified = state.get("verified_citations", [])
        parts.append(f"{len(verified)} verified, {len(flagged)} flagged citation(s).")
    return " ".join(parts) if parts else ""
```

---

## Task 4: Upgrade CLI review command for structured feedback

**File:** `muse/cli.py` (modify existing)

**Why:** The current `review` command only accepts `--approve` (bool) and `--comment`
(str). With structured interrupts that include options, the CLI must display the
available options, accept an `--option` argument, and pass structured feedback back.

**TDD steps:**

1. Write tests in `tests/test_cli.py` (extend existing):
   - `test_review_with_option_flag` -- `muse review --run-id X --stage research --option continue`
     produces feedback `{"stage": "research", "approved": True, "option": "continue"}`.
   - `test_review_with_comment_and_option` -- both `--option` and `--comment` are
     preserved in the feedback dict.
   - `test_review_backward_compat_approve_flag` -- old `--approve` still works
     without `--option`.
   - `test_graph_response_includes_structured_fields` -- `_graph_response` extracts
     `question`, `options`, and `context` from interrupt value.
2. Run tests: new tests fail.
3. Modify `cli.py`.
4. Run tests: all pass.

**Changes to `muse/cli.py`:**

```python
# --- In build_parser(), modify the review subparser ---
    review = sub.add_parser("review", help="Record HITL decision for a run")
    review.add_argument("--run-id", required=True)
    review.add_argument("--stage", required=True, choices=_STAGE_CHOICES)
    review.add_argument("--approve", action="store_true")
    review.add_argument("--option", default=None, help="Selected option label from interrupt choices")
    review.add_argument("--comment", default="")
    review.set_defaults(func=cmd_review)


# --- Modify cmd_review ---
def cmd_review(args: argparse.Namespace) -> int:
    runtime = _runtime_from_args(args)
    option = getattr(args, "option", None)
    # If an option is provided, infer approval from the option itself.
    approved = bool(args.approve) or (option is not None)
    feedback: dict[str, Any] = {
        "stage": str(args.stage),
        "approved": approved,
        "comment": args.comment,
    }
    if option is not None:
        feedback["option"] = option
    runtime.store.append_hitl_feedback(args.run_id, feedback)
    print(json.dumps({"run_id": args.run_id, "saved": True, "feedback": feedback}, ensure_ascii=False, indent=2))
    return 0


# --- Modify _graph_response to include structured fields ---
def _graph_response(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    if result.get("__interrupt__"):
        interrupt_value = getattr(result["__interrupt__"][0], "value", {})
        resp: dict[str, Any] = {
            "status": "waiting_hitl",
            "stage": _interrupt_stage(result),
            "thread_id": thread_id,
        }
        if isinstance(interrupt_value, dict):
            if interrupt_value.get("question"):
                resp["question"] = interrupt_value["question"]
            if interrupt_value.get("options"):
                resp["options"] = interrupt_value["options"]
            if interrupt_value.get("context"):
                resp["context"] = interrupt_value["context"]
            if interrupt_value.get("clarification_type"):
                resp["clarification_type"] = interrupt_value["clarification_type"]
        return resp
    return {
        "status": "completed",
        "thread_id": thread_id,
        "output_filepath": result.get("output_filepath", ""),
    }
```

---

## Task 5: Integration test -- HITL in ReAct sub-graph context

**File:** `tests/test_structured_hitl_integration.py` (new file)

**Why:** End-to-end verification that the structured HITL flow works across the full
stack: ReAct agent emits `ask_clarification` tool call, ClarificationMiddleware
intercepts and fires `interrupt()`, CLI-side resume with structured feedback gets
injected back as a ToolMessage, and the agent continues.

**TDD steps:**

1. Write `tests/test_structured_hitl_integration.py` with these tests:
   - `test_top_level_interrupt_has_structured_payload` -- run the graph with
     `auto_approve=False`, verify the first interrupt has `question`, `options`, `context`.
   - `test_resume_with_option_label` -- resume the interrupt with
     `{"stage": "research", "approved": True, "option": "continue"}` and verify
     the graph continues to the next interrupt.
   - `test_resume_with_freetext_comment` -- resume with a `comment` field instead
     of an `option` and verify the graph continues.
   - `test_backward_compat_bool_resume` -- resume with bare `True` (pre-Phase-2
     behavior) and verify the graph does not crash.
2. Run tests: all fail.
3. Wire everything together (middleware integration in the ReAct path, if Phase 1
   sub-graphs exist; otherwise test against top-level interrupts only).
4. Run tests: all pass.

**Test file:**

```python
# tests/test_structured_hitl_integration.py
"""Integration tests for structured HITL flow."""

import tempfile
import unittest

from muse.config import Settings


class _StubSearch:
    def search_multi_source(self, topic, discipline, extra_queries=None):
        return (
            [{"ref_id": "@test2024", "title": "Test", "authors": ["A"], "year": 2024,
              "doi": "10.1/t", "venue": "V", "abstract": "A.", "source": "stub",
              "verified_metadata": True}],
            extra_queries or [topic],
        )


class _StubLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "search queries" in system.lower() or "Generate 7" in system:
            return {"queries": ["test query"]}
        if "Analyze this research topic" in system:
            return {"research_gaps": ["g"], "core_concepts": ["c"],
                    "methodology_domain": "cs", "suggested_contributions": ["s"]}
        if "Generate a thesis outline" in system:
            return {"chapters": [{"chapter_id": "ch_01", "chapter_title": "Intro",
                     "target_words": 500, "complexity": "low",
                     "subsections": [{"title": "Background"}]}]}
        return {}


class _StubServices:
    def __init__(self):
        self.llm = _StubLLM()
        self.search = _StubSearch()
        self.local_refs = []
        self.rag_index = None


class StructuredHitlIntegrationTests(unittest.TestCase):
    def _make_graph(self, tmp):
        from muse.graph.launcher import build_graph

        settings = Settings(
            llm_api_key="x", llm_base_url="http://localhost", llm_model="stub",
            model_router_config={}, runs_dir=tmp, semantic_scholar_api_key=None,
            openalex_email=None, crossref_mailto=None, refs_dir=None, checkpoint_dir=None,
        )
        return build_graph(settings, services=_StubServices(), thread_id="hitl-int", auto_approve=False)

    def test_top_level_interrupt_has_structured_payload(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp:
            graph = self._make_graph(tmp)
            result = invoke(
                graph,
                {"project_id": "hitl-int", "topic": "Test", "discipline": "cs",
                 "language": "zh", "format_standard": "GB/T 7714-2015", "output_format": "markdown"},
                thread_id="hitl-int",
            )
            self.assertIn("__interrupt__", result)
            value = result["__interrupt__"][0].value
            self.assertIn("question", value)
            self.assertIn("options", value)
            self.assertIn("context", value)
            self.assertEqual(value["stage"], "research")

    def test_resume_with_option_label(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp:
            graph = self._make_graph(tmp)
            invoke(
                graph,
                {"project_id": "hitl-opt", "topic": "Test", "discipline": "cs",
                 "language": "zh", "format_standard": "GB/T 7714-2015", "output_format": "markdown"},
                thread_id="hitl-int",
            )
            result = invoke(
                graph, None, thread_id="hitl-int",
                resume={"stage": "research", "approved": True, "option": "continue"},
            )
            self.assertIn("__interrupt__", result)
            self.assertEqual(result["__interrupt__"][0].value["stage"], "outline")

    def test_resume_with_freetext_comment(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp:
            graph = self._make_graph(tmp)
            invoke(
                graph,
                {"project_id": "hitl-cmt", "topic": "Test", "discipline": "cs",
                 "language": "zh", "format_standard": "GB/T 7714-2015", "output_format": "markdown"},
                thread_id="hitl-int",
            )
            result = invoke(
                graph, None, thread_id="hitl-int",
                resume={"stage": "research", "approved": True, "comment": "Looks good"},
            )
            self.assertIn("__interrupt__", result)

    def test_backward_compat_bool_resume(self):
        from muse.graph.launcher import invoke

        with tempfile.TemporaryDirectory() as tmp:
            graph = self._make_graph(tmp)
            invoke(
                graph,
                {"project_id": "hitl-bool", "topic": "Test", "discipline": "cs",
                 "language": "zh", "format_standard": "GB/T 7714-2015", "output_format": "markdown"},
                thread_id="hitl-int",
            )
            result = invoke(
                graph, None, thread_id="hitl-int",
                resume=True,
            )
            # Should not crash; bare True triggers the bool fallback
            self.assertIn("__interrupt__", result)


if __name__ == "__main__":
    unittest.main()
```

---

## File inventory

| Action | Path |
|--------|------|
| Create | `muse/tools/__init__.py` |
| Create | `muse/tools/orchestration.py` |
| Create | `muse/middlewares/__init__.py` |
| Create | `muse/middlewares/clarification_middleware.py` |
| Modify | `muse/graph/nodes/review.py` |
| Modify | `muse/cli.py` |
| Create | `tests/test_ask_clarification_tool.py` |
| Create | `tests/test_clarification_middleware.py` |
| Create | `tests/test_structured_hitl_integration.py` |

## Verification

After all 5 tasks are complete, run:

```bash
python -m pytest tests/test_ask_clarification_tool.py tests/test_clarification_middleware.py tests/test_structured_hitl_integration.py tests/test_hitl_interrupt.py -v
```

All tests (new + old) must pass. Zero regressions in existing HITL behavior.

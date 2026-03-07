import tempfile
import unittest

from muse.config import Settings
from muse.engine import ThesisEngine, EngineContext
from muse.runtime import Runtime
from muse.schemas import new_thesis_state
from muse.store import RunStore


class RuntimeFlowTests(unittest.TestCase):
    def _make_runtime(self, runs_dir: str) -> Runtime:
        settings = Settings(
            llm_api_key="x",
            llm_base_url="http://localhost",
            llm_model="stub",
            model_router_config={},
            runs_dir=runs_dir,
            semantic_scholar_api_key=None,
            openalex_email=None,
            crossref_mailto=None,
            refs_dir=None,
        )
        return Runtime(settings)

    def test_stops_at_hitl_without_auto_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            store.save_state(run_id, state)

            def stage1(ctx: EngineContext):
                ctx.state["current_stage"] = 1
                return "hitl"

            engine = ThesisEngine(
                store=store,
                stages={1: stage1},
            )

            result = engine.run(run_id=run_id, start_stage=1, auto_approve=False)
            self.assertEqual(result["status"], "waiting_hitl")
            self.assertEqual(result["stage"], 1)

    def test_continues_with_auto_approve(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            store.save_state(run_id, state)

            def stage1(ctx: EngineContext):
                ctx.state["current_stage"] = 1
                return "ok"

            def stage2(ctx: EngineContext):
                ctx.state["current_stage"] = 2
                return "done"

            engine = ThesisEngine(
                store=store,
                stages={1: stage1, 2: stage2},
            )

            result = engine.run(run_id=run_id, start_stage=1, auto_approve=True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["stage"], 2)

    def test_engine_propagates_blocked_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(base_dir=tmp)
            run_id = store.create_run(topic="topic")
            state = new_thesis_state(
                project_id=run_id,
                topic="topic",
                discipline="cs",
                language="zh",
                format_standard="GB/T 7714-2015",
            )
            store.save_state(run_id, state)

            def stage6(ctx: EngineContext):
                ctx.state["current_stage"] = 6
                return "blocked"

            engine = ThesisEngine(store=store, stages={6: stage6})
            result = engine.run(run_id=run_id, start_stage=6, auto_approve=True)

            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["stage"], 6)

    def test_build_engine_rejects_docx_output_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._make_runtime(tmp)
            run_id = runtime.store.create_run(topic="topic")

            with self.assertRaises(ValueError):
                runtime.build_engine(run_id=run_id, output_format="docx")


if __name__ == "__main__":
    unittest.main()

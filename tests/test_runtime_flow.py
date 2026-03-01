import tempfile
import unittest

from thesis_agent.engine import ThesisEngine, EngineContext
from thesis_agent.schemas import new_thesis_state
from thesis_agent.store import RunStore


class RuntimeFlowTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

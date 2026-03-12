import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from muse.cli import _graph_response, build_parser, cmd_review


class CliSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_prog_name_is_muse(self):
        self.assertEqual(self.parser.prog, "muse")

    def test_output_format_rejects_docx(self):
        commands = [
            ["run", "--topic", "Test", "--output-format", "docx"],
            ["resume", "--run-id", "r1", "--output-format", "docx"],
            ["export", "--run-id", "r1", "--output-format", "docx"],
        ]

        for argv in commands:
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    self.parser.parse_args(argv)

    def test_template_option_removed(self):
        commands = [
            ["run", "--topic", "Test", "--template", "template.docx"],
            ["resume", "--run-id", "r1", "--template", "template.docx"],
            ["export", "--run-id", "r1", "--template", "template.docx"],
        ]

        for argv in commands:
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    self.parser.parse_args(argv)

    def test_output_format_still_accepts_markdown_latex_and_pdf(self):
        commands = [
            ["run", "--topic", "Test", "--output-format", "markdown"],
            ["resume", "--run-id", "r1"],
            ["export", "--run-id", "r1", "--output-format", "pdf"],
        ]

        for argv in commands:
            with self.subTest(argv=argv):
                args = self.parser.parse_args(argv)
                if hasattr(args, "output_format"):
                    self.assertIn(args.output_format, {"markdown", "latex", "pdf"})
                self.assertFalse(hasattr(args, "template"))


class _FakeStore:
    def __init__(self):
        self.saved: list[tuple[str, dict]] = []

    def append_hitl_feedback(self, run_id: str, feedback: dict) -> None:
        self.saved.append((run_id, feedback))


class _FakeRuntime:
    def __init__(self):
        self.store = _FakeStore()


class CliReviewTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_review_with_option_flag(self):
        args = self.parser.parse_args(
            ["review", "--run-id", "run-1", "--stage", "research", "--option", "continue"]
        )
        runtime = _FakeRuntime()
        with patch("muse.cli._runtime_from_args", return_value=runtime):
            with redirect_stdout(io.StringIO()) as buffer:
                exit_code = cmd_review(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(runtime.store.saved[0][0], "run-1")
        self.assertEqual(
            runtime.store.saved[0][1],
            {"stage": "research", "approved": True, "comment": "", "option": "continue"},
        )
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["feedback"]["option"], "continue")

    def test_review_with_comment_and_option(self):
        args = self.parser.parse_args(
            [
                "review",
                "--run-id",
                "run-2",
                "--stage",
                "draft",
                "--option",
                "guide_revision",
                "--comment",
                "Please strengthen citations.",
            ]
        )
        runtime = _FakeRuntime()
        with patch("muse.cli._runtime_from_args", return_value=runtime):
            with redirect_stdout(io.StringIO()):
                cmd_review(args)

        self.assertEqual(
            runtime.store.saved[0][1],
            {
                "stage": "draft",
                "approved": True,
                "comment": "Please strengthen citations.",
                "option": "guide_revision",
            },
        )

    def test_review_backward_compat_approve_flag(self):
        args = self.parser.parse_args(
            ["review", "--run-id", "run-3", "--stage", "final", "--approve"]
        )
        runtime = _FakeRuntime()
        with patch("muse.cli._runtime_from_args", return_value=runtime):
            with redirect_stdout(io.StringIO()):
                cmd_review(args)

        self.assertEqual(
            runtime.store.saved[0][1],
            {"stage": "final", "approved": True, "comment": ""},
        )

    def test_graph_response_includes_structured_fields(self):
        result = {
            "__interrupt__": [
                SimpleNamespace(
                    value={
                        "stage": "research",
                        "question": "References collected. How proceed?",
                        "options": [{"label": "continue", "description": "Proceed"}],
                        "context": "1 reference found.",
                        "clarification_type": "risk_confirmation",
                    }
                )
            ]
        }

        response = _graph_response(result, "run-4")
        self.assertEqual(response["status"], "waiting_hitl")
        self.assertEqual(response["question"], "References collected. How proceed?")
        self.assertEqual(response["options"][0]["label"], "continue")
        self.assertEqual(response["context"], "1 reference found.")
        self.assertEqual(response["clarification_type"], "risk_confirmation")


if __name__ == "__main__":
    unittest.main()

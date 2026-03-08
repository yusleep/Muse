"""Tests for the ask_clarification orchestration tool."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class AskClarificationToolTests(unittest.TestCase):
    def test_tool_schema_has_required_fields(self):
        from muse.tools.orchestration import ask_clarification

        schema = ask_clarification.args_schema.model_json_schema()
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

        schema = ask_clarification.args_schema.model_json_schema()
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
            {
                "question": "How many chapters?",
                "clarification_type": "missing_info",
            }
        )
        self.assertIn("CLARIFICATION PENDING", result)
        self.assertIn("missing_info", result)

    def test_options_schema_structure(self):
        from muse.tools.orchestration import AskClarificationInput, ClarificationOption

        option = ClarificationOption(label="Plan A", description="Five chapters")
        payload = AskClarificationInput(
            question="Which plan?",
            clarification_type="approach_choice",
            options=[option],
        )
        self.assertEqual(payload.options[0].label, "Plan A")
        self.assertEqual(payload.options[0].description, "Five chapters")

    def test_tool_uses_runtime_handler_when_configured(self):
        from muse.tools.orchestration import ask_clarification, set_clarification_handler

        with patch(
            "muse.tools.orchestration._normalize_clarification_response",
            return_value="Use five chapters.",
        ):
            try:
                set_clarification_handler(lambda **_: "ignored")
                result = ask_clarification.invoke(
                    {
                        "question": "How many chapters?",
                        "clarification_type": "missing_info",
                    }
                )
            finally:
                set_clarification_handler(None)

        self.assertEqual(result, "Use five chapters.")


if __name__ == "__main__":
    unittest.main()

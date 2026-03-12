import json
import tempfile
import unittest
from copy import deepcopy

from muse.config import Settings


class _ReviewLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        if "strict thesis reviewer" not in system:
            raise AssertionError(f"unexpected prompt: {system}")
        return {
            "scores": {
                "coherence": 3,
                "logic": 4,
                "citation": 4,
                "term_consistency": 4,
                "balance": 4,
                "redundancy": 4,
            },
            "review_notes": [
                {
                    "subtask_id": "sub_01",
                    "issue": "衔接不足",
                    "instruction": "补充过渡段。",
                    "severity": 2,
                },
                {
                    "subtask_id": "sub_02",
                    "issue": "轻微措辞",
                    "instruction": "统一术语。",
                    "severity": 1,
                },
            ],
        }


class _PolishLLM:
    def structured(self, *, system, user, route="default", max_tokens=2500):
        payload = json.loads(user)
        if "Polish the academic thesis chapter" in system:
            return {
                "final_text": f"[polished]{payload.get('text', '')}",
                "polish_notes": ["统一了术语。"],
            }
        if "你是一位学术论文摘要撰写专家" in system:
            return {
                "abstract": "这是一段中文摘要。",
                "keywords": ["图工作流", "检查点"],
            }
        if "You are an academic abstract writer" in system:
            return {
                "abstract": "This is an English abstract.",
                "keywords": ["graph workflow", "checkpoint"],
            }
        raise AssertionError(f"unexpected prompt: {system}")


class _ReviewServices:
    def __init__(self):
        self.llm = _ReviewLLM()


class _PolishServices:
    def __init__(self):
        self.llm = _PolishLLM()


class GraphBridgeCleanupTests(unittest.TestCase):
    def test_chapter_review_node_builds_revision_instructions_from_review_notes(self):
        from muse.graph.nodes.review import build_chapter_review_node

        node = build_chapter_review_node(_ReviewServices())
        result = node(
            {
                "chapter_plan": {"chapter_title": "绪论"},
                "merged_text": "已有草稿。",
            }
        )

        self.assertEqual(result["quality_scores"]["coherence"], 3)
        self.assertEqual(result["revision_instructions"], {"sub_01": "补充过渡段。"})
        self.assertEqual(len(result["review_notes"]), 8)

    def test_polish_node_preserves_existing_chapter_results_shape(self):
        from muse.graph.nodes.polish import build_polish_node

        chapter_results = [
            {
                "chapter_id": "ch_01",
                "chapter_title": "绪论",
                "merged_text": "第一章内容。",
                "subtask_results": [{"subtask_id": "sub_01"}],
            },
            {
                "chapter_id": "ch_02",
                "chapter_title": "系统设计",
                "merged_text": "第二章内容。",
                "subtask_results": [{"subtask_id": "sub_02"}],
            },
        ]
        state = {
            "topic": "LangGraph thesis automation",
            "language": "zh",
            "format_standard": "GB/T 7714-2015",
            "paper_package": {"chapter_results": deepcopy(chapter_results)},
            "final_text": "",
        }

        node = build_polish_node(_PolishServices())
        result = node(state)

        self.assertIn("[polished]第一章内容。", result["final_text"])
        self.assertIn("[polished]第二章内容。", result["final_text"])
        self.assertEqual(result["abstract_zh"], "这是一段中文摘要。")
        self.assertEqual(result["abstract_en"], "This is an English abstract.")
        self.assertEqual(state["paper_package"]["chapter_results"], chapter_results)

    def test_export_node_blocks_only_on_contradiction_flagged_citations(self):
        from muse.graph.nodes.export import build_export_node

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                llm_api_key="x",
                llm_base_url="http://localhost",
                llm_model="stub",
                model_router_config={},
                runs_dir=tmp,
                semantic_scholar_api_key=None,
                openalex_email=None,
                crossref_mailto=None,
                refs_dir=None,
                checkpoint_dir=None,
            )
            node = build_export_node(settings)

            soft_result = node(
                {
                    "project_id": "run-soft",
                    "paper_package": {"chapter_results": []},
                    "final_text": "# Title\n\ncontent",
                    "flagged_citations": [
                        {
                            "cite_key": "@x",
                            "reason": "metadata_mismatch",
                            "detail": "metadata_mismatch",
                        }
                    ],
                    "references": [],
                    "citation_uses": [],
                    "output_format": "markdown",
                }
            )

            self.assertTrue(soft_result["output_filepath"].endswith("thesis.md"))
            self.assertTrue(soft_result["paper_package"]["export_artifacts"] == {})

            hard_result = node(
                {
                    "project_id": "run-hard",
                    "paper_package": {"chapter_results": []},
                    "final_text": "# Title\n\ncontent",
                    "flagged_citations": [
                        {
                            "cite_key": "@y",
                            "reason": "unsupported_claim",
                            "detail": "entailment result=contradiction",
                        }
                    ],
                    "references": [],
                    "citation_uses": [],
                    "output_format": "markdown",
                }
            )

            self.assertEqual(hard_result["output_filepath"], "")
            self.assertEqual(hard_result["paper_package"]["export_artifacts"], {})

    def test_polish_node_generates_final_text_and_abstracts_without_stage_bridge(self):
        import muse.graph.nodes.polish as polish_module

        node = polish_module.build_polish_node(_PolishServices())
        result = node(
            {
                "topic": "LangGraph thesis automation",
                "language": "zh",
                "format_standard": "GB/T 7714-2015",
                "paper_package": {
                    "chapter_results": [
                        {"chapter_id": "ch_01", "chapter_title": "绪论", "merged_text": "第一章内容。"}
                    ]
                },
                "final_text": "",
            }
        )

        self.assertFalse(hasattr(polish_module, "stage5_polish"))
        self.assertIn("[polished]第一章内容。", result["final_text"])
        self.assertEqual(result["abstract_zh"], "这是一段中文摘要。")
        self.assertEqual(result["abstract_en"], "This is an English abstract.")

    def test_export_node_writes_markdown_artifacts_without_stage_bridge(self):
        import muse.graph.nodes.export as export_module

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                llm_api_key="x",
                llm_base_url="http://localhost",
                llm_model="stub",
                model_router_config={},
                runs_dir=tmp,
                semantic_scholar_api_key=None,
                openalex_email=None,
                crossref_mailto=None,
                refs_dir=None,
                checkpoint_dir=None,
            )
            node = export_module.build_export_node(settings)
            result = node(
                {
                    "project_id": "run-export",
                    "paper_package": {"chapter_results": []},
                    "final_text": "# Title\n\ncontent",
                    "flagged_citations": [],
                    "references": [],
                    "citation_uses": [],
                    "output_format": "markdown",
                }
            )

            self.assertFalse(hasattr(export_module, "stage6_export"))
            self.assertTrue(result["output_filepath"].endswith("thesis.md"))
            self.assertEqual(result["paper_package"]["export_warnings"], [])


if __name__ == "__main__":
    unittest.main()

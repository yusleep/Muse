"""Best-effort PDF visual validation node."""

from __future__ import annotations

import os
from typing import Any

from muse.prompts.visual_check import visual_check_prompt


_TARGET_PAGES = (1, 5, 10, 15, 20)


def _page_summaries(pdf_path: str) -> list[dict[str, Any]]:
    import fitz

    document = fitz.open(pdf_path)
    try:
        summaries: list[dict[str, Any]] = []
        for page_number in _TARGET_PAGES:
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(document):
                continue
            page = document.load_page(page_index)
            blocks = page.get_text("blocks")
            text = page.get_text()
            rect = getattr(page, "rect", None)
            summaries.append(
                {
                    "page": page_number,
                    "width": getattr(rect, "width", 0),
                    "height": getattr(rect, "height", 0),
                    "block_count": len(blocks) if isinstance(blocks, list) else 0,
                    "text_preview": str(text).strip()[:400],
                }
            )
        return summaries
    finally:
        document.close()


def build_visual_check_node(*, services: Any):
    def visual_check(state: dict[str, Any]) -> dict[str, Any]:
        warnings = list(state.get("export_warnings", []) or [])
        if str(state.get("output_format", "")).strip().lower() != "latex":
            return {"visual_issues": [], "export_warnings": warnings}

        export_artifacts = state.get("export_artifacts", {})
        pdf_path = ""
        if isinstance(export_artifacts, dict):
            pdf_path = str(export_artifacts.get("pdf_path", "") or "").strip()
        if not pdf_path or not os.path.isfile(pdf_path):
            warnings.append("visual check skipped: pdf artifact unavailable")
            return {"visual_issues": [], "export_warnings": warnings}

        try:
            page_summaries = _page_summaries(pdf_path)
        except ImportError:
            warnings.append("visual check skipped: PyMuPDF (fitz) is not installed")
            return {"visual_issues": [], "export_warnings": warnings}
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"visual check skipped: {exc}")
            return {"visual_issues": [], "export_warnings": warnings}

        llm = getattr(services, "llm", None)
        if llm is None or not hasattr(llm, "structured") or not page_summaries:
            return {"visual_issues": [], "export_warnings": warnings}

        system, user = visual_check_prompt(page_summaries)
        try:
            payload = llm.structured(
                system=system,
                user=user,
                route="default",
                max_tokens=1000,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"visual check skipped: {exc}")
            return {"visual_issues": [], "export_warnings": warnings}

        issues = payload.get("issues", []) if isinstance(payload, dict) else []
        normalized = [dict(issue) for issue in issues if isinstance(issue, dict)]
        return {
            "visual_issues": normalized,
            "export_warnings": warnings,
            "paper_package": {
                "visual_issues": normalized,
                "export_warnings": warnings,
            },
        }

    return visual_check

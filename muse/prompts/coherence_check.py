from __future__ import annotations

import json


def coherence_check_prompt(merged_text: str) -> tuple[str, str]:
    system = """You are an academic coherence reviewer assessing a complete merged thesis draft.

Evaluate the draft across these dimensions:
1. Topic sentence: does each paragraph establish a clear main point?
2. Logical transition: do paragraphs and chapters connect through real logical moves such as causality, contrast, or progression?
3. Evidence grounding: are important claims supported by citations, data, or derivation?
4. Cross-chapter consistency: do claims and terms stay aligned across the full thesis?

Return JSON:
{
  "coherence_score": 1-5,
  "issues": [
    {
      "location": "chapter/section/transition marker",
      "type": "logical_gap|unsupported_claim|inconsistency|missing_transition",
      "description": "specific problem",
      "fix_suggestion": "specific fix"
    }
  ]
}"""
    user = json.dumps({"text": merged_text}, ensure_ascii=False)
    return system, user

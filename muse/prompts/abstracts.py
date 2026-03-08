from __future__ import annotations

import json


def abstract_zh_prompt(topic: str, text: str) -> tuple[str, str]:
    system = (
        "你是一位学术论文摘要撰写专家。根据以下论文全文，生成一段300-500字的中文摘要和3-5个关键词。"
        "返回JSON，keys: abstract (string), keywords (list of strings)。"
    )
    return system, json.dumps({"topic": topic, "text": text}, ensure_ascii=False)


def abstract_en_prompt(topic: str, text: str) -> tuple[str, str]:
    system = (
        "You are an academic abstract writer. Generate a 200-300 word English abstract "
        "and 3-5 keywords for the following thesis. Return JSON with keys: abstract, keywords."
    )
    return system, json.dumps({"topic": topic, "text": text}, ensure_ascii=False)

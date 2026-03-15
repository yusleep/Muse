"""Few-shot outline examples for exploratory outline generation."""

from __future__ import annotations


COMPUTER_SCIENCE_OUTLINE_EXAMPLES = [
    {
        "discipline": "computer_science",
        "title": "面向分布式系统的任务编排平台设计与实现",
        "chapters": ["绪论", "相关技术与理论基础", "系统需求分析", "系统设计与实现", "实验与结果分析", "总结与展望"],
    },
    {
        "discipline": "computer_science",
        "title": "基于深度学习的网络流量识别系统研究",
        "chapters": ["绪论", "相关工作", "模型设计", "系统实现", "实验评估", "总结与展望"],
    },
    {
        "discipline": "computer_science",
        "title": "云边协同视频处理系统的架构优化",
        "chapters": ["绪论", "技术背景", "需求分析", "系统架构设计", "关键模块实现", "性能测试", "总结与展望"],
    },
    {
        "discipline": "computer_science",
        "title": "面向物联网的异常检测平台设计",
        "chapters": ["绪论", "相关技术", "需求分析与总体设计", "核心算法设计", "平台实现", "实验分析", "总结与展望"],
    },
    {
        "discipline": "computer_science",
        "title": "高并发网络服务的可靠性治理方法研究",
        "chapters": ["绪论", "理论基础与相关工作", "问题建模", "方法设计", "系统实现", "实验与讨论", "总结与展望"],
    },
]

GENERIC_OUTLINE_EXAMPLES = [
    {
        "discipline": "generic",
        "title": "问题驱动的毕业论文通用结构",
        "chapters": ["绪论", "文献综述", "研究设计", "分析与实现", "结果与讨论", "总结与展望"],
    },
    {
        "discipline": "generic",
        "title": "应用型课题通用论文结构",
        "chapters": ["绪论", "背景与现状", "方案设计", "实现过程", "评估分析", "总结与展望"],
    },
]


def get_examples_for_discipline(discipline: str) -> list[dict[str, object]]:
    normalized = str(discipline or "").strip().lower()
    if "computer" in normalized or "cs" in normalized or "计算机" in normalized:
        return list(COMPUTER_SCIENCE_OUTLINE_EXAMPLES)
    return list(GENERIC_OUTLINE_EXAMPLES)

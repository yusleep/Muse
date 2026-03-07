# Muse — 毕业论文写作 Agent 完整技术方案 v2

> **核心策略**：基于 LangGraph 从零构建，借鉴 OpenClaw 的 Lane Queue、混合记忆搜索、分层上下文组装等生产级模式，实现 Chapter Agent + SubAgent 两层写作架构。

---

## 一、系统总览

### 1.1 设计目标

面向毕业论文/学位论文的全流程写作 Agent，支持生成 20,000–50,000 字的中英文学术论文。覆盖四大核心环节：文献检索与综述、大纲/结构生成、分层写作与迭代润色、引用管理与格式化。

### 1.2 核心设计原则

- **从简到繁**：先用最小可行管道验证核心写作环节，再逐步叠加功能
- **分治写作**：每个原子写作单元（SubAgent）仅产出 1,000–2,000 字，Chapter Agent 负责整合与迭代
- **引用零幻觉**：LLM 永远不从记忆中生成引用，只从预验证的文献数据库中选取
- **人类始终在环**：每个阶段设置 HITL 检查点，AI 辅助而非替代研究者
- **借鉴但不依赖 OpenClaw**：提取其 Lane Queue、混合搜索、分层上下文等模式，但不引入其消息平台架构

### 1.3 六阶段管道架构

```
用户输入 (研究主题 + 学校格式要求)
    │
    ▼
┌──────────────────────────────────┐
│  Dissertation Orchestrator        │  ← LangGraph StateGraph
│  (维护全局 ThesisState)            │  ← 检查点持久化 + HITL 中断
└────────┬─────────────────────────┘
         │
    ┌────▼────────────────────────┐
    │ Stage 1: 文献检索 Agent       │  并行: Semantic Scholar + arXiv
    │ (Literature Search)          │  + OpenAlex + CrossRef
    └────────┬────────────────────┘
         │ ← HITL #1: 确认文献列表
    ┌────▼────────────────────────┐
    │ Stage 2: 大纲生成 Agent       │  结构化输出 → 层级 JSON
    │ (Outline Generation)         │
    └────────┬────────────────────┘
         │ ← HITL #2: 确认/修改大纲
    ┌────▼────────────────────────┐
    │ Stage 3: 两层写作系统          │  Chapter Agent → SubAgents
    │ (Two-Layer Writing)          │  每个 SubAgent 仅写 1000-2000 字
    └────────┬────────────────────┘
         │ ← HITL #3: 审阅初稿
    ┌────▼────────────────────────┐
    │ Stage 4: 引用验证 Agent       │  DOI 验证 + 语义验证
    │ (Citation Verification)      │
    └────────┬────────────────────┘
    ┌────▼────────────────────────┐
    │ Stage 5: 跨章节润色 Agent     │  全局一致性 + 学术语言打磨
    │ (Cross-Chapter Polish)       │
    └────────┬────────────────────┘
         │ ← HITL #4: 最终确认
    ┌────▼────────────────────────┐
    │ Stage 6: 编译输出 Agent       │  → Markdown / LaTeX 项目 / PDF
    │ (Compilation & Export)       │
    └──────────────────────────────┘
```

---

## 二、全局状态管理（ThesisState）

### 2.1 状态数据结构

所有 Agent 共享一个中心化的 `ThesisState`，由 LangGraph 的 checkpointer 持久化。任何阶段失败都可以从上一个检查点恢复。

```python
from typing import TypedDict, List, Optional, Dict
from enum import Enum

class StageStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    HITL_REVIEW = "hitl_review"
    COMPLETED = "completed"
    FAILED = "failed"

class Reference(TypedDict):
    ref_id: str                  # 稳定的引用键, e.g. "@wang2024deep"
    title: str
    authors: List[str]
    year: int
    doi: Optional[str]
    venue: Optional[str]
    abstract: str
    full_text_chunks: List[str]  # GROBID 解析后的分块文本
    verified: bool               # 是否通过 DOI/API 验证
    source_api: str              # "semantic_scholar" | "openalex" | "arxiv"

class SubTaskResult(TypedDict):
    subtask_id: str
    title: str
    target_words: int
    output_text: str
    actual_words: int
    citations_used: List[str]
    key_claims: List[str]
    transition_out: str
    glossary_additions: Dict[str, str]
    confidence: float
    weak_spots: List[str]
    needs_revision: bool
    revision_notes: Optional[str]  # Chapter Agent 填写
    iteration_round: int

class ChapterResult(TypedDict):
    chapter_id: str
    chapter_title: str
    target_words: int
    complexity: str              # "low" | "medium" | "high"
    subtask_results: List[SubTaskResult]
    merged_text: str
    quality_scores: Dict[str, int]  # 6 维度, 各 1-5 分
    iterations_used: int
    status: StageStatus

class ThesisState(TypedDict):
    # ── 用户输入 ──
    topic: str
    discipline: str
    degree_level: str            # "本科" | "硕士" | "博士"
    format_standard: str         # "GB/T 7714" | "APA" | 自定义
    university_template: Optional[str]
    language: str                # "zh" | "en" | "zh-en"
    user_requirements: str

    # ── Stage 1: 文献检索 ──
    search_queries: List[str]
    references: List[Reference]
    literature_summary: str
    stage1_status: StageStatus

    # ── Stage 2: 大纲 ──
    outline_json: dict           # 层级化大纲结构
    chapter_plans: List[dict]    # 每章的详细计划
    stage2_status: StageStatus

    # ── Stage 3: 写作 ──
    chapter_results: List[ChapterResult]
    terminology_glossary: Dict[str, str]  # 关键术语表
    thesis_state_summary: str    # 动态更新的全局摘要
    stage3_status: StageStatus

    # ── Stage 4: 引用 ──
    verified_citations: List[str]
    flagged_citations: List[str]  # 需人工审查的引用
    bibliography_text: str
    stage4_status: StageStatus

    # ── Stage 5: 润色 ──
    polish_notes: List[str]
    final_text: str
    stage5_status: StageStatus

    # ── Stage 6: 输出 ──
    output_format: str           # "markdown" | "latex" | "pdf"
    output_filepath: str
    stage6_status: StageStatus

    # ── 全局元数据 ──
    current_stage: int
    hitl_feedback: List[dict]    # 每次人工审查的反馈记录
    audit_trail: List[dict]      # JSONL 审计日志 (借鉴 OpenClaw)
    total_llm_calls: int
    total_tokens_used: int
```

### 2.2 借鉴 OpenClaw 的 JSONL 审计日志

OpenClaw 将每个 Agent 会话记录为 append-only 的 JSONL 文件，确保完全可追溯。我们在 ThesisState 中也采用相同模式：

```python
import json
from datetime import datetime

def append_audit_log(state: ThesisState, event: dict):
    """追加不可变的审计日志条目"""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "stage": state["current_stage"],
        "event_type": event["type"],  # "llm_call" | "hitl_decision" | "citation_verified"
        "agent": event["agent"],
        "input_summary": event.get("input_summary", ""),
        "output_summary": event.get("output_summary", ""),
        "tokens_used": event.get("tokens_used", 0),
        "model": event.get("model", ""),
    }
    state["audit_trail"].append(entry)
    
    # 同时写入磁盘上的 JSONL 文件
    with open("thesis_audit.jsonl", "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

---

## 三、Stage 1 — 文献检索与 RAG 知识库

### 3.1 多源并行检索

```python
import asyncio

async def search_literature(state: ThesisState) -> ThesisState:
    """并行查询多个学术数据库"""
    topic = state["topic"]
    
    # LLM 提取 3-5 组搜索关键词
    queries = await extract_search_queries(topic, state["discipline"])
    
    # 并行检索 (借鉴 OpenClaw Lane Queue 的并发控制, 每源一个 lane)
    results = await asyncio.gather(
        search_semantic_scholar(queries, max_results=30),
        search_openalex(queries, max_results=30),
        search_arxiv(queries, max_results=20),
        return_exceptions=True
    )
    
    # 按 DOI 去重 → 按引用数排序 → 取 top 50
    all_papers = deduplicate_by_doi(flatten(results))
    ranked_papers = rank_by_citation_count(all_papers)[:50]
    
    # Semantic Scholar 推荐 API 做滚雪球扩展
    seed_ids = [p["paper_id"] for p in ranked_papers[:5]]
    recommended = await semantic_scholar_recommendations(seed_ids)
    ranked_papers = deduplicate_by_doi(ranked_papers + recommended)[:60]
    
    state["references"] = ranked_papers
    state["search_queries"] = queries
    return state
```

### 3.2 PDF 解析与知识库构建

```
PDF 文件
  │
  ▼
GROBID (Docker) → 结构化 TEI-XML (保留章节层级)
  │                    ↘ 失败时回退: PyMuPDF / MinerU (中文)
  ▼
分块策略:
  Level 1: 摘要 (用于粗粒度检索, 定位相关论文)
  Level 2: 章节块 (512-1024 tokens, 128 overlap)
           每块附带: {paper_id, section, authors, year, doi}
  │
  ▼
双模型 Embedding:
  SPECTER2 → 论文级相似度 (title + abstract)
  BGE-M3   → 块级检索 (多语言, 8192 tokens)
  │
  ▼
混合搜索存储 (借鉴 OpenClaw 的 70/30 融合公式):
  ┌─────────────────────────────────────────┐
  │  SQLite + sqlite-vec + FTS5             │
  │                                         │
  │  final_score = 0.7 × cosine_similarity  │
  │              + 0.3 × bm25_score         │
  │                                         │
  │  Union (非交集) 取结果                    │
  └─────────────────────────────────────────┘
  
  生产环境可升级为 Qdrant (更好的元数据过滤)
```

**为什么选 SQLite 起步？** OpenClaw 用 SQLite 支撑了 240K+ stars 的全球部署，证明了 sqlite-vec + FTS5 的混合搜索对于中等规模（<100K 文档块）是完全够用的。原型阶段零外部依赖，快速验证。生产阶段如果需要更强的元数据过滤和分布式能力再迁移到 Qdrant。

### 3.3 文献综述生成

采用 **per-citation prompting** 策略：每次只生成一个论点+引用，确保每条引用都可追溯到具体文献。LLM 永远不从记忆中生成引用。

---

## 四、Stage 2 — 大纲生成

### 4.1 三步层级展开

```python
async def generate_outline(state: ThesisState) -> ThesisState:
    """三步层级展开: 总体结构 → 子节展开 → 用户审阅"""
    
    # Step 1: 高层结构 (JSON 结构化输出)
    outline = await llm_structured_output(
        system=f"""你是{state['discipline']}领域的论文指导专家。
        为{state['degree_level']}论文生成标准结构大纲。
        论文主题: {state['topic']}
        文献概况: {state['literature_summary']}""",
        output_schema=ThesisOutlineSchema,  # Pydantic model
    )
    
    # Step 2: 逐节展开 (使用相邻节上下文防止内容重叠)
    for i, chapter in enumerate(outline["chapters"]):
        prev_chapter = outline["chapters"][i-1] if i > 0 else None
        next_chapter = outline["chapters"][i+1] if i < len(outline["chapters"])-1 else None
        
        chapter["subsections"] = await expand_chapter(
            chapter=chapter,
            prev_context=prev_chapter,
            next_context=next_chapter,
            relevant_refs=get_refs_for_chapter(chapter, state["references"]),
        )
        
        # 为每章计算 SubAgent 分配
        chapter["subtask_plan"] = plan_subtasks(
            target_words=chapter["target_words"],
            complexity=chapter["complexity"],
            subsections=chapter["subsections"],
        )
    
    state["outline_json"] = outline
    state["stage2_status"] = StageStatus.HITL_REVIEW  # 等待人工确认
    return state
```

### 4.2 SubAgent 动态分配算法

```python
def plan_subtasks(target_words: int, complexity: str, subsections: list) -> list:
    """根据章节字数和复杂度动态分配 SubAgent 数量"""
    
    words_per_task = {"low": 1500, "medium": 1500, "high": 1200}[complexity]
    base_count = max(2, round(target_words / words_per_task))
    actual_wpt = target_words // base_count
    
    # 确保每个 SubAgent 在 1000-2000 字甜区
    while actual_wpt > 2000 and base_count < 10:
        base_count += 1
        actual_wpt = target_words // base_count
    while actual_wpt < 1000 and base_count > 2:
        base_count -= 1
        actual_wpt = target_words // base_count
    
    # 将 subsections 分配到 subtasks
    # 优先按自然分节点切割, 每个 subtask 对应 1-2 个 subsection
    subtasks = distribute_subsections_to_subtasks(subsections, base_count)
    
    return [{
        "subtask_id": f"sub_{i+1:02d}",
        "title": st["title"],
        "target_words": st["target_words"],
        "assigned_refs": st["relevant_refs"],
        "writing_instructions": st["instructions"],
        "predecessor_hint": st.get("predecessor_hint", ""),
        "successor_hint": st.get("successor_hint", ""),
    } for i, st in enumerate(subtasks)]
```

---

## 五、Stage 3 — 两层写作系统 (核心模块)

### 5.1 架构总览

```
Orchestrator
│
│  (借鉴 OpenClaw Lane Queue: 每章一个 lane, 章间可并行)
│
├── Lane: 绪论 ─────────────── Chapter Agent: 绪论
│                                ├── SubAgent 1 (串行) ──→ 1500字
│                                └── SubAgent 2 (串行) ──→ 1500字
│                                → 整合 → 迭代 → 输出
│
├── Lane: 文献综述 (依赖绪论) ── Chapter Agent: 文献综述
│                                ├── SubAgent 1 ──→ 1500字
│                                ├── SubAgent 2 ──→ 2000字
│                                ├── SubAgent 3 ──→ 2000字
│                                ├── SubAgent 4 ──→ 1500字
│                                └── SubAgent 5 ──→ 1000字
│                                → 整合 → 迭代(2轮) → 输出
│
├── Lane: 研究方法 (依赖文献综述)
│   ...
├── Lane: 结果分析 (依赖研究方法)
│   ...
├── Lane: 讨论 (依赖结果分析)
│   ...
└── Lane: 结论 (依赖讨论)
    ...
```

**关键决策：章间串行，章内 SubAgent 也串行。** 毕业论文的章节之间有严格的逻辑依赖（文献综述引导研究方法，研究方法决定结果分析），不适合并行。每章内的 SubAgent 也必须串行执行，因为 SubAgent N 需要看到 N-1 的完整输出来保持连贯性（AgentWrite ICLR 2025 的核心发现）。

### 5.2 Chapter Agent 实现

```python
from langgraph.graph import StateGraph, START, END

class ChapterState(TypedDict):
    # 从 Orchestrator 接收
    chapter_id: str
    chapter_title: str
    target_words: int
    complexity: str
    subtask_plans: List[dict]
    thesis_state_summary: str     # 之前所有章节的压缩摘要
    terminology_glossary: Dict[str, str]
    relevant_refs: List[Reference]
    
    # Chapter Agent 维护
    subtask_results: List[SubTaskResult]
    merged_text: str
    quality_scores: Dict[str, int]  # 6 维度评分
    current_iteration: int
    max_iterations: int             # 默认 3
    revision_instructions: Dict[str, str]  # subtask_id → 修改指令

# ─── 定义 Chapter Agent 的图节点 ───

async def plan_phase(state: ChapterState) -> ChapterState:
    """Phase 1: 细化 SubAgent 任务分配"""
    # 如果是第一轮, 使用大纲阶段预生成的计划
    # 如果是迭代轮, 跳过此阶段
    if state["current_iteration"] == 0:
        state["subtask_results"] = [
            SubTaskResult(
                subtask_id=plan["subtask_id"],
                title=plan["title"],
                target_words=plan["target_words"],
                output_text="",
                actual_words=0,
                citations_used=[],
                key_claims=[],
                transition_out="",
                glossary_additions={},
                confidence=0.0,
                weak_spots=[],
                needs_revision=False,
                revision_notes=None,
                iteration_round=0,
            )
            for plan in state["subtask_plans"]
        ]
    return state

async def execute_subtasks_serial(state: ChapterState) -> ChapterState:
    """Phase 2: 串行执行 SubAgents, 上下文累积"""
    accumulated_full_text = ""      # SubAgent N-1 的完整文本
    accumulated_summaries = ""      # SubAgent N-2 及更早的压缩摘要
    
    for i, result in enumerate(state["subtask_results"]):
        plan = state["subtask_plans"][i]
        
        # 迭代轮次中, 跳过不需要修改的 SubAgent
        if (state["current_iteration"] > 0 
            and result["subtask_id"] not in state["revision_instructions"]):
            # 仍需更新上下文累积
            if i > 0:
                accumulated_summaries += f"\n[第{i}节摘要] " + summarize_text(
                    state["subtask_results"][i-1]["output_text"], max_words=200
                )
            accumulated_full_text = result["output_text"]
            continue
        
        # ─── 构建 SubAgent Prompt (借鉴 OpenClaw 分层上下文组装) ───
        # 
        # Layer 1 (TOP — 高注意力区): 全局上下文
        # Layer 2: 前文累积
        # Layer 3: RAG 检索的相关文献
        # Layer 4 (END — 高注意力区): 具体任务指令
        
        prompt = build_subagent_prompt(
            # Layer 1: 全局
            thesis_title=f"论文主题: {state.get('topic', '')}",
            chapter_outline=format_chapter_outline(state["subtask_plans"]),
            glossary=state["terminology_glossary"],
            position=f"第{i+1}节/共{len(state['subtask_results'])}节",
            
            # Layer 2: 前文
            prev_full_text=accumulated_full_text,
            prev_summaries=accumulated_summaries,
            
            # Layer 3: RAG
            rag_passages=retrieve_for_subtask(
                plan, state["relevant_refs"], top_k=8
            ),
            
            # Layer 4: 任务
            task_title=plan["title"],
            task_instructions=plan["writing_instructions"],
            target_words=plan["target_words"],
            required_citations=plan["assigned_refs"],
            transition_in=plan["predecessor_hint"],
            transition_out=plan["successor_hint"],
            revision_notes=state["revision_instructions"].get(
                result["subtask_id"], None
            ),
        )
        
        # 调用 LLM (强制 JSON 结构化输出)
        raw_result = await call_writing_llm(prompt, response_schema=SubTaskOutput)
        
        # 更新结果
        result.update({
            "output_text": raw_result["text"],
            "actual_words": count_words(raw_result["text"]),
            "citations_used": raw_result["citations_used"],
            "key_claims": raw_result["key_claims"],
            "transition_out": raw_result["transition_out"],
            "glossary_additions": raw_result.get("glossary_additions", {}),
            "confidence": raw_result["self_assessment"]["confidence"],
            "weak_spots": raw_result["self_assessment"]["weak_spots"],
            "needs_revision": raw_result["self_assessment"]["needs_revision"],
            "iteration_round": state["current_iteration"],
        })
        
        # 更新术语表
        state["terminology_glossary"].update(raw_result.get("glossary_additions", {}))
        
        # 更新上下文累积
        if i > 0:
            accumulated_summaries += f"\n[第{i}节摘要] " + summarize_text(
                accumulated_full_text, max_words=200
            )
        accumulated_full_text = raw_result["text"]
        
        # 审计日志
        append_audit_log(state, {
            "type": "llm_call",
            "agent": f"SubAgent-{result['subtask_id']}",
            "model": "claude-sonnet-4-20250514",
            "tokens_used": raw_result.get("usage", {}).get("total_tokens", 0),
        })
    
    return state

async def integrate_and_review(state: ChapterState) -> ChapterState:
    """Phase 3: 合并所有 SubAgent 输出, 评估 6 个质量维度"""
    
    # 合并文本
    merged = "\n\n".join(r["output_text"] for r in state["subtask_results"])
    state["merged_text"] = merged
    
    # LLM 作为审阅者, 评估 6 个维度 (各 1-5 分)
    review_prompt = f"""你是一位严格的学术论文审阅专家。请评估以下章节的质量。

## 待审阅章节: {state['chapter_title']}

{merged}

## 评估维度 (各维度 1-5 分, 5 为最高):

1. **衔接性**: 各小节之间的过渡是否自然流畅
2. **论证连贯**: 论点展开是否有逻辑性, 前后是否一致
3. **引用覆盖**: 核心论点是否都有文献支撑
4. **术语一致**: 专业术语使用是否前后统一
5. **字数平衡**: 各小节篇幅是否合理, 有无过薄或过胖
6. **重复冗余**: 小节之间是否存在重复内容

请以 JSON 格式输出评分和每个需要修改的 SubAgent 的具体修改建议。"""

    review = await call_review_llm(review_prompt, response_schema=ChapterReviewOutput)
    
    state["quality_scores"] = review["scores"]
    state["current_iteration"] += 1
    
    return state

async def generate_revisions(state: ChapterState) -> ChapterState:
    """Phase 4: 生成定向修改指令, 只针对需要修改的 SubAgent"""
    
    # 从审阅结果中提取修改指令
    revision_map = {}
    for note in state.get("review_notes", []):
        if note["score"] < 4:
            revision_map[note["subtask_id"]] = note["revision_instruction"]
    
    state["revision_instructions"] = revision_map
    return state

def should_iterate(state: ChapterState) -> str:
    """条件路由: 继续迭代 or 完成"""
    scores = state["quality_scores"]
    min_score = min(scores.values()) if scores else 0
    
    if min_score >= 4:
        return "done"
    if state["current_iteration"] >= state["max_iterations"]:
        return "done"  # 达到上限, 标记待人工审查
    return "revise"

# ─── 构建 Chapter Agent 图 ───

chapter_graph = StateGraph(ChapterState)
chapter_graph.add_node("plan", plan_phase)
chapter_graph.add_node("write", execute_subtasks_serial)
chapter_graph.add_node("review", integrate_and_review)
chapter_graph.add_node("revise", generate_revisions)

chapter_graph.add_edge(START, "plan")
chapter_graph.add_edge("plan", "write")
chapter_graph.add_edge("write", "review")
chapter_graph.add_conditional_edges("review", should_iterate, {
    "revise": "revise",
    "done": END,
})
chapter_graph.add_edge("revise", "write")  # 回到写作节点, 只重写需要修改的部分

chapter_agent = chapter_graph.compile()
```

### 5.3 SubAgent Prompt 分层组装

借鉴 OpenClaw 的 `AGENTS.md → SOUL.md → TOOLS.md → history → memory` 分层模式：

```python
def build_subagent_prompt(
    # Layer 1: 全局 (放在 prompt 开头, 最高注意力)
    thesis_title: str,
    chapter_outline: str,
    glossary: dict,
    position: str,
    # Layer 2: 前文累积
    prev_full_text: str,
    prev_summaries: str,
    # Layer 3: RAG 文献
    rag_passages: list,
    # Layer 4: 任务 (放在 prompt 末尾, 最高注意力)
    task_title: str,
    task_instructions: str,
    target_words: int,
    required_citations: list,
    transition_in: str,
    transition_out: str,
    revision_notes: str = None,
) -> str:
    """
    利用 "Lost in the Middle" 效应:
    - 全局上下文放 TOP (高注意力)
    - 中间放前文和 RAG (中等注意力)
    - 具体任务放 END (高注意力)
    """
    
    sections = []
    
    # ── TOP: 全局上下文 ──
    sections.append(f"""# 论文写作任务

{thesis_title}
当前位置: {position}

## 本章完整大纲
{chapter_outline}

## 关键术语表
{format_glossary(glossary)}""")
    
    # ── MIDDLE: 前文 + RAG ──
    if prev_summaries:
        sections.append(f"""## 前序各节摘要
{prev_summaries}""")
    
    if prev_full_text:
        sections.append(f"""## 上一节完整内容 (请保持衔接)
{prev_full_text}""")
    
    sections.append(f"""## 相关文献资料 (引用时只能使用以下文献)
{format_rag_passages(rag_passages)}""")
    
    # ── END: 具体任务 ──
    task_block = f"""## 你的写作任务

**标题**: {task_title}
**目标字数**: {target_words} 字
**写作要求**: {task_instructions}

**必须引用的文献**: {', '.join(required_citations)}
**承上 (与上一节的衔接)**: {transition_in}
**启下 (为下一节铺垫)**: {transition_out}

**输出格式**: 请以 JSON 格式输出, 包含 text, citations_used, key_claims, 
transition_out, glossary_additions, self_assessment 字段。

**重要约束**:
- 只使用上面提供的文献, 不要生成任何不在列表中的引用
- 保持与前文一致的术语和论证风格
- 字数严格控制在目标范围 ±10%"""
    
    if revision_notes:
        task_block += f"""

**⚠️ 修改要求 (本轮迭代)**:
{revision_notes}
请在保留上一版本优点的基础上, 针对上述问题进行修改。"""
    
    sections.append(task_block)
    
    return "\n\n---\n\n".join(sections)
```

### 5.4 上下文窗口预算

| 层级 | 输入预算 | 明细 |
|------|---------|------|
| **SubAgent** | ~30K tokens | 大纲+术语表(3K) + 章节大纲(1K) + 上一节全文(3K) + 更早摘要(2K) + RAG文献(8K) + 任务指令(2K) = ~19K 输入, ~3K 输出 |
| **Chapter Agent** (审阅) | ~60K tokens | 完整章节(15-20K) + 论文状态(5K) + 评审标准(2K) + SubAgent 元数据(3K) = ~30K |
| **Orchestrator** | ~15K tokens | 各章摘要(500字×6=3K) + 论文状态 + 大纲 |

均在 Claude 200K / DeepSeek 128K 上下文窗口之内。

### 5.5 完整数据流示例: 28,000 字论文

```
Orchestrator → 串行执行 6 个 Chapter Agent

Chapter Agent: 绪论 (3,000字, complexity=low)
  ├── SubAgent 1: 研究背景 → 1,500字 ✓
  └── SubAgent 2: 研究目的与意义 → 1,500字 ✓
  → 审阅: 全部 ≥4 分 → 完成 (1 轮迭代)
  → 产出: 3,000 字 + 术语表更新 + 章节摘要

Chapter Agent: 文献综述 (8,000字, complexity=high)
  ├── SubAgent 1: 核心概念界定 → 1,500字 ✓
  ├── SubAgent 2: 国内研究现状 → 2,000字 ⚠️ (衔接性=3)
  ├── SubAgent 3: 国外研究现状 → 2,000字 ⚠️ (与Sub2重复)
  ├── SubAgent 4: 研究述评与不足 → 1,500字 ✓
  └── SubAgent 5: 本研究的切入点 → 1,000字 ✓
  → 审阅: Sub2 衔接性=3, Sub3 重复冗余=3
  → 迭代轮 1: 仅重写 Sub2, Sub3 (节省 60% tokens)
  → 审阅: 全部 ≥4 分 → 完成 (2 轮迭代)

Chapter Agent: 研究方法 (4,000字, complexity=medium)
  ├── SubAgent 1: 研究设计与框架 → 1,500字 ✓
  ├── SubAgent 2: 数据收集方法 → 1,500字 ✓
  └── SubAgent 3: 数据分析方法 → 1,000字 ✓
  → 审阅: 全部 ≥4 分 → 完成 (1 轮迭代)

Chapter Agent: 结果分析 (6,000字, complexity=medium)
  ├── SubAgent 1: 描述性统计 → 1,500字 ✓
  ├── SubAgent 2: 主要发现一 → 1,500字 ✓
  ├── SubAgent 3: 主要发现二 → 1,500字 ✓
  └── SubAgent 4: 综合分析 → 1,500字 ✓
  → 审阅: 全部 ≥4 分 → 完成 (1 轮迭代)

Chapter Agent: 讨论 (5,000字, complexity=medium)
  ├── SubAgent 1: 结果解释与理论对话 → 1,500字 ✓
  ├── SubAgent 2: 与已有研究对比 → 2,000字 ⚠️ (引用覆盖=3)
  └── SubAgent 3: 理论与实践意义 → 1,500字 ✓
  → 迭代轮 1: 仅重写 Sub2, 补充引用
  → 完成 (2 轮迭代)

Chapter Agent: 结论 (2,000字, complexity=low)
  ├── SubAgent 1: 主要结论 → 1,000字 ✓
  └── SubAgent 2: 不足与展望 → 1,000字 ✓
  → 完成 (1 轮迭代)

总计: 28,000 字
      19 个 SubAgent 首次调用
      + 3 个 SubAgent 修改调用 (仅 Sub2,3 和讨论 Sub2)
      + 6 次章节审阅
      ≈ 28 次 LLM 写作调用 + 8 次审阅调用 = 36 次总调用
```

---

## 六、Stage 4 — 引用验证与格式化

### 6.1 三层验证管线

```python
async def verify_all_citations(state: ThesisState) -> ThesisState:
    """三层验证: 存在性 → 元数据 → 语义"""
    
    all_cited = extract_all_citations(state["chapter_results"])
    verified = []
    flagged = []
    
    for cite_key in all_cited:
        ref = find_reference(cite_key, state["references"])
        if not ref:
            flagged.append({"key": cite_key, "reason": "不在文献库中"})
            continue
        
        # Layer 1: DOI 存在性验证
        if ref.get("doi"):
            exists = await verify_doi_crossref(ref["doi"])
            if not exists:
                flagged.append({"key": cite_key, "reason": "DOI 无法解析"})
                continue
        
        # Layer 2: 元数据校验 (作者、年份、期刊)
        metadata_ok = await crosscheck_metadata(ref)
        if not metadata_ok:
            flagged.append({"key": cite_key, "reason": "元数据不匹配"})
            continue
        
        # Layer 3: 语义验证 (引用声明是否真的由该文献支持)
        for claim in get_claims_using_citation(cite_key, state["chapter_results"]):
            passage = retrieve_relevant_passage(cite_key, claim, state["references"])
            nli_result = await check_entailment(premise=passage, hypothesis=claim)
            if nli_result == "contradiction" or nli_result == "neutral":
                flagged.append({
                    "key": cite_key,
                    "claim": claim,
                    "reason": f"引用未能支撑该论点 (NLI: {nli_result})",
                })
                break
        else:
            verified.append(cite_key)
    
    state["verified_citations"] = verified
    state["flagged_citations"] = flagged
    return state
```

### 6.2 CSL 格式化

```python
# GB/T 7714-2015 (数字标注) — 通过官方 CSL 文件, 无需自定义实现
# APA 7th, IEEE, Chicago 等 — 同一管线, 切换 CSL 文件即可

from citeproc import CitationStylesStyle, CitationStylesBibliography
from citeproc.source.bibtex import BibTeX

def format_bibliography(references: list, style: str = "gb-t-7714-2015-numeric"):
    """使用 CSL 处理器格式化参考文献列表"""
    csl_file = CSL_STYLES[style]  # 映射到 .csl 文件路径
    bib_source = convert_refs_to_bibtex(references)
    
    style = CitationStylesStyle(csl_file)
    bibliography = CitationStylesBibliography(style, bib_source)
    
    # 注册所有引用
    for ref in references:
        bibliography.register(Citation([CitationItem(ref["ref_id"])]))
    
    return bibliography.bibliography()
```

---

## 七、Stage 5 — 跨章节润色

### 7.1 全局一致性检查

Stage 3 中 Chapter Agent 只负责章内一致性。Stage 5 由 Orchestrator 执行全局检查：

```python
async def cross_chapter_polish(state: ThesisState) -> ThesisState:
    """全局视角的跨章节一致性检查与润色"""
    
    full_thesis = merge_all_chapters(state["chapter_results"])
    
    # 检查 1: 术语一致性 (跨章节)
    # 检查 2: 论点不矛盾 (绪论的假设 vs 结论的发现)
    # 检查 3: 交叉引用正确 ("如第三章所述" 是否指向正确内容)
    # 检查 4: 摘要与正文一致
    
    # 由于全文可能超过 50K tokens, 采用滑动窗口审阅
    # 每次审阅 2 个相邻章节 + 全局术语表 + 论文摘要
    
    issues = []
    chapters = state["chapter_results"]
    for i in range(len(chapters) - 1):
        pair_review = await review_adjacent_chapters(
            chapter_a=chapters[i]["merged_text"],
            chapter_b=chapters[i+1]["merged_text"],
            glossary=state["terminology_glossary"],
            thesis_abstract=state.get("abstract", ""),
        )
        issues.extend(pair_review["issues"])
    
    # 对发现的问题生成修改建议 → 人工审阅后执行
    state["polish_notes"] = issues
    return state
```

### 7.2 中文学术润色约束

```python
CHINESE_POLISH_SYSTEM = """你是中文学术论文润色专家。请遵循以下规则:

1. 仅修改确实存在问题的表达, 包括:
   - 口语化表达 → 替换为书面学术用语
   - 语法错误 → 修正
   - 逻辑跳跃 → 补充过渡句
   - 被动句过多 → 适当调整句式

2. 严禁修改:
   - 原文逻辑通顺、用词准确时, 禁止强行替换同义词
   - 保留作者的论证风格和行文习惯
   - 不改变任何实质性论点

3. 输出格式: 返回修改后的文本, 并附带修改说明列表。"""
```

---

## 八、Stage 6 — 编译输出

```python
def export_to_latex_project(state: ThesisState):
    """生成 BUPT LaTeX 论文工程并打包为 Overleaf 可上传项目"""
    project_dir = copytree("muse/templates/bupt_latex", "output/latex_project")

    render_metadata(project_dir / "config" / "info.tex", state)
    render_abstracts(project_dir / "Chapter", state)
    render_chapters(project_dir / "Chapter", project_dir / "main.tex", state["chapter_results"])
    write_bibliography(project_dir / "Bib" / "thesis.bib", state)
    copy_assets(project_dir / "resources", state)

    make_zip_archive(project_dir, "output/latex_project.zip")
    try_compile_pdf(project_dir, "output/thesis.pdf")
```

---

## 九、技术栈汇总

| 层面 | 技术选型 | 备注 |
|------|---------|------|
| **Agent 框架** | LangGraph | 状态图 + 检查点 + HITL |
| **写作 LLM (中文)** | DeepSeek-V3.2 / Qwen3 | 成本低, 中文质量高 |
| **写作 LLM (英文)** | Claude Sonnet 4 | 长文连贯性最强 |
| **审阅/推理 LLM** | DeepSeek-R1 / Claude extended thinking | 结构化评估 |
| **多模型调度** | LiteLLM | 统一 API, 模型 failover |
| **论文级 Embedding** | SPECTER2 | 学术文献专用 |
| **多语言 Embedding** | BGE-M3 | 中英文, 8192 tokens |
| **向量+全文搜索** | SQLite + sqlite-vec + FTS5 | 原型; 生产迁移 Qdrant |
| **混合搜索策略** | 70% vector + 30% BM25 | 借鉴 OpenClaw |
| **PDF 解析** | GROBID (Docker) | 回退: MinerU (中文) |
| **学术搜索** | Semantic Scholar + OpenAlex + arXiv | 三源并行 |
| **引用格式** | CSL + citeproc-py | GB/T 7714 + APA 等 |
| **引用管理** | Zotero (pyzotero) | BetterBibTeX 稳定 key |
| **Overleaf 项目导出** | vendored LaTeX template + `zipfile` | 生成完整 BUPT 论文工程 |
| **LaTeX/PDF 编译** | `latexmk` + XeLaTeX | 本地可选编译 |
| **审计日志** | JSONL append-only | 借鉴 OpenClaw |
| **后端** | FastAPI | 异步, 高性能 |
| **前端** | Next.js + TipTap 编辑器 | 原型可用 Streamlit |

---

## 十、从 OpenClaw 借鉴的模式总结

| OpenClaw 模式 | 我们的实现 | 用途 |
|--------------|-----------|------|
| **Lane Queue** (per-session FIFO) | 每章一个 LangGraph state channel | 防止同章内写作冲突 |
| **混合搜索** (70/30 vector+BM25) | SQLite sqlite-vec + FTS5 score fusion | 文献检索: 语义+精确匹配 |
| **分层上下文组装** (AGENTS.md→SOUL.md→history) | 全局→前文→RAG→任务 四层 prompt | SubAgent 的上下文构建 |
| **JSONL 审计日志** | append-only 事件流 | 可追溯性 + AI 使用声明 |
| **SubAgent 模型降级** | 审阅用强模型, 格式化用弱模型 | 成本优化 |
| **Context compaction** (自动压缩旧对话) | 前文章节自动摘要 (200字/章) | 长文上下文管理 |
| **NOT 借鉴**: Gateway/消息平台/WebSocket/深度限制 | 直接使用 LangGraph 图 | 避免不必要的复杂度 |

---

## 十一、实施路线图

### Phase 1: 最小可行管道 (Week 1-2)

```
目标: 验证核心写作环节
范围: 单章节 (绪论 3000 字) + 2 个 SubAgent
技术: LangGraph + Claude API + 硬编码大纲

Day 1-3: SubAgent prompt 调优
  - 实现 build_subagent_prompt()
  - 测试串行执行 + 上下文累积
  - 验证结构化 JSON 输出

Day 4-5: Chapter Agent 审阅循环
  - 实现 6 维度评分
  - 实现定向修改指令
  - 验证迭代循环 (plan→write→review→revise)

Day 6-7: 基础 RAG
  - Semantic Scholar API 集成
  - SQLite + sqlite-vec 最小实现
  - per-citation prompting 验证

Week 2: 端到端流程
  - 3 个章节串行生成
  - 基础引用格式化 (CSL)
  - 输出为 Markdown → LaTeX 项目打包
```

### Phase 2: 完整管道 (Week 3-5)

```
目标: 全 6 章毕业论文端到端生成
范围: 全部 Stage 1-6

Week 3: 文献模块 + 大纲模块
  - 多源并行检索 (Semantic Scholar + OpenAlex + arXiv)
  - GROBID PDF 解析 + 分块入库
  - 混合搜索 (70/30)
  - 大纲三步展开 + HITL 审阅

Week 4: 写作模块完善
  - 全 6 章 Chapter Agent 串行执行
  - 动态 SubAgent 分配
  - 跨章节状态传递 (thesis_state_summary)
  - LangGraph 检查点持久化

Week 5: 引用 + 润色 + 输出
  - 三层引用验证管线
  - CSL 格式化 (GB/T 7714)
  - 跨章节润色
  - BUPT LaTeX 项目打包 + 可选 PDF 编译
```

### Phase 3: 优化与上线 (Week 6-8)

```
Week 6: 质量优化
  - Prompt 调优 (基于真实论文对比)
  - 成本优化 (审阅用 R1, 格式化用 Haiku)
  - 错误处理和重试机制

Week 7: 用户界面
  - Streamlit 原型 / Next.js + TipTap
  - HITL 审阅界面
  - 进度追踪仪表盘

Week 8: 测试
  - 3-5 个不同学科的完整论文生成测试
  - 与真实论文对比评估
  - 性能和成本基准测试
```

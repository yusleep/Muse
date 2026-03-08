# Muse 重构调研：框架选型、架构设计与迁移路线

> 作者：Claude Opus 4.6 (综合 Codex 调研 + 独立技术验证)
> 日期：2026-03-08
> 范围：框架选型 + 四层目标架构 + 迁移路线图

---

## 执行摘要

> **以 `LangGraph` 作为主编排内核，以 `LlamaIndex` 作为知识/文档平面，保留并重构现有 Muse 的 provider、citation、latex、store 等领域模块。**

这个结论来自两轮独立调研的交叉验证：

- **Codex 调研**：从框架定位、评估标准体系、架构分层、创意设计点四个维度给出方向判断
- **Claude 独立验证**：对 LangGraph/CrewAI 的关键 API 做技术级核实，对现有 Muse 4200 行代码做逐模块拆解

两轮调研在所有核心结论上完全一致，本文将它们合并为一份可执行的工程文档。

一句话定位（引自 Codex，我完全同意）：

> **Muse 不应该重构成"另一个多角色 Agent 框架示例"，而应该重构成"一个以论文状态机为核心、以证据与成稿为一等对象的学术写作运行时"。**

---

## 一、框架选型：从 Muse 的需求出发，而不是从热度出发

### 论文写作 Agent 的 8 条选型标准

（出自 Codex 调研，经独立验证后全部采纳）

1. **能否原生支持循环**——outline → critique → rewrite, chapter draft → reviewer → revise
2. **能否可靠 checkpoint / resume**——论文 run 不是几秒的 action，需要中断恢复
3. **能否把状态建模成清晰对象**——paper state、chapter state、citation evidence state
4. **能否优雅插入 HITL**——用户确认选题/大纲/章节/导出
5. **能否支持强结构化输出**——标题/摘要/章节/citation evidence 的结构约束
6. **能否把文献与文档层处理好**——PDF/doc ingest、metadata extraction、retrieval、quote grounding
7. **能否适配现有 Muse 的模块资产**——providers、citation、latex_export、store、rag
8. **是否适合长期维护**——可追踪、可审计、新人能读懂

### 框架终审

#### LangGraph — 主编排内核（两轮调研一致推荐）

Codex 从定位与能力模型层面推荐，我从 API 层面独立验证：

| 能力 | API | 验证状态 | 对 Muse 的意义 |
|------|-----|---------|---------------|
| 状态图 | `StateGraph(TypedDict)` | 文档确认，TypedDict 优于 Pydantic（避免缓存问题） | 替代 `engine.py` 的 for 循环 |
| 并行 fan-out | `Send("node", payload)` | 官方 Deep Research 项目实证 | N 章并行写作，运行时动态数量 |
| Reducer | `Annotated[list, operator.add]` | 文档确认，支持自定义 merge 函数 | 并行章节结果安全合并 |
| HITL | `interrupt(value)` + `Command(resume=feedback)` | 文档确认，基于 checkpoint 持久化 | 替代 `hitl_feedback.json` 手动方案 |
| Checkpoint | `SqliteSaver` / `PostgresSaver` | 每个节点执行后自动保存，崩溃可恢复 | 替代 `store.save_state()` |
| 条件路由 | `add_conditional_edges(node, fn, map)` | 文档确认 | Reflexion 循环声明式定义 |
| 时间旅行 | `graph.get_state_history(config)` | 文档确认 | 任意 checkpoint 回滚/重放 |

LangGraph 把 agent 应用当作 graph + state + checkpoint + replay + interrupt 来对待——这几乎就是 Muse 真正需要的抽象。

**LangChain Open Deep Research 项目的关键发现：**

> 当多个 agent 并行写报告各节时，输出会变得"disjoint"。解决方案：并行做 research，但 writing 阶段用压缩后的 brief 做单 pass 或有序生成。

启示：**并行写章节后，必须有 merge + coherence check 节点**。

**依赖量（可控）：**
```
langgraph                       # 核心（~2MB）
langgraph-checkpoint-sqlite     # SQLite 持久化
langchain-core                  # 基础类型（被 langgraph 依赖）
```
不需要完整 `langchain` 生态。节点内直接调用 Muse 现有的 `LLMClient`，不需要改造为 LangChain ChatModel。

#### LlamaIndex — 知识/文档平面（采纳 Codex 定位）

Codex 指出 LlamaIndex 到 2026 已不仅是 RAG 工具箱，而是覆盖 documents + parsing + workflows + query planning + checkpointing + document agents 的完整知识平台。

对 Muse 来说，它最有价值的不是"替代 LangGraph"，而是：

- 帮 Muse 把文献/PDF/docs/metadata/retrieval 这一层做标准化
- 把 citation grounding 做得更扎实
- 提供更成熟的文档知识平面

**与现有 `rag.py` + `refs_loader.py` 的能力对比：**

| 能力 | Muse 现有 | LlamaIndex 能提供 | 差距 |
|------|-----------|-------------------|------|
| PDF/DOCX/TXT 解析 | pdfminer/pypdf/python-docx | LlamaParse — 更强的表格/图片/数学公式解析 | 中（学术论文有复杂排版） |
| 文本分块 | 300 词固定窗口 + 50 词重叠 | 语义分块、RecursiveCharacterTextSplitter | 小 |
| Embedding 检索 | MiniLM-L12 + cosine similarity | 多种 embedding 模型 + 向量数据库 | 小 |
| 元数据提取 | 文件名推断 title/year | 自动提取 title/authors/abstract/sections | **大** |
| Query planning | 无 | 多步骤检索策略、sub-question decomposition | **大** |
| Reranking | 无 | Cross-encoder reranking | 中 |

**核心差距在元数据提取和 query planning**——这两点直接影响论文写作质量。现有 `rag.py` 靠文件名猜测 title/year，对学术文献来说太粗糙。

Codex 建议它在 Muse 里承担：
- `CorpusService`
- `ReferenceIngestionService`
- `RetrieverService`
- `CitationContextAssembler`
- `EvidenceLookupService`

**迁移策略：** 现有 `rag.py` 通过 adapter 接口逐步迁移为 LlamaIndex 后端。Phase 0 先定义 `RetrievalService` Protocol，Phase 2 引入 LlamaIndex 实现。

#### CrewAI — 降级为借鉴（技术缺陷实证）

Codex 判断其"适合原型与局部子流程"，我通过技术验证进一步强化降级理由：

| 缺陷 | 来源 | 影响 |
|------|------|------|
| `async_execution=True` 性能退化 | 社区 issue + CrewAI Community 帖 | 并行章节不可靠 |
| 非真正持久执行 | Diagrid 独立基准 | 崩溃后无法恢复 agent 内部推理状态 |
| 无原生循环支持 | 文档确认 | Reflexion 需手动脚本模拟 |
| Memory 无修剪 | 文档 + 社区反馈 | 6 阶段流水线后期 stale context |

根本原因（Codex 原文）：

> 它最自然的建模方式仍然是"团队协作"，而 Muse 最自然的建模方式是"论文状态机"。

**结论：借鉴角色分工思想（Researcher/Writer/Critic），不作为 Muse 主框架。**

#### AutoGen — 排除

官方 README 明确引导新用户至 Microsoft Agent Framework。2026-02 博客确认 MAF 为 AutoGen + Semantic Kernel 的 successor。**不作为 Muse 依赖。**

#### Microsoft Agent Framework — 持续关注

Release Candidate 已发布，定位为 AutoGen + Semantic Kernel 的 successor。但 Muse 是 Python-first + 学术写作密集型系统，当前更天然贴合 LangGraph + LlamaIndex。**值得观察，不是当前首选。**

#### 轻量自研图 — 排除

| 自研需实现 | LangGraph 免费获得 |
|-----------|-------------------|
| 状态快照 + 序列化 | SqliteSaver 自动持久化 |
| 并行 fan-out + 结果合并 | Send API + Reducer |
| 条件路由引擎 | conditional_edges |
| 中断/恢复协议 | interrupt() / Command(resume=) |
| 执行追踪 | LangSmith 集成 |

自研本质是重写 LangGraph 的子集，且缺少社区验证。**不值得。**

### 框架排序（Codex 结论 + Claude 验证）

1. **LangGraph** — 主编排器
2. **LlamaIndex** — 知识/文档层
3. **CrewAI** — 借鉴角色分工，不作为主内核
4. **Microsoft Agent Framework** — 持续观察
5. **AutoGen** — 不推荐

---

## 二、现有架构诊断

### 模块职责分析

```
providers.py   1174 行  ← 混合 4 类职责：HTTP、LLM路由、学术搜索、引文元数据
stages.py       712 行  ← 混合 6 个阶段 + prompt + 辅助函数
latex_export.py 671 行  ← BUPT 模板 16 section 填充（高度定制，保留）
rag.py          234 行  ← embedding + BM25 混合检索（逐步迁移到 LlamaIndex）
schemas.py      203 行  ← 30 字段单体 TypedDict，无写时校验
runtime.py      186 行  ← 将 stage 函数包装为闭包注入 engine
engine.py        63 行  ← 仅是 for 循环，无条件分支/回环/并行
```

### 核心限制

| 限制 | 根因 | 影响 |
|------|------|------|
| 章节串行写作 | `engine.py` 是简单 for 循环 | 5 章 × 3 subtask × 3 iteration = 可能 45 次 LLM 调用串行执行 |
| Reflexion 硬编码 | `stage3_write` 内嵌 `while current_iteration < max_iterations` | 无法配置不同章节的迭代策略 |
| Prompt 不可维护 | 每个 stage 函数内 inline 定义 system/user prompt | 修改 prompt 需读懂整个 stage 逻辑 |
| 状态不安全 | stage 函数直接 mutate `state` dict | 无法追踪哪个 stage 修改了哪个字段 |
| HITL 笨重 | 返回 `"hitl"` 字符串 → 进程退出 → CLI resume | 无法在进程内交互 |
| 文献元数据粗糙 | `refs_loader.py` 靠文件名猜 title/year | 学术文献的作者/摘要/节信息丢失 |

### 值得保留的资产

| 模块 | 行数 | 保留理由 | 目标位置 |
|------|------|---------|---------|
| `_ModelRouter` + `LLMClient` | ~600 行 | Codex OAuth SSE、多 provider fallback、route 映射 | `services/providers.py` |
| `citation.py` | 139 行 | 三层验证逻辑（DOI → 元数据 → NLI 蕴含） | `services/citation.py` |
| `latex_export.py` | 671 行 | BUPT 模板 16 section 填充，高度定制 | `services/latex.py` |
| `rag.py` | 234 行 | embedding + BM25 混合检索 | `adapters/llamaindex/` (逐步迁移) |
| `planning.py` | 103 行 | subtask 拆分算法 | `services/planning.py` |
| `store.py` | 75 行 | RunStore artifact 管理 | `services/store.py` |
| `audit.py` | 76 行 | append-only JSONL 审计日志 | `services/audit.py` |

---

## 三、目标架构：四层结构

### 3.1 一句话版本

> **LangGraph orchestration kernel + LlamaIndex knowledge plane + Muse domain services + artifact/checkpoint store**

### 3.2 四层职责划分

#### 第一层：Orchestration Kernel（LangGraph）

回答：现在在哪一步？下一步跑什么？哪些节点可以回滚？哪些要人工批准？

- 定义整体论文工作流（顶层 graph）
- 管理节点与边（条件路由、循环）
- checkpoint / resume（SqliteSaver）
- interrupt / HITL（interrupt + Command）
- chapter-level subgraphs（Send API fan-out）
- reflection loops（conditional edges）

#### 第二层：Knowledge Plane（LlamaIndex）

回答：我从哪里获得知识？哪些上下文可供生成？哪条结论由哪些来源支撑？

- 文献 ingest（PDF/DOCX/TXT 解析 + 元数据提取）
- retrieval / reranking（语义检索 + cross-encoder）
- query planning（多步检索策略）
- citation context assembly（证据段落定位）
- note / excerpt / source grounding

#### 第三层：Domain Services（保留并重构 Muse 模块）

回答：我怎样调用模型？我怎样核引文？我怎样导出论文？

- `providers.py` → 模型网关与路由层
- `citation.py` → citation normalization / verification
- `latex_export.py` → 终态导出服务
- `store.py` → artifact store / run manifest
- `planning.py` → chapter subtask 规划
- `audit.py` → append-only 审计日志

#### 第四层：Artifacts & Checkpoints

回答：我生成过什么？哪一步失败了？哪些结果可重放？

```
runs/<run_id>/
  graph/
    checkpoints/          # LangGraph SqliteSaver
    state_snapshots/      # 关键节点的 state dump
  artifacts/
    search/               # 文献检索结果
    outline/              # 大纲 JSON
    chapters/             # 各章草稿/修订版
    reviews/              # 审稿记录
    citations/            # citation ledger
    paper/                # 成稿（composition 输出）
    export/               # 最终 LaTeX/PDF/Markdown
  audit/
    audit.jsonl           # 全流程审计日志
```

### 3.3 目录结构

（采纳 Codex 的分层目录方案）

```
muse/
  graph/                          # ── 第一层：编排 ──
    state.py                      # MuseState TypedDict + reducers
    launcher.py                   # graph compile + checkpointer 配置
    main_graph.py                 # 顶层 graph 定义（nodes + edges）
    subgraphs/
      chapter.py                  # 章节子图（write → review → revise 循环）
      citation.py                 # 引文子图（extract → lookup → verify → repair）
      composition.py              # 成稿子图（术语统一 → 交叉引用 → 摘要拼接）
    nodes/
      initialize.py               # 初始化 run state
      ingest.py                   # 触发 knowledge plane 文献导入
      search.py                   # 学术搜索（多源检索 + 去重）
      idea.py                     # 研究方向生成（topic analysis）
      outline.py                  # 大纲生成 + subtask 规划
      draft.py                    # 章节草稿写作（子图内节点）
      review.py                   # 章节审稿 + 路由逻辑（子图内节点）
      compose.py                  # 全文成稿（composition 子图入口）
      export.py                   # 最终导出（Markdown/LaTeX/PDF）

  services/                       # ── 第三层：领域服务 ──
    providers.py                  # LLMClient + _ModelRouter（从 providers.py 迁移）
    http.py                       # HttpClient + post_json / post_json_sse
    search.py                     # AcademicSearchClient（Semantic Scholar/OpenAlex/arXiv）
    citation.py                   # citation verification（DOI + 元数据 + NLI 蕴含）
    citation_meta.py              # CitationMetadataClient（Crossref）
    latex.py                      # BUPT DOCX 模板导出（从 latex_export.py 迁移）
    store.py                      # RunStore + artifact 管理
    audit.py                      # JsonlAuditSink
    planning.py                   # plan_subtasks

  schemas/                        # ── 状态契约 ──
    run.py                        # 顶层 run 配置 schema
    chapter.py                    # 章节状态 / 子图状态
    citation.py                   # citation ledger schema
    export.py                     # export manifest schema

  adapters/                       # ── 外部框架适配 ──
    llamaindex/
      retriever.py                # LlamaIndex retrieval adapter（替代 rag.py）
      ingestion.py                # LlamaIndex document ingestion（替代 refs_loader.py）
    external_search/
      semantic_scholar.py         # Semantic Scholar API adapter
      openalex.py                 # OpenAlex API adapter
      arxiv.py                    # arXiv API adapter

  prompts/                        # ── Prompt 模板 ──
    search_queries.py             # → (system, user) tuple
    topic_analysis.py
    outline_gen.py
    section_write.py
    chapter_review.py
    polish.py
    abstracts.py

  config.py                       # Settings（保留，增加 checkpoint/llamaindex 配置）
  cli.py                          # CLI（适配 graph API + thread_id）
  runtime.py                      # Runtime wiring（简化，构建 graph 而非 engine）
  __init__.py
  __main__.py
```

**分层原则：**
- `graph/` 负责编排——回答"跑什么、怎么跑"
- `services/` 负责领域能力——回答"怎么调模型、怎么核引文、怎么导出"
- `schemas/` 负责强状态契约——回答"数据长什么样"
- `adapters/` 负责外部框架和数据源——回答"怎么接 LlamaIndex、怎么接搜索 API"
- `prompts/` 负责提示词——回答"对 LLM 说什么"

**与现有代码的映射：**

| 现有文件 | 行数 | 目标位置 | 操作 |
|---------|------|---------|------|
| `engine.py` | 63 | 删除 | 被 `graph/main_graph.py` + `graph/launcher.py` 替代 |
| `stages.py` | 712 | 删除 | 拆分为 `graph/nodes/*.py` + `prompts/*.py` |
| `orchestrator.py` | 51 | 删除 | gate 逻辑移入 `graph/nodes/export.py` |
| `providers.py` | 1174 | 拆分 | → `services/providers.py` + `services/http.py` + `services/search.py` + `services/citation_meta.py` |
| `schemas.py` | 203 | 拆分 | 子类型 → `schemas/*.py`，顶层状态 → `graph/state.py` |
| `rag.py` | 234 | 迁移 | → `adapters/llamaindex/retriever.py`（先保留接口，逐步替换） |
| `refs_loader.py` | 145 | 迁移 | → `adapters/llamaindex/ingestion.py` |
| `citation.py` | 139 | 移动 | → `services/citation.py` |
| `latex_export.py` | 671 | 移动 | → `services/latex.py` |
| `chapter.py` | 81 | 合并 | 逻辑并入 `graph/nodes/review.py` |
| `planning.py` | 103 | 移动 | → `services/planning.py` |
| `store.py` | 75 | 移动 | → `services/store.py` |
| `audit.py` | 76 | 移动 | → `services/audit.py` |
| `config.py` | 102 | 保留 | 增加 checkpoint / llamaindex 配置项 |
| `cli.py` | 176 | 保留 | 适配 graph API（thread_id + interrupt/resume） |
| `runtime.py` | 186 | 简化 | 构建 graph 而非 engine |

### 3.4 图拓扑

#### 顶层 graph

```
START
  │
  ▼
initialize_run ──► ingest_corpus ──► search_and_expand_refs
                                          │
                                          ▼
                                  interrupt: review_refs  (HITL)
                                          │
                                          ▼
                                    generate_idea ──► plan_outline
                                                          │
                                                          ▼
                                                  interrupt: approve_outline  (HITL)
                                                          │
                                                          ▼
                                                    draft_paper  ← Send API fan-out
                                                    ┌───┼───┐
                                                    ▼   ▼   ▼
                                              [chapter_subgraph × N]  (并行)
                                                    │   │   │
                                                    └───┼───┘
                                                        ▼
                                                  merge_chapters  (coherence check)
                                                        │
                                                        ▼
                                                interrupt: review_draft  (HITL)
                                                        │
                                                        ▼
                                                  [citation_subgraph]
                                                        │
                                                        ▼
                                                  [composition_subgraph]
                                                        │
                                                        ▼
                                                interrupt: approve_submission  (HITL)
                                                        │
                                                        ▼
                                                    latex_export
                                                        │
                                                        ▼
                                                       END
```

#### 章节子图 `chapter_subgraph`

每章内部是一个 Reflexion 循环：

```
chapter_plan ──► chapter_draft ──► chapter_review ──┐
                      ▲                             │
                      │         score < threshold   │
                      └── chapter_revise ◄──────────┘
                                                    │
                                         score >= threshold
                                                    │
                                                    ▼
                                              chapter_done
```

#### 引文子图 `citation_subgraph`

```
extract_claims ──► lookup_evidence ──► match_support ──► flag_weak_claims
                                                              │
                                                  has repairable claims?
                                                    │yes            │no
                                                    ▼               ▼
                                              repair_claims     done
```

#### 成稿子图 `composition_subgraph`

```
unify_terminology ──► align_cross_refs ──► stitch_abstract_intro_conclusion
                                                       │
                                                       ▼
                                             normalize_figures_tables ──► prepare_paper_package
```

### 3.5 状态设计

```python
# muse/graph/state.py
from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
import operator

def _merge_dict(current: dict | None, new: dict) -> dict:
    """Reducer: 合并 dict（后者覆盖前者）"""
    result = (current or {}).copy()
    result.update(new)
    return result

class MuseState(TypedDict):
    """LangGraph 全局状态。Reducer 定义并行节点如何安全合并更新。"""

    # ── 输入（初始化后不变）──
    project_id: str
    topic: str
    discipline: str
    language: str
    format_standard: str
    output_format: str                                    # markdown | latex | pdf

    # ── 文献检索 ──
    references: Annotated[list, operator.add]             # reducer: 追加
    search_queries: list[str]
    literature_summary: str

    # ── 大纲 ──
    outline: dict                                         # {chapters: [...]}
    chapter_plans: list[dict]                             # 含 subtask_plan

    # ── 章节写作（并行安全）──
    chapters: Annotated[dict, _merge_dict]                # {ch_id: {merged_text, scores, ...}}
    citation_ledger: Annotated[dict, _merge_dict]         # {claim_id: {cite, score, evidence, status}}
    claim_text_by_id: Annotated[dict, _merge_dict]        # {claim_id: claim_text}

    # ── 引用验证 ──
    verified_citations: list[str]
    flagged_citations: list[dict]

    # ── 成稿与导出 ──
    paper_package: dict                                   # composition 输出（与 export 分离）
    final_text: str
    polish_notes: Annotated[list, operator.add]
    abstract_zh: str
    abstract_en: str
    keywords_zh: list[str]
    keywords_en: list[str]
    output_filepath: str

    # ── 控制 ──
    review_feedback: Annotated[list, operator.add]        # HITL 反馈累积
    rag_enabled: bool
    local_refs_count: int
```

**对比当前 `ThesisState` 的关键变化：**
- 删除 `stage1_status` ~ `stage6_status`、`current_stage` — graph checkpoint 替代
- 删除 `hitl_feedback` — `review_feedback` + interrupt 替代
- 删除 `audit_events` — audit.py 独立管理
- 新增 `citation_ledger: Annotated[dict, _merge_dict]` — 引文升为一等公民
- 新增 `paper_package: dict` — composition 与 export 分离
- `chapter_results: list` → `chapters: Annotated[dict, _merge_dict]` — 支持并行安全合并

### 3.6 关键实现模式

#### 并行章节写作（Send API）

```python
# muse/graph/nodes/draft.py
from langgraph.types import Send

def fan_out_chapters(state: MuseState) -> list[Send]:
    """为每章派发一个 chapter_subgraph 实例。运行时动态数量。"""
    return [
        Send("chapter_subgraph", {
            "chapter_plan": plan,
            "references": state["references"],
            "topic": state["topic"],
            "language": state["language"],
        })
        for plan in state["chapter_plans"]
    ]
```

#### Reflexion 循环（条件边）

```python
# muse/graph/subgraphs/chapter.py
def should_revise(state: ChapterState) -> str:
    scores = state.get("quality_scores", {})
    min_score = min(scores.values()) if scores else 0
    iteration = state.get("iteration", 0)
    if min_score < 4 and iteration < 3:
        return "revise"
    return "done"

chapter_graph.add_conditional_edges(
    "chapter_review", should_revise,
    {"revise": "chapter_draft", "done": END}
)
```

#### HITL 中断（interrupt）

```python
# muse/graph/nodes/outline.py 之后
from langgraph.types import interrupt

def approve_outline(state: MuseState) -> dict:
    feedback = interrupt({
        "stage": "outline",
        "chapter_count": len(state["chapter_plans"]),
        "chapters": [p["chapter_title"] for p in state["chapter_plans"]],
    })
    return {"review_feedback": [feedback]}
```

#### Prompt 模板化

```python
# muse/prompts/section_write.py
import json

def section_write_prompt(
    topic: str, chapter_title: str, subtask: dict,
    refs_snapshot: list[dict], language: str,
    revision_instruction: str | None = None,
    local_context: list[dict] | None = None,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt) 元组。"""
    system = (
        "Write one thesis subsection with citations. "
        "IMPORTANT: for citations_used, use ONLY ref_id values from the available_references list. "
        "Return JSON with keys: text, citations_used, key_claims, transition_out, "
        "glossary_additions, self_assessment."
    )
    user_payload = {
        "topic": topic, "chapter_title": chapter_title,
        "subtask": subtask, "language": language,
        "available_references": refs_snapshot,
        "revision_instruction": revision_instruction,
    }
    if local_context:
        user_payload["local_context"] = local_context
    return system, json.dumps(user_payload, ensure_ascii=False)
```

---

## 四、5 个创意设计点

（出自 Codex 调研，经技术可行性评估后全部采纳）

### 1. Citation Ledger：引文升为一等公民

不要只在最终阶段检查引用。为每一章维护一个 `citation_ledger`：

```python
citation_ledger[claim_id] = {
    "claim": "...",
    "cited_source": "ref_id",
    "support_score": 0.85,          # NLI 蕴含得分
    "evidence_excerpt": "...",       # 来源文本片段
    "confidence": "high",
    "repair_status": "verified",     # verified | flagged | repaired
}
```

**实现方式：** `citation_ledger` 作为 `Annotated[dict, _merge_dict]` 字段加入 MuseState。章节写作时同步维护（write 节点写入 claim + cite，review 节点验证 support score）。

**效果：** Muse 从"先写再补引用"变成"带证据写作"。

### 2. Thesis Twin State：工作态与提交态双轨

维护两种状态：

- **working draft** — 可随时修改，允许不完整
- **submission candidate** — 只有满足以下条件才能推进：
  - outline stable
  - chapter reviews pass (min_score >= 4)
  - citation ledger clean (no unrepaired flags)
  - editorial checks pass

**实现方式：** composition 子图的入口节点检查 gate 条件。通过 → 生成 `paper_package`；未通过 → 返回修改建议，触发对应章节的 revise 循环。

### 3. Chapter-local memory + Paper-global memory

拆成两级 memory：

- **chapter-local**：本章术语、核心论点、已写事实、待修项（存在子图 state 中）
- **paper-global**：全文术语规范、研究问题、贡献点、统一参考文献池（存在顶层 MuseState 中）

**实现方式：** LangGraph 的子图天然隔离 state。chapter_subgraph 有自己的 `ChapterState`，通过 reducer 向顶层 MuseState 汇报结果。术语表 `terminology_glossary` 放在顶层 state，每章写作时只读引用。

### 4. Editorial Board Pattern：多维审稿

不让一个 critic 包打天下。chapter_review 节点内调用 4 个专项 prompt：

- `logic_reviewer` — 论证逻辑、因果链
- `style_reviewer` — 学术文风、表述规范
- `citation_reviewer` — 引用支撑度、citation ledger 一致性
- `structure_reviewer` — 段落结构、过渡衔接

最后由一个 `editorial_synthesizer` 汇总为统一的 scores + review_notes。

**实现方式：** 一个节点内 4 次 LLM 调用，不需要 4 个独立 agent。避免过度设计，但获得多维度审稿质量。

### 5. Paper Composition 与 Export 分离

这是 Codex 最重要的架构贡献之一：

- **composition** — 决定论文内容组织与一致性（术语统一、交叉引用对齐、摘要/绪论/结论拼接）
- **export** — 只是把既定 paper package 渲染成目标格式（BUPT LaTeX 工程 / PDF / Markdown）

**效果：**
- 支持多模板输出（不同学校模板只需换 export adapter）
- 支持同一 paper package 的不同投稿版本
- composition 问题和 export 问题可以独立调试

---

## 五、迁移路线图

### Phase 0：冻结领域边界

**目标：** 让现有领域模块变成 graph nodes 可调用的 service，不急着删旧代码。

1. `providers.py` 拆分为 `services/providers.py` + `services/http.py` + `services/search.py` + `services/citation_meta.py`
2. `citation.py` → `services/citation.py`（纯移动）
3. `latex_export.py` → `services/latex.py`（纯移动）
4. `store.py` → `services/store.py`（纯移动）
5. `rag.py` 抽取 `RetrievalService` Protocol 接口
6. 验证：现有 `runtime.py` + `stages.py` 路径仍可正常运行

### Phase 1：搭 LangGraph 外壳

**目标：** graph 能 compile 并跑通最简单的 3 个节点。

1. `pip install langgraph langgraph-checkpoint-sqlite`
2. 创建 `graph/state.py`：MuseState TypedDict + reducers
3. 创建 `graph/launcher.py`：SqliteSaver 配置
4. 创建 `graph/main_graph.py`：`START → initialize → search → outline → END`
5. 创建 `graph/nodes/initialize.py` + `search.py` + `outline.py`
6. 从 `stages.py` 提取 prompt 到 `prompts/*.py`
7. 验证：graph 跑通 initialize → search → outline 三节点

### Phase 2：章节子图 + 知识平面

**目标：** 核心写作循环 graph-native 化，引入 LlamaIndex 知识平面。

1. 创建 `graph/subgraphs/chapter.py`（draft → review → revise 循环）
2. 创建 `graph/nodes/draft.py` + `review.py`（Editorial Board Pattern）
3. 添加 Send API fan-out（并行章节）+ Reducer 合并
4. 添加 HITL interrupt 点（approve_outline、review_draft）
5. 引入 `adapters/llamaindex/retriever.py` + `ingestion.py`（替代 rag.py + refs_loader.py）
6. 创建 `schemas/chapter.py` + `schemas/citation.py`
7. 验证：`python -m muse run --topic "..." --auto-approve` 全流程跑通

### Phase 3：Citation Ledger + Composition

**目标：** 可信度层 + 成稿层完成，Muse 与普通写作 agent 拉开差距。

1. 创建 `graph/subgraphs/citation.py`（extract → lookup → verify → repair）
2. Citation Ledger 集成到章节写作流程（write 时写入，review 时验证）
3. 创建 `graph/subgraphs/composition.py`（术语统一 → 交叉引用 → 摘要拼接）
4. Composition 与 Export 分离（`compose.py` → `paper_package`，`export.py` 只渲染）
5. 验证：citation ledger 有效，paper_package 完整

### Phase 4：CLI + 测试 + 下线旧路径

**目标：** 所有用户操作正常，测试通过，旧代码安全下线。

1. `cli.py` 适配：run 创建 thread_id → resume 基于 checkpoint → review 用 Command(resume=)
2. `runtime.py` 简化：构建 graph 而非 engine
3. 删除 `engine.py`、`orchestrator.py`、`stages.py`、`chapter.py`（逻辑已迁移）
4. 迁移测试用例：`test_stages.py` → `test_nodes.py`，`test_runtime_flow.py` → `test_graph.py`
5. 新增 `test_parallel_chapters.py`、`test_reflexion.py`、`test_citation_ledger.py`
6. 验证：`python -m pytest tests/ -v` 全绿，CLI 全命令正常

### Phase 5：收尾

**条件：** 以下全部满足后才算完成。

- 新 graph path 完整跑通（auto-approve + HITL 两种模式）
- run artifacts 按新目录结构可读
- citation ledger 有效
- export path 稳定（Markdown + LaTeX + PDF）
- 关键测试迁移完成
- 旧 `runtime/stages` 主路径已下线

---

## 六、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| LangGraph API breaking change | 低 | 中 | 锁定版本 `langgraph>=0.2,<0.3` |
| Send API 并行章节输出 disjoint | 中 | 高 | merge 节点做 coherence check + composition 子图做术语统一 |
| interrupt 恢复后节点从头执行 | 确定 | 低 | 确保 interrupt 前的代码幂等 |
| LlamaIndex 依赖量大 | 确定 | 中 | Phase 0 用 adapter 接口隔离，Phase 2 才引入实现 |
| Codex OAuth SSE 与 LangGraph 不兼容 | 无 | 无 | 节点内直接调用 Muse LLMClient，不走 LangChain |
| 现有 runs/ 状态文件不兼容 | 确定 | 低 | 全面重写不保留旧格式 |

---

## 七、总结

```
┌──────────────────────────────────────────────────────────────┐
│                     Muse v2 架构公式                          │
│                                                              │
│  Layer 1: LangGraph StateGraph                               │
│           ├── Send API (并行章节)                             │
│           ├── conditional_edges (Reflexion)                   │
│           ├── interrupt / Command (HITL)                      │
│           └── SqliteSaver (checkpoint)                        │
│                                                              │
│  Layer 2: LlamaIndex Knowledge Plane                         │
│           ├── Document ingestion (PDF/DOCX 元数据提取)        │
│           ├── Semantic retrieval + reranking                  │
│           └── Query planning                                 │
│                                                              │
│  Layer 3: Muse Domain Services                               │
│           ├── LLMClient / _ModelRouter (保留)                 │
│           ├── Citation verification (保留)                    │
│           ├── BUPT LaTeX export (保留)                        │
│           └── RunStore + Audit (保留)                         │
│                                                              │
│  Layer 4: Artifacts & Checkpoints                            │
│           └── runs/<run_id>/{graph,artifacts,audit}/          │
│                                                              │
│  + Citation Ledger (引文一等公民)                              │
│  + Editorial Board Pattern (多维审稿)                         │
│  + Paper Composition ↔ Export 分离                            │
│  + Thesis Twin State (工作态/提交态)                          │
│  + Chapter-local / Paper-global memory                       │
│                                                              │
│  = 论文状态机 × 证据驱动 × 可扩展                              │
└──────────────────────────────────────────────────────────────┘
```

### 不建议的路线（两轮调研一致）

1. **完全改成 CrewAI 项目** — Muse 需要论文状态机，不是 team simulation
2. **直接 fork HKUDS AI-Researcher** — 它不是 LangGraph 模板，更偏研究原型
3. **继续在 runtime/stages.py 上打补丁** — 复杂度已到拐点，每增加一个 review loop / chapter graph / citation branch 都会指数上升
4. **把 LlamaIndex 当主编排器** — 它是最佳知识平面，但全局编排不如 LangGraph

### 参考来源

**官方来源：**
- LangGraph GitHub: `https://github.com/langchain-ai/langgraph`
- LangGraph Durable Execution: `https://docs.langchain.com/oss/python/langgraph/durable-execution`
- LangGraph Interrupts: `https://docs.langchain.com/oss/python/langgraph/interrupts`
- LangGraph Persistence: `https://docs.langchain.com/oss/python/langgraph/persistence`
- LangGraph Deep Research: `https://blog.langchain.com/open-deep-research/`
- LangGraph Reflection Agents: `https://blog.langchain.com/reflection-agents/`
- CrewAI GitHub: `https://github.com/crewAIInc/crewAI`
- CrewAI Flows: `https://docs.crewai.com/en/concepts/flows`
- AutoGen GitHub: `https://github.com/microsoft/autogen`
- Microsoft Agent Framework RC: `https://devblogs.microsoft.com/semantic-kernel/migrate-your-semantic-kernel-and-autogen-projects-to-microsoft-agent-framework-release-candidate/`
- LlamaIndex GitHub: `https://github.com/run-llama/llama_index`
- LlamaIndex Agent Workflows: `https://developers.llamaindex.ai/python/llamaagents/workflows/`
- HKUDS AI-Researcher: `https://github.com/HKUDS/AI-Researcher`

**独立参考：**
- Diagrid — Checkpoints Are Not Durable Execution: `https://www.diagrid.io/blog/checkpoints-are-not-durable-execution-why-langgraph-crewai-google-adk-and-others-fall-short-for-production-agent-workflows`
- Medium 基准: `https://medium.com/@lukasz.marcin.grochal/i-benchmarked-5-ai-agent-frameworks-heres-what-actually-matters-fd5782578cc0`
- LangGraph Send API: `https://dev.to/sreeni5018/leveraging-langgraphs-send-api-for-dynamic-and-parallel-workflow-execution-4pgd`
- LangGraph Multi-Agent Systems: `https://reference.langchain.com/python/langgraph/supervisor/`

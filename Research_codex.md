# Muse 重构调研：2026 论文写作 Agent 框架选型与整体迁移路线

> 面向对象：`Muse`  
> 范围：框架选型 + 目标架构 + 迁移路线图  
> 日期：`2026-03-08`

## 执行摘要

如果 `Muse` 接受 **整体迁移**，我当前最推荐的方向不是“直接变成 CrewAI 项目”，也不是“照着 HKUDS AI-Researcher 重写”，而是：

> **以 `LangGraph` 作为主编排内核，以 `LlamaIndex` 作为知识 / 文档平面，保留并重构现有 `Muse` 的 provider、citation、latex、store 等领域模块。**

更具体地说：

- **主框架首选：`LangGraph`**
  - 原因：论文写作 Agent 的核心是**长流程状态机**，而不是单纯的多角色聊天
  - 它在官方定位上就强调：
    - durable execution
    - human-in-the-loop
    - stateful memory
    - production-ready deployment
- **辅助框架首选：`LlamaIndex`**
  - 不作为主编排器，而作为：
    - PDF / 文档 ingestion
    - retrieval / query planning
    - knowledge-grounded generation
    - citation context assembly
- **不建议作为最终主内核：`CrewAI`**
  - 它在 2026 年已经比过去成熟很多，`Flows + State + Persistence + Memory` 都有了
  - 但对 `Muse` 这种论文写作系统来说，它仍更适合：
    - 快速原型
    - 角色分工验证
    - 局部 team-style subflow
  - 不如 `LangGraph` 适合作为整个系统的底层操作系统
- **不建议作为长期主依赖：`AutoGen`**
  - 不是因为它差，而是因为到 2026-03 官方自己已经明确提示：**新用户优先看 Microsoft Agent Framework**
  - 这使它更像“仍可用、但不是战略首选”的框架

还需要强调一个容易被热度叙事掩盖的事实：

- 我查到的一篇 2026 独立基准对比并没有得出“某个框架在最终质量上碾压所有对手”的结论
- 它更强调：真正拉开差距的是
  - 状态控制
  - 执行一致性
  - 开发复杂度
  - token / 时延成本

所以这份文档会刻意把“谁最火”降级成参考信息，把“谁最适合论文写作状态机”提升为主决策标准。

一句话判断：

> **`Muse` 应从“手写 6-stage pipeline”升级成“以 LangGraph 为主状态机、以 LlamaIndex 为知识平面、以 Muse 领域模块为执行与产物层”的论文写作运行时。**

## 一、先纠偏：对你给我的那段判断，哪些成立，哪些要修正

你给出的判断整体方向并不差，但如果要拿它指导 `Muse` 重构，有几处必须先纠偏。

### 1. “LangGraph 是 2026 年最稳的生产级候选”——基本成立

这一点我认为**成立**。

我核到的官方信号包括：

- LangGraph 官方 GitHub 当前把自己定位为 **long-running, stateful agents** 的低层 orchestration framework
- README 明确列出核心收益：
  - durable execution
  - human-in-the-loop
  - comprehensive memory
  - production-ready deployment
- 官方文档还专门强调：
  - checkpoint / persistence
  - thread-based resume
  - deterministic replay
  - side-effect/idempotency 设计

这组能力与论文写作 Agent 的需求高度吻合，因为论文系统天然需要：

- 草稿中断后恢复
- 按章节循环修稿
- 审稿人/导师人工插点
- 可追踪的状态与产物

### 2. “CrewAI 是 2026 年快速原型与角色团队强者”——成立，但需要降级定位

这点也**大体成立**，但我不会把它放到 `Muse` 的主架构核心。

为什么？

- 它官方仓库和文档现在已经明确不是只有 `Crews`
- 还强调：
  - `Flows`
  - state management
  - `@persist` state persistence
  - unified memory
  - HITL examples
- 这说明 CrewAI 到 2026 已经明显从“角色型玩具框架”进化成“可以做复杂工作流”的系统

但对于 `Muse` 来说，它的问题是：

- 它的**最佳心智模型仍然是 roles + tasks + crews**
- `Muse` 的最佳心智模型更像：
  - paper state
  - chapter subgraph
  - citation ledger
  - review loop
  - export terminal nodes

也就是说，CrewAI 很适合模拟“论文小组协作”，但 `Muse` 更需要“论文状态机”。

### 3. “AutoGen 仍是前三主力”——要谨慎

如果只看 GitHub 星数和历史影响力，AutoGen 当然还是头部框架。  
但如果看 **2026-03 的战略位置**，这个判断需要明显降级。

我核到的关键信号是：

- AutoGen 官方 README 明确写着：
  - **if you are new to AutoGen, please checkout Microsoft Agent Framework**
  - AutoGen 仍会维护，但主要是 bug fixes 和关键安全补丁
- Microsoft 官方博客在 **2026-02-19** 进一步明确：
  - **Microsoft Agent Framework** 已到 **Release Candidate**
  - 它被定位为 **Semantic Kernel 和 AutoGen 的 successor**

所以对 `Muse` 来说，更准确的判断应该是：

- **AutoGen 本身不再适合作为长期主押注**
- 如果真要走微软栈，应直接评估 `Microsoft Agent Framework`

### 4. “HKUDS AI-Researcher 底层就是 LangChain/LangGraph 栈，可直接 fork 改”——不成立

这一条是我最明确要纠正的。

我本地核了 `HKUDS/AI-Researcher` 的源码与依赖，结论是：

- 它的核心依赖是：
  - `litellm`
  - `instructor`
  - 各类浏览器 / 文档 / 工具包
- 它并没有体现出明显的 `LangGraph` / `LangChain` 核心编排依赖
- 实际上它更像是：
  - `MetaChain`
  - `FlowModule`
  - `AgentModule`
  - `ToolModule`
  - 自研 environment / cache / tool orchestration

所以：

- 它**不是**一个“LangGraph 应用模板”
- 它更适合被当作：
  - **架构灵感来源**
  - **研究代理分层范例**
- 不适合被当作 `Muse` 直接 fork 的宿主架构

### 5. “LlamaIndex 不是独立框架，只能当配角”——不完全准确

这句话也要修正。

`LlamaIndex` 到 2026 年其实已经不仅是 RAG 工具箱，它官方明确在强调：

- Workflows
- Human in the loop
- Reliable structured generation
- Query planning
- Checkpointing workflows
- End-to-end document agents

所以更准确的说法是：

- `LlamaIndex` **可以**做 workflow / agent
- 但对 `Muse` 来说，它**最合适的位置不是主编排器**
- 它最适合作为 **知识与文档平面**

## 二、论文写作 Agent 该按什么标准选框架

如果目标是：

- 文献检索
- idea 生成
- 大纲
- 逐节草稿
- 审稿 / 反思
- 引文校验
- 最终 LaTeX 工程输出

那框架不该按“谁最火”选，而该按下面 8 个标准选：

### 1. 能否原生支持循环而不是只擅长 DAG

论文写作不是线性流水线。  
至少会有这些循环：

- outline -> critique -> rewrite
- chapter draft -> reviewer -> revise
- citation check -> fix unsupported claims -> rerun
- final paper compose -> editorial review -> polish

### 2. 能否可靠 checkpoint / resume

论文 run 不是几秒钟的 agent action。  
它需要：

- 持久化状态
- 中断恢复
- 人工批准后继续
- 出错后局部重跑

### 3. 能否把状态建模成清晰对象

`Muse` 最终需要的不只是消息历史，而是：

- paper state
- outline state
- chapter state
- citation evidence state
- review state
- export state

### 4. 能否优雅插入 HITL

论文系统天然需要 HITL：

- 用户确认选题/大纲
- 用户审阅章节草稿
- 用户处理 citation flags
- 用户确认最终导出

### 5. 能否支持强结构化输出

论文系统不能接受“基本像 JSON”。  
它需要：

- 标题/摘要/章节的结构约束
- citation evidence 的结构约束
- reviewer comments 的结构约束
- export manifest 的结构约束

### 6. 能否把文献与文档层处理好

论文系统不是普通任务 agent。  
它有大量：

- PDF / doc / notes ingest
- metadata extraction
- retrieval
- quote grounding
- bibliography normalization

### 7. 能否适配现有 `Muse` 的模块资产

现在 `Muse` 已经有：

- `muse/providers.py`
- `muse/rag.py`
- `muse/citation.py`
- `muse/latex_export.py`
- `muse/store.py`

最好的框架不是“功能最多”，而是“能把这些模块包进去而不是全部推倒”。

### 8. 是否适合长期维护

真正决定成败的不是 demo，而是：

- 未来半年是否还想维护它
- 新人是否能读懂
- 出问题能否 trace
- 产物是否可审计

## 三、框架比较：从 Muse 的角度，而不是从热度的角度

## 3.1 LangGraph：最适合作为 `Muse` 的主编排内核

### 我为什么推荐它

LangGraph 的核心优势不是“它是 LangChain 生态”，而是它把 agent 应用当作：

- graph
- state
- checkpoint
- replay
- interrupt

来对待。

这几乎就是 `Muse` 真正需要的抽象。

对 `Muse` 特别重要的点：

- `StateGraph` 很适合 paper/chapter/citation 这种显式状态
- durable execution 很适合长流程论文生成
- human-in-the-loop 很适合导师审批/修稿节点
- memory + persistence 很适合多轮论文迭代
- 官方现在也强调 production-ready deployment 与 observability

### 它最适合 Muse 的哪些部分

- 整个顶层 orchestration graph
- chapter-level subgraphs
- critique/reflection loops
- approval interrupts
- citation audit branch
- paper composition branch
- export terminal nodes

### 它的代价

- 心智模型确实比 CrewAI 更偏工程
- 需要认真设计 state schema
- 需要认真处理 side effects 与 idempotency

但这恰恰是 `Muse` 应该做的事情，而不是缺点。

### 结论

如果 `Muse` 要整体迁移，**LangGraph 是我最推荐的主框架**。

## 3.2 CrewAI：优秀，但更适合作为原型框架或局部模式来源

### 它现在比很多人印象中成熟得多

2026 的 CrewAI 已经不只是：

- roles
- tasks
- delegation

它也有：

- Flows
- state access/update
- `@persist`
- memory
- plots / visualization
- human-in-the-loop examples

所以不能再把它当成“只能玩角色扮演”的框架。

### 但它为什么不适合当 Muse 的终局底座

根本原因不是功能少，而是：

> 它最自然的建模方式仍然是“团队协作”，而 `Muse` 最自然的建模方式是“论文状态机”。

论文系统的关键对象不是：

- 研究员
- 作者
- 编辑
- 审稿人

而是：

- paper graph
- chapter draft state
- citation support ledger
- review gate
- export manifest

CrewAI 可以表达这些，但不是它最优雅的表达方式。

### 我建议怎么用 CrewAI

不是完全不用，而是：

- 借鉴它的角色划分思路
- 借鉴它的“book flow / chained crews”示例
- 必要时把它用于某些局部 team-style 子流程

但不要把整个 `Muse` 建在它上面。

### 结论

**CrewAI 适合原型与局部子流程，不适合作为 Muse 的最终主内核。**

## 3.3 AutoGen：仍强，但不适合做 Muse 的长期框架押注

### 为什么我不推荐它做 Muse 核心

AutoGen 的对话式 agent 协作仍很强，尤其适合：

- brainstorming
- multi-agent discussion
- code-execution style agent loops

但 `Muse` 最关键的痛点不是“agent 之间能不能聊”，而是：

- 论文 run 能否稳定保存
- chapter revision 能否结构化追踪
- human review 能否优雅插入
- artifacts 能否可审计

更重要的是，官方产品路线已经给出明确信号：

- 新用户优先看 `Microsoft Agent Framework`
- AutoGen 继续维护，但不是新功能主战场

### 对 Muse 的建议

- 不把 AutoGen 作为长期核心依赖
- 如果你非常想吸收它的价值，可以只借鉴：
  - message-passing
  - debate / discussion loop patterns

### 结论

**不推荐把 Muse 整体迁到 AutoGen。**

## 3.4 Microsoft Agent Framework：值得关注，但不建议现在作为 Muse 主框架

### 它为什么必须进入视野

到 2026-03，Microsoft Agent Framework 已经不能忽略：

- 官方明确把它定位为 AutoGen + Semantic Kernel 的 successor
- Release Candidate 已发布
- 对 .NET / Python 都是稳定 API 面

如果你做的是：

- 企业工作流
- 多语言系统
- 微软生态集成
- 企业治理/部署

它很值得严肃评估。

### 但为什么我仍然不选它作为 Muse 的当前目标

`Muse` 当前更像一个：

- Python-first
- 学术写作/检索/文档处理密集
- 需要深度整合 citation、RAG、LaTeX、chapter loop

的系统。

这类系统现在更天然贴合：

- LangGraph 的状态图抽象
- LlamaIndex 的文档与 retrieval 能力

MAF 对 `Muse` 并非不可行，但：

- 当前迁移摩擦会更大
- 可直接复用的论文/研究工作流范式也较少

### 结论

**值得持续关注，但不是 Muse 当前最佳主选项。**

## 3.5 LlamaIndex Workflows：不做主内核，但必须进入最终方案

### 它的真实价值

LlamaIndex 到 2026 已经不仅是“RAG 工具箱”，而是：

- documents + parsing
- workflows
- human in the loop
- reflection loops
- query planning
- checkpointing
- document agents

从 `Muse` 的角度看，它最有价值的不是“替代 LangGraph”，而是：

- 帮你把文献 / PDF / docs / metadata / retrieval 这一层做标准化
- 把 citation grounding 做得更扎实
- 提供更成熟的文档知识平面

### 为什么不直接让它做主编排器

因为 `Muse` 的主问题仍是全局 paper workflow orchestration。  
LlamaIndex 的 workflows 很强，但它在这方面不如 LangGraph 那么“以状态机为核心”。

### 我建议它在 Muse 里的位置

- `CorpusService`
- `ReferenceIngestionService`
- `RetrieverService`
- `CitationContextAssembler`
- `EvidenceLookupService`

### 结论

**LlamaIndex 是 Muse 未来架构的必选配角，但不建议单独扛主编排。**

## 四、我的最终推荐：Muse 的目标架构

## 4.1 一句话版本

> **LangGraph orchestration kernel + LlamaIndex knowledge plane + Muse domain modules + artifact/checkpoint store**

## 4.2 四层结构

### 第一层：Orchestration Kernel（LangGraph）

职责：

- 定义整体论文工作流
- 管理节点与边
- checkpoint / resume
- interrupt / HITL
- chapter-level subgraphs
- reflection loops

这一层回答的是：

- 现在在哪一步？
- 下一步跑什么？
- 哪些节点可以回滚/重试？
- 哪些节点要人工批准？

### 第二层：Knowledge Plane（LlamaIndex）

职责：

- 文献 ingest
- PDF / 文档解析
- retrieval / reranking
- query planning
- note / excerpt / source grounding

这一层回答的是：

- 我从哪里获得知识？
- 哪些上下文可供生成？
- 哪条结论由哪些来源支撑？

### 第三层：Domain Services（保留并重构现有 Muse 模块）

可保留并重构的模块包括：

- `muse/providers.py`
  - 作为模型网关与路由层
- `muse/citation.py`
  - 作为 citation normalization / verification service
- `muse/latex_export.py`
  - 作为终态导出服务
- `muse/store.py`
  - 作为 artifact store / run manifest / index 层
- `muse/rag.py`
  - 逐步迁成面向 LlamaIndex 的 adapter

这一层回答的是：

- 我怎样调用模型？
- 我怎样核引文？
- 我怎样导出论文？

### 第四层：Artifacts & Checkpoints

建议未来的 run 目录明确拆成：

```text
runs/<run_id>/
  graph/
    checkpoints/
    state_snapshots/
  artifacts/
    search/
    outline/
    chapters/
    reviews/
    citations/
    paper/
    export/
  audit/
    audit.jsonl
```

这一层回答的是：

- 我生成过什么？
- 哪一步失败了？
- 哪些结果可重放？
- 哪些章节已经通过审稿？

## 4.3 建议的目录形态

如果最后走我推荐的方案，`Muse` 的目录不应该继续只围绕 `runtime.py + stages.py`，而更适合逐步演化成：

```text
muse/
  graph/
    state.py
    launcher.py
    main_graph.py
    subgraphs/
      chapter.py
      citation.py
      composition.py
    nodes/
      initialize.py
      ingest.py
      search.py
      idea.py
      outline.py
      draft.py
      review.py
      citation.py
      compose.py
      export.py
  services/
    providers.py
    retrieval.py
    citation.py
    latex.py
    store.py
  schemas/
    run.py
    chapter.py
    citation.py
    export.py
  adapters/
    llamaindex/
    external_search/
  legacy/
    runtime.py
    stages.py
```

这不是要求一步到位重命名，而是一个目标方向：

- `graph/` 负责编排
- `services/` 负责领域能力
- `schemas/` 负责强状态契约
- `adapters/` 负责外部框架和数据源
- `legacy/` 只是过渡期容器，最终应逐步缩小

## 五、我建议的 Muse 新图结构

我不会再把 `Muse` 设计成“一个 `stages.py` 管全部阶段”的样子。  
更优雅的结构应该是：

### 顶层 graph

- `initialize_run`
- `ingest_corpus`
- `search_and_expand_refs`
- `generate_idea`
- `plan_outline`
- `human_approve_outline`
- `draft_paper`
- `review_and_reflect`
- `citation_audit`
- `paper_compose`
- `human_approve_submission`
- `latex_export`

### 章节子图 `chapter_subgraph`

每章内部是一个小循环：

- `chapter_plan`
- `chapter_draft`
- `chapter_self_review`
- `chapter_critic`
- `chapter_revise`
- `chapter_done?`

### 引文子图 `citation_subgraph`

- `extract_claims`
- `lookup_evidence`
- `match_support`
- `flag_weak_claims`
- `repair_or_escalate`

### 成稿子图 `paper_compose_subgraph`

- `unify_terminology`
- `align_cross_refs`
- `stitch_abstract_intro_conclusion`
- `normalize_figures_tables`
- `prepare_latex_package`

## 六、创意但可落地的 5 个设计点

这里我给的不是炫技点，而是我认为对 `Muse` 真有价值的增强设计。

### 1. Citation Ledger：把引文从“附属信息”升级为一等公民

不要只在最终阶段检查引用。  
建议为每一章维护一个 `citation ledger`：

- claim
- cited source
- support score
- evidence excerpt
- confidence
- repair status

这样 `Muse` 的论文生成就不再是“先写，再补引用”，而是“带证据写作”。

### 2. Thesis Twin State：工作态与提交态双轨

建议把整篇论文维护为两种状态：

- **working draft**
- **submission candidate**

只有当：

- outline stable
- chapter reviews pass
- citation ledger clean
- editorial checks pass

时，才能把草稿态推进到提交态。

这会让 `Muse` 的行为非常像真正的论文生产系统，而不是一次性文本生成器。

### 3. Chapter-local memory + Paper-global memory

不要只有一个全局 memory。

应该拆成：

- **chapter-local memory**
  - 本章术语、核心论点、已写事实、待修项
- **paper-global memory**
  - 全文术语规范、研究问题、贡献点、统一参考文献池

这样可以显著减少跨章节串味。

### 4. Editorial Board Pattern

不要让一个 critic 节点包打天下。  
更优雅的方式是一个轻量“编辑委员会”：

- `logic_reviewer`
- `style_reviewer`
- `citation_reviewer`
- `structure_reviewer`

最后再由一个 `editorial_synthesizer` 汇总。

这比“一个 reviewer 既管逻辑又管文风又管引用”更稳定。

### 5. Paper Composition Layer 与 Export Layer 分离

这个点非常重要。

未来一定要把：

- **paper composition**
- **latex export**

拆开。

也就是：

- `paper composition` 决定论文内容组织与一致性
- `latex export` 只是把既定 paper package 渲染成 BUPT LaTeX 工程、zip、pdf

这样 `Muse` 以后才能支持：

- 多模板输出
- 同一 paper package 的不同投稿版本
- 不同学校模板切换

## 七、迁移路线图：怎么从现在的 Muse 走到目标架构

这里我给的是 **整体迁移** 路线，但不会让你一次性爆炸重写。

### Phase 0：冻结领域边界，不急着删旧代码

先做的不是写新图，而是冻结现有领域接口：

- `providers`
- `citation`
- `latex_export`
- `store`
- `refs_loader`

目标是让它们变成未来 graph nodes 可调用的 service，而不是继续和 `runtime/stages` 强耦合。

### Phase 1：先搭 LangGraph 外壳，不迁所有业务

第一阶段只做：

- `graph state`
- `graph launcher`
- `run checkpointing`
- 一两个最简单节点

建议先迁：

- `initialize_run`
- `search_and_expand_refs`
- `plan_outline`

此时旧 `runtime.py` 仍可保留。

### Phase 2：迁移大纲与章节草稿循环

这是最关键的一步。

建议引入：

- `chapter_subgraph`
- `critic/revise loop`
- `outline approval interrupt`

迁移后，`Muse` 的核心已经不再是旧 6-stage 逻辑，而是 graph-native 章节工作流。

### Phase 3：迁移 citation ledger 与 review gates

这一阶段把“可信度层”补上：

- claim extraction
- evidence lookup
- support verification
- flagged claim repair

这是 `Muse` 与普通写作 agent 真正拉开差距的地方。

### Phase 4：迁移 paper composition 与 export

这一步才把：

- abstract/introduction/conclusion stitching
- related work synthesis
- terminology normalization
- latex project export

统一接到 graph 终态。

### Phase 5：下线旧 `runtime/stages` 主路径

只有在以下条件满足后才下线旧路径：

- 新 graph path 可以完整跑通
- run artifacts 可读
- citation ledger 有效
- export path 稳定
- 关键测试迁移完成

## 八、我不建议走的路线

### 路线 1：完全改成 CrewAI 项目

不建议。

原因不是 CrewAI 不好，而是 `Muse` 的主问题不是 team simulation，而是 long-running paper state orchestration。

### 路线 2：直接 fork HKUDS AI-Researcher

不建议。

原因：

- 它不是 LangGraph 模板
- 它更偏研究原型系统
- 它的编排与状态处理对 `Muse` 不够工程化

### 路线 3：继续在现有 `runtime/stages.py` 上无限打补丁

最不建议。

因为你已经到了一个拐点：

- 继续补丁，短期快
- 但以后每增加一个：
  - review loop
  - chapter graph
  - citation branch
  - human interrupt
  
复杂度都会指数上升

### 路线 4：把 LlamaIndex 当主编排器用到底

我不推荐。

它很强，但更适合作为知识平面，而不是全局 orchestration kernel。

## 九、最终判断

如果今天就要为 `Muse` 下注一个中长期架构，我的答案是：

### 推荐结论

- **主编排内核：`LangGraph`**
- **知识 / 文档层：`LlamaIndex`**
- **保留并重构的 Muse 领域模块：**
  - `providers`
  - `citation`
  - `latex_export`
  - `store`
  - `refs_loader`

### 一句话定位

> **Muse 不应该重构成“另一个多角色 Agent 框架示例”，而应该重构成“一个以论文状态机为核心、以证据与成稿为一等对象的学术写作运行时”。**

这也是为什么，我最终把框架排序写成：

1. **LangGraph** —— 最适合当 `Muse` 的主编排器
2. **LlamaIndex** —— 最适合当 `Muse` 的知识 / 文档层
3. **CrewAI** —— 很适合借鉴角色分工和快速原型，但不适合作为最终主内核
4. **Microsoft Agent Framework** —— 值得持续观察，但不适合作为 `Muse` 当前主选
5. **AutoGen** —— 仍有价值，但不适合作为当前长期押注

## 参考来源

### 官方来源

- LangGraph GitHub  
  `https://github.com/langchain-ai/langgraph`
- LangGraph 文档：Durable execution  
  `https://docs.langchain.com/oss/python/langgraph/durable-execution`
- CrewAI GitHub  
  `https://github.com/crewAIInc/crewAI`
- CrewAI Flows 文档  
  `https://docs.crewai.com/en/concepts/flows`
- AutoGen GitHub  
  `https://github.com/microsoft/autogen`
- Microsoft Agent Framework RC 博客  
  `https://devblogs.microsoft.com/semantic-kernel/migrate-your-semantic-kernel-and-autogen-projects-to-microsoft-agent-framework-release-candidate/`
- LlamaIndex GitHub  
  `https://github.com/run-llama/llama_index`
- LlamaIndex Agent Workflows 文档  
  `https://developers.llamaindex.ai/python/llamaagents/workflows/`
- HKUDS AI-Researcher GitHub  
  `https://github.com/HKUDS/AI-Researcher`

### 独立参考

- Medium 基准：`I Benchmarked 5 AI Agent Frameworks — Here’s What Actually Matters`  
  `https://medium.com/@lukasz.marcin.grochal/i-benchmarked-5-ai-agent-frameworks-heres-what-actually-matters-fd5782578cc0`

## 备注

这份文档刻意做了两件事：

1. **把“热度叙事”降级为参考信息**
2. **把“架构契合度”提升为主决策标准**

如果后续你要，我下一步可以基于这个研究结果，继续给出一版：

- **面向代码目录的 `Muse` 新架构草图**
- 或者 **从当前仓库到 LangGraph 版本的具体文件级迁移表**

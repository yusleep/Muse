# Muse 重构调研 V2：更稳、更收敛的框架选择与迁移判断

> 日期：`2026-03-08`  
> 输入来源：`Research_codex.md` + `Research_claude.md` + 已核验的官方资料  
> 目标：给 `Muse` 一份更适合作为**架构决策依据**的研究结论，而不是实现说明书

## 执行摘要

如果 `Muse` 接受整体迁移，我当前最稳的结论仍然是：

> **以 `LangGraph` 作为主编排内核，以 `LlamaIndex` 作为知识 / 文档平面，保留并重构现有 `Muse` 的 provider、citation、latex、store 等领域模块。**

这份 V2 相比前两版，做了两件事：

1. **保留 Codex 版的收敛判断**
2. **只吸收 Claude 版中真正会增强决策信心的工程证据**

因此它的定位不是：

- “把所有框架都讲一遍”
- “把未来实现细节提前写成施工图”

而是：

- 明确 `Muse` 为什么该主选 `LangGraph`
- 明确 `LlamaIndex` 为什么该是知识平面，而不是主编排器
- 明确哪些现有 `Muse` 模块该保留
- 明确迁移应该怎么分阶段推进

一句话版本：

> **`Muse` 不该重构成“另一个多角色 agent demo”，而应该重构成“一个以论文状态机为核心、以证据与成稿为一等对象的学术写作运行时”。**

## 为什么 V2 比前两版更可靠

### 相比 `Research_codex.md`

V2 补上了两类缺失信息：

- 更直接的工程证据
  - 为什么现有 `engine.py` / `runtime.py` / `stages.py` 已经不适合继续扩展
  - 为什么 `providers.py` / `citation.py` / `latex_export.py` / `store.py` 值得保留
- 更明确的目标目录形态
  - 让“LangGraph + LlamaIndex + Muse 服务层”不再停留在口号级

### 相比 `Research_claude.md`

V2 去掉了三类会拉散主线的内容：

- 过细的 API 列举
- 容易使文档滑向实现说明的长表格
- 热度叙事或“技术上虽正确但当前阶段不影响决策”的细节

因此 V2 更适合当前用途：**先做架构决策，而不是立刻施工。**

## 一、先纠偏：哪些流行说法需要修正

### 1. “LangGraph 是 2026 年最稳的生产级候选”——成立

这一点在前两版中结论一致，我维持不变。

它最重要的不是“最火”，而是它的能力模型与论文写作系统高度对齐：

- graph
- state
- checkpoint
- interrupt
- replay
- conditional routing

对 `Muse` 这意味着：

- 章节循环不是补丁逻辑，而是原生图结构
- HITL 不是 CLI 特殊分支，而是图上的 interrupt
- resume 不再依赖手工状态文件，而是原生 checkpoint

### 2. “CrewAI 是最适合论文系统的主框架”——不成立，但仍值得借鉴

这条在 V2 里进一步收敛。

我接受以下判断：

- CrewAI 在 2026 年已经明显成熟
- 它在原型速度、角色建模、局部协作流程上仍很强

但我不接受“因此它更适合 Muse 主架构”的跳跃结论。  
原因仍然很简单：

> `Muse` 的根问题是**论文状态机**，不是**角色团队模拟**。

因此对 `Muse` 来说，CrewAI 的最优定位应是：

- 原型阶段参照物
- 角色语义设计来源
- 局部 team-style 子流程候选

而不是底层 orchestration kernel。

### 3. “AutoGen 仍然是安全的长期押注”——不成立

这一点 V2 比前两版写得更明确：

- `AutoGen` 仍重要
- 但它在 2026-03 的官方定位已经明显退到“维护中”
- 真要走微软路线，应直接评估 `Microsoft Agent Framework`

所以：

- `AutoGen` 不是当前 `Muse` 的长期主押注
- 最多只值得借鉴某些对话/辩论 loop 模式

### 4. “HKUDS AI-Researcher 可直接作为 Muse 的宿主框架”——不成立

这一点在两版中结论一致，V2 继续维持。

`AI-Researcher` 更像：

- `LiteLLM + 自研 MetaChain/FlowModule/AgentModule`
- 自研 environment / tool / cache orchestration

它值得借鉴的是：

- 分层方式
- 研究链与成稿链拆分
- agent/tool/flow 之间的边界意识

但它不适合作为 `Muse` 的直接 fork 基底。

### 5. “LlamaIndex 只能做 RAG 配角”——不准确

更准确的说法是：

- `LlamaIndex` 到 2026 已经是完整的知识 / 文档工作流平台
- 但对 `Muse` 而言，它最适合的定位仍然不是主编排器
- 它最适合承担：
  - 文献 ingest
  - query planning
  - retrieval / reranking
  - citation grounding
  - document-aware knowledge services

## 二、Muse 需要的不是“多 agent 框架”，而是“论文状态机”

这是整份 V2 的核心判断。

如果目标是：

- 文献检索
- idea 生成
- 大纲
- 逐节草稿
- 审稿反思
- 引文核验
- 最终 LaTeX 工程输出

那么真正决定框架优劣的不是角色系统或宣传热度，而是以下问题：

### 1. 能否原生支持循环

论文系统天然包含：

- outline -> critique -> rewrite
- chapter draft -> reviewer -> revise
- citation audit -> repair -> rerun
- compose -> editorial review -> polish

### 2. 能否 checkpoint / resume

论文 run 长、状态多、人工插点多，必须支持：

- 中断恢复
- 局部重跑
- 人工批准后继续

### 3. 能否把状态建模为清晰对象

`Muse` 需要的不是一堆消息，而是：

- paper state
- chapter state
- citation evidence state
- review state
- export state

### 4. 能否优雅插入 HITL

例如：

- 审大纲
- 审章节
- 处理 citation flags
- 确认导出

### 5. 能否支持强结构化输出

例如：

- outline schema
- section draft schema
- citation ledger schema
- export manifest schema

### 6. 能否处理文档与知识层

论文系统一定涉及：

- PDF / DOCX / notes ingest
- metadata extraction
- retrieval / reranking
- quote grounding
- bibliography normalization

### 7. 能否兼容 `Muse` 现有资产

这里是 V2 吸收 Claude 版后最重要的一条增强。

当前 `Muse` 里**明确值得保留**的资产包括：

- `muse/providers.py`
  - 多 provider 路由、fallback、Codex OAuth 适配，这些不是重写框架后就该丢掉的东西
- `muse/citation.py`
  - 三层 citation verification 思路本身就是 `Muse` 的关键差异化
- `muse/latex_export.py`
  - BUPT 模板适配与完整 LaTeX 工程输出属于高价值领域资产
- `muse/store.py`
  - run artifact 与持久化思路可直接升级，而不是废弃
- `muse/planning.py`
  - 章节/subtask 拆分逻辑仍然有保留价值

这意味着：  
最佳框架不是“替代一切”，而是“把这些模块包进一个更合适的 orchestration 模型中”。

## 三、最终框架判断

## 3.1 `LangGraph`：主编排内核

V2 最终仍把 `LangGraph` 放在第一位，而且理由比前两版更凝练：

> 它不是“功能最多”，而是“最贴合论文状态机问题定义”。

### 为什么它最契合 Muse

- `StateGraph` 适合显式 paper/chapter/citation state
- conditional edges 适合 reflection loop
- interrupt 适合 HITL
- checkpointer 适合 durable execution
- replay / state history 适合调试、审计和回滚
- fan-out / merge 模式适合多章并行研究与有控制的收敛

### 从 Claude 版吸收的一个关键工程点

`LangGraph` 的价值不只在“能做图”，而在于它把 `Muse` 当前最难处理的几件事变成了**框架原语**：

- 并行章节：`Send`
- 结果安全合并：reducer
- 人工中断：`interrupt(...)`
- checkpoint 持久化：`SqliteSaver` / `PostgresSaver`

这说明如果 `Muse` 继续靠手写 `engine.py` 演化，实际上是在重复发明这些框架原语。

### 结论

**`LangGraph` 应作为 `Muse` 的主编排器。**

## 3.2 `LlamaIndex`：知识 / 文档平面

这部分 V2 也收敛得更清楚：

> `LlamaIndex` 不需要和 `LangGraph` 争主编排器位置，它最有价值的地方在知识平面。

### 为什么它重要

因为 `Muse` 的论文质量最终很大程度取决于：

- reference ingestion
- metadata quality
- retrieval quality
- evidence grounding
- query planning

而这正是 `LlamaIndex` 到 2026 年最成熟的部分。

### 从 Claude 版吸收的一个关键判断

现有 `rag.py` 的问题不在“不能检索”，而在：

- metadata 粗糙
- query planning 缺失
- reranking 缺失
- 文档结构理解偏弱

因此未来更合理的方向是：

- 不立即删除 `rag.py`
- 而是把它逐步包成 `RetrievalService` adapter
- 让底层后端可以渐进切到 `LlamaIndex`

### 结论

**`LlamaIndex` 应作为 `Muse` 的知识 / 文档平面，而不是主编排器。**

## 3.3 `CrewAI`：借鉴，不主押

V2 对它的定位是：

- 强原型
- 强角色语义
- 强团队式表达

但仍然不适合作为 `Muse` 的全局底层。

### 最值得借鉴的地方

- `Researcher / Writer / Critic / Editor` 这种角色切分
- 对外表达上更贴近用户理解

### 不该让它扛的地方

- 全局论文状态机
- 复杂 revision loop
- citation ledger 驱动的结构化纠错链
- 大规模 artifact/checkpoint 体系

### 结论

**可以借鉴 CrewAI 的角色建模，不建议把 `Muse` 建在 CrewAI 上。**

## 3.4 `AutoGen` / `Microsoft Agent Framework`

V2 把这部分并到一起看：

- `AutoGen`：不再建议作为主押注
- `Microsoft Agent Framework`：值得持续观察，但现在不是 `Muse` 当前最佳主选

原因不是它们不强，而是：

- `Muse` 当前是 Python-first
- 学术写作与文档处理密集
- 现有资产与 LangGraph/LlamaIndex 的整合摩擦更低

### 结论

**当前阶段不作为 `Muse` 主架构优先路线。**

## 四、对现有 Muse 的关键诊断

这一部分是 V2 从 Claude 版吸收后保留下来的最重要增强。

### 当前限制不在“模型不够强”，而在“编排层过薄”

现有结构大体是：

- `runtime.py`
- `stages.py`
- `engine.py`

它的问题不是不能工作，而是扩展性太差。

### 核心限制

#### 1. `engine.py` 太薄

它本质上更接近一个顺序执行器，而不是 orchestration runtime。

这会直接导致：

- 章节并行难
- 条件路由难
- 反思循环难扩展
- checkpoint / replay 不自然

#### 2. `stages.py` 承担了过多责任

它既包含：

- 阶段逻辑
- prompt 组织
- 状态变更
- 流程控制

这会让后续任何复杂演化都变成“往一个大文件里继续堆逻辑”。

#### 3. `runtime.py` 与阶段实现耦合过深

这会使：

- 流程定义难以声明式化
- 同一能力难以在不同 flow 中复用
- 局部替换节点成本高

### 值得保留的领域资产

这部分是 V2 的关键结论之一：

- `muse/providers.py`
  - 这是**服务层资产**
- `muse/citation.py`
  - 这是**可信度层资产**
- `muse/latex_export.py`
  - 这是**终态输出资产**
- `muse/store.py`
  - 这是**artifact 层资产**
- `muse/planning.py`
  - 这是**章节规划资产**

所以 `Muse` 的重构方向不应该是：

> “把现有代码推平，全部换成新框架”

而应该是：

> “把现有高价值领域模块，从旧编排壳中解耦出来，挂到新的 graph runtime 下”。

## 五、推荐目标架构

V2 维持 Codex 的四层结构，但把 Claude 版的工程细节压缩成更清晰的决策图。

## 5.1 四层结构

### 第一层：Orchestration Kernel

建议：`LangGraph`

职责：

- 定义顶层论文工作流
- 管理循环、条件边、checkpoint、resume、interrupt

### 第二层：Knowledge Plane

建议：`LlamaIndex`

职责：

- ingestion
- retrieval
- reranking
- query planning
- citation grounding

### 第三层：Domain Services

建议：保留并重构 `Muse` 现有模块

职责：

- providers
- citation
- latex export
- planning
- store

### 第四层：Artifacts & Checkpoints

建议：run 目录升级为 graph-native artifact store

职责：

- checkpoint
- state snapshot
- chapter artifacts
- review artifacts
- citation ledger
- export manifest

## 5.2 目标目录形态

V2 保留 Claude 版里最有价值的一部分，但不把它当成必须一步到位的重命名任务。

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

这个结构的价值在于它把几个概念彻底分开：

- graph nodes
- domain services
- schemas
- external adapters
- legacy compatibility

## 六、迁移路线图

V2 保留 Codex 版的阶段性迁移思路，但吸收 Claude 版的资产保留逻辑，使路线更可信。

### Phase 0：冻结服务边界

先不要大改逻辑，先把这些模块定义成稳定服务接口：

- `providers`
- `citation`
- `latex_export`
- `store`
- `planning`
- `rag` / retrieval

### Phase 1：先搭 LangGraph 壳，不迁所有业务

先迁最容易 graph 化的部分：

- initialize
- search / expand refs
- outline planning

这一步的目标不是替代旧 runtime，而是证明 graph runtime 可以接住 `Muse` 的状态与 artifact。

### Phase 2：迁章节写作循环

这是最关键的一步。

引入：

- chapter subgraph
- critique / revise loop
- outline approval interrupt

### Phase 3：迁 citation ledger

把 `Muse` 最关键的可信度层挂到 graph runtime 中：

- claim extraction
- evidence lookup
- support verification
- repair / escalate

### Phase 4：迁 paper composition 与 export

这一阶段把：

- abstract / intro / conclusion stitching
- terminology normalization
- paper package assembly
- BUPT LaTeX export

统一接到新图中。

### Phase 5：下线旧 runtime 主路径

只有在这些条件全部满足后才做：

- graph path 可完整跑通
- 关键 artifact 可追踪
- citation ledger 稳定
- export path 稳定
- 关键测试迁移完成

## 七、创意但可落地的设计点

V2 保留 Codex 版最强的 4 个创意点，并删掉不影响决策的装饰性内容。

### 1. Citation Ledger

把引文变成一等对象：

- claim
- source
- evidence
- support score
- repair status

这是 `Muse` 与普通写作 agent 最容易拉开差距的地方。

### 2. Thesis Twin State

维护两种论文状态：

- working draft
- submission candidate

只有当：

- review pass
- citation clean
- composition pass

时，才能推进到提交态。

### 3. Chapter-local / Paper-global 双层记忆

- chapter-local：本章问题、术语、待修项
- paper-global：全局贡献点、术语规范、统一参考文献池

这能减少跨章节串味。

### 4. Paper Composition Layer 与 Export Layer 分离

这点必须明确保留：

- `paper composition`
  - 负责组织与一致性
- `latex export`
  - 负责渲染输出

未来如果你要支持多模板、多投稿版本，这个分层是必须的。

## 八、当前阶段可以先锁定的最小决策

这是 V2 新增的一节，用来避免“研究一做完就默认要全面开工”。

即使你现在还不马上重构，也已经可以先锁定这 4 件事：

### 1. 主编排方向锁定为 `LangGraph`

不是因为立刻要迁，而是为了避免继续在旧 `engine.py` 思维上投资过多。

### 2. 知识平面方向锁定为 `LlamaIndex`

不是因为立刻要引入全部组件，而是为了避免继续把 retrieval 能力和 graph 编排混成一个问题。

### 3. 现有高价值模块确定保留

至少以下模块不应被轻率推倒：

- `providers.py`
- `citation.py`
- `latex_export.py`
- `store.py`
- `planning.py`

### 4. `AI-Researcher` 只作为灵感源，不作为宿主框架

这能避免后面在错误前提上做 fork 或大迁移判断。

## 九、最终建议

如果必须把这份 V2 压缩成一句最终决策，那就是：

> **Muse 的正确方向，不是“换一个更流行的多 agent 框架”，而是“升级成一个以 LangGraph 为编排内核、以 LlamaIndex 为知识平面、以现有 Muse 高价值模块为领域服务层的论文状态机系统”。**

这也是为什么我最终把排序收敛为：

1. **LangGraph** —— 主编排器
2. **LlamaIndex** —— 知识 / 文档平面
3. **CrewAI** —— 借鉴角色分工，不主押
4. **Microsoft Agent Framework** —— 持续观察
5. **AutoGen** —— 不作为当前长期依赖

## 参考来源

- `Research_codex.md`
- `Research_claude.md`
- LangGraph GitHub / Docs
- CrewAI GitHub / Docs
- AutoGen GitHub
- Microsoft Agent Framework RC 博客
- LlamaIndex GitHub / Docs
- HKUDS AI-Researcher GitHub

# AI-Researcher 对 Muse 的架构 / Agent 编排启示调研

> 调研对象：`HKUDS/AI-Researcher`  
> 官方仓库：`https://github.com/HKUDS/AI-Researcher`  
> 官方论文入口：`https://arxiv.org/abs/2505.18705`  
> 调研时间：`2026-03-08`

## 结论先行

如果只看 **架构 / agent 编排**，`AI-Researcher` 对 `Muse` 最大的启示不是“多 agent 更强”，而是它把一个复杂研究系统拆成了几层比较清楚的结构：

1. **顶层 flow**：按任务模式切换不同编排链，而不是所有请求都走同一条流水线。
2. **agent 抽象层**：把 agent 视为“有指令、模型、工具、上下文”的可调用单元。
3. **tool / environment 层**：浏览器、Docker、代码工作区、文件浏览等环境被单独组织。
4. **cache / resume 层**：agent 与 tool 的中间产物可以缓存和恢复。
5. **paper generation 层**：研究执行和论文成稿是分开的，不是一个超长链条全部包办。

对当前 `Muse` 来说，我认为**最值得借鉴**的是下面四点：

- 在 `muse/runtime.py` 与 `muse/stages.py` 之上，加一层 **flow recipe / orchestration recipe**
- 在 `muse/store.py` 中增加 **agent 级与 tool 级 artifact/cache**
- 将“研究推进链”和“成稿输出链”进一步解耦
- 利用现有 `muse/providers.py` 的模型路由能力，扩展成 **角色级 agent profile**，而不只是阶段级模型选择

但也有几处**不建议照搬**：

- `AI-Researcher` 顶层入口仍然偏脚本化，依赖 `global_state.py` 这样的进程级标志
- 它大量依赖长 prompt 串行推进，状态契约并不总是强类型
- `flowcache.py` 的缓存恢复是交互式菜单，适合实验型工作流，不一定适合 `Muse` 当前 CLI / HITL 模式
- 一些结果通过从自然语言输出里“抠 JSON”来提取，工程鲁棒性一般

下面我按“源码里确实看到的实现”与“对 Muse 的推断建议”分别展开。

## 一、AI-Researcher 的真实编排结构

### 1. 顶层是“多模式入口”，不是单一流水线

从 `main_ai_researcher.py` 可以直接看到，系统把入口拆成三种模式：

- `Detailed Idea Description`
- `Reference-Based Ideation`
- `Paper Generation Agent`

对应路由也很明确：

- 前两种进入 `research_agent`，分别走 `run_infer_plan.py` 和 `run_infer_idea.py`
- 最后一种进入 `paper_agent/writing.py`

这说明 `AI-Researcher` 的整体编排思想不是“一条固定 research pipeline”，而是**先按任务形态分流，再进入不同 flow**。  
对 `Muse` 的意义是：当前 `Muse` 已经有稳定的 6-stage pipeline，但未来如果要支持不同工作模式，比如：

- “从题目直接起草”
- “从参考文献反推创新点”
- “仅对已有 run 做成稿 / 导出 / 审校”
- “只做 citation audit / review”

那么单纯在 `stage1..stage6` 上堆条件分支，会越来越别扭。更合适的方向是：**Stage 保持基础能力，Flow 决定用哪些 Stage、以什么顺序、带哪些 agent profile 运行。**

### 2. 它有一层很明确的 agent 运行时抽象

`research_agent/inno/core.py` 和 `research_agent/inno/types.py` 是这个项目最值得看的地方之一。

我看到它至少做了三件事：

- 用 `Agent` 抽象 agent 的名字、模型、instructions、functions、tool_choice、examples 等属性
- 用 `Response` / `Result` 统一 agent 返回值
- 用 `MetaChain` 统一模型调用、tool call 处理、上下文变量传递、重试逻辑

也就是说，在 `AI-Researcher` 里，agent 不是松散的 prompt 模板，而是一个**带运行协议的对象**。  
这和 `Muse` 当前架构的差别是：

- `Muse` 现在更偏 **stage-driven runtime**
- `AI-Researcher` 更偏 **agent-driven orchestration**

两者并不冲突。对 `Muse` 来说，最自然的吸收方式不是“推倒重来改成 agent-first”，而是：

- 保留 `muse/stages.py` 作为业务阶段边界
- 在阶段内部引入更明确的 `AgentSpec` / `AgentProfile` / `AgentRun` 抽象
- 让每个阶段不是只知道“调用一个 provider”，而是知道“我这一步由哪个角色 agent 执行，它有哪些工具、使用哪个模型路由、产出什么结构化结果”

如果做成这样，`Muse` 的 `providers.py` 就不再只是“模型入口”，而能成为 **agent role -> provider route** 的一部分。

### 3. 它把 Flow、Agent、Tool 进一步拆层

`research_agent/inno/workflow/flowcache.py` 里有三个很关键的抽象：

- `FlowModule`
- `AgentModule`
- `ToolModule`

这套设计虽然不复杂，但非常有启发性：

- `FlowModule` 负责组织完整流程
- `AgentModule` 负责执行某个 agent，并带缓存 / 恢复能力
- `ToolModule` 负责执行某个工具，并能缓存结果

在 `run_infer_plan.py` / `run_infer_idea.py` 中，这种模式非常明显：  
`InnoFlow` 会依次组织：

- `load_instance`
- `github_search`
- `prepare_agent`
- `download_paper`
- `survey` / `idea`
- `code_survey`
- `plan_agent`
- `ml_agent`
- `judge_agent`
- `exp_analyser`

它本质上是一个**显式编排图**，虽然目前写法偏串行脚本，但模块边界很清楚。

对 `Muse` 的启示非常直接：

- 现在 `muse/runtime.py` + `muse/stages.py` 其实已经有 `FlowModule` 的雏形
- 但 `Muse` 还缺少更细粒度的 `AgentModule` / `ToolModule` 产物管理
- 如果后面要让 `Muse` 更容易做：
  - 中途恢复
  - 失败重试
  - 单步重跑
  - 某个 agent 输出人工修订后继续
  
那么比起继续把所有状态都塞进一个大 `ChapterState` / run state，**更好的路径是把 agent run artifact 单独落盘**。

### 4. 它非常强调“环境”是独立层

`run_infer_plan.py` 和 `run_infer_idea.py` 里都会实例化这些对象：

- `DockerEnv`
- `BrowserEnv`
- `RequestsMarkdownBrowser`

这说明在它的设计里，“模型”和“执行环境”不是一个东西。  
模型负责推理，环境负责：

- 浏览网页
- 读文件
- 搜代码
- 在工作目录里执行实验
- 跑容器化任务

这个分层对 `Muse` 很有价值，因为 `Muse` 未来如果变成更强的研究代理，很容易把下面几件事混在一起：

- LLM provider 路由
- 检索 API 调用
- 本地资料读取
- 外部论文 / 网页抓取
- 导出 / 编译 / 构建

`AI-Researcher` 给出的启示是：**这些应该被视为 environment / tool adapters，而不是 provider 层的附属品。**

也就是说，对 `Muse` 来说更合理的长期结构可能是：

- `providers.py`：只管模型访问与路由
- `tools/` 或 `environments/`：只管浏览、检索、文件系统、TeX 编译、引用校验资源访问
- `stages.py`：编排业务阶段
- `runtime.py`：编排 run 生命周期

### 5. 它把“研究执行链”和“论文成稿链”分开了

`paper_agent/writing.py` 这条链单独存在，并且它内部流程也很清晰：

- methodology
- related work
- experiments
- introduction
- conclusion
- abstract
- tex 清理
- LaTeX 编译

虽然这部分实现比较脚本化，但它体现出的架构决策很重要：  
**研究环节与论文环节不是一回事。**

这和 `Muse` 当前状态相比，有两个直接启发：

1. `Muse` 现在已经有 Stage 6 导出，但“成稿生成”和“格式导出”还比较近  
   以后可以考虑把“论文章节编排 / 成稿整合 / 语言统一 / 图表组织”单列成一个 paper composition 子系统
2. `latex_export.py` 解决的是“最后导出什么产物”，不一定等于“最终论文是如何组织出来的”

换句话说，`AI-Researcher` 提醒我们：**paper composition 不应只被视为 export。**

## 二、对 Muse 最有价值的 5 个启示

### 启示 1：在 Stage 之上增加 Flow Recipe 层

这是我认为最值得做的一点。

当前 `Muse` 的主结构是：

- `muse/runtime.py`：run 生命周期
- `muse/stages.py`：阶段逻辑
- `muse/store.py`：持久化
- `muse/schemas.py`：状态结构

这套结构已经足够清晰，但它仍然更像**单主线流水线**。  
如果借鉴 `AI-Researcher`，可以在上面再加一层：

- `StandardThesisFlow`
- `ReferenceDrivenFlow`
- `ReviewOnlyFlow`
- `PolishAndExportFlow`

每个 flow recipe 决定：

- 使用哪些 stage
- 每个 stage 采用哪些 agent profile
- 哪些 stage 必须 HITL
- 哪些 artifact 要被单独保存

这样一来，`Muse` 会从“一个固定 pipeline”进化成“一个有多个研究工作流入口的 runtime”。

### 启示 2：把 agent run artifact 从总状态里拆出来

`AI-Researcher` 的 `FlowModule + AgentModule + ToolModule` 设计虽然朴素，但有一个非常实用的结果：  
**每一步的输入、输出、上下文是相对可追踪的。**

`Muse` 现在的 run 目录已经不错，但如果继续增强，我建议增加类似目录：

```text
runs/<run_id>/agents/
  stage2_outline/
    input.json
    prompt.txt
    output.json
    trace.jsonl
  stage3_writer_ch1/
    ...
runs/<run_id>/tools/
  semantic_scholar_search/
  openalex_lookup/
  latex_compile/
```

这会直接改进：

- debug 能力
- resume 粒度
- 局部重跑
- 审计与复盘
- future benchmark / evaluation

### 启示 3：把“角色”作为模型路由的上层语义

`AI-Researcher` 明确存在：

- prepare agent
- survey agent
- idea agent
- coding plan agent
- ml agent
- judge agent
- experiment analyser

这说明它的编排不是按“模型”思考，而是按“角色职责”思考。  
而 `Muse` 其实已经具备做这件事的底层条件：`muse/providers.py` 已经有多模型路由。

下一步更值得做的是把路由键从纯技术性名字升级成职责性名字，比如：

- `research_search`
- `outline_planner`
- `chapter_writer`
- `chapter_reviewer`
- `citation_auditor`
- `cross_chapter_polisher`
- `paper_composer`

这会让：

- 配置更清晰
- 评估更清晰
- A/B 测试更容易
- agent 级 fallback 更自然

### 启示 4：显式引入 environment adapters

如果 `Muse` 未来要扩展，例如：

- 更深入网页检索
- 本地参考资料抽取
- 读取 PDF / DOCX / 笔记库
- 编译 TeX / 生成图表 / 调脚本

那么建议提前引入 `environment adapters` 概念，而不是让这些逻辑散落在 `stages.py` 和工具函数里。

一个可能的方向是：

- `muse/environments/web.py`
- `muse/environments/files.py`
- `muse/environments/latex.py`
- `muse/environments/search.py`

然后每个 stage / agent 通过显式依赖注入来使用它们。  
这样做最大的好处不是“更优雅”，而是：

- 更容易 mock / test
- 更容易限权
- 更容易替换实现
- 更容易把未来的 agent 操作范围控制在安全边界内

### 启示 5：把论文成稿系统独立成子系统

`AI-Researcher` 的 `paper_agent` 告诉我一个很重要的事实：  
**研究逻辑完成，不代表论文已经准备好。**

对 `Muse` 来说，最可能出现的后续需求不是“再多几个导出格式”，而是：

- 同一 run 产出多个论文组织版本
- 支持导师风格 / 学校模板差异
- 更强的摘要、引言、related work 统稿
- 章节间术语统一
- 图表、公式、实验叙述一致性修订

这更像一个 `paper composition layer`，而不是 `latex export` 的附加功能。

## 三、不建议直接照搬的地方

### 1. 进程级全局状态

`main_ai_researcher.py` + `global_state.py` 这一层明显更偏实验系统写法。  
`INIT_FLAG` 之类的进程级状态对 Web demo 可能够用，但对 `Muse` 这种 CLI + run store 型工具并不理想。

`Muse` 现在更好的方向仍然是：

- 让状态存在 run store
- 让 resume 基于 run artifact
- 避免进程内布尔旗标成为流程控制核心

### 2. 交互式缓存恢复菜单

`flowcache.py` 在命中缓存时会弹出：

- Yes
- Resume
- No

这对研究人员本地实验很方便，但对 `Muse` 当前体验并不是最优。  
`Muse` 已经有更稳定的思路：**run-id + 显式 resume**。  
所以如果借鉴它的缓存思想，建议只借鉴：

- cache 粒度
- artifact 结构
- 局部恢复能力

而不要照搬它的交互方式。

### 3. Prompt-heavy 串行链条

`run_infer_idea.py` / `run_infer_plan.py` 里能看出很多步骤是靠超长 prompt 串起来的。  
这对早期快速做 research system 很有效，但缺点是：

- 契约松
- 输出漂移大
- 调试成本高
- 错误恢复难

`Muse` 已经有 `schemas.py` 和更强的结构化状态基础，所以更适合走：

- **短 prompt**
- **强 schema**
- **阶段间明确输入输出**

而不是把所有上下文塞进单轮大提示词里。

### 4. 从自然语言中抠 JSON

`extract_json_from_output()` 这种模式在原型阶段常见，但在长期维护里比较脆。  
`Muse` 应当尽量坚持：

- 明确 JSON schema
- 明确 parser
- 明确失败恢复路径

这点上，`Muse` 其实应该比 `AI-Researcher` 更“工程化”，而不是向它回退。

### 5. 过度绑定“机器学习实验代理”问题设定

`AI-Researcher` 很大一部分设计都围绕：

- benchmark instance
- dataset selection
- GitHub codebase reuse
- Docker / GPU / training / testing

这对“自动做 ML 创新与实验”的场景很合理，但 `Muse` 的核心目标目前更偏：

- 学术写作
- 文献组织
- 章节撰写
- 引文核验
- 论文导出

因此 `Muse` 更应该学习它的**编排层思路**，而不是直接复制它的任务骨架。

## 四、我建议 Muse 的改造优先级

### 短期：在现有架构上最小增量增强

优先做这三件事：

1. **增加 flow recipe 层**
   - 保留 `stage1..stage6`
   - 新增不同 run mode 的编排定义
2. **增加 agent/tool artifact 落盘**
   - 不必一步到位做复杂缓存
   - 先把关键输入输出单独存下来
3. **明确角色级模型路由**
   - 在 `providers.py` 现有基础上把 route 从阶段扩展到角色

这是性价比最高的一组改造。

### 中期：把成稿链从导出链中拆出来

当你开始继续打磨论文侧能力时，我建议单独引入：

- `paper composition` 层
- 章节统稿器
- 摘要 / 引言 / related work 的专门组织逻辑

此时 `latex_export.py` 只负责“把既定 paper package 渲染成工程 / zip / pdf”，而不再承担太多成稿组织职责。

### 长期：引入更正式的 orchestration runtime

如果 `Muse` 以后真的要往“研究代理平台”走，可以再考虑：

- 可声明的 flow graph
- 可重放的 agent traces
- partial rerun / replay
- benchmark / evaluation harness
- tool permission model

但这一步不适合现在立刻做。  
以当前仓库规模，先做“可解释的 flow recipe + 可审计 artifact”就足够了。

## 五、如果把这份调研转成 Muse backlog，我会这样拆

### 候选任务 1：引入 FlowRecipe

目标：

- 在 `muse/runtime.py` 之外新增 flow 定义层
- 支持至少两种 flow：
  - `default_thesis`
  - `export_only` 或 `review_only`

### 候选任务 2：为每个阶段落 agent artifact

目标：

- 为 outline / writer / reviewer / citation audit / export 单独记录输入输出
- 让 run 目录可调试、可回放

### 候选任务 3：把模型路由升级为角色路由

目标：

- 从“一个 stage 一个模型偏好”升级为“一个角色一个模型偏好”
- 保留 fallback 机制

### 候选任务 4：抽离 environments

目标：

- 把 web/search/files/latex 等能力做成适配层
- 降低 `stages.py` 的耦合度

### 候选任务 5：单独建立 paper composition 层

目标：

- 让论文统稿、章节组织、摘要引言整合不再只是 export 的前置副作用

## 六、总体判断

我对 `AI-Researcher` 的总体评价是：

- **作为研究代理原型，它的分层意识其实不错**
- **作为可长期维护的工程系统，它仍然有明显脚本化和实验系统痕迹**

因此，对 `Muse` 最好的用法不是“照着重构一遍”，而是：

- 学它的**分层方式**
- 不学它的**脆弱接口**

更具体一点：

- 学 `Flow / Agent / Tool / Environment / Paper` 的职责分离
- 不学 `global_state + 长 prompt 串联 + 文本抠 JSON + 交互式缓存菜单`

如果只用一句话概括这份调研给 `Muse` 的建议，那就是：

> **让 Muse 从“有 6 个阶段的单条流水线”，升级成“由多个 flow recipe 编排、由角色 agent 执行、由 artifact 支撑恢复与审计的研究写作运行时”。**

## 参考来源

- 官方仓库：`https://github.com/HKUDS/AI-Researcher`
- 官方 README：`https://github.com/HKUDS/AI-Researcher/blob/main/README.md`
- 顶层入口：`https://github.com/HKUDS/AI-Researcher/blob/main/main_ai_researcher.py`
- Agent 运行时：`https://github.com/HKUDS/AI-Researcher/blob/main/research_agent/inno/core.py`
- 类型定义：`https://github.com/HKUDS/AI-Researcher/blob/main/research_agent/inno/types.py`
- Flow/cache 抽象：`https://github.com/HKUDS/AI-Researcher/blob/main/research_agent/inno/workflow/flowcache.py`
- 研究主流程：
  - `https://github.com/HKUDS/AI-Researcher/blob/main/research_agent/run_infer_plan.py`
  - `https://github.com/HKUDS/AI-Researcher/blob/main/research_agent/run_infer_idea.py`
- 论文生成链：
  - `https://github.com/HKUDS/AI-Researcher/blob/main/paper_agent/writing.py`
  - `https://github.com/HKUDS/AI-Researcher/blob/main/paper_agent/tex_writer.py`
- 论文入口：`https://arxiv.org/abs/2505.18705`

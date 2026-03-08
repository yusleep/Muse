# Research Cognitive Engine — Design Draft (PENDING)

> **Status:** Draft — 待 Phase 0-5 代码更新完成后再讨论和细化。
> **Date:** 2026-03-08
> **Depends on:** Phase 0-5 (DeerFlow-Inspired Upgrade) 全部完成

---

## 问题诊断

Muse 当前的 4 个核心短板：

| 短板 | 表现 |
|-----|------|
| **写作质量不达标** | 浅层拼接、缺乏深度论证、文本像"材料堆砌"而非"有思想的论述" |
| **缺乏研究思维** | Agent 不会发现问题、形成假设、设计实验、分析结果、连接知识 |
| **学术规范认知不足** | 不了解审稿标准、学科惯例，写出的论文不像"学术论文" |
| **工程可靠性** | 从搜索到输出太慢、太贵、过程中容易出错卡死 |

**根因**：Muse 是一个"文本拼接流水线"，不是一个"研究者"。Phase 0-5 解决了工程架构
（工具调用、中间件、子 agent、MCP、沙箱、记忆），但没有解决**认知架构**。

**核心场景**：硕士/博士学位论文（5-8 万字），后续扩展到顶会论文和非研究型学术写作。

---

## 设计方向：B + A 混合

方向 B（研究认知引擎）解决根本问题，方向 A（Prompt/Skills 优化）作为实现手段。

---

## 1. Research Storyline Graph（研究故事线图）

### 问题
当前 Muse 直接从大纲跳到章节写作，没有构建研究的"故事线"。
结果：各章节独立存在，缺乏贯穿全文的论证主线。

### 设计思路
在 outline 和 chapter_draft 之间增加一个 **storyline construction** 阶段：

```
search → outline → [NEW: storyline] → chapter_draft → ...
```

Storyline 是一个结构化的论证骨架：

```yaml
storyline:
  research_question: "如何在保持生成质量的同时降低 GAN 训练不稳定性？"

  motivation:
    gap: "现有稳定化方法牺牲了生成多样性"
    evidence: ["@ref_1: spectral norm 限制了模型容量", "@ref_2: 梯度惩罚增加训练成本"]

  hypothesis: "通过自适应正则化可以在稳定性和多样性之间取得平衡"

  contribution_chain:
    - claim: "提出 AdaReg 方法"
      supports: hypothesis
      chapter: "ch_03"
      evidence_type: "theoretical + experimental"

    - claim: "AdaReg 在 CIFAR-10/CelebA 上优于 baseline"
      supports: "claim_1"
      chapter: "ch_04"
      evidence_type: "experimental"

    - claim: "消融实验表明自适应机制是关键"
      supports: "claim_2"
      chapter: "ch_04"
      evidence_type: "ablation"

  chapter_roles:
    ch_01: "establish problem + motivation"
    ch_02: "survey existing approaches → identify gap"
    ch_03: "present method → theoretical justification"
    ch_04: "experimental validation → analysis"
    ch_05: "conclusions + limitations + future work"
```

**每个章节必须服务于 storyline 中的一个环节**。写作时 agent 可以查询 storyline
确认"我这一节在论证什么、需要什么证据"。

---

## 2. Critical Literature Analysis（批判性文献分析）

### 问题
当前文献综述只是列举：A 做了 X，B 做了 Y，C 做了 Z。
缺乏批判性分析：这些方法的共同假设是什么？局限在哪？留下了什么 gap？

### 设计思路
将文献综述从"列举"升级为"知识图谱构建"：

```
[NEW: literature_analysis] 阶段输出:

knowledge_map:
  approaches:
    - name: "Spectral Normalization"
      solves: ["训练不稳定"]
      assumes: ["判别器 Lipschitz 约束足够"]
      limitations: ["限制模型容量", "不适用于大规模生成"]
      refs: ["@miyato2018", "@zhang2019"]

    - name: "Gradient Penalty"
      solves: ["模式坍塌"]
      assumes: ["梯度惩罚点的选取代表数据分布"]
      limitations: ["计算开销大", "超参敏感"]
      refs: ["@gulrajani2017"]

  gap_analysis:
    identified_gaps:
      - gap: "稳定性和多样性的 trade-off 未被解决"
        evidence: ["@ref_1 的表 3 显示 IS 下降", "@ref_2 的图 5 显示模式减少"]
        your_contribution: "AdaReg 通过自适应正则化解决此 trade-off"

  evolution_timeline:
    - period: "2014-2017"
      theme: "基础架构探索"
      key_works: [...]
    - period: "2017-2020"
      theme: "训练稳定化"
      key_works: [...]
    - period: "2020-present"
      theme: "大规模生成 + 质量控制"
      key_works: [...]
```

**文献综述基于 knowledge_map 生成**，而非直接从文献列表拼接。

---

## 3. Argumentation Chain（论证链条）

### 问题
当前每个小节独立写作，claim 之间没有逻辑关联。
结果：读起来像维基百科条目，不像学术论证。

### 设计思路
每个 claim 必须有完整的论证结构：

```yaml
argument:
  claim: "AdaReg 方法有效降低了训练不稳定性"

  premises:
    - "训练不稳定性的主要原因是判别器梯度的剧烈波动（@ref_1）"
    - "自适应正则化可以根据训练状态动态调整约束强度"

  reasoning: "因此，当判别器梯度过大时增强正则化，过小时放松，
              可以在不牺牲模型容量的前提下维持稳定训练"

  evidence:
    - type: "experimental"
      content: "表 1 显示 AdaReg 的 FID 下降 15% 且训练 loss 曲线更平稳"
    - type: "ablation"
      content: "移除自适应机制后 FID 上升 23%（表 3）"

  counter_arguments:
    - "固定正则化也能稳定训练"
      rebuttal: "但会限制模型容量（图 4 对比）"
```

**写作工具升级**：`write_section` tool 不只生成文本，还要输出该段的
argumentation chain。Review tool 检查论证链的完整性（有无跳步、有无反例未处理）。

---

## 4. Cross-Chapter Coherence Engine（跨章节连贯引擎）

### 问题
各章独立并行写作后拼接，术语不一致、逻辑断裂、前后矛盾。
现有 composition subgraph 只是 placeholder。

### 设计思路
一个专门的 **Coherence Agent** 负责维护全文一致性：

**职责**：
1. **术语表管理**：维护全文统一术语表，检查所有章节使用一致
2. **论证主线追踪**：确保每章都在推进 storyline，没有偏题
3. **前后引用验证**：第 4 章提到"如第 3 章所述"时，确认第 3 章确实有此内容
4. **过渡段生成**：在章节之间生成自然的过渡，而非硬切换
5. **重复检测**：避免不同章节重复论述相同内容

**执行时机**：
- 每章写完后立即做 coherence check（增量）
- 全文合并后做全局 coherence pass

---

## 5. Academic Review Simulation（学术审稿模拟）

### 问题
当前 review 只有 multi-lens 打分（logic/style/citation/structure），
无法发现深层问题（论证不充分、实验设计缺陷、创新点不清晰）。

### 设计思路
写完后用 LLM 模拟 **3 种审稿人**，生成 structured review：

| 审稿人角色 | 关注点 |
|-----------|--------|
| **Reviewer 1: 严格审稿人** | 论证漏洞、逻辑跳步、未处理的反例、过度声明 |
| **Reviewer 2: 方法论专家** | 实验设计合理性、baseline 是否公平、消融实验是否充分 |
| **Reviewer 3: 领域专家** | 对现有工作的理解是否准确、贡献是否显著、与 SOTA 的关系 |

**Review 输出格式**（参考真实审稿）：

```yaml
review:
  reviewer: "strict"
  overall_score: 5  # 1-10
  summary: "论文提出了 AdaReg 方法，但论证链有明显跳步..."

  strengths:
    - "问题动机清晰"
    - "实验覆盖了多个数据集"

  weaknesses:
    - severity: "major"
      location: "ch_03, Section 3.2"
      issue: "从'梯度波动导致不稳定'到'自适应正则化有效'之间缺乏理论推导"
      suggestion: "补充收敛性分析或至少提供直觉解释"

    - severity: "major"
      location: "ch_04, Table 1"
      issue: "未与最新的 StyleGAN3 对比"
      suggestion: "补充 StyleGAN3 baseline 或解释为何不适用"

    - severity: "minor"
      location: "ch_02"
      issue: "文献综述中对 Gradient Penalty 的描述不够准确"
      suggestion: "修正公式 (3) 的约束条件"

  questions:
    - "自适应正则化的超参 λ 是如何选择的？敏感性分析？"
    - "方法是否适用于非图像生成任务？"
```

**修改循环**：Review → 识别 major issues → 自动修改 → 再次 review → 直到无 major issues。

---

## 6. 实现路径（与现有 Phase 0-5 的关系）

### 前置依赖
- Phase 0-C（Skills）：认知框架通过 SKILL.md 注入
- Phase 1（ReAct）：storyline/literature/coherence 作为工具或 agent
- Phase 2（HITL）：storyline 审批、review 反馈需要结构化交互
- Phase 3（Subagent）：审稿模拟可以作为并行子 agent

### 建议的实现阶段（Phase 0-5 完成后）

```
Phase 6-A: Research Storyline Graph
  - storyline 数据结构
  - storyline construction 节点（outline 之后、draft 之前）
  - 写作工具查询 storyline

Phase 6-B: Critical Literature Analysis
  - knowledge_map 数据结构
  - literature_analysis 节点（search 之后、outline 之前）
  - 文献综述基于 knowledge_map 生成

Phase 6-C: Argumentation Chain
  - argument 数据结构
  - write_section 工具升级（输出 argument chain）
  - review 工具检查论证完整性

Phase 6-D: Cross-Chapter Coherence Engine
  - Coherence Agent（composition subgraph 升级）
  - 术语表、前后引用、过渡段

Phase 6-E: Academic Review Simulation
  - 3 种审稿人 prompt
  - Review → Fix 循环
  - 与 HITL 集成（用户可以选择接受/拒绝修改建议）
```

### 依赖图

```
Phase 0-5 (DeerFlow Upgrade, 全部完成)
  └── Phase 6-A (Storyline) ──┐
  └── Phase 6-B (Literature)  ├── Phase 6-D (Coherence)
  └── Phase 6-C (Argument) ──┘        │
                                       └── Phase 6-E (Review Simulation)
```

---

## 待讨论事项（代码更新后）

1. Storyline 是由 LLM 一次性生成还是通过多轮对话与用户共建？
2. Knowledge map 的粒度——到论文级还是到 claim 级？
3. 审稿模拟是否需要接入真实的 paper review 数据作为 few-shot？
4. Coherence Engine 是独立 agent 还是融入 composition subgraph？
5. 这些新阶段是否需要额外的 HITL 中断点？
6. 对于非 CS 学科（如社会科学、人文），认知框架需要怎样调整？

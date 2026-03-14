# PROGRESS.md — Muse Chapter Graph 改进进度

## 总览

- **状态**：进行中
- **总进度**：1/5 phases, 11/28 steps
- **当前 Wave**：Wave 2
- **下一步**：Phase 2 Step 2.4

---

## Wave 1 — Quick Wins

### Phase 1: Prompt & Logic Fixes (8/8)
- [x] Step 1.1: 重写 chapter_review_prompt — 丰富评审 rubric (~80 行)
- [x] Step 1.2: 字数控制反馈 — 偏差 >30% 自动重试 (~20 行)
- [x] Step 1.3: Revision 指令合并 — 同 subtask 多条 note 不覆盖 (~8 行)
- [x] Step 1.4: Citation allowlist 硬校验 (~12 行)
- [x] Step 1.5: References 展示优化 — 扩大 agent 可见范围 (~10 行)
- [x] Step 1.6: Tool JSON 双重序列化修复 (~5 行)
- [x] Step 1.7: Subtask 范围提示增强 (~5 行)
- [x] Step 1.8: 移除 abstract 截断 — 扩大参考文献上下文 (~10 行)

---

## Wave 2 — Reliability

### Phase 2: Reliability Hardening (3/5)
- [x] Step 2.1: ReAct 递归上限 — 累积器 + 补写缺失 subtask (~60 行)
- [x] Step 2.2: Revision stall 检测 — 文本相似度 + 分数趋势 (~12 行)
- [x] Step 2.3: Self-assessment 利用 — confidence 驱动优先修订 (~20 行)
- [ ] Step 2.4: Web search / image search stub 替换 (~15 行)
- [ ] Step 2.5: PaperIndexService — LlamaIndex 全文索引 + 段落级语义检索 (~250 行)

---

## Wave 3 — Review Architecture

### Phase 3: Global Review (Post-Merge) (0/5)
- [ ] Step 3.1: 生成-评审闭环 — Reviewer 从修订效果中学习 (~80 行)
- [ ] Step 3.2: 多视角 Critique + Judge — 3 Persona 独立评审 (~200 行)
- [ ] Step 3.3: 分层深度修订 — Structural → Content → Line (~250 行)
- [ ] Step 3.4: 智能模型路由优化 (~25 行)
- [ ] Step 3.5: Coherence Check — 合并后连贯性验证 (~110 行)

---

## Wave 4 — Cross-Chapter Intelligence

### Phase 4: Cross-Chapter Intelligence (0/5)
- [ ] Step 4.1: Citation 硬门禁 (~80 行)
- [ ] Step 4.2: Memory Keeper — 跨章术语/引用一致性追踪 (~120 行)
- [ ] Step 4.3: Persistent Reflection Bank — 修订经验积累 (~130 行)
- [ ] Step 4.4: Reference Briefs — 章节级参考文献分析 (~150 行)
- [ ] Step 4.5: Argument Planning — 预写结构化论证规划 (~100 行)

---

## Wave 5 — Exploratory

### Phase 5: Exploratory Enhancements (0/5)
- [ ] Step 5.1: STORM 模拟对话视角发现 (~265 行)
- [ ] Step 5.2: 单 Pass 写作模式 (~98 行)
- [ ] Step 5.3: Agent 自进化 — LLM 分析弱项生成 prompt 改进 (~200 行)
- [ ] Step 5.4: 导出后视觉验证 — PDF → VLM 检查 (~190 行)
- [ ] Step 5.5: 人类大纲 Heuristics — Few-shot 模板注入 (~95 行)

---

## Notes

- 2026-03-14: 初始化 Chapter Graph 改进计划，基于 evo/writing-agents-research.md 研究
- 2026-03-14: 前序 V2.1 计划（6 Phases, 39 Tasks）已全部完成
- 2026-03-14: Phase 2 Step 2.5 设计升级为 PaperIndexService（LlamaIndex），替代原 PaperContentClient（pymupdf）
- 2026-03-14: 完成 Phase 2 Step 2.1，删除 chapter fallback graph，改为 partial recovery + 显式失败
- 2026-03-14: 完成 Phase 2 Step 2.2，为 should_iterate 增加文本/分数 stall 检测
- 2026-03-14: 完成 Phase 2 Step 2.3，接通 self-assessment 到 review notes 和 MuseState

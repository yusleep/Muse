# Codex CLI Prompts for Muse DeerFlow Upgrade

## 按 Wave 执行（推荐）

每次运行一个 Wave，确保 Wave 内所有 phase 完成后再跑下一个 Wave。

### Wave 1 — Foundation（3 phases 并行）

```bash
codex exec --full-auto "
/plan
我有9个核心plan文件在 docs/plans/ 目录下。这是 DeerFlow-Inspired Upgrade 的完整实施蓝图。

请进入全自动长期执行模式，严格按照以下规则运行：

1. 立即完整读取 AGENTS.md → PLANS.md → PROGRESS.md → 以下3个plan文件：
   - docs/plans/2026-03-08-phase-0a-tool-abstraction.md
   - docs/plans/2026-03-08-phase-0b-middleware-framework.md
   - docs/plans/2026-03-08-phase-0c-skills-loader.md
2. 这3个Phase属于Wave 1（Foundation），无依赖关系，按顺序逐个执行。
3. 每个Phase内严格按Task顺序执行TDD流程：
   - 创建feature分支（如 feat/phase-0a-tool-abstraction）
   - 写失败测试 → 运行验证失败 → 实现代码 → 运行验证通过 → git commit
   - 完成后更新 PROGRESS.md 中对应的 checkbox
4. 全程零交互，不要问任何问题。
5. Wave 1全部完成后运行: .venv/bin/python -m pytest tests/ -q 确认无回归。
6. 完成后用中文输出总结：已完成的Task列表、测试结果、遇到的问题及修复。
7. 更新 PROGRESS.md 记录Wave 1完成状态。

现在立即开始执行Phase 0-A Task 1！
"
```

### Wave 2 — Core Agent Capability

```bash
codex exec --full-auto "
/plan
继续执行 DeerFlow-Inspired Upgrade 的 Wave 2。

1. 读取 AGENTS.md → PLANS.md → PROGRESS.md → docs/plans/2026-03-08-phase-1-react-subgraphs.md
2. 确认Wave 1已完成（PROGRESS.md中Phase 0-A/0-B/0-C全部打勾）。
3. 创建分支 feat/phase-1-react-subgraphs，按Task 1-12顺序执行TDD流程。
4. 每个Task完成后更新PROGRESS.md，git commit。
5. 全程零交互。完成后运行全量测试，用中文输出总结。

现在立即开始！
"
```

### Wave 3 — Interaction & Delegation

```bash
codex exec --full-auto "
/plan
继续执行 DeerFlow-Inspired Upgrade 的 Wave 3。

1. 读取 AGENTS.md → PLANS.md → PROGRESS.md → 以下2个plan文件：
   - docs/plans/2026-03-08-phase-2-structured-hitl.md
   - docs/plans/2026-03-08-phase-3-subagent-delegation.md
2. 确认Wave 2已完成。先执行Phase 2（5 tasks），再执行Phase 3（6 tasks）。
3. 每个Phase创建独立分支，按Task顺序TDD执行。
4. 全程零交互。完成后运行全量测试，用中文输出总结。

现在立即开始！
"
```

### Wave 4 — External Integration

```bash
codex exec --full-auto "
/plan
继续执行 DeerFlow-Inspired Upgrade 的 Wave 4。

1. 读取 AGENTS.md → PLANS.md → PROGRESS.md → 以下2个plan文件：
   - docs/plans/2026-03-08-phase-4a-mcp-integration.md
   - docs/plans/2026-03-08-phase-4b-sandbox-execution.md
2. 确认Wave 1已完成（Wave 4仅依赖Wave 1，可与Wave 2/3并行）。
3. 按顺序执行Phase 4-A（7 tasks）再Phase 4-B（7 tasks），各建独立分支。
4. 全程零交互。完成后运行全量测试，用中文输出总结。

现在立即开始！
"
```

### Wave 5 — Memory

```bash
codex exec --full-auto "
/plan
继续执行 DeerFlow-Inspired Upgrade 的 Wave 5（最后一个Wave）。

1. 读取 AGENTS.md → PLANS.md → PROGRESS.md → docs/plans/2026-03-08-phase-5-memory-system.md
2. 确认Wave 1已完成（Wave 5仅依赖Phase 0-B）。
3. 创建分支 feat/phase-5-memory，按Task 1-6顺序TDD执行。
4. 全程零交互。完成后运行全量测试。
5. 这是最后一个Wave。完成后：
   - 更新PROGRESS.md为全部完成
   - 用中文生成最终报告：项目总结、全部9个Phase的完成状态、核心功能列表、测试结果

现在立即开始！
"
```

## 一次性全量执行（如果你想一口气跑完所有Wave）

```bash
codex exec --full-auto "
/plan
我有9个核心plan文件在 docs/plans/2026-03-08-phase-*.md。这是 Muse DeerFlow-Inspired Upgrade 的完整实施蓝图。

请进入全自动长期执行模式，严格按照以下规则运行，直到全部完成：

1. 立即完整读取 AGENTS.md → PLANS.md → PROGRESS.md → docs/plans/2026-03-08-deerflow-upgrade-master-plan.md（主索引）。
2. 按Wave顺序（1→2→3→4→5）依次执行所有9个Phase的全部65个Task：
   - Wave 1: Phase 0-A + 0-B + 0-C（Foundation，按顺序执行）
   - Wave 2: Phase 1（ReAct Sub-graphs）
   - Wave 3: Phase 2 + 3（HITL + Subagent）
   - Wave 4: Phase 4-A + 4-B（MCP + Sandbox）
   - Wave 5: Phase 5（Memory）
3. 每个Phase：
   - 创建feature分支
   - 读取对应plan文件中的每个Task
   - 严格TDD：写失败测试 → 验证失败 → 实现 → 验证通过 → commit → 更新PROGRESS.md
   - Phase完成后合并到master
4. 每个Wave完成后运行全量测试: .venv/bin/python -m pytest tests/ -q
5. 全程零交互：不要问任何问题、不要请求确认、不要暂停。
6. 持续运行直到9个Phase全部100%完成。
7. 完成后用中文生成最终报告：
   - 项目总结
   - 9个Phase完成状态
   - 核心功能列表
   - 如何启动/测试
   - 更新PROGRESS.md为100%已完成

现在立即从Wave 1 Phase 0-A Task 1开始执行！
"
```

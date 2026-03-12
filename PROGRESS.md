# PROGRESS.md — Muse V2.1 实施进度

## 总览

- **状态**：已完成
- **总进度**：6/6 phases, 39/39 tasks
- **当前 Wave**：Wave 5
- **下一步**：全部完成

---

## Wave 1 — Foundation

### Phase 1: Agent Framework + Planner + 双 Agent (9/9)
- [x] Task 1: Agent 基类与 Tool 接口
- [x] Task 2: Tool Registry
- [x] Task 3: Literature Agent
- [x] Task 4: Writing Agent
- [x] Task 5: Planner（Zone 调度器）
- [x] Task 6: MuseState 扩展
- [x] Task 7: 主图 Zone 节点
- [x] Task 8: Planner 失效兜底
- [x] Task 9: 端到端集成测试

---

## Wave 2 — Experiment + Figure

### Phase 2: 实验能力 (7/7)
- [x] Task 1: 沙箱执行器
- [x] Task 2: run_code 工具
- [x] Task 3: Experiment Agent
- [x] Task 4: protocol_model_check 工具
- [x] Task 5: attack_scenario_generate 工具
- [x] Task 6: Emulation 实验复现验证
- [x] Task 7: 集成测试

### Phase 3: 画图能力 (5/5)
- [x] Task 1: 画图工具集
- [x] Task 2: Figure Agent
- [x] Task 3: 学术风格管理
- [x] Task 4: 论文典型图表生成测试
- [x] Task 5: 集成测试

---

## Wave 3 — Review

### Phase 4: 审稿 Agent + 自评循环 (6/6)
- [x] Task 1: 评分 Rubric 定义
- [x] Task 2: Review Agent
- [x] Task 3: 评分校准样本
- [x] Task 4: refinement_zone 审稿循环
- [x] Task 5: 稳定化测试集
- [x] Task 6: 集成测试

---

## Wave 4 — Knowledge + Meta

### Phase 5: Knowledge Base + Meta Layer (7/7)
- [x] Task 1: Knowledge Base（四命名空间）
- [x] Task 2: Meta Layer 候选策略生成
- [x] Task 3: 验证门
- [x] Task 4: 熔断与灰度机制
- [x] Task 5: 策略版本化与回滚
- [x] Task 6: Provenance 追溯
- [x] Task 7: 集成测试

---

## Wave 5 — Full Planner + Web UI

### Phase 6: Planner 全 Zone + Web UI (5/5)
- [x] Task 1: Planner 全 Zone 覆盖
- [x] Task 2: Web UI 后端
- [x] Task 3: Web UI 前端
- [x] Task 4: 进化报告展示
- [x] Task 5: 端到端评估

---

## Notes

- 2026-03-09: 初始化 V2.1 实施计划，替换旧 DeerFlow 计划
- 2026-03-09: 已验证 `feat/v2-phase-1-agent-framework` 基线（Phase 1 相关测试 23 通过）
- 2026-03-09: `feat/v2-phase-2-experiment` 完成 Task 1（sandbox executor），相关 sandbox 测试 48 通过
- 2026-03-09: `feat/v2-phase-2-experiment` 完成 Task 2（run_code 工具），相关 registry+sandbox 测试 55 通过
- 2026-03-09: `feat/v2-phase-2-experiment` 完成 Task 3（Experiment Agent），相关 agent 回归测试 22 通过
- 2026-03-09: `feat/v2-phase-2-experiment` 完成 Task 4（protocol_model_check 工具），相关 protocol+registry 测试 17 通过
- 2026-03-09: `feat/v2-phase-2-experiment` 完成 Task 5（attack_scenario_generate 工具），相关 attack+protocol 测试 14 通过
- 2026-03-09: `feat/v2-phase-2-experiment` 完成 Task 6（Emulation 实验复现验证），相关 Phase 2 回归测试 65 通过
- 2026-03-09: `feat/v2-phase-2-experiment` 完成 Task 7（集成测试），`drafting_zone` 已可调度 `ExperimentAgent`，相关回归测试 72 通过
- 2026-03-09: `feat/v2-phase-3-figure` 完成 Task 1（画图工具集），图表工具与注册表测试 7 通过
- 2026-03-09: `feat/v2-phase-3-figure` 完成 Task 2（Figure Agent），被动/主动模式与图表相关回归测试 9 通过
- 2026-03-09: `feat/v2-phase-3-figure` 完成 Task 3（学术风格管理），风格模块与图表工具回归测试 9 通过
- 2026-03-09: `feat/v2-phase-3-figure` 完成 Task 4（论文典型图表生成测试），论文图表场景与 FigureAgent 回归测试 11 通过
- 2026-03-09: `feat/v2-phase-3-figure` 完成 Task 5（集成测试），`drafting_zone` 已可调度 `FigureAgent`，相关回归测试 14 通过
- 2026-03-09: Wave 2 gate 通过，`.venv/bin/python -m pytest tests/ -q` 结果为 `674 passed, 8 warnings, 21 subtests passed`
- 2026-03-10: `feat/v2-phase-4-review` 完成 Task 1（评分 Rubric 定义），新增 `tests/test_review_rubric.py`，相关测试 2 通过
- 2026-03-10: `feat/v2-phase-4-review` 纠正 Phase 基线：Phase 4 分支已 rebase 到 `feat/v2-phase-3-figure`，当前 stacked worktree 全量测试为 `676 passed, 8 warnings, 21 subtests passed`
- 2026-03-10: `feat/v2-phase-4-review` 完成 Task 2（Review Agent），新增 `muse/agents/review.py` 与 `tests/test_review_agent.py`，相关 review 测试 4 通过
- 2026-03-10: `feat/v2-phase-4-review` 完成 Task 3（评分校准样本），新增 `muse/agents/review_calibration.py` 与 `muse/data/calibration_samples/`，相关 review 回归测试 5 通过
- 2026-03-10: `feat/v2-phase-4-review` 完成 Task 4（refinement_zone 审稿循环），新增 `tests/test_refinement_zone.py`，并扩展 planner / writing / main_graph，相关相邻回归测试 28 通过
- 2026-03-10: `feat/v2-phase-4-review` 完成 Task 5（稳定化测试集），新增 `muse/data/stabilization_tests/` 与 `tests/test_review_stability.py`，相关 review 测试 6 通过
- 2026-03-10: `feat/v2-phase-4-review` 完成 Task 6（集成测试），新增 `tests/test_review_integration.py` 并将 refinement 结果桥接到 `review_feedback`，相关 integration 测试 12 通过
- 2026-03-10: Wave 3 gate：`.venv/bin/python -m pytest tests/ -q` 结果为 `684 passed, 8 warnings, 21 subtests passed`
- 2026-03-10: Wave 3 gate：`.venv/bin/python -m muse --config /home/planck/gradute/Muse/config.yaml check` 返回 `{\"llm\": true, \"semantic_scholar\": false, \"openalex\": true, \"crossref\": true, \"ok\": false}`；当前阻塞来自本地 `semantic_scholar` 配置/凭据而非代码回归
- 2026-03-10: `feat/v2-phase-5-knowledge-meta` 完成 Task 1（Knowledge Base 四命名空间），新增 `muse/knowledge/base.py` 与 `tests/test_knowledge_base.py`，相关 memory+knowledge 测试 28 通过
- 2026-03-10: `feat/v2-phase-5-knowledge-meta` 完成 Task 2（Meta Layer 候选策略生成），新增 `muse/meta/layer.py` 与 `tests/test_meta_layer.py`，相关 knowledge+meta 测试 7 通过
- 2026-03-10: `feat/v2-phase-5-knowledge-meta` 完成 Task 3（验证门），新增 `muse/meta/verification.py` 与 `tests/test_verification_gate.py`，相关 knowledge+meta 测试 9 通过
- 2026-03-10: `feat/v2-phase-5-knowledge-meta` 完成 Task 4（熔断与灰度机制），新增 `muse/meta/circuit_breaker.py` 与 `tests/test_circuit_breaker.py`，相关 knowledge+meta 测试 12 通过
- 2026-03-10: `feat/v2-phase-5-knowledge-meta` 完成 Task 5（策略版本化与回滚），扩展 `muse/meta/policy.py` 并新增 `tests/test_policy.py`，相关 knowledge+meta 测试 15 通过
- 2026-03-10: `feat/v2-phase-5-knowledge-meta` 完成 Task 6（Provenance 追溯），新增 `muse/meta/provenance.py` 与 `tests/test_provenance.py`，相关 knowledge+meta 测试 17 通过
- 2026-03-10: `feat/v2-phase-5-knowledge-meta` 完成 Task 7（集成测试），新增 `tests/test_meta_integration.py`，相关 knowledge+meta 集成测试 18 通过
- 2026-03-10: Wave 4 gate：`.venv/bin/python -m pytest tests/ -q` 结果为 `702 passed, 8 warnings, 21 subtests passed`
- 2026-03-10: `feat/v2-phase-6-planner-webui` 完成 Task 1（Planner 全 Zone 覆盖），新增 `research_zone` Planner 节点与 `tests/test_full_zone_planner.py`，相关回归测试通过
- 2026-03-10: `feat/v2-phase-6-planner-webui` 完成 Task 2（Web UI 后端），新增 `muse/web/` FastAPI 后端与 `tests/test_web_api.py`，相关 API/WS 测试通过
- 2026-03-10: `feat/v2-phase-6-planner-webui` 完成 Task 3（Web UI 前端），新增 `muse/web/static/` 单页前端与 `tests/test_web_static.py`，可渲染首页并加载静态资源
- 2026-03-10: `feat/v2-phase-6-planner-webui` 完成 Task 4（进化报告展示），新增 `/runs/{run_id}/evolution` 页面与 `tests/test_web_evolution_page.py`
- 2026-03-10: `feat/v2-phase-6-planner-webui` 完成 Task 5（端到端评估），新增 `tests/test_v2_full_e2e.py` 与 `evolution_report` 状态字段
- 2026-03-10: Wave 5 gate：`.venv/bin/python -m pytest tests/ -q` 结果为 `711 passed, 8 warnings, 21 subtests passed`
- 2026-03-11: 安全：在 `.gitignore` 中忽略 `/config.yaml`，降低误提交本地 API key 的风险
- 2026-03-11: 修复：LLM 路由仅对 `chatgpt.com/backend-api` 开启 SSE streaming，避免 Anthropic 等非 SSE provider 被错误走 `post_json_sse`
- 2026-03-11: 修复：`HttpClient.post_json_sse()` 支持解析 `delta.tool_calls`（工具调用流可无文本输出），并返回标准 `choices[0].message.tool_calls` 形状给上层使用
- 2026-03-11: Phase 1 Review 修复：在 `master` 提交 `paper_package` 合并 reducer 与 export/polish/latex/provider 稳定性修复；并将 `feat/v2-phase-1-agent-framework` rebase 到最新 `master` 后修复 async/降级/日志/Reducer 等问题，全量测试 `664 passed, 8 warnings, 21 subtests passed`
- 2026-03-11: master 全量测试通过：`.venv/bin/python -m pytest tests/ -q` 结果为 `641 passed, 6 warnings, 21 subtests passed`（补齐 `load_settings(env)` 去除隐式 config.yaml 依赖，并清理本地 AGENTS/PLANS 对 `docs/plans/` 的引用）
- 2026-03-12: 修复 citation ReAct 合约，新增结构化 `record_citation_assessment` / `finalize_citation_review` 工具、显式 `citation_worklist`、严格工具集与无 fallback 的硬失败路径；相关集成测试改为注入可 finalize 的假 React agent，全量测试 `649 passed, 21 subtests passed`
- 2026-03-12: 修复 React tool runtime schema 与 citation 递归死循环：将运行时参数改为显式 injected tool arg、补充 OpenAI tool schema 回归测试、移除 citation agent 的 `tool_choice=required`/`update_plan` 干扰项；相关回归测试为 `47 passed, 11 subtests passed`，并通过真实 `reasoning` 路由验证 citation prompt 首轮会直接调用 `verify_doi`

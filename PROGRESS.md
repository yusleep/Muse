# Muse Progress — DeerFlow-Inspired Upgrade

- 状态: 进行中
- 总进度: 5 / 9 phases (41 / 65 tasks)
- 当前 Wave: Wave 3 — Interaction & Delegation
- 下一步: Phase 3 Task 3

## Wave 1 — Foundation

### Phase 0-A: Tool Abstraction Layer (8/8)
- [x] Task 1: Install LangChain dependencies
- [x] Task 2: Create MuseChatModel adapter
- [x] Task 3: Extend _build_request_payload for tools
- [x] Task 4: Create model factory
- [x] Task 5: Create ToolRegistry
- [x] Task 6: Create academic_search tool
- [x] Task 7: Create citation tools
- [x] Task 8: Integration test

### Phase 0-B: Middleware Framework (7/7)
- [x] Task 1: Middleware protocol + MiddlewareChain
- [x] Task 2: LoggingMiddleware
- [x] Task 3: RetryMiddleware
- [x] Task 4: SummarizationMiddleware
- [x] Task 5: DanglingToolCallMiddleware
- [x] Task 6: Integration with main_graph
- [x] Task 7: Settings config

### Phase 0-C: Skills Loader (7/7)
- [x] Task 1: Skill dataclass + SkillLoader
- [x] Task 2: SkillRegistry
- [x] Task 3: academic-writing skill
- [x] Task 4: citation-gb-t-7714 skill
- [x] Task 5: thesis-structure-zh skill
- [x] Task 6: deep-research skill
- [x] Task 7: Integration test

## Wave 2 — Core Agent Capability

### Phase 1: Sub-graph ReAct Conversion (12/12)
- [x] Task 1: Writing tools
- [x] Task 2: Review tools
- [x] Task 3: Research tools
- [x] Task 4: File tools
- [x] Task 5: Orchestration tools
- [x] Task 6: Chapter subgraph ReAct
- [x] Task 7: Update fan_out_chapters
- [x] Task 8: Citation tools
- [x] Task 9: Citation subgraph ReAct
- [x] Task 10: Composition tools
- [x] Task 11: Composition subgraph ReAct
- [x] Task 12: Integration test

## Wave 3 — Interaction & Delegation

### Phase 2: Structured HITL (5/5)
- [x] Task 1: ask_clarification tool
- [x] Task 2: ClarificationMiddleware
- [x] Task 3: Upgrade interrupt nodes
- [x] Task 4: Upgrade CLI review command
- [x] Task 5: Integration test

### Phase 3: Subagent Delegation (2/6)
- [x] Task 1: SubagentExecutor
- [x] Task 2: SubagentResult protocol
- [ ] Task 3: spawn_subagent tool
- [ ] Task 4: SubagentLimitMiddleware
- [ ] Task 5: Built-in agent configs
- [ ] Task 6: Integration test

## Wave 4 — External Integration

### Phase 4-A: MCP Integration (0/7)
- [ ] Task 1: Install langchain-mcp-adapters
- [ ] Task 2: extensions.yaml config + loader
- [ ] Task 3: OAuthTokenManager
- [ ] Task 4: MCP tool loader
- [ ] Task 5: Tool cache
- [ ] Task 6: ToolRegistry integration
- [ ] Task 7: Integration test

### Phase 4-B: Sandbox Execution (0/7)
- [ ] Task 1: Sandbox ABC + ExecResult
- [ ] Task 2: LocalSandbox
- [ ] Task 3: DockerSandbox
- [ ] Task 4: VFS path mapping
- [ ] Task 5: Sandbox tools
- [ ] Task 6: Dockerfile.sandbox
- [ ] Task 7: Integration test

## Wave 5 — Memory

### Phase 5: Memory System (0/6)
- [ ] Task 1: MemoryEntry + MemoryStore
- [ ] Task 2: Memory formatting
- [ ] Task 3: MemoryMiddleware
- [ ] Task 4: Confidence lifecycle
- [ ] Task 5: Extraction triggers
- [ ] Task 6: Integration test

## Notes

- 2026-03-08: 旧 6 phase 计划（冻结服务边界 → 测试迁移）已 100% 完成。
- 2026-03-08: 新增 DeerFlow-inspired upgrade 设计和 9 phase 实施计划。
- 2026-03-08: Phase 1 Task 6 完成，新增 chapter ReAct prompt，并为 chapter 子图加入 ReAct + fallback 双模入口。
- 2026-03-08: Phase 1 Task 7 完成，补充 fan_out 契约测试并确认现有 Send 负载可直接兼容新 chapter agent。
- 2026-03-08: Phase 1 Task 8 完成，为 citation 子图补充 verify/crosscheck/entailment/flag/repair 五个 ReAct 工具，并保留旧工厂接口。
- 2026-03-08: Phase 1 Task 9 完成，新增 citation ReAct prompt，并为 citation 子图加入 ReAct + fallback 双模入口。
- 2026-03-08: Phase 1 Task 10 完成，新增 terminology/cross-ref/transition/rewrite 四个 composition 工具。
- 2026-03-08: Phase 1 Task 11 完成，新增 composition ReAct prompt，为 composition 子图加入 dual-mode，并将 main_graph 的 composition 节点切到新构造器。
- 2026-03-08: Phase 1 Task 12 完成，新增 dual-mode/fallback 集成测试，Wave 2 进入全量回归验证。
- 2026-03-08: Wave 2 Gate 通过：`.venv/bin/python -m pytest tests/ -q` → `343 passed, 1 skipped, 21 subtests passed`。
- 2026-03-08: Phase 2 Task 1 完成，新增 `ask_clarification` 工具及结构化 schema。
- 2026-03-08: Phase 2 Task 2 完成，新增 ClarificationMiddleware 并导出到 middleware 公共表面。
- 2026-03-08: Phase 2 Task 3 完成，为 top-level interrupt 节点加入 question/options/context 等结构化字段，并保留布尔 resume 兼容。
- 2026-03-08: Phase 2 Task 4 完成，CLI `review` 命令支持 `--option`，`_graph_response` 能返回 question/options/context/clarification_type。
- 2026-03-08: Phase 2 Task 5 完成，structured HITL 集成测试覆盖结构化 payload、option/comment resume 与旧布尔 resume。
- 2026-03-08: Phase 3 Task 1 完成，新增 `muse.agents` 包、最小 `SubagentResult` 骨架与并发/超时安全的 `SubagentExecutor`。
- 2026-03-08: Phase 3 Task 2 完成，`SubagentResult` 支持默认值、`to_dict`/`from_dict` 与 `summary()`。
- 测试基线: `.venv/bin/python -m pytest tests/ -q` → `343 passed, 1 skipped, 21 subtests passed`

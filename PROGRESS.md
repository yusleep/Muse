# Muse Progress — DeerFlow-Inspired Upgrade

- 状态: 进行中
- 总进度: 2 / 9 phases (21 / 65 tasks)
- 当前 Wave: Wave 1 — Foundation
- 下一步: Phase 0-C Task 7

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

### Phase 0-C: Skills Loader (6/7)
- [x] Task 1: Skill dataclass + SkillLoader
- [x] Task 2: SkillRegistry
- [x] Task 3: academic-writing skill
- [x] Task 4: citation-gb-t-7714 skill
- [x] Task 5: thesis-structure-zh skill
- [x] Task 6: deep-research skill
- [ ] Task 7: Integration test

## Wave 2 — Core Agent Capability

### Phase 1: Sub-graph ReAct Conversion (0/12)
- [ ] Task 1: Writing tools
- [ ] Task 2: Review tools
- [ ] Task 3: Research tools
- [ ] Task 4: File tools
- [ ] Task 5: Orchestration tools
- [ ] Task 6: Chapter subgraph ReAct
- [ ] Task 7: Update fan_out_chapters
- [ ] Task 8: Citation tools
- [ ] Task 9: Citation subgraph ReAct
- [ ] Task 10: Composition tools
- [ ] Task 11: Composition subgraph ReAct
- [ ] Task 12: Integration test

## Wave 3 — Interaction & Delegation

### Phase 2: Structured HITL (0/5)
- [ ] Task 1: ask_clarification tool
- [ ] Task 2: ClarificationMiddleware
- [ ] Task 3: Upgrade interrupt nodes
- [ ] Task 4: Upgrade CLI review command
- [ ] Task 5: Integration test

### Phase 3: Subagent Delegation (0/6)
- [ ] Task 1: SubagentExecutor
- [ ] Task 2: SubagentResult protocol
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
- 测试基线: `.venv/bin/python -m pytest tests/ -q` → `150 passed, 1 skipped`

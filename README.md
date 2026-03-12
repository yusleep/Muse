# Muse

[![CI](https://github.com/yusleep/Muse/actions/workflows/ci.yml/badge.svg)](https://github.com/yusleep/Muse/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-6f42c1)
![FastAPI](https://img.shields.io/badge/Web_UI-FastAPI-009688)
![CLI](https://img.shields.io/badge/Interface-CLI-333333)
![HITL](https://img.shields.io/badge/Human--in--the--Loop-HITL-0ea5e9)
![MCP](https://img.shields.io/badge/Extensions-MCP-10b981)
![Sandbox](https://img.shields.io/badge/Execution-Sandbox-f59e0b)
![Memory](https://img.shields.io/badge/Memory-SQLite-a855f7)
![Export](https://img.shields.io/badge/Export-Markdown%20%7C%20LaTeX%20%7C%20PDF%20%7C%20Overleaf-ef4444)

Muse 是一个面向学术论文 / 毕业论文场景的**多智能体写作框架**。它以 LangGraph 为编排内核，通过 **Zone 调度 + 专家 Agent** 架构，将文献检索、实验复现、图表生成、章节写作、审稿评分、引用核验与终稿导出串成一条**可恢复、可扩展、可审计**的执行链路。

## Why Muse

- **多智能体协作**：5 个专家 Agent（文献、写作、实验、画图、审稿）+ Zone-local Planner 自主调度
- **真能跑完整流程**：覆盖 `research → outline → draft → review → refine → citation → export`
- **支持中途暂停与恢复**：HITL 阶段可人工审核，状态保存在 `runs/<run_id>/`
- **自适应进化**：Knowledge Base + Meta Layer 学习历史运行经验，自动生成改进策略
- **代码与实验**：沙箱执行器支持在隔离环境中运行代码、协议验证、攻击模拟
- **学术画图**：matplotlib/TikZ/表格生成，内置学术风格管理
- **多格式导出**：Markdown、LaTeX（北邮模板，可直接导入 Overleaf）、PDF
- **Web UI 监控**：FastAPI + WebSocket 实时查看运行状态与进化报告

## Quick Start

### 1. 安装

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml，填入 API Key 和模型配置
```

优先级：`CLI 参数 > 环境变量 > config.yaml > 默认值`

最小配置只需：
- `auth.profiles` 中填好 API Key
- `providers` 中启用模型提供方
- `routes.default` 指向一个可用模型

### 3. 验证

```bash
.venv/bin/python -m muse --help
.venv/bin/python -m muse check          # 连通性检查
.venv/bin/python -m pytest tests/ -q    # 711 passed
```

### 4. 运行

```bash
.venv/bin/python -m muse run \
  --topic "多智能体系统在学术写作中的应用" \
  --discipline "Computer Science" \
  --language zh \
  --format-standard "GB/T 7714-2015" \
  --output-format markdown \
  --auto-approve                        # 跳过 HITL 审核
```

运行产物落在 `runs/<run_id>/`。

## Architecture

### 整体架构

```mermaid
flowchart TB
    U[User] --> CLI[muse CLI]
    U --> WEB[Web UI]

    CLI --> RT[Runtime]
    WEB --> RT

    RT --> G[LangGraph Main Graph]
    RT --> MW[Middleware Chain]
    RT --> KB[Knowledge Base]
    RT --> META[Meta Layer]
    RT --> SBX[Sandbox Executor]
    RT --> MEM[SQLite Memory]
    RT --> MCP[MCP Tools]
    RT --> SRCH[Search Services]

    G --> RZ[Research Zone]
    G --> DZ[Drafting Zone]
    G --> REF[Refinement Zone]
    G --> CI[Citation Subgraph]
    G --> CO[Composition Subgraph]
    G --> EX[Export]

    RZ --> LA[Literature Agent]
    DZ --> LA2[Literature Agent]
    DZ --> WA[Writing Agent]
    DZ --> EA[Experiment Agent]
    DZ --> FA[Figure Agent]
    REF --> RA[Review Agent]
    REF --> WA2[Writing Agent]
    REF --> FA2[Figure Agent]

    EX --> OUT[Markdown / LaTeX (Overleaf) / PDF]
```

### 管线流程

```
initialize → research_zone → review_refs(HITL)
  → outline → approve_outline(HITL)
  → drafting_zone → review_draft(HITL)
  → refinement_zone → citation_subgraph → polish → composition_subgraph
  → approve_final(HITL) → export
```

### Zone 调度模型

每个 Zone 拥有独立的 Planner 实例，按当前状态自主调度专家 Agent：

| Zone | Agent 池 | 职责 |
|------|----------|------|
| **research_zone** | LiteratureAgent | 文献检索、综述生成、研究空白分析 |
| **drafting_zone** | LiteratureAgent, WritingAgent, ExperimentAgent, FigureAgent | 章节撰写、实验执行、图表生成 |
| **refinement_zone** | ReviewAgent, WritingAgent, FigureAgent | 审稿评分 → 定向修改 → 再评估循环 |

Planner 在 LLM 失败时自动降级为确定性路由，确保管线不中断。

### 专家 Agent

| Agent | 工具 | 能力 |
|-------|------|------|
| **LiteratureAgent** | `academic_search`, `arxiv_fulltext` | 多源检索、文献分析、综述生成 |
| **WritingAgent** | `formula_check` | 章节生成、已有内容扩写、修订 |
| **ExperimentAgent** | `run_code`, `protocol_model_check`, `attack_scenario_generate` | 沙箱代码执行、协议验证、攻击模拟 |
| **FigureAgent** | `plot_figure`, `render_tikz`, `render_table`, `diagram_generate` | 学术图表生成，支持主动建议 |
| **ReviewAgent** | — | 8 维度评分（novelty/rigor/clarity/...），校准稳定 |

### Knowledge Base + Meta Layer

```
Knowledge Base（4 命名空间）          Meta Layer
┌──────────────────────┐    ┌────────────────────────────┐
│ global_methodology   │    │ 候选策略生成                │
│ user_profile         │───▶│ 验证门（Verification Gate） │
│ run_history          │    │ 熔断器（Circuit Breaker）   │
│ policy_memory        │◀───│ 策略版本化与回滚            │
└──────────────────────┘    └────────────────────────────┘
```

- **Knowledge Base**：SQLite 持久化，按命名空间隔离全局方法论、用户偏好、运行历史、学习策略
- **Meta Layer**：分析审稿评分中的最弱维度，生成改进策略，经验证门通过后写入 policy_memory
- **熔断机制**：策略连续失败时自动熔断，防止无效策略反复执行
- **Provenance**：策略的完整来源追溯，支持审计与回滚

## Web UI

```bash
# 启动 Web 服务
.venv/bin/python -m muse web --port 8000
```

- 运行状态实时查看（WebSocket 推送）
- 进化报告展示（`/runs/{run_id}/evolution`）
- API 接口：运行列表、状态查询、反馈提交

## CLI Commands

| 命令 | 说明 |
|------|------|
| `check` | 检查 LLM / 检索服务连通性 |
| `debug-llm` | LLM 调试探针 |
| `run` | 发起新运行 |
| `resume` | 恢复中断的运行 |
| `review` | 人工审核（approve / 提交反馈） |
| `export` | 导出结果（markdown / latex / pdf；LaTeX 可导入 Overleaf） |

### 人工审核

```bash
.venv/bin/python -m muse review \
  --run-id <run_id> \
  --stage research \
  --approve \
  --comment "继续"
```

### 恢复执行

```bash
.venv/bin/python -m muse resume --run-id <run_id>
```

### 导出

```bash
.venv/bin/python -m muse export --run-id <run_id> --output-format markdown
.venv/bin/python -m muse export --run-id <run_id> --output-format latex
.venv/bin/python -m muse export --run-id <run_id> --output-format pdf
```

导出 LaTeX 后，将输出目录打包上传到 Overleaf 即可在线编译与协作。

## Repo Map

```text
muse/
├── agents/           # 专家 Agent（Literature, Writing, Experiment, Figure, Review）
│   └── tools/        # Agent 工具实现（run_code, plot_figure, render_tikz...）
├── graph/            # LangGraph 主图与子图
│   ├── nodes/        # 节点工厂函数
│   └── subgraphs/    # chapter / citation / composition 子图
├── knowledge/        # Knowledge Base（4 命名空间，SQLite 持久化）
├── meta/             # Meta Layer（策略生成 / 验证门 / 熔断器 / 版本化）
├── web/              # Web UI（FastAPI + WebSocket + 静态页面）
├── sandbox/          # 沙箱执行器（local / Docker）
├── memory/           # SQLite 记忆存储
├── middlewares/       # 中间件链（Logging, Retry, Memory, DanglingToolCall...）
├── services/         # LLM / 检索 / 元数据服务
├── mcp/              # MCP 工具扩展
├── templates/        # 导出模板（LaTeX 北邮模板）
├── cli.py            # CLI 入口
├── runtime.py        # 服务装配与运行时
└── config.py         # 统一配置加载
tests/                # 711 tests（单元 / 集成 / E2E）
runs/                 # 运行产物与 checkpoint
```

## Advanced Configuration

- **统一配置文件**：`config.example.yaml` → `config.yaml`
- **显式配置路径**：`--config /path/to/config.yaml` 或 `MUSE_CONFIG`
- **本地参考资料**：`--refs-dir ./refs` 或 `MUSE_REFS_DIR`
- **MCP 扩展**：`extensions.yaml` / `MUSE_EXTENSIONS_PATH`
- **中间件调优**：`MUSE_MIDDLEWARE_*`
- **导出增强**：本地 `pandoc` + `xelatex` + `latexmk`
- **Docker 沙箱**：安装 Docker 后自动使用容器化执行

## Notes

- CI badge 对应 `.github/workflows/ci.yml`
- 当引用核验发现关键矛盾项时，导出会被阻断，避免错误内容进入终稿
- 沙箱执行默认限制：300s 超时 / 512MB 内存 / 禁止网络

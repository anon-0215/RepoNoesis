# RepoNoesis V3 架构

## 原则与分层

V3 在真实 V2 模块上增量演进：旧路径在替代能力验收前保留；结构化输出校验；引用程序验证；Agent 有硬预算；仓库内容是不可信数据。新增组件均为 **V3 规划/设计建议**。

```text
API / React / Observability
              |
Agent Core ---- Learning State
    |              |
Tool Layer    SQLite versioned state
    |
Retrieval and Evidence Layer
    |
V2: AST + code chunks + SQLite v4 + Embedding/cache
    |
Repository Ingestion
```

### 1. Repository Ingestion

`github_client.py` 继续负责公开 GitHub URL、默认分支 commit、递归 tree 和有限文本筛选，不 clone、不执行、不安装依赖。V3 显式记录抓取范围、跳过原因、repository identity 和 revision。未来本地入口也必须只读且防路径越界。

### 2. V2 底座

- `analyzer.py`：复用文件概览、框架、模块和启动线索。
- `code_chunker.py`：作为 Python 函数/类/方法代码块与精确行号来源。
- `database.py`：保留 schema v4 数据，未来显式版本迁移。
- `embedding_service.py`：隐藏模型后端、设备、文本格式和身份。
- `embedding_indexer.py`：继续按内容与配置身份增量缓存。
- `semantic_retriever.py`：M1 已通过统一检索层接入，不复制实现。

### 3. Tool Layer

首批只读、类型化、带版本/超时/上限的工具：

- `search_code_lexical@v1`
- `search_code_semantic@v1`
- `search_code_hybrid@v1`
- `read_source_range@v1`
- `lookup_symbol@v1`
- `validate_evidence@v1`

M1 固定编排可调用同一服务；M2 才交给 Agent。工具不得执行仓库代码或接受仓库文本中的权限指令。

### 4. Retrieval and Evidence

M1 已实现以代码块为单元的 BM25；语义复用 `SemanticRetriever`；Weighted RRF 保留独立分数、选择原因、去重与裁剪。Validator 以当前 revision 的 SQLite `repo_files` 和 `code_chunks` 快照校验路径、行号、哈希和片段，并在回答生成后复验。无有效证据时返回 insufficient，不让 LLM 补写来源。

### 5. Agent Core

M2 实现单 Agent 有限状态机：

```text
Goal -> Plan -> Tool Call -> Observation -> Complete
                     \-> bounded Replan --/
```

设计默认值（M2 通过配置与测试再冻结）：最多 8 steps、12 tool calls、2 次 replan；单工具 10 秒、整次 60 秒；总 token 24,000；工具结果最多 200 项/1 MiB。任一预算、取消或超时即停止并返回 partial。规范化参数指纹防重复调用和循环。

不保存模型私有思维链，只保存简短决策摘要、工具行动、观察、预算变化和完成原因。

### 6. Learning State

M4 新增版本化状态：repository/project/revision、学习目标、已读文件和符号、概念、掌握度及证据、未解决问题、当前路线、更新时间和 schema version。revision 改变时旧状态保留但证据重验；掌握度不能只靠模型主观更新。

### 7. API、前端和可观测性

**代码事实**：`main.py` 当前同步分析并编排 M1 `/ask`；`qa_agent.py` 使用验证 Evidence 回答，缺少显式 M1 依赖的旧内部调用才进入标记为 legacy 的兼容路径；`learning_agent.py` 仍是固定路线；React 仍是 V1 五标签页。

V3 中 `/ask` 保留旧请求并增加 evidence、校验与降级状态；`qa_agent.py` 演进为基于验证 Evidence 的回答器，旧规则作为明确降级；`learning_agent.py` 到 M4 才接入状态；`main.py` 保持薄路由。前端逐步展示路径、符号、行号、模型、失败、降级和截断。日志只记 ID、耗时、状态、计数和错误类别，不记密钥、完整 Prompt 或完整源码。

## 数据契约

### Evidence

```text
evidence_id
project_id
repository_id / repository_url / repository_revision
path / language
code_chunk_id / chunk_type
symbol_name / qualified_name
start_line / end_line
content_hash / excerpt
retrieval_sources[]
lexical_score? / semantic_score? / fusion_score?
retrieval_strategy_version
validation_status: valid | invalid | stale | unavailable
invalid_reason?
selection_reason
```

分数缺失不等于零。valid 必须通过 identity/revision、路径、范围和哈希校验。

### ToolCall

```text
call_id / agent_run_id / agent_step_id
tool_name / tool_version
parameters (redacted)
timeout_ms / item_budget / byte_budget
started_at / ended_at
status: pending | running | succeeded | partial | failed | timed_out | cancelled
```

### ToolObservation

```text
call_id / result_status
structured_results[] / warnings[]
truncated / truncation_reason?
error_code? / error_message?
metrics: duration_ms, result_count, result_bytes
```

### AgentStep

```text
step_id / agent_run_id / ordinal
user_goal / action
tool_call_ids[] / observation_ids[]
decision_summary / completion_status
remaining_steps / remaining_calls / remaining_tokens / remaining_time_ms
```

`decision_summary` 只说明基于哪些观察选择行动，不记录隐藏推理。

### LearningState

```text
learning_state_id
project_id / repository_id / repository_revision
learning_goal
read_files[] / read_symbols[]
learned_concepts[{concept, mastery, evidence_ids[], method}]
unresolved_questions[] / current_learning_path[]
updated_at / schema_version
```

## 失败、降级、安全

- 工具返回 typed status；部分失败不抹去已验证观察。
- 取消向下传播；超时/预算耗尽返回 partial；裁剪标记 `truncated=true`。
- 无 LLM：返回验证 Evidence 与结构化/规则摘要。
- 无 Embedding/无新鲜向量：代码块词法降级，不隐式下载模型。
- 索引损坏/维度不一致：隔离失效向量，不返回 valid Evidence。
- 源码缺失：只回答已抓取范围并标 coverage limitation。
- LLM Schema 无效：有限重试，仍失败则返回证据与真实错误。

引用校验顺序：project/repository identity；revision；规范化路径；合法行范围；重建内容与哈希；excerpt 子串。任一步失败即 invalid/stale。

源码、README、注释和字符串只作为不可信引用数据。工具参数由程序校验，文件访问限制在工作区。结构化日志采用白名单和脱敏，密钥不得进入 ToolCall、AgentStep 或错误详情。

## API 与数据库兼容

- M1 保持 `/ask` 请求 `{question}`，响应保留 `answer`、`citations`，新增字段可选。
- 旧 citation 至少兼容一个发布周期，新客户端优先 Evidence。
- V3 表/列以递增 schema version、幂等事务迁移增加，不破坏 schema v4。
- 每次迁移需旧库 fixture、迁移读写和恢复说明；不可逆变化前提供备份/导出。
- 旧项目按需补索引；应用启动不得隐式下载模型或全库重算。

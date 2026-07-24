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
- `database.py`：M3 以幂等迁移升级到 schema v5；保留 v4 数据并新增正式关系节点、
  edge 和 index-run 状态。
- `embedding_service.py`：隐藏模型后端、设备、文本格式和身份。
- `embedding_indexer.py`：继续按内容与配置身份增量缓存。
- `semantic_retriever.py`：M1 已通过统一检索层接入，不复制实现。

### 3. Tool Layer

M2/M3 共用静态白名单 `ToolRegistry` 和五个只读、类型化、带版本/协作式超时/上限的工具：

- `search_code@1`
- `lookup_symbol@1`
- `read_source@1`
- `validate_evidence@1`
- `expand_relations@1`

`search_code` 包装 M1 hybrid/lexical 降级与 EvidenceBuilder；`lookup_symbol` 读取
schema v4 `code_chunks` 定义；`read_source` 读取 SQLite `repo_files` 抓取快照；
`validate_evidence` 复用 M1 CitationValidator。输入 schema 禁止额外字段，
project/repository/revision 只从服务器上下文绑定。工具不得执行仓库代码、访问
网络、修改目标仓库或接受仓库文本中的权限指令。

`expand_relations` 只接受当前请求 Evidence 或绑定 revision 的 relation node，
调用独立关系服务执行默认一跳、最大二跳的稳定 BFS；发现的 chunk 仍经统一
EvidenceBuilder 和 CitationValidator，不创建第二套 citation。

### 4. Retrieval and Evidence

M1 已实现以代码块为单元的 BM25；语义复用 `SemanticRetriever`；Weighted RRF 保留独立分数、选择原因、去重与裁剪。Validator 以当前 revision 的 SQLite `repo_files` 和 `code_chunks` 快照校验路径、行号、哈希和片段，并在回答生成后复验。无有效证据时返回 insufficient，不让 LLM 补写来源。

### 5. Agent Core

M2 已实现请求级单 Agent 有限状态机：

```text
Goal -> Plan -> Tool Call -> Observation -> Complete
                     \-> bounded Replan --/
```

冻结默认值为最多 5 steps、8 tool calls、每步 1 call、同工具 3 calls、连续
2 步无进展停止；单工具 15 秒、整次 60 秒；Planner 每步/总输出 512/2,048
估算 token；observation 64 KiB；源码 200 行/32 KiB。任一预算、取消或超时即
停止并复验已有 Evidence。规范化参数指纹防同参重复与 A→B→A 简单循环。

不保存模型私有思维链，只保存简短决策摘要、工具行动、观察、预算变化和完成原因。
同步工具采用协作式 timeout/cancellation，不宣称可抢占 Python 同步操作；不创建
后台线程。正式 `/ask` 默认进入 Agent Core，无 LLM/Planner 失败时回 M1
deterministic fallback。

M3 把新 edge/node/path/chain/Evidence 纳入进展判断和规范化 fingerprint。最终回答
前后都验证 relation chain；关系在生成期间变化时丢弃关系依赖文本。无关系索引时
保持 M2 retrieval-only，不让 `/ask` 整体失败。

### 6. Static Relation Layer

`relation_analysis.py` 从 SQLite snapshot 的 Python AST 抽取 imports、calls、
references 和 defines，状态为 resolved、ambiguous、unresolved、external 或
unsupported，并记录确定性 resolution rule。`relation_graph.py` 负责双向查询、
有界 BFS、请求级 Evidence chain 和关系身份/内容复验。

schema v5 的 `relation_nodes`、`code_relations`、`relation_index_runs` 全部绑定
project/repository revision；node/edge ID 包含内容或 revision 身份。默认 depth 1、
最大 2，且有 seed、neighbor、node、edge、path、Evidence 和字节硬上限。该层不
宣称恢复运行时调用图。

### 7. Learning State

M4 新增版本化状态：repository/project/revision、学习目标、已读文件和符号、概念、掌握度及证据、未解决问题、当前路线、更新时间和 schema version。revision 改变时旧状态保留但证据重验；掌握度不能只靠模型主观更新。

### 8. API、前端和可观测性

**代码事实**：`main.py` 当前同步分析并编排 M2 bounded Agent `/ask`；
`qa_agent.py` 的 `answer_from_evidence()` 是 M1/M2 共用的最终强制校验与回答边界；
无 LLM 时 Agent Core 进入 M1 确定性降级。`learning_agent.py` 仍是固定路线；
React 仍是 V1 五标签页。

`/ask` 保留旧请求和 M1/旧响应字段，M2 新增 agent schema/mode/status/trace/budget；
M3 新增 relation schema、analysis mode、受限 Evidence chain 和 relation summary；
`learning_agent.py` 到 M4 才接入状态；`main.py` 保持薄路由。前端逐步展示路径、
符号、行号、模型、失败、降级和截断。日志只记 ID、耗时、状态、计数和错误类别，
不记密钥、完整 Prompt 或完整源码。

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

### RelationEdge / EvidenceChain

```text
RelationEdge:
edge_id / project_id / repository_revision / relation_type
source_node_id / source path+symbol+line+hash
target_node_id? / target path+symbol+line+hash?
raw_target_name / resolution_status / resolution_rule / language

EvidenceChain:
chain_id / request owner / project / repository_revision
seed Evidence IDs / supporting Evidence IDs
ordered node IDs / ordered edge IDs / relation types
resolution_status / truncated / warnings
```

关系 path 不是 citation；最终 citation 仍只能来自 valid Evidence。

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
- M3 database schema 为 v5，保留并迁移 v4 数据；Evidence/Agent/relation API
  schema 分别独立报告。
- 每次迁移需旧库 fixture、迁移读写和恢复说明；不可逆变化前提供备份/导出。
- 旧项目按需补索引；应用启动不得隐式下载模型或全库重算。

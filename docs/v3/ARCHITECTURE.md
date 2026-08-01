# RepoNoesis V3 架构

## 原则与分层

V3 在真实 V2 模块上增量演进：旧路径在替代能力验收前保留；结构化输出校验；引用程序验证；Agent 有硬预算；仓库内容是不可信数据。本文中 M1—M4 章节描述当前已提交或本轮实现的代码事实，M5 仍是规划。

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

Retrieval v2 Phase 2 仅在请求显式选择 `retrieval_version=v2` 时复用 QueryAnalyzer、dense、
lexical 和 symbol 三源，并以 `weighted_rrf_v2@1` 做 exact chunk identity fusion。Phase 3
继续保持 v1 和 plain v2 冻结；只有请求同时选择 `hierarchy_mode=normalize_v1`，才在 RRF
之后、final top-k 之前执行 `hierarchy_normalization_v1@1`。该层只识别同 project、revision、
normalized path 内的精确 span hierarchy，不使用 relation graph，也不传播或伪造检索分数。
resolver 查询、深度、derived candidate、family occupancy 和最终结果均有硬上限；查询不完整或
metadata 冲突时保留 direct candidates 并记录 warning。最终选中的每个成员仍是 SQLite 中的
真实完整 chunk，继续进入原 Citation/Relation/Evidence 边界。

Retrieval v2 Phase 4 在上述冻结输出之后增加独立的请求级 `relation_mode`。默认值 `off`
完全不调用新 expander；只有 `retrieval_version=v2 + relation_mode=expand_v1` 才执行
`relation_expansion_v1@1` 和 `relation_selection_v1@1`。它只读取 M3 已持久化的
`imports/calls/references/defines` edge，不从同名 symbol、文本、embedding、LLM 或 hierarchy
猜测关系。查询同时绑定 project/revision、edge type、seed node、确定排序和显式 LIMIT；v1
固定为一跳，默认最多 12 seeds、每 seed 8 edges、总计 96 rows、24 unique targets、每 target
8 paths、16 warnings。node 必须通过权威 `code_chunk_id` 和 path/span/hash/qualified name
唯一映射到真实完整 chunk；external、unresolved、ambiguous、stale 或 scope conflict 只进入
内部 audit/warning，不进入 Evidence。

relation path priority 使用独立的 `relation_priority_v1@1`：
`1 / (1 + seed_selection_rank) * relation_type_weight * depth_decay`，不覆盖 raw score、source
rank、RRF contribution、fused score 或 Phase 3 group priority。`relation_selection_v1@1` 在
`top_k >= 3` 时最多分配 3 个且不超过 30% 的 relation slots，并限制每 seed 1 个、每 relation
family 2 个；relation 不足时按冻结顺序回填 direct candidates。direct、hierarchy、relation
provenance 分离，多 seed/edge/path 命中同一 target 时按原 chunk identity 去重但保留有界路径。
选中 relation chunk 仍经 EvidenceBuilder、CitationValidator 和带准确 direction 的
RelationValidator 前后复验。大型 expansion/selection trace 只保留在内部 retrieval audit，
不注入回答正文。

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

Phase 4 的 `relation_mode` 同样进入请求级 ToolContext 与工具指纹，Planner 和 tool arguments
都不能设置或覆盖。显式 `expand_v1` 请求由 `search_code@1` 的 retrieval-time expander 持有
唯一的一套 relation budget；该请求内再次调用 M3 `expand_relations@1` 会被拒绝以避免重复
扩展。`relation_mode=off` 时 M3 tool 的 schema 和行为保持不变。

### 6. Static Relation Layer

`relation_analysis.py` 从 SQLite snapshot 的 Python AST 抽取 imports、calls、
references 和 defines，状态为 resolved、ambiguous、unresolved、external 或
unsupported，并记录确定性 resolution rule。`relation_graph.py` 负责双向查询、
有界 BFS、请求级 Evidence chain 和关系身份/内容复验。

schema v5 的 `relation_nodes`、`code_relations`、`relation_index_runs` 全部绑定
project/repository revision；node/edge ID 包含内容或 revision 身份。默认 depth 1、
最大 2，且有 seed、neighbor、node、edge、path、Evidence 和字节硬上限。该层不
宣称恢复运行时调用图。

Phase 4 不重建或改写上述 graph。它把原 edge 的 source→target 定义为 outgoing view，把同一
edge 的 target→source 定义为 incoming view；inverse view 不创建第二条 edge identity。
relation-derived Evidence chain 固定一跳，并额外校验 seed/target Evidence 与对应 chunk node、
原 edge type、方向、project/revision 和内容身份。

### 7. Learning State

M4 已实现 local-single-user 的结构化学习层。数据库 schema v6 保存 goal、versioned
plan、ordered DAG step、revision-bound target/task、bounded attempt、validated
evaluation、immutable event 和 derived target state。状态只由
verified assessment、explicit self-report、system observation 与 revision
revalidation 的有限事件确定性投影；不读取完整聊天，也不让模型直接写 mastery。

attempt/evaluation/event/state/plan adaptation 在同一 SQLite 写事务中完成；event 有
幂等 identity、单调 event order 和禁止 UPDATE/DELETE 的触发器。revision 变化时按
path/symbol/hash 唯一映射，changed/missing/ambiguous 进入 needs_review 并生成新 plan
version。`learning_validation.py` 在提交前独立重读 event history 并复算物化投影。
正式细节见 `M4_DECISIONS.md`。

### 8. API、前端和可观测性

**代码事实**：`main.py` 当前同步分析并编排 M2—M4 bounded Agent `/ask`；
`qa_agent.py` 的 `answer_from_evidence()` 是 M1/M2 共用的最终强制校验与回答边界；
无 LLM 时 Agent Core 进入 M1 确定性降级。`learning_agent.py` 仍生成兼容固定路线，
正式动态状态由 `learning_service.py` 负责；
React 仍是 V1 五标签页。

`/ask` 保留旧请求和 M1/旧响应字段，M2 新增 agent schema/mode/status/trace/budget；
M3 新增 relation schema、analysis mode、受限 Evidence chain 和 relation summary；
M4 在同一 Registry 新增只读 `get_learning_context@1`，并提供类型化学习 API；
`main.py` 保持薄路由。前端仍未接入 M4 API。前端逐步展示路径、
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

### Learning State and Event

```text
learner / project / repository / revision
goal / plan version / ordered DAG step / target
task / bounded rubric / attempt / validated evaluation
immutable event / provenance / idempotency / event_order
mastery: unseen|introduced|practicing|demonstrated|mastered|needs_review
availability: current|changed|missing|ambiguous|stale
derived projection / update rule version / updated_at
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

## M5 隔离评测平面

M5 是默认不加载的 CLI 评测平面，不新增生产路由或数据库迁移：

```text
versioned dataset -> strict validator -> fixed local snapshot
-> existing analyzer/chunker/relation/embedding index
-> existing retriever or run_bounded_agent
-> existing CitationValidator/RelationValidator
-> deterministic metrics -> atomic checkpoint/result/report
```

fixed 模式直接复用 `LexicalRetriever`、`SemanticRetriever`、`HybridRetriever`、
`EvidenceBuilder` 和 `answer_from_evidence`。Agent 模式向唯一 `ToolRegistry` 传入服务器定义的
受限视图；benchmark 输入不能注册工具或关闭 validator。adaptive sequence 在 run 目录内的
独立 SQLite 中调用正式 `LearningService`，不接触生产 learner。

真实 provider 只通过 CLI `--live` 加环境 gate 启用；正常 FastAPI 启动不会运行 benchmark、
加载 M5 模型、访问外部仓库或写 artifacts。生产 database schema 保持 v6；独立
benchmark/metric schema 均为 1。

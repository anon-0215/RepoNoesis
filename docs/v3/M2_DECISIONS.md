# M2：工具层与有限 Agent Core 决策

## 范围、基线与结论

M2 基于 `v3-agent-development` 的 M1 提交
`08bc459fdd2992be1346a454d7bab20ba77c6dd2` 增量实现。正式 `/ask`
默认进入请求级 Agent Core；无 LLM 或 Planner 决策连续无效时，Agent Core
调用保留的 M1 单次证据问答链路。

M2 没有增加数据库表、列或轨迹持久化，`SCHEMA_VERSION` 继续为 4。它没有实现
调用图、import/依赖图、定义—引用图、跨文件多跳、学习状态、多 Agent、开放插件、
Shell、目标仓库代码执行或代码修改。

## 正式调用链

M1：

```text
/ask
-> answer_question
-> HybridRetriever
-> EvidenceBuilder
-> CitationValidator
-> grounded/deterministic answer
-> CitationValidator final pass
```

M2：

```text
/ask
-> run_bounded_agent
-> bind project + repository + revision from server state
-> structured Planner decision
-> static Tool Registry
-> M1 retrieval/evidence/validator or V2 SQLite AST chunks
-> bounded observation/replan loop
-> answer_from_evidence
-> mandatory CitationValidator pass before and after generation
-> M1-compatible response + M2 audit summary
```

无 LLM：

```text
/ask -> Agent Core -> deterministic M1 search -> mandatory validation -> answer/fallback
```

该路径返回 `agent_mode=deterministic_fallback`，不会使 `/ask` 不可用。

## Tool Registry

实现：`backend/app/services/agent_tools.py`。

注册表是进程内静态白名单，不扫描目录、不动态导入、不支持第三方安装或远程 MCP。
每个 `ToolSpec` 明确声明名称、版本、描述、Pydantic 输入 schema、handler、协作式
超时、结果数量和字节上限。重复注册和未知工具都拒绝；工具列表按名称稳定排序。
所有输入 model 使用 `extra="forbid"`，所以模型附加的 project、revision 或其他
未知字段不会进入 handler。

正式注册四个工具：

| 工具 | 版本 | 真实复用 |
| --- | --- | --- |
| `search_code` | `1` | `HybridRetriever`、`EvidenceBuilder`、请求级 Evidence store |
| `lookup_symbol` | `1` | SQLite schema v4 的 `code_chunks` AST 定义数据 |
| `read_source` | `1` | SQLite `repo_files` 当前抓取快照 |
| `validate_evidence` | `1` | M1 `CitationValidator` |

### `search_code`

接受 query、可选 path/language/symbol 和 `top_k<=20`。project/repository/revision
只从 `ToolContext` 注入。它不复制 BM25、Embedding、RRF 或 Evidence 算法；
Embedding 不可用时沿用 M1 lexical 降级并保留 warning。Observation 只返回
Evidence 身份、位置、检索来源和指标，不重复返回源码 excerpt。

### `lookup_symbol`

从当前 project/revision 的 `code_chunks` 查询 `symbol_name` 和
`qualified_name`，支持 exact、prefix 和受控 fuzzy；支持 path/language 过滤，
按匹配级别、path、start line、qualified name、chunk id 稳定排序。结果携带
symbol kind、行号、revision、hash 和可关联的 chunk identity。它只查定义，
不构建引用或调用关系。

### `read_source`

读取 `repo_files` 中已抓取的只读快照，不访问或执行本机目标仓库。它拒绝绝对路径、
反斜杠、`.`/`..` 和非规范化路径，检查绑定 revision，限制 200 行和 32 KiB
源码内容，并在读取前后比较文件身份 hash。由于事实源是 SQLite 文本快照而不是
文件系统路径，不会跟随符号链接；“符号链接逃逸”在该存储模型中不可发生。
读取结果不是 Evidence，最终回答不能仅凭该 Observation 引用。

### `validate_evidence`

只接受当前 request-owned Evidence ID。伪造 ID、其他请求 ID、过期内容或
project/revision/path/hash/line 不一致均返回结构化 invalid。即使 Agent 未调用
此工具或工具曾返回 valid，最终回答仍由服务器再次调用同一个 Validator。

## 运行时契约

实现：`backend/app/services/agent_contracts.py`。

`ToolCall` 包含 call/step ID、工具名和版本、清理后的参数、timeout、budget、
开始/结束时间和状态。状态支持 pending、running、succeeded、failed、
timed_out、cancelled、rejected。

`ToolObservation` 包含 call ID、状态、内部结构化结果、warnings、truncated、
清理后的 error，以及 duration/result count/output bytes 指标。序列化失败是
结构化 tool failure；结果按 item/byte 上限安全裁剪。

`AgentStep` 包含 step ID、user goal、action、calls、observations、最长 240
字符的 decision summary、completion status 和 remaining budget。API
`agent_trace` 只由 `to_public_dict()` 产生，不返回参数、源码、Planner prompt、
原始模型输出或私有推理。

所有这些对象只存在于当前请求内；M2 不保存完整轨迹。

## Planner 决策

结构化 schema：

```json
{
  "status": "continue",
  "action": "search_code",
  "arguments": {"query": "authenticate_user", "top_k": 5},
  "decision_summary": "先检索最相关的函数定义。"
}
```

`status` 仅允许 `continue`、`answer`、`insufficient_evidence`。额外字段被拒绝；
continue 必须有 action，终止决策不能携带 action/arguments。LLM Planner prompt
使用任务名 `bounded_repository_planner` 和版本 `m2-v1`，明确区分服务器约束、
用户目标与不可信 observation。非 JSON、未知字段或非法状态最多修复一次；
第二次仍失败进入确定性 M1 降级。未知工具和非法工具参数在 Registry 层仍会再次
拒绝，Planner 不能绕过白名单。

当前 OpenAI-compatible provider 不返回统一 usage，Planner token 使用量按输出
字符数除以 4 向上估算，API 标记 `planner_token_enforcement=estimated`。
`LLMClient` 向 provider 发送每步 `max_tokens`；最终生成发送
`max_final_answer_tokens`。这不是 provider tokenizer 的精确计数，属于已记录限制。

## Agent 状态机与终止

实现：`backend/app/services/agent_core.py`。

请求状态跟踪 request ID、绑定身份、goal、step/call 数、deadline、Planner token、
Evidence、symbol/source/validation 进展键、规范化 fingerprint、连续无进展、
warning 和 completion status。每步最多一个工具：

```text
bind -> plan -> validate decision -> execute one tool -> observe
     -> progress/repeat checks -> replan
     -> complete/insufficient/degraded/budget/cancel
     -> mandatory final validation -> constrained answer
```

正式 `/ask` 在 route 入口只创建一个基于 `time.monotonic()` 的请求级绝对
`request_deadline_at`。work/planning cutoff 由该 deadline 减去 final-answer reserve
派生；Planner 首次调用、repair、HTTP retry/backoff、普通工具和检索共享同一 cutoff，
不能逐步重新取得完整预算。final-answer Provider 使用请求剩余总预算，但仍受同一
绝对 request deadline 约束。

任何 step、call、Planner token、work cutoff 或 request deadline 耗尽后不再启动
新的 Planner、repair 或工具。deadline 快失败返回空 `answer/citations/evidence`；
不再在预算失败后构造或持久化“确定性部分答案”。reserve 只是 final answer 的启动
门禁和预算保留，不保证同步 Provider、校验或持久化一定能在 deadline 前完成。

## 默认预算与配置

实现：`AgentLimits`、`get_agent_limits()`、`.env.example`。

| 预算 | 默认/服务器最大值 |
| --- | ---: |
| max agent steps | 5 |
| max tool calls | 8 |
| max calls per step | 1 |
| max same tool calls | 3 |
| max no-progress steps | 2 |
| total deadline | 60,000 ms |
| default tool timeout | 40,000 ms |
| minimum final-answer reserve | 5,000 ms (`AGENT_FINAL_ANSWER_RESERVE_MS`, bounded 100–30,000 ms) |
| search results | 20 |
| observation bytes | 65,536 |
| source lines / bytes | 200 / 32,768 |
| accumulated Evidence excerpt bytes | 49,152 |
| Planner tokens per step / total | 512 / 2,048 |
| final answer tokens | 1,600 |

环境变量可以在安全范围内缩小预算，不能提高服务器最大值；非法、越界或不可解析值
回退到文档默认值。请求 schema 没有增加可提高预算的字段。

## 重复调用、循环与进展

fingerprint 由工具名、规范化参数、服务器绑定 project 和 revision 生成。已成功的
同参调用不再执行；同参失败不能立即原样重试；被拒绝调用仍计入全局 call 预算。
每个工具最多 3 次。因为成功 fingerprint 会被记住，A→B→A 的第三步 A 被拒绝。

进展只承认新的 Evidence ID、symbol chunk identity、源码范围/hash 或
Evidence validation 状态。连续 2 步无新进展即停止；只有仍满足统一成功门禁时才
能返回并持久化答案，否则返回有界安全失败。所有路径仍受全局
step/call/work-cutoff/request-deadline 限制。

## timeout 与 cancellation 的真实边界

同步 SQLite、BM25 和本地 handler 采用调用前后 deadline/cancellation 检查与
耗时复核。单工具自身上限先到返回 canonical `tool_timeout`（HTTP 503）；真实请求
deadline 先到或同边界到达返回 `deadline_exceeded`（HTTP 504）。诊断分别记录
tool deadline overrun 与 request deadline overrun。Python 同步函数无法安全抢占，
所以这仍是 cooperative timeout：只能在启动前阻止、阶段间停止或返回后发现
overrun，不能宣称真正取消。M2 没有创建后台线程、future 或残留 Provider 请求。

`LLMClient` 使用 provider HTTP timeout（最大 45 秒）；Planner/repair 的每个
attempt 与 backoff 都重新计算 work cutoff 的剩余时间，final answer 则重新计算请求
deadline 的剩余时间。Agent Core 提供可测试的 `CancellationToken` 并在 Planner
前后、工具前后检查。当前同步 FastAPI 路由没有低风险接入客户端断开检测，所以
网络断开不能抢占模型加载、Embedding encode、lexical/semantic SQLite 查询、
BM25/评分、symbol/hierarchy/relation expansion 或同步 SQLite 持久化；这是已知限制。
成功路径在 `save_chat_answer()` 调用前立即复核同一 request deadline；已到期则 504
且零写入。同步 SQLite 写入一旦开始仍不能被安全硬抢占，本轮没有把它描述成严格
硬中止。

## Prompt Injection 与安全

Planner prompt 把源码、README、注释、文档、字符串、文件名、符号和 observation
全部标记为不可信。Planner 只看到受限 observation 摘要，不看到 `read_source`
完整内容。Tool schema 和 Registry 再次拒绝 unknown tool、Shell、环境读取、
project/revision 切换、预算字段和校验跳过。工具无网络 handler、无动态 import、
无 `eval`/`exec`/子进程、无写入目标仓库。

日志只记录 request/step/call ID、工具名/版本、状态、耗时、计数、截断、预算和
完成状态；不记录 API key、Authorization、完整 Prompt、原始 Planner 输出或源码。

## 降级

- 无 LLM：`deterministic_fallback`，执行 M1 search/validate/answer。
- Planner 两次 schema 失败：`deterministic_fallback` 并附降级 warning。
- 无 Embedding：M1 lexical code-chunk 检索，`retrieval_mode=lexical`。
- 单工具失败：结构化 Observation；预算允许时继续重规划。
- budget/cancel：停止新 Planner/工具/LLM generation，复验已有 Evidence。
- 无 valid Evidence：固定回答“当前源码证据不足，无法可靠回答。”，无 citation。

## API 与兼容

正式请求仍支持原 `{question}`，没有新增必填字段。M1 与旧字段继续保留：

```text
answer / citations
evidence_schema_version / evidence
grounding_status / retrieval_mode / warnings
```

新增：

```text
agent_schema_version = 1
agent_mode = bounded | deterministic_fallback
agent_status = completed | insufficient_evidence | degraded |
               budget_exhausted | cancelled | failed
agent_trace
budget_usage
```

旧 citation 仍只由最终 valid Evidence 派生。`answer_from_evidence()` 是 M1/M2
共用的最终回答边界，避免 `/ask` 与工具层维护两套校验规则。

所有项目类型在 `save_chat_answer()` 前执行同一失败门禁。deadline、work/reserve、
Provider、Evidence、Citation、Relation、tool timeout 或持久化异常都不会作为成功
`AskResponse` 返回，也不会写入聊天历史；legacy 的正常确定性降级成功仍保持兼容并
只写入一次。终止性 request deadline 高于较早记录的 citation/relation/tool/provider
次级失败，历史阶段计数和有界失败码仍可保留在 diagnostics 中。

## 确定性工程评测

标注：`backend/tests/fixtures/m2_agent_eval.json`；执行：
`backend/tests/test_m2_evaluation.py`。

冻结 14 个场景：4 个一步检索、3 个 symbol/source、2 个证据不足、2 个工具失败/
降级、1 个预算/循环、2 个 Prompt Injection。每条记录 goal、answerable、
fake decisions、允许/禁止工具、预期序列、Evidence、状态和预算。

验收断言：

- 允许工具之外的实际执行 0；
- 目标仓库代码执行 0；
- 超出 step/call 预算 0；
- 最终 Evidence 复验通过率 100%；
- invalid Evidence 进入 citation 0；
- unanswerable 编造事实 0。

这些只是不访问网络、不调用真实 LLM、不下载模型的确定性 fixture 结果，不能外推
为真实模型或任意真实仓库上的总体成功率。M1 的 16 条冻结评测继续单独回归。

## 与准备文档的差异

`ARCHITECTURE.md` 的准备版曾建议 8 steps、12 calls、单工具 10 秒、24,000 token、
1 MiB observation，并列出 lexical/semantic/hybrid 三个独立 search 工具。本次
M2 指令随后冻结为 5 steps、8 calls、15 秒、2,048 Planner token、64 KiB
observation，并要求最小工具名 `search_code`。实际实现按最新冻结决策收敛为一个
包装 M1 hybrid/lexical 降级的 `search_code@1`，避免暴露第二套检索编排。

`ROADMAP.md` 准备版写 Agent “显式启用”和可新增 trace 表；实际验收要求正式
`/ask` 默认进入 Agent Core、schema 原则上保持 v4，因此本次默认启用且不建表。

# M3：关系图扩展与跨文件多跳证据分析决策

## 范围、基线与结论

M3 基于 `v3-agent-development` 的 M2 提交
`739f46540f07e045a71d94a67ed69efe85d23b8d` 增量实现。它只分析服务器已经
持久化的仓库 snapshot，不 clone、不 import、不执行目标代码、不下载依赖，也不把
README、注释、字符串或 LLM 输出当作关系事实。

正式能力限定为 Python 静态关系：

- module/import dependency；
- 保守可解析的 call；
- definition/reference；
- 复用 `code_chunks` 层级的 defines/contains；
- 默认一跳、最大二跳的稳定有界图扩展；
- 请求级 Evidence chain、关系复验和原 CitationValidator 复验。

本里程碑没有实现运行时调用图、完整类型/控制流/数据流、动态派发、通用多语言图、
长期学习状态、前端重构、代码修改、M4 或 M5。

## 数据库 v4 → v5

schema v4 没有 revision 绑定、可事务替换和可索引查询的正式关系存储，目录级
`modules.depends_on_json` 和 `repo_files.imports_json` 也不含符号、行号、解析状态
或稳定 edge identity。因此 M3 升级到数据库 schema v5，而 Evidence schema 和
Agent schema 仍分别为 1。

v5 为 `projects` 增加 `repository_revision`，并新增：

### `relation_nodes`

保存 `node_id`、project/revision、language、node type、path、可空
`code_chunk_id`、symbol/qualified name、行范围和 content hash。文件节点引用
`repo_files` snapshot 的内容身份；符号节点复用 `code_chunks`，不复制第二套源码。

### `code_relations`

保存 `edge_id`、project/revision、relation type、source/target node、path、
可空 chunk、symbol、源码行、raw target、resolution status/rule、language 和两端
content hash。未解析、外部关系的 target 可以为空。

### `relation_index_runs`

以 project/revision 为主键记录 `complete|partial`、成功/失败/不支持文件数、
node/edge 数、受限 warning 和索引时间。没有记录时 `/ask` 明确进入
`retrieval_only`。

实际查询索引：

- revision + node path；
- revision + qualified symbol；
- revision + relation source；
- revision + relation target；
- revision + relation type；
- revision + source/target symbol。

迁移幂等；版本号只在迁移成功后更新。测试覆盖 fresh v5、v4→v5、数据保留、
重复启动和强制迁移失败不伪报 v5。

## 稳定身份

node ID 是以下规范化字段 JSON 的 SHA-256：

```text
project + revision + language + node_type + path
+ qualified_name + start/end line + content_hash
```

edge ID 是：

```text
project + revision + relation_type + source_node_id + source_line
+ target_node_id或null + raw_target_name
+ resolution_status + resolution_rule
```

chain ID 还绑定 request owner、project/revision、排序后的 seed/supporting Evidence
以及有序 node/edge 序列。重复索引得到相同 ID；内容或 revision 改变会改变身份。
最终 validator 会重新计算 node、edge 和 chain ID，拒绝伪造对象。

## Python AST 抽取与解析

实现：`backend/app/services/relation_analysis.py`。

### import

支持 `import a`、`import a.b as alias`、`from a import b`、相对和多级相对 import。
路径按 Python 模块名映射；`__init__.py` 表示 package，`src/`、`lib/` 只增加一个
可审计候选别名。同名候选不任选一个：

- 唯一内部模块/符号：`resolved`；
- 多个内部候选：`ambiguous`；
- 内部前缀存在但目标缺失或相对越界：`unresolved`；
- 不属于抓取 snapshot 的顶层依赖：`external`。

外部节点没有可遍历源码，不访问第三方依赖。

### call

解析优先级为：

1. 最近词法作用域定义；
2. 当前模块定义；
3. 显式 import symbol/alias；
4. module alias attribute；
5. 当前类的 `self.method()` / `cls.method()`；
6. 唯一明确类名的 `Class.method()`。

参数和局部赋值会遮蔽外层同名定义。任意对象 attribute、`getattr()`、链式动态调用、
多候选、`eval`/`exec` 生成目标都不会被强行唯一解析。它们记录为
`ambiguous`、`unresolved` 或 `external`，不能用于肯定的运行时断言。

### definition/reference

`defines` 复用现有 chunk qualified name、parent symbol 和行号：

- file defines top-level class/function；
- class/function defines method/nested symbol。

`references` 只解析当前词法作用域、模块定义和显式 import binding。它不会按全仓库
同名字符串猜测；call-site 单独以 `calls` 表达。built-in 标为 external，参数/局部
遮蔽和无法绑定的 Name/Attribute 标为 unresolved。

### 状态与依据

状态固定为 `resolved`、`ambiguous`、`unresolved`、`external`、`unsupported`。
依据是可复现规则，例如 `same_local_scope`、`same_module`、`explicit_import`、
`relative_import`、`import_alias`、`module_alias`、`self_method`、
`class_qualified`、`dynamic_attribute`；没有模型生成的 confidence。

## 正式索引生命周期

`main.analyze_project` 的真实顺序：

```text
fetch snapshot
-> analyze + extract code_chunks
-> save repo_files/code_chunks
-> index_project_relations（独立事务）
-> optional Embedding index
```

关系抽取只读取 SQLite snapshot。关系阶段失败时 M1/M2 数据仍保留，并返回结构化
warning；关系表不会伪装完成。重建采用当前 revision replace-all，先去重再写入。
`save_analysis` 或直接 code-chunk 替换会先使旧关系/index run 失效，防止删除文件、
删除符号或内容变化后留下 ghost edge。单文件 SyntaxError 产生 partial coverage，
其 parser error 不是 Evidence。

## 图查询与预算

实现：`backend/app/services/relation_graph.py`。

使用稳定 BFS，支持 outbound、inbound、both 和关系类型白名单。node/edge/path
去重；path 不重复进入已有节点，因此自环和 A→B→A 会终止。默认与硬上限：

| 项目 | 默认 | 硬上限 |
| --- | ---: | ---: |
| depth | 1 | 2 |
| seed nodes | 8 | 8 |
| neighbors per node | 20 | 20 |
| nodes | 64 | 64 |
| edges | 128 | 128 |
| paths | 24 | 24 |
| observation bytes | 65,536 | 65,536 |
| relation Evidence | 16 | 16 |

模型不能提高硬上限；Pydantic、tool handler 和底层 traversal 都重新限制。
超限返回 `truncated=true` 和 warning，不产生半条 edge。图查询仍计入 M2 的
5 steps、8 calls、60 秒总 deadline 和单工具协作式 timeout。

## `expand_relations@1`

继续使用 M2 `ToolRegistry`，没有第二套 Registry。

输入：

```text
seed_evidence_ids[] / seed_symbol_ids[]（至少一种）
relation_types[]
direction: outbound | inbound | both
max_depth
per_node_limit
```

Evidence 必须属于当前 request；symbol node 必须属于服务器绑定的
project/revision。额外字段、绝对身份字段、未知类型/方向、超限和伪造 seed 都拒绝。
Evidence seed 在 import 查询时还会安全加入其 owning file node，才能从函数证据
追到文件级 import。

Observation 只返回受限 node/edge/path/chain 摘要、supporting Evidence ID、
解析统计、truncation/warning 和 metrics；不返回 SQL 主键、绝对路径、Planner
原文、私有思维或源码正文。

## Evidence chain 与最终校验

关系节点的 chunk 通过 `EvidenceBuilder.build_from_code_chunks()` 转换成原
Evidence schema，注册到当前 `EvidenceStore`，随后仍由 CitationValidator 校验。
关系 path 本身不是 citation。

请求级 chain 保存：

```text
chain_id / owner / project / revision
seed Evidence IDs / supporting Evidence IDs
ordered node IDs / ordered edge IDs / relation types
resolution status / truncated / warnings
```

最终回答前后都执行：

1. CitationValidator 校验所有 seed/supporting Evidence；
2. RelationValidator 校验 owner、project/revision、chain identity；
3. 重读 node/edge 并重算身份；
4. 校验 node content hash 与当前 repo_files/code_chunks；
5. 校验有序 edge endpoint；
6. 丢弃失效 chain 和仅由失效 chain 引入的 Evidence；
7. `answer_from_evidence` 再做生成前后 CitationValidator。

如果关系在生成期间变化，丢弃关系依赖的生成文本，使用仍有效 Evidence 重新生成
确定性回答。ambiguous chain 保留其状态，但回答不得把它表述为唯一运行时目标。

## Agent 与 `/ask`

正式链路：

```text
/ask
-> run_bounded_agent
-> server-bound project/repository/revision
-> Planner m3-v1
-> search_code / lookup_symbol
-> expand_relations
-> optional read_source
-> request Evidence chain
-> RelationValidator
-> CitationValidator
-> answer_from_evidence
-> second relation/evidence validation
-> compatible response
```

Agent 新进展包括新 edge、node、path、chain、Evidence 和验证状态；重复排序、warning
文本变化、相同未解析结果不算进展。关系 fingerprint 包含 tool/version、规范化 seeds、
排序后的 relation types、direction、有效 depth/limit 和 project/revision，因此参数
顺序或空白不能绕过重复与 A→B→A 防护。

## API 与兼容

原请求没有新增必填字段。M1/M2 字段和旧 citations 保留。新增：

```text
relation_schema_version = 1
analysis_mode = retrieval_only | relation_expanded
evidence_chains[]
relation_summary
```

`relation_summary` 返回 seed、resolved/ambiguous/unresolved/external edge、
validated chain、truncated 和受限 warning 统计。health 单独报告数据库 schema 5；
数据库、Evidence、Agent 和 relation API 版本不混用。

无关系索引返回 retrieval-only；无 LLM 保持 M2 deterministic fallback，不声称执行
多跳；无 Embedding 保持 M1 lexical 降级。

## 安全边界

AST 解析只构造数据，不执行节点行为。源码、README、docstring、字符串、文件名、
模块和符号均是不可信数据，不能修改 seed、project/revision、relation type、预算、
validator 或工具白名单。Registry 没有 Shell、动态 import、网络和写仓库 handler。
常规测试使用临时 SQLite、fake planner/LLM/Embedding，不访问 GitHub、不下载模型、
不 import fixture 仓库。

## 工程评测

冻结文件：`backend/tests/fixtures/m3_relation_eval.json`；执行：
`backend/tests/test_m3_evaluation.py`。

20 条场景分类：4 import、4 call、3 definition/reference、4 跨文件多跳、
2 ambiguous/unresolved、1 retrieval-only、2 Prompt Injection/预算。

确定性 fixture 达到：

- 禁止工具实际执行、目标代码执行、跨 revision edge、预算越界：0；
- exact/resolvable call edge precision/recall：100% / 100%；
- ambiguous 错误唯一解析：0；
- bounded gold path 找回率：100%；
- 最终 Evidence 校验和有效 chain 校验：100%；
- invalid Evidence/chain 进入 citation：0；
- M1 Hit@5 = 100%，mock MRR@10 >= 0.80；
- M2 14 条冻结场景保持通过。

这些指标只代表本地确定性工程 fixture，不代表真实 Python 生态运行时调用图、
真实 LLM Agent 成功率或任意真实仓库总体质量。

## 已知限制与剩余风险

- 静态解析不能证明运行时分支或动态派发；
- import root 只支持保守的仓库路径和常见 `src/lib` layout，复杂 packaging 可能
  unresolved/ambiguous；
- 当前作用域模型不是完整 Python 编译器符号表；
- 同步 SQLite/AST timeout 是协作式，不能抢占正在执行的 Python 同步指令；
- file-level import 到 symbol-level call 的组合依赖有界 seed 扩展，不是通用知识图谱；
- 正式真实 LLM、真实 BGE-M3、网络仓库和大规模性能评测未执行，留到 M5。

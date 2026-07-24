# RepoNoesis V3 Schema 与 API

## 版本边界

| 契约 | 当前版本 |
| --- | ---: |
| SQLite database schema | 5 |
| Evidence schema | 1 |
| Agent schema | 1 |
| Relation API schema | 1 |

版本互相独立，不能用 database v5 代替 Evidence/Agent/relation API 版本。

## SQLite v5

v5 保留 v4 的 `projects`、`repo_files`、`modules`、`learning_steps`、
`chat_answers`、`code_chunks` 和 `code_chunk_embeddings` 数据。`projects` 新增
`repository_revision`；新增 `relation_nodes`、`code_relations` 和
`relation_index_runs`。正式字段、稳定身份、索引、迁移与生命周期见
`M3_DECISIONS.md`。

## `/api/projects/{project_id}/ask`

请求兼容：

```json
{"question": "函数 a 跨文件调用了什么？"}
```

可选的 M1 filter 仍为 `path`、`language`、`symbol`、`evidence_count`；没有新增
关系预算或 project/revision 参数。

响应保留：

```text
answer / citations
evidence_schema_version / evidence / grounding_status / retrieval_mode / warnings
agent_schema_version / agent_mode / agent_status / agent_trace / budget_usage
```

M3 新增：

```text
relation_schema_version = 1
analysis_mode = retrieval_only | relation_expanded
evidence_chains[]
relation_summary
```

`evidence_chains[]` 只公开 chain ID、relation types、path length、seed/supporting
Evidence IDs、resolution status 和 truncated。它不公开完整图、内部 SQL、绝对路径、
源码、Planner prompt/output 或其他请求数据。

## analyze 与 health

`POST /api/projects/analyze` 可返回受限 `relation_index` 摘要。关系索引失败时分析数据
仍保存，摘要为 warning。

`GET /api/health` 的 `database_schema_version=5` 只表示 SQLite schema，不表示
关系覆盖完整或真实运行时行为。

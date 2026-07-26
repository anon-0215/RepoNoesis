# RepoNoesis V3 Schema 与 API

## 版本边界

| 契约 | 当前版本 |
| --- | ---: |
| SQLite database schema | 6 |
| Evidence schema | 1 |
| Agent schema | 1 |
| Relation API schema | 1 |
| Learning API schema | 1 |

版本互相独立。database v6 不代表 Evidence、Agent、relation 或 learning API 的版本。

## SQLite v6

v6 保留 v5 的全部 M1/M2/M3 表和数据，新增 `learner_profiles`、`learning_goals`、
`learning_targets`、`learning_plans`、`learning_plan_steps`、
`learning_step_prerequisites`、`learning_tasks`、`learning_task_evidence`、
`learning_rubric_criteria`、`learning_attempts`、`learning_evaluations`、
`learning_events` 和 `learner_target_states`。正式字段、索引、触发器和状态规则见
`M4_DECISIONS.md`。

## `/api/projects/{project_id}/ask`

旧请求保持兼容：

```json
{"question": "函数 a 跨文件调用了什么？"}
```

M1 filter `path`、`language`、`symbol`、`evidence_count` 仍是可选字段；不接受 learner、
project、repository、revision、图预算或学习预算。

响应继续保留 M1/M2/M3 字段，并新增：

```text
learning_schema_version = 1
learning_mode = disabled | profiled | adaptive | degraded
learning_context_summary {
  goal_id, plan_version, current_step, explanation_depth,
  demonstrated_target_count, mastered_target_count, needs_review_count
}
learning_plan_summary {
  plan_id, version, status, current_step_id,
  completed_step_count, remaining_step_count, adapted, adaptation_reason
}
recommended_next_action
learning_warnings[]
```

学习摘要不是源码 Evidence。最终源码事实仍必须通过 RelationValidator 和
CitationValidator；旧 revision Evidence 不能进入当前 response citation。

## Learning API

所有 learner identity 均由服务器绑定为 local single user。mutation request 使用
`extra=forbid`，并带 8—120 字符的 client idempotency key。

| 方法 | 路径 | 能力 |
| --- | --- | --- |
| POST/GET | `/api/projects/{project_id}/learning/goals` | 创建或列出 goal |
| PATCH | `/api/projects/{project_id}/learning/goals/{goal_id}` | active/completed/cancelled |
| POST | `/api/projects/{project_id}/learning/plans` | 按 expected version 创建 plan |
| GET | `/api/projects/{project_id}/learning/plans/current` | 当前 plan/version/steps |
| GET | `/api/projects/{project_id}/learning/state` | 受限 target state |
| POST | `/api/projects/{project_id}/learning/tasks` | 创建 Evidence/rubric 绑定 task |
| GET | `/api/projects/{project_id}/learning/tasks/{task_id}` | 读取受限 task |
| POST | `/api/projects/{project_id}/learning/tasks/{task_id}/attempts` | 提交 bounded attempt |
| POST | `/api/projects/{project_id}/learning/self-reports` | explicit self-report |
| POST | `/api/projects/{project_id}/learning/events/{event_id}/corrections` | append correction event |
| GET | `/api/projects/{project_id}/learning/next-action` | 推荐下一动作 |
| POST | `/api/projects/{project_id}/learning/revalidate` | 当前 revision 重验证 |

API 不返回 learner ID、完整 event log、全部答案、raw evaluator/Planner output、聊天、
私有思维、SQL 行主键、绝对路径、完整源码、其他 learner 数据或无效 Evidence。

## `get_learning_context@1`

输入为空 JSON `{}`，未知字段拒绝。输出最多 16 states、8 recent verified outcomes、
12 plan steps 和 16,384 bytes；每次 Agent run 最多调用一次并计入原 step/call/time/
observation budget。输出只用于教学深度和下一步建议。

## analyze 与 health

`POST /api/projects/analyze` 的 M3 relation index 行为不变。`GET /api/health` 的
`database_schema_version=6` 只表示 SQLite schema，不表示 relation coverage、learning
profile 完整、真实 evaluator 质量或用户教学效果。

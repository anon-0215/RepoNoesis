# 源鉴 RepoNoesis V3·LP2.3｜跨版本学习连续性契约

> 状态：离线实现冻结契约。本文只定义 LP2.3 的跨 revision 学习连续性、目标映射、
> 保守状态派生和最小 UI；不修改 M5/Phase 6，也不代表真实用户教学效果验收。

## 1. 身份与所有权

- `learner_id` 继续使用稳定的 `learner-local-single-user-v1`。LP2.3 不引入用户、权限或
  浏览器生成的 learner identity。
- `workspace_id` 是跨 revision 的长期仓库工作区身份；`project_id` 仍是不可变的
  revision snapshot 身份。历史 project 被显式访问时只返回该 project 自己的 M4 历史。
- P2.2 激活 B 的同一事务，以旧 active project A、新 project B、新 activation version、
  learner 和 `learning-continuity-v1@1` 保证一条唯一 `pending` transition。客户端不能指定
  A、B、activation version、learner 或内部外键。
- goal 是可跨 revision 延续的用户意图。LP2.3 为 B 建立带 lineage 的派生 goal 和最小
  revision plan；step 只在其 target 有安全映射时进入新 plan，受修改的 step 进入
  `needs_review`，删除或无法映射的 step 只保留在 A 的历史 plan 与影响摘要中。
- target、task、attempt、evaluation、Evidence、Citation、Relation 和旧 learning event
  严格绑定各自 project/revision。task 不跨 revision 复制；B 的新 task 必须重新绑定 B 的
  Evidence。
- 旧 `learner_target_states` 不更新。B 的状态只能由新的
  `continuity_state_derived/revision_continuity` 系统事件投影，并通过 mapping row 追溯到
  transition、旧 target 和旧 state；该事件没有 attempt/evaluation 外键，不冒充用户行为。

## 2. Transition 状态机

```text
pending -> running -> succeeded
                   -> failed --显式 retry--> pending
```

- 创建、claim、失败、retry 和发布均使用 SQLite `BEGIN IMMEDIATE` 事务。
- 相同 workspace/source/target/learner/config 及同一 activation version 由唯一约束幂等。
- 遗留的 `pending` 或 `running` 在后端重启时确定性转为
  `failed/continuity_interrupted`；不会被当作成功，且只能显式 retry。
- mapping 先在内存中确定性计算，随后在一个发布事务内写入 target、mapping、派生 event、
  state、goal/plan lineage 和最终统计。失败时不暴露半完成 mapping 或状态。
- 发布事务再次校验 target project 仍是该 workspace 当前 active project，activation version
  仍与 transition 一致。旧 activation 的延迟工作不能污染更新后的 active snapshot。
- continuity 失败不会回滚已经正确激活的代码 snapshot。普通 `/ask` 继续使用 B；B 的
  learning context 在安全发布前保持 disabled/profiled，不会偷用 A 的 mastery。

## 3. 确定性目标映射

映射只读取 A/B 已持久化的 manifest、repo file、AST chunk 与冻结身份；不调用 LLM、
Embedding、Provider、网络或目标仓库代码。

| 状态 | 确定性条件 | 状态规则 |
| --- | --- | --- |
| `unchanged_exact` | 同一规范化 path/qualified name，且内容 hash 严格一致；行号可变化。 | 可继承旧 target state。 |
| `renamed_exact` | 旧内容 hash 在 B 中只有一个候选，内容严格一致且不存在多对一碰撞。 | 可继承，并在 mapping 中保留旧/新 path 与 qualified name。 |
| `modified` | 同一代码身份仍唯一可定位但内容变化；或同文件唯一 AST 结构目标仅名称/实现变化。 | 一律派生 `needs_review`，不得保持 mastered。 |
| `deleted` | B 中没有同身份或严格内容候选。 | B 不创建有效 target/state；A 历史保留。 |
| `ambiguous` | 多个严格候选、多个结构候选，或多个旧 target 争用同一新 target。 | 不创建 B state，不继承 mastery。 |
| `unmapped` | bounded concept 等缺少可证明的代码身份。 | 不继承 mastery。 |
| `incompatible` | A/B parser/chunker identity 不一致。 | 所有受影响代码 target 拒绝继承。 |

文件 rename 只有唯一内容 hash 才成立。符号改名导致内容身份变化时最多分类为
`modified`；不会仅凭名称、路径、行号、显示文本或相似度升级为 exact。新增代码 target
不会生成历史 mastery、attempt 或 evaluation。映射按 source target ID 排序，结果与输入
顺序无关；目标唯一索引禁止两个旧 target 同时发布到同一新 target。

## 4. Mastery 与学习计划

- `unchanged_exact` / `renamed_exact` 可保守继承 `unseen`、`introduced`、`practicing`、
  `demonstrated`、`mastered` 或既有 `needs_review`，并保留旧 verified/qualifying count 作为
  系统派生 provenance；这些计数不是 B 的新 attempt。
- `modified` 始终得到 `needs_review` 和 `revision_modified`；用户必须在 B 上完成新的真实
  task/attempt/evaluation 才能重新证明 mastery。
- `deleted`、`ambiguous`、`unmapped`、`incompatible` 不产生 B target state。
- 派生 plan 不复制 task、rubric、task Evidence、attempt 或 evaluation。严格等价 step 可保留
  进度；modified step 进入 `needs_review`；无安全 target 的 step 不进入 B 的有效 plan。
- B 的 `get_learning_context@1`、`/ask` learning summary 和 next action 只读取 B 的派生/后续
  真实记录。存在 `needs_review` 时优先返回 foundational review。

## 5. Schema v11

schema v10 additive 升级为 v11，不删除、改名或重写旧表：

- `learning_continuity_transitions`：workspace、A/B project/revision、activation version、
  learner、mapping config、状态、稳定错误码、计数和时间；两组唯一约束保证请求与 activation
  幂等。
- `learning_continuity_mappings`：每个旧 target 的分类、规则、旧/新 target、旧/新内容 hash、
  相对 path、qualified name、源/派生 mastery 与 review reason；部分唯一索引禁止多对一发布。
- `learning_continuity_goal_lineage`：source/target goal 与可选 source/target plan 的 additive
  lineage，状态为 `carried`、`needs_review` 或 `history_only`。
- 新状态仍使用 M4 的 `learning_targets`、`learning_events` 和 `learner_target_states`，但由明确
  continuity event 产生；`trg_learning_events_no_update/no_delete` 保持不变。

统一 migration 事务覆盖空库、v10 非空 M4 数据、重复启动和故障回滚。历史 project、goal、
plan、step、task、attempt、evaluation、event、Evidence、Citation 和 Relation 继续可读。

## 6. API 与前端

- `GET /api/workspaces/{workspace_id}/learning-continuity`
- `GET /api/workspaces/{workspace_id}/learning-continuity/{transition_id}`
- `GET /api/workspaces/{workspace_id}/learning-continuity/{transition_id}/targets`
- `POST /api/workspaces/{workspace_id}/learning-continuity/{transition_id}/retry`

workspace 详情同时返回当前 activation 的 continuity 安全摘要，支持页面刷新和后端重启恢复。
响应只包含 revision、状态、计数、相对 target identity、稳定错误码和有限说明；不返回源码、
Evidence 正文、Prompt、messages、reasoning、Embedding、凭据、完整本地绝对路径或内部
source/target project/learner 外键。读取 API 不触发 Provider、Embedding 或仓库分析。

前端保持 P2.1 项目库和 P2.2 检查更新入口，显示 pending/running/succeeded/failed、保留、
需复习、历史删除和未继承计数。失败时只提供显式 retry；普通代码问答不被阻断，学习页明确
提示旧 mastery 没有被当作新 revision 的用户证明。

## 7. 验证边界

LP2.3 默认验证只使用临时 SQLite、临时/fake repository、fake Embedding 和确定性 evaluator。
没有运行网络、真实 BGE-M3、真实 Provider、Gate A/B/C、P2 live Gate 或 M5 live pilot。
因此结果只证明离线实现、迁移、映射、幂等、失败恢复和兼容回归，不证明真实远程仓库的长期
学习效果、真实用户收益或生产规模可靠性。

2026-08-08 封存记录：测试先行红测命令
`python -B -m unittest tests.test_learning_continuity` 因生产模块尚不存在得到
`Ran 1 test / FAILED (errors=1)`；实现后的同一模块为 `Ran 12 tests / OK`。完整后端回归为
`Ran 557 tests / OK`；前端 Vitest `9 passed`，TypeScript `tsc --noEmit` 与 Vite production
build 通过。OpenAPI continuity 路由、无内部身份 request body、安全响应字段、稳定
404/409/422 和显式 retry 均由离线测试覆盖。

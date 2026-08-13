# M4：结构化学习者状态与自适应学习闭环决策

## 范围、基线与结论

M4 基于 `v3-agent-development` 的 M3 提交
`3bf29b54784544008c625f33f98c231aad1cd629` 增量实现。开发前独立确认分支、HEAD、
干净工作区、worktree 和完整后端基线；基线为 186/186。

本阶段实现的是本地单用户、结构化、可审计、跨进程的学习闭环：

```text
goal
-> versioned plan / ordered DAG steps
-> revision-bound task / rubric / learning Evidence
-> bounded answer / structured evaluator
-> immutable event
-> deterministic target-state projection
-> new plan version / next action
-> later /ask and service restart reuse
```

M4 没有实现完整前端学习中心、OAuth/RBAC、多租户、代码执行、补丁运行、模型训练、
真实用户实验、Fixed RAG 对比、正式消融或 M5。旧 `learning_steps` 和五阶段页面仍保留。

## 身份与 local single-user 边界

系统此前没有可信用户或会话身份。因此 M4 明确采用 `local_single_user`，服务器创建
稳定 profile `learner-local-single-user-v1`。API 请求、Planner、源码、README、用户
回答和 self-report 均不能提交或切换 learner ID。

身份链如下：

```text
learner_id (server only)
+ project_id (本地分析实例)
+ repository_id (owner/repo)
+ repository_revision
+ stable domain ID
```

goal、plan、step、task、attempt、evaluation、event 和 state 均校验 learner/project；
涉及源码的 target、task 和 learning Evidence 还绑定 revision/content hash。该边界不
伪装成云端多租户安全模型。

## 数据库 v5 → v6

v5 只有固定 `learning_steps`，没有 owner、正式 goal/plan version、task/attempt、
不可变 event、投影 state、幂等键或事务化路线适配，因此升级到数据库 schema v6。
Evidence、Agent、Relation API 和 Learning API schema 仍各自为 1。

v6 新增：

| 表 | 作用 |
| --- | --- |
| `learner_profiles` | 稳定 local learner 和启停状态 |
| `learning_goals` | 有限 goal type、文本、project/repository/revision、幂等身份 |
| `learning_targets` | repository/module/file/symbol/concept 的观察身份与 availability |
| `learning_plans` | goal 下的不可静默覆盖 plan version 和 supersession |
| `learning_plan_steps` | 有序、类型化、状态化步骤 |
| `learning_step_prerequisites` | 规范化 DAG 前置边 |
| `learning_tasks` | owner/revision/plan/step/target/rubric 绑定 |
| `learning_task_evidence` | 指向现有 `code_chunks` 的稳定 learning Evidence 身份 |
| `learning_rubric_criteria` | 有限 criterion、权重、critical 与 Evidence 白名单 |
| `learning_attempts` | 最长 12,000 字符的显式任务回答和幂等键 |
| `learning_evaluations` | 只保存校验后的有限 breakdown，不保存原始模型输出 |
| `learning_events` | append-only 审计来源和每 target 单调 `event_order` |
| `learner_target_states` | 可由 event history 重建的物化投影 |

实际索引覆盖 active goal、goal/version、plan/order、learner/target、revision/target、
task/attempt、event owner/time、event target/order 和 review/stale state。数据库触发器
拒绝 `learning_events` 的 UPDATE/DELETE。迁移使用原 `Database.connect()` 与同一事务，
版本号只在所有 v6 DDL 成功后更新；重复初始化幂等。

`learning_task_evidence.code_chunk_id` 在源码块被替换时设为 NULL，而不删除历史任务；
后续 validation 会把它识别为 stale。源码、chunk、relation 和 citation 没有第二份副本。

## 记忆边界与隐私

项目事实继续只来自 `repo_files`、`code_chunks`、relation tables 和请求级 Evidence。
学习记忆只保存未来动作需要的结构化 goal、plan、task、受限回答、validated evaluation、
event、state 和 next action。

不把以下内容写入长期学习记忆：完整聊天、原始 Planner prompt/output、原始 evaluator
output、Agent 私有思维、整段重复源码、无限历史 trace 或向量化对话。旧 `chat_answers`
仍是兼容的 V1 表，但不参与 M4 状态投影或 `get_learning_context@1`。

## 正式对象与稳定身份

所有公开领域 ID 都是由服务器绑定字段和幂等 identity 计算的稳定 SHA-256 前缀 ID；
不公开内部 SQLite 行号。learning Evidence ID 绑定 project/revision/path/symbol/line/hash，
表内仍只引用原 `code_chunks`。

源码 target 记录 observed revision、规范化 path、qualified name、content hash 和可空
chunk reference。跨 revision 只在程序证明唯一映射时保留状态；不宣称同路径或同名
symbol 天然是同一实体。

## 学习状态和确定性投影

掌握状态为：

```text
unseen -> introduced -> practicing -> demonstrated -> mastered
                                      \-> needs_review
```

availability 独立为 `current | changed | missing | ambiguous | stale`，不与掌握等级混用。
更新规则版本是 `state_update_rule_version=1`：

- 普通 `/ask`、检索、阅读不写 event，不提升状态；
- explicit self-report 只允许 `unseen -> introduced`；
- 首次有效 attempt 可进入 practicing；
- 一个通过的、结构化且 Evidence 有效的不同 task 进入 demonstrated；
- mastered 至少需要两个不同 task ID 的 pass，且至少一个为解释、静态追踪、关系、
  影响分析或事实/推断/未知边界任务；
- 后续有效 fail 保留成功 event，并进入 needs_review；
- ungradable 不写权威 event，不更新 state；
- correction 新增 `evaluation_corrected` event，原 event 不修改；
- revision changed/missing/ambiguous 强制 needs_review。

投影严格按事务内生成的每 target 单调 `event_order` 重放，避免同一秒事件按哈希 ID
误排序。独立的 `learning_validation.py::LearningStateValidator` 会重新读取 target、event
history 和物化 state，按纯投影函数重算并逐字段比较；stale target 显示 mastered 或
任何投影差异都会使事务失败。测试还会故意破坏物化 state 验证拒绝，并用新
Database/Service 实例验证恢复。

## task、rubric 与 evaluator 契约

支持不执行代码的 symbol 解释、静态关系追踪、定义定位、模块关系说明、change impact
和“源码事实/静态推断/未知运行时行为”区分任务。

任务只能由当前 active plan/version/step 创建。服务器从 target 和当前 relation index
解析最多 8 条 learning Evidence；调用方不能提交 code chunk SQL ID。rubric 最多 8 条，
criterion ID 唯一，类型白名单、权重上限、critical 和 supporting Evidence 集合均验证。

Evaluator 使用 `bounded_learning_evaluator / m4-v1`，只允许 schema v1：pass、partial、
fail、ungradable、原 rubric criterion、允许 Evidence、有限反馈/缺失概念/误区/warning。
程序重新计算 verdict：critical 未满足不能 pass；通过的仓库 criterion 必须引用允许的
Evidence；evaluator 不能提交 state、mastered、event 或 plan。无 LLM 返回 ungradable。

## 不可变 event、事务、幂等与并发

graded attempt 的以下步骤位于一个 `BEGIN IMMEDIATE` 事务：

```text
revalidate task / current plan / revision / Evidence
-> insert attempt
-> insert validated evaluation
-> append immutable learning event
-> rebuild and upsert target state
-> create adapted plan version
-> supersede old plan
```

Evaluator 在事务外运行，写事务开始后会再次校验 task/revision/Evidence，覆盖
“evaluation 后 Evidence 失效”的竞态。任一步异常全部回滚。同一 task/idempotency key
返回原 attempt；SQLite 写锁、唯一键和二次查询保证并发重复提交只产生一个 event。
旧 plan version task 不能覆盖新 plan。

## 路线适配

- pass：完成当前 step，激活下一个可用 step；mastered 时跳过同 target 的重复入门阅读；
- partial：保留未满足 criterion，新增/激活 review step；
- fail：保留目标和历史成功，新增基础 Evidence remediation；
- correction：append correction event 后重建状态；
- revision unchanged：建立绑定新 revision 的 plan version；
- revision changed/missing/ambiguous：旧 step 进入 needs_review/invalid，并插入 review；
- 每个 event/revision 使用唯一 adaptation key，重复处理不增长 plan version。

计划输入经 owner、target、revision、步骤数、前置边、DAG、稳定排序和 optimistic version
校验。最大 20 steps、40 prerequisite edges、每 event 最多一次适配。

## revision 重验证

`revalidate_project()` 对当前 project snapshot 执行：

- canonical target 唯一且 hash 未变：current，保留 verified state并记录 event；
- canonical target 存在但 hash 改变：changed + needs_review；
- canonical identity 消失但相同 hash 在当前 revision 唯一：允许唯一重命名映射；
- 相同 hash 有多个候选：ambiguous，不自动选择；
- 无候选：missing，历史 event 保留；
- old-revision task/Evidence 标 stale；
- active plan 生成绑定当前 revision 的新版本并按需插入 review。

当前公开 `/api/projects/analyze` 仍沿用既有行为，每次导入创建新的 project 实例；M4 不会
把不同 project 自动合并成一个 learner history。正式重验证服务针对同一 project 的
snapshot revision 生命周期，这是当前的已知产品边界。

## `get_learning_context@1` 与 Agent Core

M4 在原 `ToolRegistry` 注册唯一只读工具 `get_learning_context@1`，没有第二套 Registry。
输入 schema 为空且 `extra=forbid`，不接受 learner/project/repository/revision、SQL、路径、
state/event/plan 或未知字段；每个 Agent run 最多调用一次，并计入原 5 step、8 call、
deadline、timeout、fingerprint 和 observation budget。

服务器硬上限：16 target states、8 recent verified outcomes、12 plan steps、16,384 bytes。
输出不含 learner ID、完整回答、完整 event log、raw evaluator、聊天、源码、绝对路径或
SQL key。学习上下文只能调整 explanation depth 与 next action，不能作为 repository
Evidence、扩大预算、绕过 RelationValidator/CitationValidator 或移除 citation。

M4 `/ask` 调用链：

```text
/ask
-> server binds local learner + project/repository/revision
-> load bounded validated learning context
-> run_bounded_agent / original Planner and BudgetState
-> optional get_learning_context@1
-> search / lookup / read / expand relations
-> EvidenceStore + EvidenceChainStore
-> RelationValidator + CitationValidator
-> answer_from_evidence(learning depth guidance only)
-> second relation/citation validation
-> M1/M2/M3/M4 compatible response
```

attempt 更新链：

```text
typed API -> ownership/current-plan/revision/Evidence validation
-> evaluator schema validation -> transactional attempt/evaluation/event/state/plan
-> next request reads new bounded context
```

## API 与降级

旧 `/ask` 请求没有新增必填字段。M4 新增 response 字段：

```text
learning_schema_version=1
learning_mode=disabled|profiled|adaptive|degraded
learning_context_summary
learning_plan_summary
recommended_next_action
learning_warnings
```

学习 API 支持 create/list/update goal、create/get current plan、get state、create/get task、
submit attempt、self-report、evaluation correction、next action 和 revision revalidation。
所有 mutation request 都 `extra=forbid`，server-bound identity 不在请求 schema 中。

- 无 goal：learning disabled，M3 原链路可用；
- 无 LLM：ask 的确定性 fallback 不变；attempt 为 ungradable，不伪造 mastery；
- 无 Embedding：M1 lexical fallback 不变；
- 无 relation index：M3 retrieval-only 不变；
- 学习状态损坏：learning degraded warning，M3 仍工作，不使用部分状态。

## Prompt Injection 与执行边界

goal、plan 文本、task prompt、答案、self-report、feedback、源码、README、注释、docstring、
字符串、文件名、符号和 memory summary 全部是不可信数据。身份、revision、rubric、
Evidence、event_order、state update、plan version、工具白名单和预算都由服务器决定。

Registry 没有 set_mastery、change identity、Shell、import、network、environment 或 code
execution 工具。测试 fixture 中包含伪造 event/mastery/JSON/Shell 指令，并验证目标仓库
代码执行次数始终为 0。

## 工程评测与已知限制

冻结文件：`backend/tests/fixtures/m4_learning_eval.json`；执行：
`backend/tests/test_m4_evaluation.py`。24 条场景分类为 4 goal/plan、6 assessment、
4 adaptation、4 persistence/revision、3 Agent、3 security/degradation。

这些测试是本地、临时 SQLite、fake planner/evaluator/LLM/Embedding 的确定性工程 fixture；
不访问网络、不下载模型、不运行仓库代码。结果不代表真实用户学习效率、真实掌握度预测、
真实 LLM evaluator 准确率、Fixed RAG 优势或大规模仓库质量。

最终实际验证结果：

- M4 定向与冻结：32/32，24 条冻结场景状态转移和路线适配均为 100%；
- M3 回归：30/30，20 条冻结场景全部通过；
- M2 回归：36/36，14 条冻结场景全部通过；
- M1 回归：12/12，Hit@5=100%，mock MRR@10 满足 `>=0.80`；
- 完整后端：218/218；
- 未授权读取/写入、直接 Planner mastery、错误 self-report mastery、重复 attempt 提升、
  非法跃迁、投影不一致、半事务、旧 plan 覆盖、stale mastered、跨 project/revision
  target、失效 Evidence/citation、预算越界和目标代码执行：fixture 观测均为 0。

开发期红测包括：v5/Registry 旧快照未更新；以及同一秒 event 用哈希 ID 排序造成后续
fail 可能先于 pass 投影。前者更新为 M4 契约，后者通过事务内单调 `event_order` 修复并
加入独立 validator 回归。最终没有失败或跳过的后端测试。

未执行真实 LLM、真实 BGE-M3、真实 GitHub/网络仓库、大规模性能、前端构建和浏览器
E2E。前四项属于本轮禁止或 M5 范围；前端没有修改，因此未运行前端检查。

剩余限制：local single-user；没有自动跨 project history merge；前端尚未接入新 API；
semantic evaluator 的真实模型质量未评测；静态关系仍受 M3 Python-only 保守解析限制；
同步 SQLite timeout 仍是协作式；正式真实 LLM/BGE-M3/网络仓库和用户实验留到 M5。

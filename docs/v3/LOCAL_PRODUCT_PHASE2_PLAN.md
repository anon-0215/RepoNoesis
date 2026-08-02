# RepoNoesis Local Product Phase 2 设计方案（建议）

> 状态：建议，尚未实施，尚未成为既有架构决策。
>
> 基线：Local Product Phase 1 代码验收提交
> `e07bfd16e16ecbb827ab002fb9f11274013b92e3`，数据库 schema v8。
>
> 唯一推荐主线：**可重新打开、revision-aware、可恢复的持久化仓库工作区生命周期**。

## 1. 前置审查结论

仓库在本方案之前没有正式定义 “Local Product Phase 2”。`ROADMAP.md`
中的 M1—M5 是能力与研究评测里程碑；Local Product Phase 1 是在已冻结研究基线之上
建立的本地产品闭环。M5 与 Local Product Phase 2 是两条不同路线：

- M5 是默认关闭、隔离存储、固定数据集和固定 revision 的可复现实验/对照评测面；
- Local Product Phase 2 是正式 FastAPI、SQLite 和 React 产品面的持续使用能力；
- Phase 2 可以复用 M5 的身份、恢复和指标经验，但不能把 Phase 2 Gate 记作 M5 live
  pilot，也不能因 Phase 2 开发修改冻结的 M5/Phase 6 基线。

现有文档对历史 M1—M5 边界基本一致，但 README 的旧“后续计划”仍混合产品、研究和
远期语言方向；M4 决策文档记录的是当时 `/analyze` 每次创建 project 的历史行为。
Phase 1 后的当前产品路径已经改变为：相同 source + revision 的 completed/analyzing
项目复用，failed 项目在原记录重分析，新 revision 则创建新的 project。历史里程碑文档
不应回写为当前产品状态，本方案和当前路线图负责说明差异。

### 1.1 当前代码事实

| 能力 | 当前真实状态 | 缺口类型 |
| --- | --- | --- |
| 项目持久化 | 项目、分析结果、索引和学习数据写入 SQLite；`GET /api/projects/{id}` 可按已知 ID 读取。 | 功能基本存在。 |
| 已有项目重新打开 | 后端可按 ID 读取，但没有项目列表 API、项目选择器或稳定浏览器恢复；前端 `projectId` 只在 React 进程内。 | 产品功能/UX 缺口。 |
| `/analyze` 幂等 | Phase 1 产品请求对相同 revision 复用；failed 可原记录重试；legacy `repo_url` 路径仍创建新项目。 | 兼容路径不一致。 |
| revision 检测 | 导入时解析 Git commit，并把 revision 纳入 source identity；没有对已有项目显式“检查更新/刷新”的 API。 | 功能缺口。 |
| 同仓库增量更新 | 新 revision 创建新 project；`save_analysis` 全量替换该 project 的文件/模块/学习步骤和关系数据。 | 核心功能/架构缺口。 |
| 仅重建变化部分 | EmbeddingIndexer 对同一 project 的 fresh cache 可避免重复编码，但新 revision 的产品编排没有端到端增量复用。 | 性能/规模及编排缺口。 |
| 多项目管理 | 数据库可保存多个 project，但没有列表、搜索、选择、归档或删除产品流程。 | 产品功能/UX 缺口。 |
| 学习状态跨会话 | M4 goal/plan/task/attempt/event/state 在同一 project 下持久化并可跨后端进程恢复。 | 已实现。 |
| 学习状态跨 revision | 新 revision 当前是新 project；不同 project 不自动合并 learner history。 | 核心功能/数据语义缺口。 |
| M1—M4 前端 | 当前 UI 暴露导入、概览、地图、旧学习路线、问答和报告；未完整暴露 M4 goal/plan/task/attempt/review API。 | 产品功能/UX 缺口。 |
| 真实 Provider/Embedding | Gate A/B/C 在 CLI smoke 中验证；Gate C 是一个很小的本地 Python fixture。没有记录真实浏览器 E2E、长期运行或多仓库矩阵。 | 验收广度/质量缺口。 |
| 语言 | 产品级 chunk、关系和 Gate 重点是 Python；旧分析器有轻量 JS/TS 结构识别，但不是同等产品级证据链。 | 兼容性边界。 |
| 规模 | 同步分析，面向中小型仓库，前端进度粗粒度；未证明大仓库性能或中断恢复。 | 规模/可靠性缺口。 |
| 平台 | 当前本地验收记录来自 Windows；启动脚本和文档以 Windows 为主。 | 平台验证边界。 |

### 1.2 必须优先处理的架构债务

1. `source_identity` 同时包含稳定来源和 revision，无法直接表达“同一仓库工作区的多个
   revision”。
2. `project_id` 当前既是用户项目身份又是单次 revision snapshot 身份；M4 学习状态绑定
   `project_id`，新 revision 因而断开学习连续性。
3. 导入是同步的一次性编排，缺少持久化 run、阶段、失败恢复、原子激活和显式取消语义。
4. 分析保存以整批替换为主；Embedding 有 freshness 机制，但产品更新链未把文件 manifest、
   chunk identity、relation impact 和 embedding reuse 组织成一个正确的增量事务。
5. 前端没有稳定项目路由/选择器，服务重启后的数据库持久化没有形成用户可见闭环。

这些债务正好位于“持续使用”主线的入口，若先扩展完整学习 UI、更多语言或大仓库规模，
会把错误的 project/revision 身份传播到更多 API 和页面。

## 2. 候选方向比较

| 候选方向 | 用户价值 | 当前准备度/依赖 | Phase 2 结论 |
| --- | --- | --- | --- |
| 项目持久化与重新打开 | 极高；决定重启后能否继续使用。 | 数据已持久化，缺列表/选择/恢复入口。 | 纳入唯一主线的 P2.1。 |
| revision 检测和增量更新 | 极高；真实仓库会持续变化。 | 身份、事务和学习连续性需要先设计。 | 唯一主线核心，P2.2。 |
| 多仓库项目管理 | 高；本地用户通常学习多个仓库。 | 依赖项目目录；高级标签/归档不是首要。 | P2.1 只做最小项目库，完整管理延后。 |
| 前端完整产品闭环 | 高。 | 范围过宽，必须建立在稳定生命周期 API 上。 | 只实现生命周期和必要 M4 连续性 UI，其余延后。 |
| 学习计划和复习流程产品化 | 高，且 M4 后端已具备。 | 跨 revision 语义未解决。 | P2.3 接入连续性所需最小流程；完整教学工作台延后。 |
| 更大真实仓库验收 | 中高。 | 同步全量更新和恢复不足会放大风险。 | 作为本主线 Gate/性能边界，不单独成为主线。 |
| 多语言支持 | 潜在价值高。 | 会扩大 chunk、关系和 validator 语义。 | 明确非目标，后续独立阶段。 |
| Provider/Embedding 配置管理 | 中。 | Phase 1 已有后端配置、示例和安全状态诊断。 | 只保持兼容，不建设密钥 UI。 |
| 运行诊断和错误恢复 | 高。 | 与持久化更新 run 不可分割。 | 作为 P2.2 横切要求纳入主线。 |

**推荐判断：** Phase 2 不应以新增模型能力为中心，而应把 Phase 1 的“一次成功导入和
问答”升级为用户每天可以重新进入、检查更新、失败后恢复并继续学习的工作区。

## 3. 一句话目标与用户结果

**一句话目标：** 用户可以从项目库重新打开一个已分析仓库，显式检查并刷新到新
revision；系统只重做有必要的工作，失败时保留原可用 snapshot，并在新 snapshot 激活后
安全重验证既有学习状态。

完成后，用户可感知的闭环为：

```text
打开 RepoNoesis
-> 从项目库选择已持久化工作区
-> 继续查看、问答或学习
-> 显式检查仓库 revision
-> 无变化时零重建
-> 有变化时可观察地增量分析/索引
-> 成功后原子切换到新 snapshot
-> 旧 Evidence 标为 current/changed/renamed/missing/ambiguous
-> 继续计划、任务与复习
```

## 4. 范围与非目标

### 4.1 明确范围

- 本地单用户的项目库、分页列表、最近打开和明确的 active snapshot。
- 按 ID 重新打开，无需重新分析；刷新浏览器和重启服务后仍可恢复。
- 对 clean local Git root 与公开 unauthenticated HTTPS Git URL 显式检查 revision。
- 相同 revision 幂等 no-op；新 revision 创建可审计的更新 run。
- 文件 manifest diff、Python 文件/chunk 增量处理和安全的 embedding cache reuse。
- relation 受影响闭包重建；不能证明局部更新正确时允许显式、可观察的全关系重建。
- 新 snapshot 完成前不替换 active snapshot；失败可重试且不破坏旧数据。
- M4 学习上下文的显式跨 revision carry-forward/revalidation，不复制或伪造 mastery。
- 前端显示项目、revision、更新时间、run 阶段、失败码、重试和学习重验证结果。
- 生产 HTTP 路径的离线、真实本地模型和一次显式授权真实 Provider 验收。

### 4.2 明确非目标

- M5/Phase 6 live pilot、对照实验或历史基线修改。
- 多用户、登录、OAuth、RBAC、云同步、协作编辑或远端托管。
- 私有 Git 凭据、SSH Git、任意端口、submodule、Git LFS 或仓库代码执行。
- 自动后台轮询或未经用户操作自动拉取；Phase 2 只支持显式 refresh。
- 完整 Git 历史浏览、分支管理、merge、工作区写入或依赖安装。
- Java/Go/Rust 或把轻量 JS/TS 分析提升为 Python 等价证据链。
- 更换 LLM/Embedding 模型、修改 thinking、Prompt、Agent/工具/token 预算或放宽
  CitationValidator、RelationValidator 和生成后复验。
- 完整 M4 教学工作台、社交功能或教学效果研究。
- 把一次大仓库 Gate 外推为任意规模的生产承诺。

## 5. 推荐架构

### 5.1 身份与 snapshot

采用“稳定 workspace + 不可变 revision snapshot”，而不是继续让
`source + revision` 直接充当用户项目身份：

- `workspace_id`：用户长期打开的稳定身份；来源键只包含 `source_type` 和规范化来源。
- `project_id`：继续表示现有 API 使用的已分析 revision snapshot，保持 M1—M4 的
  revision-bound 语义。
- `active_project_id`：workspace 当前可读 snapshot；只在新 snapshot 全部完成后原子更新。
- 相同 workspace/revision 唯一；重复 refresh 返回已有 run/snapshot，不重复工作。
- schema v8 中每个既有 project 先迁移成一个独立 workspace，不自动合并看似相同来源的
  历史记录，避免错误合并学习历史。后续是否合并必须由单独、显式设计处理。

建议使用 additive schema migration（目标版本由实现阶段最终冻结），至少包含：

- `repository_workspaces`：稳定来源、显示名、active project、最近打开时间和状态；
- `workspace_revisions`：workspace、project、revision、parent project、manifest identity、
  activation 状态和时间；
- `repository_update_runs`：幂等键、期望 revision、固定阶段、状态、安全计数、稳定错误码
  和重试关联；
- 必要的 manifest/change-set 存储，以及可按最终 embedding 输入和模型身份验证的共享缓存
  引用。

迁移只新增表/列/索引，不删除 v8 数据。降级到旧代码时，旧 API 仍能读取原 `projects`
snapshot；新表可以被忽略。不得通过破坏性 down migration 回滚用户数据。

### 5.2 更新编排

建议固定阶段：

```text
source_preflight
-> revision_resolution
-> manifest_diff
-> source_analysis
-> chunk_update
-> relation_update
-> embedding_update
-> learning_revalidation
-> snapshot_validation
-> activation
```

规则：

1. source 和 revision 在服务端解析；前端不能提交数据库身份或伪造 resolved revision。
2. 同 revision 返回 `unchanged`，不创建重复 snapshot，不重新编码 embedding。
3. 新 revision 先建立 staging snapshot；旧 active snapshot 在全过程保持可读。
4. added/changed/deleted/renamed 文件由 tracked-file manifest 和内容哈希确定。
5. 仅新增/变化的 Python 内容重新 chunk；未变化 chunk 必须保留可审计 provenance。
6. Embedding 仅在最终输入哈希、模型/revision、prefix、max length、normalize 和格式版本均
   相同时复用；否则重新编码。不得仅按路径复用。
7. relation 局部重建必须覆盖变化文件及其入/出依赖影响闭包；出现动态或不明确依赖时，
   允许退化为该 snapshot 的全关系重建，并记录原因，不能静默保留陈旧边。
8. 完成 chunk、relation、embedding、Evidence 可解析性和数据库一致性检查后，单事务激活。
9. 学习重验证失败时不能伪造成功；是否阻止 snapshot 激活应在 P2.3 冻结为显式状态。
   推荐允许代码 snapshot 激活，但把 learning 标为 `needs_review`，保留旧事件且禁止自动
   mastered。
10. 临时 checkout、staging 行和失败 run 按有界保留策略清理；清理失败可重试且不删除
    active snapshot。

### 5.3 API 与前端入口

建议新增而不破坏现有路由：

- `GET /api/workspaces`：分页列出安全摘要；
- `GET /api/workspaces/{workspace_id}`：workspace、active snapshot 和最近 run；
- `POST /api/workspaces/{workspace_id}/open`：可选地更新最近打开时间；
- `POST /api/workspaces/{workspace_id}/refresh`：显式检查并启动/复用更新 run；
- `GET /api/workspaces/{workspace_id}/runs/{run_id}`：阶段与安全诊断；
- 现有 `/api/projects/{project_id}`、`/ask`、map、report 和 M4 API 继续针对明确的
  snapshot project ID。

前端影响范围：`frontend/src/App.tsx`、`frontend/src/lib/api.ts`、
`frontend/src/types.ts`、`frontend/src/lib/product.ts` 和样式/离线测试。项目选择应进入 URL
或受限本地偏好，而 API key、源码、Evidence 或回答正文不得为恢复目的写入浏览器持久存储。

### 5.4 核心服务影响

预计涉及：

- 入口/编排：`backend/app/main.py`，并建议新增独立 workspace/update service，避免继续
  扩大路由函数；
- 来源与 revision：`backend/app/services/repository_import.py`；
- 持久化与 migration：`backend/app/database.py`；
- 分析/chunk：`analyzer.py`、`code_chunker.py`；
- relation：`relation_analysis.py`、`relation_graph.py`；
- embedding：`embedding_indexer.py`、`embedding_service.py`；
- 学习连续性：`learning_service.py`、`learning_validation.py`；
- 只读消费：`qa_agent.py`、EvidenceStore、bounded Agent 和 validators 原则上不改决策
  语义，只接收已激活 snapshot。

## 6. 兼容性、安全、恢复和可观察性

### 6.1 M1—M4 兼容性

- 旧分析请求和项目读取响应保留；新字段必须可选，旧数据库必须自动向前迁移。
- bounded Agent、工具白名单、工具/step/token/deadline 预算保持不变。
- EvidenceStore、CitationValidator、RelationValidator 和生成后复验继续绑定明确的
  project/revision；禁止跨 snapshot 混用 Evidence。
- M4 immutable event 不改写。跨 revision 只能追加 carry-forward/revalidation 事件，并由
  现有确定性投影重新计算状态；旧通过记录保留历史意义，但不能自动证明新 revision mastery。
- 关闭 Phase 2 workspace UI/refresh 功能后，Phase 1 的导入和按 project ID 使用路径仍可用。

### 6.2 安全边界

- 不执行、import 或安装目标仓库代码/依赖；只读取 Git tracked files。
- local source 仍要求 clean Git root；refresh 不修改或 checkout 用户工作树。
- public source 继续只允许无凭据 HTTPS、公共地址、443、无 query/fragment；禁止 hooks、
  submodule、LFS smudge、交互式凭据和不可信 Git 配置。
- README、注释、字符串、路径和 Git 元数据都是不可信数据，不能改变阶段、预算、identity、
  validator 或激活决策。
- API key 只在后端环境中；workspace/run/API/日志/浏览器均不得保存 key、Header、prompt、
  Provider 正文、reasoning content 或其派生特征。
- 所有路径必须限制在配置的工作区/checkout 根内，symlink 和 traversal 必须拒绝。

### 6.3 幂等与失败恢复

- refresh 接受/生成稳定 idempotency key；相同 workspace + resolved revision + config identity
  只能有一个有效 run。
- 进程中断后，`running` run 必须可判定为 resumable 或 safely failed，不能永远假装运行中。
- resume 逐阶段验证 checkpoint identity；不匹配即拒绝恢复，不拼接两个 revision 的结果。
- 任一阶段失败时 active snapshot 不变，用户仍可问答和学习旧 revision。
- 重试复用已验证 staging 产物；是否复用必须由 hash/config identity 决定，不能由文件名或
  仅凭 run 状态决定。
- activation 使用事务和 optimistic version；并发 refresh 只有一个胜出，另一个返回稳定
  conflict/reused 状态。

### 6.4 可观察性

持久化并展示白名单元数据：固定阶段、run 状态、resolved revision、文件/chunk 的
added/changed/deleted/reused 数量、relation 局部/全量模式、embedding generated/cached 数量、
开始/结束时间、稳定错误码和是否可重试。

禁止记录源码、diff、问题、回答、Prompt、Evidence 正文、工具参数/结果、Provider 正文、
reasoning content、Header、凭据及其长度/片段/哈希。诊断必须有大小上限，且不得参与 Agent、
学习投影或 snapshot 激活决策。

## 7. 子阶段、输入、输出、验收与回滚

### P2.1：项目库与可重新打开工作区

**输入：** schema v8 数据库中的现有 projects，以及 Phase 1 的 local/public import。

**输出：** additive workspace migration；项目分页列表/详情 API；active snapshot；前端项目
选择、最近打开和刷新/重启后的恢复；当前 revision 和状态展示。

**验收：**

- v8 fixture 迁移后行数、project ID、分析结果和 M4 记录不丢失；重复迁移 no-op；
- 至少两个项目可列表、选择，浏览器刷新与后端重启后可按 URL/最近项目重新打开且不分析；
- 不知道 project ID 的用户也能完成重新打开；空库和损坏/不存在项目返回稳定错误；
- 旧 `/api/projects/{id}`、`/ask` 和 M1—M4 回归不变；
- 普通打开不加载 BGE 模型、不访问网络、不调用 Provider。

**回滚边界：** 新路由/UI 受单一可关闭能力边界控制；旧项目路由继续可读。回滚代码时保留
新增表，不执行破坏性 schema 回退。

### P2.2：revision-aware 增量刷新与恢复

**输入：** P2.1 workspace、当前 active snapshot、显式 refresh 请求和 local/public source。

**输出：** revision resolution、持久化 update run、manifest diff、staging snapshot、增量
chunk/embedding、正确 relation 重建、原子 activation、进度/错误/重试 UI。

**验收：**

- 相同 revision 连续 refresh 返回 unchanged/reused，不新增 snapshot，document encode 为 0；
- 单文件变化只重新 chunk/encode 必需内容；删除/重命名不会留下可检索陈旧 chunk；
- relation 结果与对新 revision 做一次干净全量构建等价；若使用全量关系退化，状态明确；
- 在每个阶段注入失败后，旧 active snapshot 仍可读、可问答，重试无重复行或混合 revision；
- 并发相同 refresh、进程中断/resume、staging 清理和磁盘/SQLite 错误均有确定性测试；
- local refresh 不修改目标 worktree；public refresh 保持 Phase 1 URL/clone 安全边界。

**回滚边界：** 禁用 refresh 后 active Phase 1 snapshots 仍工作；未激活 staging 可按 run ID
安全清理；已激活 snapshot 不通过删除用户历史来回滚，而是显式重新选择上一个有效 snapshot。

### P2.3：学习连续性与最小持续学习 UI

**输入：** P2.2 的 previous/new snapshot lineage，以及既有 M4 goal、plan、task、attempt、
event 和 state。

**输出：** 跨 revision revalidation/carry-forward 服务；current/changed/renamed/missing/
ambiguous 结果；needs-review 计划；前端最小 goal、当前 plan、任务、attempt、复习和 warning
界面。

**验收：**

- 旧 immutable events 和 attempts 原样保留，重复 revalidation 不增加重复事件/plan version；
- hash 未变的唯一目标可保持 verified，变化/缺失/多候选目标不能自动 mastered；
- 旧 Evidence/citation 不得进入新 snapshot 回答，新任务只绑定 active revision Evidence；
- 浏览器/后端重启后 goal、plan、state、next action 和 review 状态一致；
- 无 LLM evaluator 时仍返回 ungradable；deterministic fallback 和旧固定学习路线保持兼容；
- 学习 revalidation 失败被清晰展示，不会伪装 workspace refresh 全部成功。

**回滚边界：** 可关闭 carry-forward 和新学习 UI，保留全部 immutable history、新 snapshot 与
旧 M4 API；不得删除或反向改写学习事件。

## 8. 测试与真实验收 Gate

### 8.1 离线测试策略

每个子阶段先做定向测试，再做完整后端回归；修改前端时运行 TypeScript 检查、生产构建和
组件/产品辅助函数测试。所有默认测试使用临时目录、临时 SQLite、fake LLM/Embedding，禁止
网络和模型下载。

测试矩阵至少覆盖：

- v8 migration、空库、重复迁移、旧数据兼容、rollback-read compatibility；
- workspace 分页/排序/选择、active snapshot、浏览器恢复和跨进程恢复；
- local/public identity normalization、same/new revision、并发幂等；
- added/changed/deleted/renamed 文件，chunk identity、embedding freshness 和 relation 等价；
- 每阶段 fault injection、事务 activation、resume、取消、清理和容量上限；
- M1/M2/M3/M4、普通 `/ask`、bounded budgets、Evidence/Citation/Relation validators；
- learning revalidation 的 unchanged/changed/renamed/missing/ambiguous 和重复事件抑制；
- Prompt Injection、路径穿越、symlink、恶意 Git metadata 和诊断脱敏；
- 目标仓库 execution/import/dependency install 次数始终为 0。

### 8.2 建议真实 Gate

| Gate | 外部能力 | 通过条件 |
| --- | --- | --- |
| P2-A：restart/reopen | 无网络、fake provider/embedding | 从 v8 迁移，导入两个 fixture，重启服务并用真实 HTTP/浏览器路径重新打开；不得重分析。 |
| P2-B：local revision refresh | real local BGE-M3，离线 | 对受控本地 Git fixture 做 unchanged、单文件 change、delete/rename；fresh 内容编码正确、同 revision encode 0、结果与 clean build 等价、旧 snapshot 可恢复。 |
| P2-C：public refresh | 显式授权网络，无付费 Provider | 对固定公开 HTTPS Python repository/revision 检查并刷新；验证 URL 安全、revision、恢复和持久化。网络不可用时只能 conditional，不伪造 pass。 |
| P2-D：grounded continuity | real local BGE-M3；一次显式授权真实 Provider | 通过正式 HTTP/浏览器流程打开已存在 workspace、刷新 revision、执行一次 bounded `llm_grounded` 问答并完成 Citation/Relation/生成后复验；学习状态得到真实 revalidation。 |

真实 Gate 的仓库、revision、配置 identity、调用次数和退出码必须落安全结构化记录；fake、
deterministic fallback 或 M5 provider 不能冒充 P2-D。未获凭据/网络授权时保持
`CONDITIONAL PASS`，不得自动重试产生费用。实现阶段再冻结唯一精确命令，本设计阶段不
臆造尚不存在的 CLI。

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| workspace/project/revision 迁移错误 | 丢失可达性或串联错误学习历史 | additive migration；每个旧 project 独立 workspace；备份/fixture；不自动合并。 |
| 增量结果与全量结果不等价 | 陈旧 Evidence、relation 或错误引用 | 逐 revision identity；golden clean-build 对比；不确定关系时明确全量重建。 |
| update 中断污染 active 数据 | 项目不可用 | staging + 原子 activation；旧 snapshot 始终可读；逐阶段 fault injection。 |
| 跨 revision 错误继承 mastery | 教学状态失真 | immutable history；唯一 hash/identity 校验；changed/missing/ambiguous 强制 review。 |
| embedding 错误复用 | 检索结果失真 | 完整 input/model/config identity；维度/finite/normalize 检查；拒绝模糊 cache。 |
| 多个 refresh 竞态 | 重复行、错误 active revision | idempotency key、唯一约束、optimistic version、单事务 activation。 |
| public Git SSRF/凭据泄露 | 安全事件 | 复用 Phase 1 URL/DNS/Git hardening；无私有凭据；重定向重新验证。 |
| 大仓库耗时/磁盘膨胀 | UX 和恢复压力 | 明确上限、阶段计数、取消/恢复、保留策略；不承诺任意规模。 |
| P2 范围膨胀为完整学习平台 | 延迟交付且风险扩散 | 只做生命周期所需最小 M4 UI；高级教学体验单列后续阶段。 |
| Phase 2 与 M5 结论混淆 | 误报科研/质量结果 | 独立 Gate、命名、存储和文档；不修改 Phase 6。 |

## 10. 推荐实施顺序与完成定义

1. **P2.1** 先冻结 workspace/snapshot 身份和 migration，交付可列表、可重新打开的项目库。
2. **P2.2** 在稳定身份上实现 update run、revision diff、增量重建、恢复和原子激活。
3. **P2.3** 最后把 M4 学习历史安全跨 revision 重验证，并补齐最小持续学习 UI。
4. 逐阶段通过离线门禁；全部代码冻结后再申请 P2-B/C/D 的真实资源授权。

Local Product Phase 2 只有在以下条件同时满足后才能称为完成：

- P2.1—P2.3 的兼容、迁移、幂等、恢复、安全和前端验收全部通过；
- 完整后端与前端回归无失败、无未说明跳过；
- P2-A、P2-B 通过；P2-C/P2-D 按授权实际执行并诚实记录，缺授权时状态保持 conditional；
- M1—M4 的 Evidence、validator、bounded budgets 和学习事件语义未放宽；
- 没有修改或借用 M5/Phase 6 结果来替代产品 Gate。

Phase 2 完成后，完整 M4 教学工作台、多项目高级组织、更大仓库性能、多语言和 M5 live
pilot 仍应作为独立后续工作评估，不能在本阶段顺带实现。

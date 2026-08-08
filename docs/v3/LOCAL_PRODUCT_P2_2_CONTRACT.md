# 源鉴 RepoNoesis V3·LP2.2 更新契约

> 状态：实施冻结契约。本文只定义 LP2.2 revision 检测、增量更新和原子激活；
> LP2.3 跨 revision 学习重验证仍未实现。

## 1. 身份与不可变边界

- `workspace_id` 继续表示 P2.1 已冻结的稳定工作区身份。
- `project_id` 继续表示一个 revision-bound snapshot。已激活或已失败的 snapshot
  不被改造成另一个 revision，也不合并到其他 workspace。
- revision identity 是来源仓库由服务端解析出的 40 位 Git commit SHA；路径、显示名、
  列表位置和客户端输入都不能充当 revision identity。
- 文件 manifest identity 是按规范化仓库相对路径排序后，对每个已纳入 snapshot 的文件
  `path + content_sha256 + size` 计算的确定性哈希。mtime 不参与任何复用判断。
- 同一 workspace 内 `(workspace_id, repository_revision)` 唯一。一个 workspace 在任意时刻
  只能有一个 `activation_status=active` 且可打开的 snapshot。

## 2. Update run 状态机

持久化状态为 `pending -> running -> succeeded | failed`。失败 run 只有通过显式 retry
才能回到 `pending`；`succeeded` 是终态。阶段按以下固定顺序单调推进：

```text
revision_resolution
-> manifest_diff
-> source_analysis
-> chunk_update
-> relation_update
-> embedding_update
-> snapshot_validation
-> activation
```

`succeeded` 的结果只允许：

- `unchanged`：目标 revision 已是 active revision；不创建 snapshot，不运行 Embedding；
- `activated`：新 snapshot 已完整验证并在同一事务中成为 active snapshot。

相同 workspace、resolved revision 和构建配置身份只产生一个 run。重复请求返回同一 run；
失败后的重新执行必须使用该 run 的显式 retry，不创建竞争 run。

## 3. 并发、激活与失败

- run 获取、状态转换、snapshot 归属和激活均在 SQLite `BEGIN IMMEDIATE` 事务中完成。
- 并发 refresh 由唯一约束收敛到同一 run；只有持有 `pending -> running` 转换的执行者工作。
- 新 project 先以 `staging` 归属 workspace。旧 `active_project_id` 在分析、chunk、relation、
  Embedding 和验证全过程保持不变。
- 激活事务同时校验旧 active pointer、递增 workspace activation version、把旧 snapshot 标为
  `superseded`、新 snapshot 标为 `active`，并把 run 标为 `succeeded/activated`。
- 任一前置阶段或激活事务失败时，新 snapshot 标为 `failed`，run 记录固定错误码；旧 active
  pointer 不变。失败 snapshot 永不被 reopen、`/ask`、检索或学习 API 自动选中。
- 进程启动时遗留的 `pending/running` run 被确定性标为 `failed/update_interrupted`，可由用户
  显式 retry；不使用内存状态冒充恢复，也不拼接未经验证的阶段产物。
- 本轮不提供自动轮询、后台定时更新、文件监听或取消 API。请求中断等同于可安全重试的
  进程中断；不会改变旧 active snapshot。

## 4. 增量与缓存复用

- 文件差异按内容哈希识别 added/modified/deleted/unchanged；只有内容哈希唯一匹配的
  deleted/added 对才可报告为 renamed。
- parser/chunker 版本相同时，unchanged 文件的 chunk 可复制到新 snapshot；唯一内容匹配的
  rename 可复制 chunk 结构但必须改写路径和 revision。added/modified 文件重新 AST chunk。
  parser/chunker 版本不同时全部重新 chunk。
- Embedding 只在最终 embedding input hash、chunk content hash、有效模型身份、resolved
  revision、维度、dtype、normalize、文本格式、prefix/max length 配置身份均相同时复制。
  rename 会改变最终输入中的 path，因此不会复用旧向量。blob 长度、finite 值或归一化校验
  失败即重新编码。
- relation 数据从不跨 project 复制。LP2.2 在新 snapshot 上全量、确定性重建关系图并记录
  `relation_mode=full`；这保留 snapshot 隔离，并作为局部闭包尚未证明时的安全退化。
- snapshot 验证要求 manifest、project revision、全部 chunk revision、relation run 和全部
  启用的 Embedding 均精确属于目标 snapshot；无法证明安全复用的数据必须重算。

## 5. 来源、安全与兼容

- local 检测与 refresh 继续要求 clean Git root，只读取 tracked files，不 checkout 或修改用户
  worktree。
- public 检测与 refresh 继续复用 P2.1/LP1 的无凭据 HTTPS、公共地址、443、禁重定向、禁
  submodule/LFS/hooks/交互凭据边界。客户端不能提交内部 project/snapshot 外键。
- API 和日志只返回白名单状态、revision、计数、固定错误码和有限错误说明；不返回源码、
  diff、Embedding blob、Evidence、Prompt、messages、reasoning、凭据或不必要绝对路径。
- 旧 project ID、P2.1 reopen、M1-M4 Evidence/relation/learning 外键继续有效。refresh 不写、
  复制、删除或重验证旧 goal、plan、task、attempt、evaluation、immutable event 或 mastery。

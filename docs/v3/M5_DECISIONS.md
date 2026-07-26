# M5：真实集成与可复现实验决策

## 范围与边界

M5 在 M4 提交 `f7ec8f16be3585053db4982a3d00ed4457c06dbc` 上增量建立真实 provider
契约、固定 revision 的真实仓库 pilot、严格 dataset validator、模式适配、可恢复 runner、
指标、消融基础和离线冻结工程评测。没有增加第二套 Agent、Evidence、Relation Graph 或
Learning State；没有修改前端、执行目标仓库、安装目标依赖、训练模型、开展用户实验或开始
M6。

生产 SQLite 继续是 schema v6。benchmark 使用 `artifacts/m5/runs/<run_id>` 下的独立
SQLite、checkpoint、cache 和结构化结果，均被 Git 忽略，不污染正式 learner 数据。
benchmark schema、metric schema 各自为 1，不与数据库或 M1—M4 API schema 混用。

## 正式复用链路

生产 `/ask` 保持：

```text
/ask -> LearningService.get_learning_context
-> run_bounded_agent -> ToolRegistry/BudgetState
-> EvidenceStore/EvidenceChainStore
-> RelationValidator/CitationValidator
-> answer_from_evidence -> second validation -> response
```

M4 attempt 保持：

```text
typed attempt -> task/plan/revision/Evidence validation
-> evaluator schema/rubric validation
-> BEGIN IMMEDIATE -> attempt/evaluation/event
-> deterministic state projection + LearningStateValidator
-> plan adaptation
```

M5 runner 复用：

```text
trusted config -> dataset validator -> fixed local snapshot ingestion
-> existing analyzer/chunker/relation index/embedding index
-> fixed modes: existing retrievers -> EvidenceBuilder -> answer_from_evidence
-> M2/M3/M4 modes: existing run_bounded_agent + restricted existing ToolRegistry
-> existing validators -> M5 deterministic metrics -> atomic checkpoint/report
```

模式白名单只能来自可信 CLI/config。问题、README、源码、注释、历史学习内容和模型输出
不能改变 mode、provider、revision、预算、gold 或 validator。

## Provider 契约

`app.m5.providers` 为 answer generator 和 structured evaluator 提供同一受限契约：provider/
model/revision/capability、timeout、最多三次 attempt、输入输出上限、取消、usage、latency、
actual model 和分类错误。错误分类包括 cancellation、timeout、connection、429、5xx、
invalid JSON 和 oversized response。配置快照递归脱敏；不保存 Authorization、API key、原始
响应或私有思维链。

OpenAI-compatible provider 保持厂商中立。真实 answer 要求 `M5_ALLOW_NETWORK=1` 和
`M5_ALLOW_REAL_LLM=1`；真实 evaluator 还要求 `M5_ALLOW_PAID_EVAL=1`。缺 key 或 gate
时拒绝，而非伪造 live。全局 ledger 限制 250 次调用及输入/输出 token。provider 不返回
usage 或没有可靠价格时成本为 `unknown`，不能写 0。

structured evaluator 使用 M4 `EvaluationOutput` 的 `extra=forbid` schema，并再次检查 rubric
criterion 和 Evidence 白名单，按权重/critical criterion 重算 verdict。非法、超长或格式错误
输出变为 `ungradable`；它不能写 mastery、event 或 plan。

## Embedding

真实 dense 复用 M1 `EmbeddingService`、`EmbeddingIndexer` 和 SQLite freshness identity。
M5 wrapper 强制独立 cache，记录 model/revision、max length、normalized、device、dtype、
dimension 和 config hash；拒绝空向量、维度变化、NaN/Infinity。未设置
`M5_ALLOW_MODEL_LOAD=1` 时不加载；未设置 network gate 时使用 `local_files_only`，缺模型
不会静默下载。离线测试使用同一 `EmbeddingService` 契约的 deterministic fake backend，
结果明确标记 fake，不能当作 BGE-M3 live 结果。

## Dataset 与身份

pilot-v1 仓库：ItsDangerous、Click、HTTPX，均为 BSD-3-Clause，shallow clone 后固定完整 SHA。
repository identity 包含 URL、SHA、Python 内容 fingerprint、分析上限、排除路径和 acquisition
method。scenario gold 使用 revision + POSIX path + qualified symbol + line span + SHA-256；不使用
数据库 Evidence ID。

36 个场景严格分为 9 locate、9 explain、9 relation、6 impact、3 unanswerable/injection；每仓库
12 个。另有 6 条受控 adaptive sequence。全部 annotation provenance 为
`agent_assisted_developer_curation`，status 为 `agent_curated_pending_human_review`。

validator 在 run 前拒绝未知字段、错误 schema、重复身份/问题、非 40 位 SHA、跨 revision、
路径穿越、symlink 越界、缺文件/符号、越界 span、hash 不符、不可解析 relation identity、
预算越界、过长文本、非法 provenance 和不一致的 unanswerable gold。

## 模式、公平性与消融

- `fixed_lexical_rag`：BM25 固定 top-k；无 dense/Planner/relation/learning。
- `fixed_dense_rag`：仅 fresh dense；不得 lexical fallback。
- `m1_hybrid_rag`：正式 M1 lexical+dense+RRF。
- `m2_bounded_agent`：正式 Agent Core，只允许 search/lookup/read/validate。
- `m3_relation_agent`：M2 加 expand_relations 和 RelationValidator。
- `m4_profiled_agent`：M3 加一次 server-bound learning context。
- `m4_adaptive_sequence`：独立 SQLite 中调用正式 M4 mutation/service loop。

所有 answer 模式共享 answer provider、temperature、输出 token、Evidence cap、citation contract、
timeout/retry 和固定 corpus。高级模式只在明确记录的 Planner/relation/learning 工具上有差异。
validator-off 消融只允许测试隔离实现；生产和正式 benchmark 回答永远不能关闭
CitationValidator 或 RelationValidator。

## Runner、恢复、缓存和统计

run ID 是 dataset/revisions/fingerprints、规范化 config、provider/model、prompt/metric/evaluator
version 和源码 tree digest 的 SHA-256，不以时间命名。checkpoint 带 records checksum，通过
临时文件加 `os.replace` 原子提交；resume 跳过成功结果，retry 记录 attempt number，损坏
checkpoint 阻止恢复。结果按 scenario/mode/attempt 稳定排序。

指标包括 Hit@1/5、MRR@10、nDCG@10、file/symbol/span recall、Evidence P/R/F1、citation/
revision、relation chain、abstention/key point、工具/步骤/预算、learning transition、p50/p95、
token/cost、paired delta 和固定 seed bootstrap 95% CI。失败样本单独计数，不进入成功质量均值；
NaN/Infinity 被拒绝；unknown cost 保持 unknown。

## 安全与限制

目标仓库只通过 Git 和文本/AST 读取，不 import、不执行、不装依赖。Registry 没有 Shell、网络、
写仓库或 validator bypass 工具。运行记录明确保存 execution/import/Shell count 为 0。

当前 annotation 尚未人工复核；静态 relation 不能证明运行时行为；fake provider 工程结果不能
代表真实模型质量；同模型 judge 必须标记 `same_model`；pilot 不能证明教学有效性、mastery
预测准确性、所有仓库优势或语言泛化。

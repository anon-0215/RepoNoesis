# RepoNoesis V3 路线图

共同约束：前一阶段验收后才开始下一阶段；保持旧 API/数据库可用；不执行或修改目标仓库。规划阈值不等于已通过。

## M1：证据问答纵向闭环

**目标与 I/O**：输入 project、问题、revision、代码块和可用向量；输出回答、验证 Evidence、独立 lexical/semantic/fusion scores、策略/模型/降级状态。

```text
问题 -> 代码块词法 -> BGE-M3 语义 -> 混合融合
     -> Evidence -> 引用校验 -> /ask -> 路径/符号/行号回答
```

**允许修改**：`qa_agent.py`、`semantic_retriever.py`、新增词法/融合/evidence 服务、`database.py`、`main.py`、类型模型、后端测试；仅在必要时改前端 API 类型和问答页。

**不属于 M1**：完整 Agent、调用图、多跳、长期状态、动态路线、前端大改。

**测试**：词法排名、分数融合、去重、Evidence Schema、路径/行号/哈希/revision 校验、各类降级；冻结 SQLite fixture 上 `/ask` 集成及旧字段兼容。

**量化验收**：M1 使用 16 条确定性工程回归标注；answerable Hit@5 = 100%、mock hybrid MRR@10 >= 0.80；最终引用有效率 100%；无依据问题错误引用率 0%；Embedding 不可用用例 100% 明确降级；完整确定性后端测试通过。50 条以上真实仓库/真实模型扩展与正式科研评测留到 M5。

**兼容/回滚/出口**：保留 `answer`、`citations`，新增字段可选；schema 仅增量；配置开关可回旧 `qa_agent`。达标且失败/降级可观察后进入 M2。

## M2：工具层与有限 Agent Core

**完成状态（2026-07-24）**：已在 `v3-agent-development` 实现静态 Tool
Registry、`search_code`/`lookup_symbol`/`read_source`/`validate_evidence`、
结构化 Planner 决策、请求级有限循环、预算/重复/no-progress/协作式
timeout/cancellation、防 Prompt Injection、最终强制引用校验和正式 `/ask`
默认接入。schema 保持 v4；没有实现 M3 关系或多跳能力。详细冻结决策见
`M2_DECISIONS.md`。

**目标与 I/O**：输入用户目标与 M1 Evidence 服务；输出受预算限制的 Agent run、ToolCall、Observation、Step 和验证回答/partial。

**允许修改**：tool registry、只读工具、Agent Core、预算/取消/日志、Pydantic 契约、必要 API/表及状态 UI。

**不属于 M2**：调用图、多跳关系、长期状态、多智能体、代码修改。

**测试**：工具白名单、Schema、预算、超时/取消、重复调用、有限 replan、脱敏、Prompt injection；模拟模型 1—8 步、无 LLM 与部分失败集成。

**量化验收**：100% 在硬上限内终止；重复同参逃逸率 0%；超时/取消 100% 明确状态；冻结任务完成率 >= 80%；引用有效率 100%。

**兼容/回滚/出口**：M1 固定闭环保留为无 LLM/Planner 失败降级；正式 `/ask`
默认经过 Agent Core；trace 仅为请求级 API 摘要，不新增数据库表；旧请求、旧
citation 和 M1 Evidence 字段保持。完成并提交 M2 后停止，M3 必须另行开始。

## M3：关系扩展与多跳仓库分析

**完成状态（2026-07-24）**：已在 `v3-agent-development` 实现 Python-only
revision-bound imports/calls/references/defines、数据库 schema v5、稳定 node/edge
身份、幂等关系索引、有界双向 BFS、`expand_relations@1`、请求级 Evidence chain、
关系与 Citation 双重最终校验、正式 `/ask` 兼容字段及 20 条冻结工程场景。默认
depth 1、硬上限 2；没有实现运行时调用图、完整类型/数据流、M4 或 M5。详细冻结
决策见 `M3_DECISIONS.md`。

**目标与 I/O**：输入 AST、import、代码块、符号、revision 和 Evidence；输出版本化调用/import/定义—引用关系及有限多跳 Evidence 图。

**允许修改**：`analyzer.py`、`code_chunker.py` 的增量扩展或新关系服务、关系表/迁移、查询工具与必要 API。

**不属于 M3**：运行时代码执行、完整动态 dispatch、全语言、无限图遍历、长期状态。

**测试**：同/跨文件调用、alias import、方法、未解析/多候选目标、循环、深度/节点预算、revision 隔离和跨文件 Evidence。

**量化验收**：按 M3 开发指令冻结的 20 条确定性场景，exact/resolvable call edge
precision/recall 均为 100%，bounded gold path 找回率 100%，预算和引用校验 100%
生效；这些结果不外推到真实 Python 生态或真实模型。

**兼容/回滚/出口**：关系表新增并绑定 revision；M1 不依赖关系表；开关关闭回 M2/M1。误差分类和图预算通过后进 M4。

## M4：长期学习状态与自适应引导

**完成状态（2026-07-26）**：已在 `v3-agent-development` 实现 database schema v6、
local-single-user identity、结构化 goal、versioned plan/DAG step、revision-bound task 与
rubric、bounded attempt、validated evaluation、immutable event、deterministic projection、
路线适配、revision 重验证、跨进程恢复、`get_learning_context@1` 和 `/ask` M4 兼容字段。
24 条冻结工程场景全部通过；没有实现完整前端、多租户、真实用户实验或 M5。详细决策见
`M4_DECISIONS.md`。

**目标与 I/O**：输入学习目标、验证阅读/问答/练习事件和 revision；输出 LearningState、掌握证据、问题和动态路线。

**允许修改**：`learning_agent.py`、learning state/event 服务、数据库迁移、学习 API 和前端工作台。

**不属于 M4**：模型训练、无证据心理画像、仓库自动修改、云端多租户。

**测试**：事件幂等、掌握规则、证据绑定、revision 变化、隐私、路线更新；跨会话恢复和旧证据 stale/revalidate。

**量化验收**：保存/恢复一致率 100%；mastery 变化 100% 有事件和方法/证据；重复事件不重复计分；陈旧证据识别率 100%；冻结场景路线符合规则 >= 90%。

**兼容/回滚/出口**：旧 `learning_steps` 保留；新状态用新表；关闭动态路线回固定五阶段且不删事件。迁移与隐私通过后进 M5。

## M5：评测、前端呈现与完整验收

**目标与 I/O**：输入 M1—M4、冻结集和对照配置；输出质量/检索/Agent/学习报告、消融、证据 UI 和最终演示。

**允许修改**：评测 harness/fixtures、报告、`frontend/src`、展示 API、文档和演示脚本；只修复验收缺陷。

**不属于 M5**：新核心方向、完整多语言、仓库执行、云多用户、模型训练。

**测试**：指标计算、数据校验、前端状态；分析—索引—问答—Agent—状态—报告 E2E，覆盖加载、失败、降级、取消、revision。

**量化验收**：M1—M4 门槛保持；引用有效率 100%；关键 UI 状态 E2E 覆盖 100%；Direct LLM、semantic-only、无关系扩展等消融各至少一次；Agent 完成率 >= 80%；完整后端测试和前端构建通过。

**兼容/回滚/完成**：UI 兼容旧响应；feature flag 可回旧五标签页；迁移有恢复说明。全部门槛、演示、限制和可复现报告通过后才称 V3 完整验收。

M1 是第一个代码开发里程碑。本设计阶段提交后停止，不提前实现 M1，更不把 Agent、调用图或 Learning State 塞入 M1。

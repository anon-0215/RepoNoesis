# RepoNoesis V3 基线

## 基线身份与口径

- 创建日期：2026-07-24（Asia/Shanghai）
- 正式名称：《源鉴（RepoNoesis）——面向真实代码仓库的证据驱动型持续学习智能体》
- 分支：`v3-agent-development`
- worktree：`D:\Project\RepoNoesis-v3`
- V2 基线：`63abe4b542f5c2d55e7b077709d46c0153e84fcd`

本地目录名变化不表示远程仓库已经改名。本阶段只冻结设计，不实现 V3 功能。

本文用以下口径避免把规划写成现状：

- **代码事实**：基线代码、配置或 schema 可直接证明。
- **本轮验证**：2026-07-24 在 V3 worktree 实际执行。
- **历史验证**：V2 迁移日志中的旧结果，本轮没有重跑对应环境。
- **V3 规划**：尚未实现的里程碑内容。
- **设计建议**：推荐约束，不代表现有行为。

## V1、V2、V3 边界

### GitLearnAgent V1

**代码事实**：已完成的仓库导读 Demo，包括公开 GitHub 抓取、文件级分析、项目地图、固定学习路线、文件级规则问答、Markdown 报告和 React/Vite 工作台。

### GitLearnAgent V2

**代码事实**：V2 已结束为“可信代码分析与检索底座”，完成 Python AST 函数、异步函数、类、方法及嵌套代码块；路径、限定名、精确行号、内容哈希和 repository revision；SQLite schema v4；可配置且延迟加载的 BGE-M3 dense Embedding；增量向量缓存；内部代码块级语义检索；无 LLM 或无 Embedding 时的降级路径。

旧 `docs/V2_PLAN.md` 中未实现的混合检索、调用图、证据问答和自适应教学不再称为“必须补完的 V2”，由 V3 重新排序和验收。

### RepoNoesis V3

**V3 规划**：复用 V2 底座，增加可验证源码证据问答、Agent 工具层、有限规划—行动—观察循环、关系扩展与多跳分析、持久化学习状态和自适应引导。V3 不从零重写，也不训练基础模型。

## 可复用资产

**代码事实**：

- `github_client.py` 获取公开 GitHub 仓库默认分支 commit 和递归 tree；最多选 45 个候选文本文件，单文件不超过 200,000 bytes，不 clone、不执行目标仓库。
- `analyzer.py` 提供文件级 Python AST、JS/TS 轻量分析、框架识别、模块、目录树和启动线索。
- `code_chunker.py` 提取 Python 函数、类、方法和嵌套符号，保存原始内容、精确行号与 SHA-256。
- `database.py` 的 `SCHEMA_VERSION = 4`；表为 `schema_versions`、`projects`、`repo_files`、`modules`、`learning_steps`、`chat_answers`、`code_chunks`、`code_chunk_embeddings`。
- `embedding_service.py` 负责延迟加载、设备、固定文本格式、配置身份、float32 与归一化。
- `embedding_indexer.py` 仅重算缺失/过期向量，批量失败后逐块重试；失败不销毁已保存分析。
- `semantic_retriever.py` 从 SQLite 读取新鲜向量，在内存点积排序，返回代码块 ID、路径、符号、行号、哈希和 semantic score。
- `main.py` 保留同步分析 API、学习路线、问答与报告编排。
- `learning_agent.py` 当前为固定五阶段路线，LLM 只可能增强第一阶段目标。

**历史验证**：

- V2 日志记录代码块相关确定性测试 16/16、当时完整后端测试 24/24。
- V2 日志记录 Embedding 相关确定性测试 61/61、当时完整后端测试 89/89。
- 历史真实 BGE-M3 smoke 使用 CPU、1024 维归一化向量，验证首次生成、缓存命中和单块增量重算；本轮不重跑真实模型。

## 已实现但未接入产品链路

**代码事实**：

- `SemanticRetriever` 已实现且有独立测试，但 `/ask` 没有调用它。
- 代码块和向量有精确来源元数据，现有问答 citation 只有 `path`、`summary`、`snippet`。
- revision 存于快照和代码块；`projects` 表没有独立 revision 列和多 revision 历史。
- 分析响应可包含 Embedding 索引统计，现有前端未完整展示。

### `/ask` 的当前事实

`main.py` 的 `/ask` 取得 `db.get_bundle()` 后直接调用 `qa_agent.answer_question()`。后者对完整文件按英文/路径 token、意图、核心文件和重要性排序并截取文本；不调用 `SemanticRetriever`，不做代码块级词法检索、混合融合或引用校验。因此当前 `/ask` 不是 V3 证据问答。

## API、schema、测试与运行环境

**代码事实**：

- FastAPI 提供 health、同步 analyze、项目详情、地图、学习路线、ask 和 report。
- Embedding 默认关闭；SQLite 默认 `backend/data/gitlearn.sqlite`，可用 `GITLEARN_DB` 配置。
- 当前是幂等建表和轻量升级，不是正式迁移框架。
- LLM 使用 OpenAI 兼容接口；无 Key 时规则降级。
- 确定性测试使用 `unittest`；真实模型 smoke 不属于默认 `discover tests`。

**本轮验证**：

- V3 创建后为 `v3-agent-development`，初始 HEAD 为精确 V2 基线，初始 worktree 干净。
- V1：`D:\Project\GitLearnAgent` / `main` / `d8a4d56`。
- V2：`D:\Project\GitLearnAgent-v2` / `v2-development` / `63abe4b`，只含受保护未跟踪文件 `docs/NEXT_AI_PROJECT_CONTEXT_PROMPT.md`。
- 本轮测试结果以最终实际命令记录为准，不以历史数字代替。

## 受保护对象

- `main`、`v2-development`、标签 `v0.1-demo`
- `D:\Project\GitLearnAgent`、`D:\Project\GitLearnAgent-v2`
- V2 用户文件 `docs/NEXT_AI_PROJECT_CONTEXT_PROMPT.md`
- `.env`、`embedding_cache/`、数据库、日志、模型权重和缓存
- Git 历史；禁止 hard reset、clean、force push、stash 用户改动、merge、rebase

## 当前不能宣传为完成

以下均为 **V3 规划**：代码块级词法与混合检索；`/ask` Evidence 与引用校验；Agent 工具层和有限循环；调用/import/定义—引用及多跳分析；持久化学习状态和动态路线；完整评测、消融、证据前端；完整多语言、目标仓库执行或修改；基础模型训练或权重“持续学习”。

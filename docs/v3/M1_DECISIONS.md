# M1：证据问答纵向闭环决策

## 范围与基线

M1 在 `v3-agent-development` 的 V2 schema v4、Python 代码块和现有 BGE-M3 缓存上增量实现。它只完成固定的“检索—Evidence—校验—回答”编排，不增加 Agent、工具调用循环、调用图、学习状态或前端重构。

正式 `/ask` 保持旧请求 `{question}` 可用；新增的 `path`、`language`、`symbol` 和 `evidence_count` 都是可选过滤/上限。旧响应 `answer`、`citations` 保持名称和类型。

## BM25 与 tokenizer

实现：`backend/app/services/lexical_retriever.py`。

- 检索单位是 SQLite `code_chunks`，不使用文件摘要代替代码块。
- BM25 参数固定为 `k1 = 1.5`、`b = 0.75`。
- 语料包含相对路径、文件名、符号名、qualified name、签名和代码块正文。
- tokenizer 对原始 token 使用 Unicode `casefold`，按路径/点号/冒号/代码标点自然分隔；保留完整标识符，同时拆分 `snake_case`、`camelCase` 和 `PascalCase`。
- 连续 CJK 文本保留完整片段，并生成字符 unigram 和 bigram；不引入 jieba 或其他依赖。
- 查询和文档使用同一函数。相同分数依次按 path、start line、end line、chunk id 稳定排序。
- project 是必选检索作用域；path、language、symbol 是查询条件，不通过重复 token 提权。

## Weighted RRF

实现：`backend/app/services/hybrid_retriever.py`。

```text
fusion_score(d) =
    1.0 / (60 + lexical_rank(d))
  + 1.0 / (60 + semantic_rank(d))
```

- rank 从 1 开始。
- lexical/semantic 候选上限各 20；默认 Evidence 5，最大 8。
- 身份键包含 project、revision、path、start/end line 和 content hash；重复代码块合并并保留两路原始 score/rank。
- 融合排序依次使用 fusion score、最佳单路 rank、path、start/end line、chunk id。
- 语义检索复用 `SemanticRetriever` 和原 SQLite 向量缓存，不建立平行索引。
- Embedding 关闭、无新鲜缓存或语义异常时返回 `retrieval_mode = lexical` 和 warning，不让 `/ask` 整体失败。
- 语义候选的 project/revision 不属于当前索引时直接丢弃。

为落实“问答不得触发模型下载”，Sentence Transformers 后端以 `local_files_only=True` 加载。已有本地模型/进程内模型仍可使用；本地缓存缺失时语义链路明确降级，不访问网络。

## Evidence 生命周期

实现：`backend/app/services/evidence.py`。

1. Hybrid candidate 转换为 Evidence，并限制 excerpt 为最多 2,000 字符。
2. `CitationValidator` 从同一 SQLite 读取事务中重新读取 project、`code_chunks` 和 `repo_files`。
3. 依次校验 repository/project、revision、安全相对路径、chunk identity、行号、文件对应行内容、SHA-256、符号和 excerpt。
4. 只把 valid Evidence 交给回答器。
5. 回答生成后再校验一次；生成期间发生变化时丢弃陈旧回答和 Evidence。
6. invalid Evidence 不进入响应 Evidence、旧 citations 或 Prompt；warning 只包含 Evidence ID 和错误类别，不包含源码。

当前 GitHub ingestion 不 clone 目标仓库，因此“文件仍存在”的真实事实源是同一抓取快照中的 `repo_files`，不是本机目标仓库路径。这个映射与现有安全边界一致，也避免执行或访问被分析仓库工作区。M1 不新增 Evidence 表，不修改 schema v4，不持久化完整问答轨迹。

## Grounded Answer 与不可信源码

实现：`backend/app/services/qa_agent.py`。

- Prompt 模板名为 `grounded_repository_answer`，版本 `m1-v1`。
- 系统消息明确把源码、注释、README、文档和字符串视为不可信数据。
- Evidence 使用 `[E1]` 等稳定编号；人类引用统一为 `relative/path.py:start-end`。
- 模型输出必须只引用已提供的 Evidence ID，且其中出现的 Python 位置必须属于有效 Evidence；否则丢弃并改用确定性说明。
- 无 LLM 时仍返回有效 Evidence 和可理解的确定性说明。
- 无有效 Evidence 时统一回答“当前源码证据不足，无法可靠回答。”，不返回 citations。
- 没有传入 M1 数据库/Embedding 依赖的旧内部 Python 调用暂时保留，但明确返回 `retrieval_mode = legacy`、`grounding_status = degraded`。正式 FastAPI `/ask` 总是传入 M1 依赖。

## API 字段与旧 citation 兼容

`backend/app/main.py` 的旧请求只含 `question` 时行为兼容。响应新增：

```text
evidence_schema_version: 1
evidence: Evidence[]
grounding_status: grounded | insufficient_evidence | degraded
retrieval_mode: hybrid | lexical | legacy
warnings: string[]
```

FastAPI 使用 `AskResponse`、`EvidenceResponse` 和 `CitationResponse` 对正式响应做 Pydantic 校验。

旧 `citations[{path, summary, snippet}]` 只由最终 valid Evidence 派生；新旧路径和片段保持一致。兼容至少持续到 M2 完成，最早在 M3 评估弃用；弃用前必须增加版本说明、兼容测试和迁移说明。

## M1 工程评测集

标注：`backend/tests/fixtures/m1_eval.json`；执行：`backend/tests/test_m1_evaluation.py`。

冻结集共 16 条：4 个精确符号、4 个行为改写、4 个过滤、2 个不可回答、1 个 Prompt Injection、1 个过期证据问题，其中至少 4 个中文问题。每条包含 question ID、可回答性、预期 repository/path/symbol/行范围/hash、forbidden evidence、预期 grounding status 和标注意见。

该集合只用于确定性工程回归：answerable `Hit@5 = 100%`、mock hybrid `MRR@10 >= 0.80`、返回引用有效率 `100%`、无效/跨仓库/过期 Evidence 进入最终回答数 `0`、unanswerable 编造引用数 `0`。不得外推为真实仓库或真实 BGE-M3 总体质量。

## 真实文件映射

| 职责 | 实现 |
| --- | --- |
| tokenizer / BM25 | `backend/app/services/lexical_retriever.py` |
| 语义身份与过滤适配 | `backend/app/services/semantic_retriever.py`、`backend/app/database.py` |
| Weighted RRF | `backend/app/services/hybrid_retriever.py` |
| Evidence / validator | `backend/app/services/evidence.py` |
| Grounded answer | `backend/app/services/qa_agent.py` |
| 正式 API 编排 | `backend/app/main.py` |
| 回归与评测 | `backend/tests/test_*m1*`、`test_lexical_retriever.py`、`test_hybrid_retriever.py`、`test_evidence.py` |

## 与准备文档的差异

准备版 `ROADMAP.md` 曾写“至少 50 个冻结问题、Recall@5 >= 0.80、MRR >= 0.65”。本次 M1 开发指令随后冻结为“至少 16 条工程评测、answerable Hit@5 = 100%、mock MRR@10 >= 0.80”，因此 M1 按后者验收并同步路线图。扩展到 50 条以上、真实模型/真实仓库对照和正式科研评测仍属于 M5，不在 M1 夸大完成。

现有架构文档建议 validator 读取“当前 revision 的持久化源码/代码块”；实际实现具体化为一次 SQLite snapshot join，并在回答后复验。没有 schema 冲突，schema v4 保持不变。

## 验证边界

常规测试全部使用临时 SQLite、fixture、Fake Embedding 和 Fake LLM，不访问 GitHub、不执行 fixture 代码、不访问网络、不下载模型。真实 BGE-M3 smoke 不是 M1 完成条件，本阶段不重复执行；M1 只验证现有缓存接入和 `local_files_only` 问答加载约束。

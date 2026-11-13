# V2 迁移记录

## 基线信息

- V1 冻结提交：`d8a4d5646e10034b46739b6e79849b16d304aed1`
- V1 标签：`v0.1-demo`
- 稳定目录：`D:\Project\GitLearnAgent`
- V2 worktree：`D:\Project\GitLearnAgent-v2`
- V2 分支：`v2-development`
- 远程分支：`GitLearnAgent/v2-development`
- 记录日期：2026-06-07

## 当前基线测试结果

已执行：

```powershell
cd backend
D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest discover tests
```

结果：

```text
Ran 8 tests in 0.002s

OK
```

补充说明：当前 shell 中 `python` 不在 PATH，直接执行 `python -B -m unittest discover tests` 失败；使用现有启动脚本中约定的 Conda Python 路径后，后端测试通过。

## 已知问题

- 根目录 `start_all.bat` 在当前 V2 worktree 中仍硬编码调用 `D:\Project\GitLearnAgent` 下的后端和前端脚本。
- `backend/run_backend.bat`、`backend/run_backend.ps1`、`frontend/run_frontend.bat`、`frontend/run_frontend.ps1` 存在本机 Conda 路径假设。
- `backend/app/services/github_client.py` 的 GitHub Token 错误提示中硬编码了 V1 目录下的 `.env` 路径。
- 当前问答引用没有行号。
- 当前 Python AST 分析没有保存函数/类代码块内容、起止行号或内容哈希。
- 当前检索没有 BGE-M3、向量库或 Embedding 缓存。
- 当前 LLM 输出没有 Schema 校验和可追踪重试记录。
- 当前数据库没有迁移框架和 schema 版本管理。

## 尚未迁移或替换的旧功能

- 文件级静态分析仍需升级为函数/类代码块分析。
- 文件级核心排序仍需补充代码块级排序和证据验证。
- 规则检索仍需升级为关键词与语义混合检索。
- 本地规则学习路线仍需升级为诊断驱动的动态学习路径。
- Markdown 报告仍需升级为包含学习过程、证据和评分细则的学习档案。
- 前端仍是 V1 标签页式工作台，尚未迁移为引导式三栏学习工作台。

## 迁移记录模板

后续每次迁移或模块替换时追加一条记录。

```markdown
## YYYY-MM-DD 模块名称

- 日期：
- 模块：
- 功能分支：
- 修改摘要：
- 数据库变化：
- 配置变化：
- 测试命令：
- 测试结果：
- 已知限制：
- 提交 SHA：
```

## 初始记录

### 2026-06-07 V2 开发规范和规划文档

- 日期：2026-06-07
- 模块：开发规范、V1 基线、V2 规划、迁移记录
- 功能分支：`v2-development`
- 修改摘要：初始化 V2 文档，不实现业务功能。
- 数据库变化：无。
- 配置变化：无。
- 测试命令：`D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest discover tests`
- 测试结果：8 个后端 unittest 通过。
- 已知限制：前端构建未执行；当前未修改前端业务代码，且禁止安装新依赖。
- 提交 SHA：待提交后补充。

### 2026-06-08 Python 函数级代码块

- 日期：2026-06-08
- 模块：Python 函数级代码块
- 功能分支：`feat/python-code-chunks`
- 修改摘要：新增独立 Python AST 代码块提取模块，支持函数、异步函数、类、方法、异步方法、嵌套符号、装饰器起始行、多行签名、CRLF、非 ASCII 内容和语法错误警告；在仓库分析成功路径中对 `.py` 文件提取代码块并保存。
- 数据库变化：新增 `schema_versions` 表记录轻量 schema 版本；新增 `code_chunks` 表，保存项目外键、仓库 revision、语言、路径、代码块类型、符号名、限定名、父符号、起止行号、原始内容、SHA-256 内容哈希和创建时间；新增项目级保存、文件级替换、按项目/路径/符号/类型查询和项目删除清理接口。当前仍使用幂等 SQLite 初始化，后续可引入正式迁移工具。
- 配置变化：无。
- 测试命令：`D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest tests.test_code_chunker tests.test_database_code_chunks`；`D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest discover tests`
- 测试结果：最小相关测试 16 个通过；完整后端 unittest 24 个通过。
- 已知限制：仅处理 Python AST 可静态定位的函数和类定义；不分析动态生成符号；未生成 Embedding，未实现混合检索或 RAG；前端暂不展示代码块。
- 提交 SHA：`0f8ccbcefdbb6f26e49d8cbc84876d248336c1ad`。

### 2026-06-08 本地 BGE-M3 稠密向量缓存与基础语义检索

- 日期：2026-06-08
- 模块：本地 BGE-M3 向量生成、缓存与基础语义检索
- 功能分支：`feat/bge-m3-semantic-retrieval`
- 修改摘要：新增独立 Embedding 服务、SQLite 向量缓存表、增量索引器和内部语义检索服务；仅实现 dense embedding，不实现关键词融合、BM25、Reranker、AST 关系扩展、RAG 问答改造或前端页面。`EMBEDDING_ENABLED=false` 时旧分析流程不加载模型。
- 数据库变化：`SCHEMA_VERSION` 升至 4；新增 `code_chunk_embeddings` 表，保存 `code_chunk_id`、`content_hash`、`embedding_input_hash`、模型标识、模型 revision、文本格式版本、`embedding_config_hash`、维度、dtype、归一化标记和 float32 little-endian `vector_blob`；通过外键随 `code_chunks` 级联删除；新增批量 upsert、fresh cache 查询、缺失/过期查询和项目向量读取接口。代码块保存逻辑改为尽量保留既有 `code_chunk_id`，以支持未变代码块复用 embedding 缓存；同一 code chunk 在同一模型/config 下输入变化时清理旧向量，避免缓存无界增长。
- 配置变化：新增 `EMBEDDING_ENABLED`、`EMBEDDING_MODEL_NAME_OR_PATH`、`EMBEDDING_MODEL_REVISION`、`EMBEDDING_DEVICE`、`EMBEDDING_BATCH_SIZE`、`EMBEDDING_MAX_LENGTH`、`EMBEDDING_NORMALIZE`、`EMBEDDING_CACHE_DIR`、`EMBEDDING_QUERY_PREFIX`、`EMBEDDING_DOCUMENT_PREFIX`。`EMBEDDING_MAX_LENGTH` 运行时限制为 16 到 8192。
- 依赖变化：`backend/requirements.txt` 新增并收窄为 `sentence-transformers>=3.0,<4.0`；真实隔离环境解析到 Python 3.12.13、torch 2.12.0+cpu、sentence-transformers 3.4.1、transformers 4.57.6、tokenizers 0.22.2。本次没有安装或升级原 `gitlearnagent` Conda 环境中的 PyTorch，也没有安装向量数据库或 LangChain/LlamaIndex 等框架。
- 模型下载边界：默认模型名为 `BAAI/bge-m3`，也支持用户配置本地模型目录；模型延迟加载，不在导入模块或应用启动时加载；缓存目录来自配置，默认 `embedding_cache`，已由 `.gitignore` 排除；单元测试使用 Fake backend，不下载真实模型。
- 测试命令：`D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest tests.test_embedding_service tests.test_database_embeddings tests.test_embedding_indexer tests.test_semantic_retriever`；`D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest discover tests`；`D:\Programme\Anaconda\envs\gitlearnagent-v2-embed-smoke\python.exe tests\smoke_bge_m3.py`。
- 测试结果：审查修复后 embedding 相关确定性测试 61 个通过；完整后端 unittest 89 个通过。真实 BGE-M3 smoke test 在隔离环境 `gitlearnagent-v2-embed-smoke` 中通过，设备为 CPU，模型 `BAAI/bge-m3` resolved revision 为 `5617a9f61b028005a4858fdac845db406aefb181`，输出 1024 维 float32，文档和查询向量范数约为 1。中文查询“用户身份是如何验证的？”排序为 `authenticate_user`、`initialize_database`、`upload_file`；SQLite 集成链路验证首次生成 3 个向量、第二次 3 个全部命中缓存、修改认证代码块后仅重算 1 个、检索返回 `src/auth.py` 的 `authenticate_user` 10-12 行。
- 是否执行真实 BGE-M3 smoke test：已执行。首次下载/加载写入已忽略的 `embedding_cache\bge_m3_smoke`，未下载本地 7B，未修改原 Conda 环境。首次模型加载约 897954 ms；缓存后复跑加载约 12857 ms；文档编码约 2537 ms，查询编码约 478 ms。进程内存占用未能可靠获取。
- 已知限制：第一版从 SQLite 读取项目向量后在内存中点积排序，适合中小型仓库；未使用 FAISS/Chroma 等外部向量库；长代码块按固定文本格式交给模型，实际截断由 Sentence Transformers/max length 处理；本地模型目录只记录目录路径标识，尚未做模型文件内容级指纹，也不会扫描并哈希数 GB 模型文件；语义检索尚未接入问答、学习路线或前端。真实依赖安装时本机 pip 配置使用 `https://pypi.tuna.tsinghua.edu.cn/simple` 镜像。
- 提交 SHA：原功能提交 `61c042383524580de58493b64a325f4ba1992a15`；审查修复提交 `669420d169bea4b26abc788509ed58b7528ecc75`；合并提交 `3d47c9ced88d8423f50d793f45dec9cd7ea688e1`。


### 2026-11-11 HyDE 与单轮列表重排序（历史阶段，最终结论见下方计分修复版）

- 功能分支：`codex/hyde-rerank`；基于稳定 main 增量开发，未提交、推送或合并。
- 增加两个独立默认关闭开关、短任务参数和共享15秒增强上限；仅增强 RAG/Agent 首次检索，复用现有 Provider、BGE、源码向量、取消、总时限与调用用量。
- 候选在最终截取前补充召回与一轮排序，保留原分数/融合排名，增强选择顺序穿过候选源码回读与证据池；假想说明不进入最终证据或回答。
- 现有执行摘要显示状态、耗时和回退；提供单独增强启动入口，不改正式 `.env`。数据库无 Schema 迁移，历史增加兼容的可选摘要字段。
- 当时阶段记录：固定12题24次首次问答，集中修复后仅复验4项，首次记录保留。该阶段完整通过7/12→8/12，额外等待配对中位数4.31秒；此成绩不是最终v2结论。首次重排3次回退另列。最终结论以计分修复版及验证摘要为准。
- 完整后端1298项中1297通过、1项既有README标题断言失败；相关后端114项、最终增强模块14项通过；网页前端50项和网页/桌面生产构建通过。
- 桌面独立源码同步并新建便携ZIP。唯一真实问答成功保存、重排成功、HyDE因复制的网页缓存配置不匹配回退；随后复用原桌面1992条向量并通过只读召回检查，未追加桌面问答。不将该次回退计为桌面增强成功。
- 说明：`docs/HYDE_RERANK.md`。本机逐题对照、真实记录、包和具体限制位于 `output/hyde-rerank-20261111/`；未涉及TrueSkill或参赛材料。


### 2026-11-11 HyDE 计分修复与 v2 收尾

- 开发验证阶段位于 `codex/hyde-rerank`；本次Git交付限定主仓库，桌面独立副本不在提交范围内。
- 最小修复：融合排序单独读取原候选分数，原检索贡献与一次HyDE贡献相加；HyDE独有候选的元数据分数不再重复相加。生成提示词、最终模型参数、候选上限及正文补全规则未修改。
- 确定性测试先复现顺序错误，再修复；最小相关集33项通过，发行规则3项通过。未重复全量测试或前端构建，复用既有生产构建。
- 离线重放12题精确还原修复前排序；12题的重排输入集合均受影响，仅增强组各复验一次，复用已保存HyDE生成结果并按原实测等待回放；原版不重跑。
- 本轮完整通过7/12，原版7题完整通过均保留，源码区间覆盖441/552→508/552；额外耗时中位数7.02秒、共享增强最长7.47秒。C3本次新增“保证底层关闭”的错误，不沿用上轮8/12，也不宣布严格质量不退步全面通过。
- 新版桌面通过一次真实RAG问答：batch1严格匹配1992条缓存，HyDE与重排均成功；E1引用与688行完整源码可打开，历史重开恢复，后台退出码0且模型关闭。
- 新包 `output/hyde-rerank-20261111/desktop-portable-v2/源鉴-Windows-HyDE重排试用-20261111-v2.zip`；旧ZIP、旧样机和原网页入口保留。本轮实际新增生成HTTP34次，tokens156913。完整记录在 `output/hyde-rerank-20261111/score-fix-v2/`。
- 脱敏的最终验证结论见 `docs/HYDE_RERANK_VALIDATION.md`。桌面同步打包不代表桌面源码已提交GitHub；旧实验目录保持本机原样，不随主仓库提交。

### 2026-11-13 百炼专用重排独立接入与共同候选池对照

- 基线 main 887d696，独立分支 codex/qwen3-rerank；本轮不提交、推送、创建PR或合并，未部署日常目录。
- 增加北京百炼 qwen3-rerank 后端及桌面重排服务选择，保留原DeepSeek默认、HyDE和Agent设置。DPAPI独立保存百炼密钥，旧设置兼容；非法返回和服务失败明确回退。
- 固定Click/HTTPX各6题，BGE有效向量分别1992/533，重算0；每题一次基础检索，A/B/C共享候选及摘录，关闭HyDE和后续补查。两组各12次全新回答，独立匿名盲评9/12与10/12，原DeepSeek通过题全部保留；百炼另有1题证据支持/完整性取舍和1题源码交付覆盖减少，维持DeepSeek默认，百炼为可选项。
- 正式48次HTTP、冒烟1次、独立桌面2次，总51次；正式无重试/修复/回退，未因成绩重复采集。桌面真实回答、引用及688行全文打开、保存与完整重开恢复、正常退出验证；日常设置、密钥密文、源码/向量/历史及快捷方式保持。
- 全后端1314项执行，只有既有README标题断言失败；前端314项、桌面相关40项、shell 7项通过，网页及桌面生产构建完成。具体记录见本机 output/qwen-rerank-20261113，产品说明见 docs/BAILIAN_RERANK.md。

### 2026-11-13 重排设置工程收尾

- 默认重排改为品牌无关的“使用当前生成模型”，复用有效生成API/基础模型；独立方式本轮只适配阿里云百炼北京qwen3-rerank。旧deepseek/bailian值兼容迁移，新模式优先，读取不落盘。
- 两层桌面设置只在开关开启时显示，独立服务的地址、模型、密钥、超时保留；密钥来源优先级明确，DPAPI留空保留/明确清除，保存不收费检测。独立配置与回答、Planner、HyDE及预算分离。
- 失败只回退基础排序，不自动转用另一服务；摘要显示实际模型、所选重排方式和中文原因。默认短任务参数保留，通用兼容接口不强加未配置的thinking扩展。
- 桌面保存限制补齐导入worker、后台更新执行及排队窗口；不会因保存设置重启进行中的问答或导入。
- 本轮只验证工程接线和部署，不重跑12题，不作算法质量提升主张。私有备份、独立验收、真实调用账本和本机恢复说明留在output/qwen-rerank-closeout-20261113，不随源码提交。

# RN-FE-01 前端源码问答工作台

## 当前状态与提交前收口（2026-09-24）

**状态：固定场景真实端到端验收通过，进入提交前收口。** 固定项目 `fdc0c6a5-c18d-4e3f-a258-a82a8af943d7`、revision `6aabf099bfdd4c1e75fe8d0e0d4241372b988ab1`、同一 CommandCollection 问题，按 RAG、Agent、RAG、Agent 串行完成 4 次真实工作台提交。4 次均成功，合计 8 次模型逻辑调用和 8 次 Provider HTTP 尝试；中间进度、最终引用与源码阅读联动，以及保存记录 62～65 的串行答案 SHA-256／引用 ID 核对通过。数据库无 request_id 列；该保存核对不构成按请求 ID 的数据库直接关联。成功响应未提供协议修复次数，不能推断为 0。服务进程四次之间未变化，不能推断首轮为冷启动。下文较早的“待修复”“READY_FOR_MANUAL_UAT／RETEST”均为当时历史状态，后续定点修复和本次验收记录优先；不抹去其事故和未知归因。

**提交前审查范围。** 两个语义槽位由共享 `EmbeddingService` 的非阻塞 semaphore 在提交前占用，最多两个已接受任务；任务仍运行时即使请求已停止等待也占槽，其他请求得到 capacity 并走受检词法候选。模型加载受锁保护，迟到结果仅保留在线程私有局部值，不能再次晋升 Evidence。普通 `/ask` 与流式入口共用 `_execute_ask`；成功响应校验、保存后才发 completed，断线可能仍完成并保存。流式进度来自受 Agent 步数／工具次数及固定检查点约束的单请求事件，客户端无自动重发。以上是代码路径与现有隔离回归支持的结论，不是对任意并发量的系统容量保证。提交前发现成功安全日志曾把缺失校验结果投影成 false，已改为保留 null，并补定点回归。

**接口与依赖。** `execution_mode` 缺省为 Agent；`citation.evidence_id`、`execution_summary` 和地图身份字段均为新增可选响应字段。`/map` 的 `core_files.content` 已移除：仓库内前端地图只消费树、模块和文件元数据，现有相关测试不依赖正文；这只能证明可见消费者未受影响，外部客户端兼容性未知。前端新增 `prismjs`、`react-markdown`、`remark-gfm` 与 `@types/prismjs`，既有 lockfile 同步变化。2026-09-24 既有 npm audit 记录为 4 项：Vitest → `@vitest/mocker` 中等风险影响测试链，Vite → PostCSS → nanoid 高风险影响开发服务／构建链。依赖风险应在发布前针对实际公告、修复版本和使用入口单独处理；本次未执行强制修复或无关升级。

**专项未验证与限制。** 真实慢语义触发的词法回退仍未观察到；历史 `citation_format_invalid` 的具体根因未知；隔离回归和这 4 次固定场景成功不等于全面可靠或整个系统全面验收。断线后后端可能继续完成并保存，当前无幂等、断线续传或服务端取消合同；既有项目删除 `cleanup_pending` 风险、地图无完整源码浏览接口、外部 `/map` 消费者未知及上述依赖审计项继续保留。下文各阶段记录按发生时间保留，不把旧待办自动视为现状，也不因这些未知项无限追加同题实验。

## 2026-09-24 有限真实端到端验收（固定场景 4/4 通过）

**授权与身份。** 在真实工作台串行提交 4 次顶层问答：RAG、Agent、RAG、Agent；固定问题“CommandCollection 如何从多个 Group 中查找和聚合子命令？”。提交前以现有项目记录和只读 SQLite 核对项目 ID `fdc0c6a5-c18d-4e3f-a258-a82a8af943d7`、本地来源 `D:\DemoRepos\click`、完整 revision `6aabf099bfdd4c1e75fe8d0e0d4241372b988ab1`、workspace `16ac11ba-c2c6-5372-8324-0e15f995a495`。预检时 8000/5173 均未监听；确认无活动请求后，使用项目 Python 环境与原启动模块启动后端，以现有 Vite 方式启动前端，四次之间未重启。Provider 已配置 `openai_compatible` / `deepseek-v4-pro`，未读取或输出密钥；总截止 60 秒、默认工具限额 15 秒、回答保留 5 秒、每逻辑调用最多 2 次 HTTP 重试，均未调整。没有导入、更新、重索引、模型下载、schema 或生产代码修改。

**实际提交。** 独立 Chromium 会话经真实模式按钮和发送按钮调用流式入口；请求守卫恰好放行 4 次，没有普通 `/ask`、额外探针或重发。四次网络响应均为 200，提交与返回的项目、revision、执行模式一致。第 1 次在最终答案前实际看到 BASE 完成、5 条新增证据和答案生成；第 4 次在 pending 时读到生成后引用检查，再看到成功终态。每条最终页面显示“请求已结束”。前端解析终态后在 `finally` 调用 `reader.cancel()`，浏览器 `requestfailed` 计数因此每次加 1；不能仅以该计数认定问答断线。第 1 次旁路 `response.text()` 无法读取已消费的流体，移除该监听后没有重发；随后依据页面消费的结构化结果、进度、网络条目和只读保存记录核对，不声称旁路掌握原始终态帧数。

| 序号 | 模式、请求 ID | BASE／语义／证据 | Planner、BASE 外工具 | 模型逻辑／HTTP 尝试 | 终态；浏览器 fetch／服务端编排 | 保存核对 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | RAG `b0ce16af-72a3-44c7-b514-108af2275b12` | 成功；hybrid／completed；5 Evidence、5 citations；模型 unready→ready | 0；0 | 1／1 | completed；15,785／15,718 ms | id 62，答案 SHA-256 及引用 ID 匹配 |
| 2 | Agent `3a9d11e3-acc6-41bd-a52b-41f7a22261c6` | 成功；hybrid／completed；5／5；模型 ready | 2；2（成功 1、规则拒绝 1），no_progress 后合法完成 | 3／3 | completed；11,466／11,406 ms | id 63，答案 SHA-256 及引用 ID 匹配 |
| 3 | RAG `97301aec-e4ff-4e8f-ba0f-8f17fa9e0976` | 成功；hybrid／completed；5／5；模型 ready | 0；0 | 1／1 | completed；5,253／5,186 ms | id 64，答案 SHA-256 及引用 ID 匹配 |
| 4 | Agent `05b1ee98-f0f9-432a-9d55-1d7d3191302f` | 成功；hybrid／completed；5／5；模型 ready | 2；2（成功 1、规则拒绝 1），no_progress 后合法完成 | 3／3 | completed；10,517／10,359 ms | id 65，答案 SHA-256 及引用 ID 匹配 |

**阶段与安全合同。** 四次词法耗时依次 593/577/547/531 ms，命中数均为 20；查询编码 469/139/141/110 ms，向量读取 1452/828/922/780 ms，语义等待 11795/1108/1202/1016 ms。第 1 次模型加载完成耗时 9733 ms，后 3 次该字段未提供；开始时模型状态依次为 unready/ready/ready/ready。显式 `fusion_ms=0`、`promotion_ms=0` 均为毫秒取整，不把子阶段直接相加当作墙钟。四次最终引用及回答后证据校验通过，关系校验字段为 `null`（本场景未提供适用结果）。每次正文 [E1]、底部 [E1]、源码栏及放大阅读均核对为当前请求 `src/click/core.py:2147–2202` / `CommandCollection`、同一完整 revision；其余 citation/Evidence 元数据逐一与当前项目、revision、身份和范围对应。身份与格式通过不代表全部自然语言结论已被独立证明。

成功响应未提供首次／最近答案协议失败子码，也未提供有界修复是否尝试、次数或结果，均记录为“未提供”，不由总逻辑调用数猜作 0 或 false。两个 Agent 摘要的 `diagnostics_truncated=true`，不推断被裁剪内容。四次模型逻辑调用合计 8、Provider HTTP 尝试合计 8；无可靠 token／计费字段，不估算费用。只读 `chat_answers` 查询限定本轮前最大 id 61、固定项目和问题，以答案 SHA-256 和 citation ID 分别匹配 id 62–65。该表无 request_id 列，所以这是串行执行加内容一致性的保存核对，并非数据库直接按 request_id 关联；未见重复保存，也未删除记录。

**证据与边界。** 安全浏览器结果、只读保存核对、fetch 计时和截图保存在仓库外 `C:\Users\Anon\.codex\visualizations\2026\09\24\01a0d2f6-a1ea-7c00-b60a-b64edc05c792\rn_live_uat_20260924_2146`，不保留 prompt、Provider 原始输出、密钥或完整源码。固定项目／revision／问题的 4 次真实端到端验收通过，未触发停止条件。历史 `b09e186a-7b07-43a9-b7f1-a4707f68bb54` 引用失败的根因仍未知；本次成功不能保证以后不再发生。四次语义均正常完成，真实慢语义触发的词法回退仍未观察到，BASE 间歇性超时未全面关闭；整个系统尚不能视为全面验收通过。

## 2026-09-24 最终答案引用协议失败定点诊断（诊断能力已补齐，原失败根因仍未知）

**历史现场。** 同项目 `click`、revision `6aabf099bfdd` 的 RAG 请求 `b09e186a-7b07-43a9-b7f1-a4707f68bb54` 已完成 BASE 并形成 5 条合法证据，最终以 `citation_validation / citation_format_invalid` 失败。随后 Agent 请求 `0ec55e80-a303-4ea8-ae08-903a043977e2` 成功，但属于独立请求。当前后端启动器仅向终端输出安全诊断，没有该请求的落盘日志；现存截图没有 `final_answer_initial_failure`、`final_answer_protocol_failure`、`final_answer_repair_failure` 的细节，且当前浏览器没有可回取历史响应的调试入口。**历史请求细分失败原因无法恢复。** 修复是否尝试及次数、首答与修复输出的各自子码均为未知；不能把缺失推成 false 或 0，也不能用后续 Agent 成功代替这条 RAG 的解释。

**核实的生产协议链。** `/ask` 与 `/ask/stream` 共用 `_execute_ask → run_bounded_agent → run_finalization_phase → answer_from_evidence`。同请求 EvidenceStore 经 CitationValidator（适用时再经 RelationValidator）重验；服务端为权威 Evidence 建立请求内 `A1..An` 别名，并在首答与一次有界修复提示中发送同一 JSON schema、别名列表和禁止模型自行写 `[E#]` 的约束。Provider 适配层只交出模型 `message.content`；`render_structured_final_answer` 严格解析 JSON 与别名，再由服务端根据 Evidence ID 和权威路径/行号渲染正文引用；最终正文引用和生成后 Evidence 再校验。初答或修复仍不合规时保持失败。成功结果在响应合同与截止时间检查后保存一次，再发布流式 `completed`；失败走唯一 `failed` 终态且不保存。稀疏编号、乱序、合法重复引用与跨项目/revision/身份拒绝由现有隔离回归覆盖。本轮没有找到可证明的首答/修复协议冲突、别名替换或保存顺序错误，不能推断原请求违反的是哪一种子码。

**本轮确认并修复的缺陷。** 原失败详情虽由 recorder 生成安全子码，但 4 KB 有界裁剪在高诊断量时可删除 `final_answer_*_failure`；前端 `projectAskFailure` 又完全丢弃这些对象和修复状态；后端把缺失的修复状态投影成 false。现在有界裁剪优先保留三个有限 `stable_code`，旧快照缺失布尔值保留为未知；前端只白名单投影子码和少量 `violation_kind`，在失败摘要显示请求 ID、提交时的 RAG/Agent 模式、首次/最终子码、修复 0/1 次及协议重验/最终结果。缺字段显示“未提供”，明确未尝试显示“未尝试”；不透传原始答案、prompt、异常正文或任意 Provider 内容。普通 `/ask` 和流式失败复用同一前端投影。**这些是诊断丢失问题的修复，并非对历史 `citation_format_invalid` 的根因修复。** 没有改动生成/修复次数、提示词、解析器、CitationValidator、RelationValidator、Evidence 身份、BASE 策略或总预算。

**可控验证。** 新增的后端缺失状态/有界裁剪回归及前端安全投影回归在修复前各失败，修复后通过。项目 Python 环境的协议、最终生成及诊断测试本轮为 76 项通过；路由、模式与可观测性另轮 77 项通过；M1 与有界编排另轮 16 项通过。前端安全投影/流式/摘要定点测试 112 项通过；加入工作台和提交行为回归的后续一轮 134 项通过，TypeScript 检查及 build 通过；`git diff --check` 退出码 0。隔离浏览器直接装载摘要组件和可控失败对象，未装载 App、未提交正式问答；确认新诊断和旧字段缺失均可辨识。浏览器首轮唯一 404 是隔离页缺 favicon，补入 data URL 后重新装载无新增错误；临时页面和本轮浏览器会话已清理。8000 监听仍属项目 `gitlearnagent` Python 启动器，自动重载工作子进程启动于本轮后端编辑后，`GET /api/health` 返回 200；5173 仍属原项目 Vite 进程，未手动重启或终止服务。以上替身结果不能归因历史请求的实际模型输出，也不能声称真实 Provider 引用格式问题已经关闭。下一次人工复验只需在刷新后的现有工作台观察新失败条目的“本次执行摘要”：记录请求 ID、模式、首次/修复/最终安全子码和最终阶段；如果模型仍返回非法格式，失败且零保存是正确行为。BASE 慢语义补丁仍独立待真实场景复验，整个源码问答验收仍未通过。

## 2026-09-24 BASE 检索可靠性差量修复（READY_FOR_MANUAL_RETEST）

**历史现场与归因边界。** 请求 `309402e5-d667-47cc-a2a2-443439e1c1ee` 的截图确认 BASE 报告超时、约 33 秒后以 `retrieval / evidence_insufficient` 结束；紧接着同题 RAG 约 5 秒取得 5 条证据，之后 Agent 也成功但出现一次补充工具失败及 `no_progress`。没有该超时请求的可归属分阶段日志，因此历史具体瓶颈仍是 **Unknown**。以下替身复现证明代码层面的脆弱路径，不构成对该请求的完整归因。本轮未向正式 `/ask`、流式入口或 Provider 发送请求，也未重索引或改动正式数据库。

**核实的生产链路与预算。** `POST /api/projects/{project_id}/ask` 和 `/ask/stream` 共用 `_execute_ask → run_bounded_agent → BASE ToolRegistry.execute(search_code) → retrieve_code`；流式只将同一执行放到工作线程，并由请求私有队列传事件。`_execute_ask` 在路由入口创建单调时钟请求截止时间；`RequestBudget` 从同一截止时间划出回答整理保留量；ToolRegistry 在 BASE 开始时取“请求截止、工作截止、工具超时”三者最早值。默认设置分别为 60 秒总预算、40 秒单工具限额、5 秒回答保留，实际可由配置覆盖。V1 `HybridRetriever` 同步词法检索后进入 `SemanticRetriever`：模型身份获取可触发按需加载，随后读取已存向量、编码查询、向量评分、融合。模型由进程内 `EmbeddingService` 复用，加载和编码受同一锁保护。原实现的 `check_active` 只在这些同步调用返回后运行；若调用耗尽工具时限，已有词法候选被整个 BASE 超时路径抛弃，无法经过 `CandidateEvidencePool → 当前项目/revision 权威重读 → EvidenceBuilder → EvidenceStore`。原工具超时是**返回后检查**，不能中断底层模型调用。V2 可选检索同样是词法后同步稠密阶段，已纳入同一有界等待修复。

**策略与安全边界。** 保持原总截止时间、ToolRegistry 权限/预算和回答保留量。词法结果成为本次请求的明确中间结果；进入语义阶段前检查活跃状态及剩余工具时间，从该剩余时间中最多保留 1 秒、至少保留 20 毫秒且按 10% 分配给融合、候选权威重读及 Evidence 晋升。语义工作由当前 `EmbeddingService` 持有的两个可复用执行槽运行，槽满立即转用本次请求的非语义候选；等待到阶段切点即停止等待。模型加载状态为 `unready/loading/ready/failed`，同一实例的加载锁阻止重复加载；加载仅用本地文件，不触发下载或索引构建。正常返回仍使用原融合、排序、去重及 V2 策略，未改变 top-k 或质量阈值。仅已知的 Embedding、维度/值及 I/O 类可恢复错误进入降级；未知编程异常继续失败。词法或 V2 符号候选仍必须经过原身份/类型/排名校验和权威重读；没有合法 Evidence 时不调用回答模型。请求过期、真正取消及 ToolRegistry 的终态门禁仍优先，不允许回退绕过。RAG 仍无 Planner 或自主补充工具；Agent 仍按原有界规则；成功一次保存、失败零次保存。

**不能中断的工作与剩余风险。** `Future.result(timeout)` 只结束请求等待，不取消已经开始的原生加载/编码。执行槽一直占用到该工作真实返回，之后自动释放并由同一服务实例复用；迟到结果仅停留在工作线程局部变量，不能写已终结请求的 EvidenceStore、终态或答案。最多两个慢工作，无逐请求无界排队；若底层调用永久不返回，该进程的两个语义槽会一直占用，后续请求走安全的非语义路径，进程关闭也可能受 Python 执行器等待影响。模型 `ready` 只指模型已加载，**不表示项目向量索引可用**。词法检索、权威数据库重读、融合及生成仍是同步/协作检查路径，极端情况下也可能越过截止时间，随后由原预算门禁拒绝；本轮没有把这些阶段宣称为可抢占。

**新增可观察性。** 原有有界 recorder 的 BASE 诊断投影增加单调时钟测得的词法命中数/耗时、模型开始与结束就绪状态、模型身份/加载完成耗时、向量读取、查询编码、向量评分、融合、Evidence 晋升耗时，以及 `semantic_status=completed/skipped_budget/timed_out/capacity/failed/disabled`、阶段剩余预算、等待耗时和实际来源。未执行或未及时结束的阶段不填写耗时，子阶段不相加充当墙钟。安全诊断仅携带计数、受限状态与耗时，不携带问题、源码、向量、prompt、密钥或任意异常正文。前端在已校验词法回退成功时说明语义未完成；摘要区分“未开始”和“执行中超时”，零合法证据继续失败，总预算耗尽继续显示预算失败；Planner `no_progress` 保留独立说明。旧请求/响应缺这些可选字段仍可显示。

**可控验证与人工复验。** 修复前隔离 SQLite 项目中，合法词法命中加 120 毫秒语义替身、60 毫秒 BASE 限额时，实际等约 125 毫秒，关键回归失败；修复后改用 300 毫秒限额与 750 毫秒替身，BASE 在等待切点回退并得到合法 Evidence。其他替身覆盖正常 V1 混合、V2 慢稠密、冷加载、热模型慢编码、加载故障、零命中、跨项目/revision/哈希拒绝、取消/过期、迟到结果不回写、两槽并发限流及单次模型加载。现有问答路由/流式回归覆盖 RAG Planner=0、Agent 预算、终态一致和成功/失败保存次数；隔离浏览器的成功/失败摘要展示正确且没有 API 请求。它们不能证明真实 BGE-M3、正式索引或 Provider 的全链路可靠性已关闭。下一次人工验收：确认后端加载本次代码，在现有项目/revision 各提交一次 RAG 与 Agent 问题；若 BASE 降级，记录请求 ID、BASE 状态、`semantic_status`、模型状态、各完成阶段耗时、实际来源、新增 Evidence 数和最终终态；若总预算耗尽，保留预算原因而不解释成“仓库无答案”。最高状态为 `READY_FOR_MANUAL_RETEST`。

本次在项目 `gitlearnagent` Python 环境中，直接相关 229 项后端回归通过；进一步收紧“已知可恢复错误”范围后，检索/BASE 49 项再次通过；最终小范围耗时字段与代码整理后，直接受影响 148 项再次通过。前端完整 165 项、build 与 `git diff --check` 通过。隔离 Chromium 页面分别展开合法回退成功和零证据失败摘要，控制台错误 0，未发 `/api` 请求。测试中曾遇到 Windows 后台读取尚持有临时 SQLite 文件的清理竞争，测试等待真实工作结束后通过；残留的单个本次测试临时目录经核对后尝试清理，但自动审批以 `blocked by policy` 拒绝，未再删除。当前 8000 后端由 RepoNoesis 临时启动器调用项目 Python 运行，自动重载子进程的启动时间晚于本次后端源码修改，健康检查 200 且无已建立的 8000 连接；5173 Vite 服务保留。本轮不手动重启、不中断用户服务。

**刷新后人工复验的新现场（19:20–19:21）。** 用户截图中，同项目 `click`、revision `6aabf099bfdd`、同题的 RAG 请求 `b09e186a-7b07-43a9-b7f1-a4707f68bb54` 已完成 BASE 并取得 5 条证据，初始/生成输入及回答后证据校验均显示通过，但最终以 `citation_validation / citation_format_invalid` 失败，没有交付答案；随后的 Agent 请求成功，BASE 同样取得 5 条证据，补充检索因 `no_progress` 止步后基于现有证据回答。这个新失败位于答案引用格式/协议阶段，不是截图所示的 BASE 超时。RAG 失败摘要未展示语义状态或实际来源，因此不能用它断言慢语义回退已通过或失败。现有截图只给汇总原因码，尚不能确定模型首答或有界修复各自违反了哪项结构化协议。后端安全失败详情已设计为保留 `final_answer_initial_failure`、`final_answer_protocol_failure`、`final_answer_repair_failure` 等有限子码；若本次终态帧或后端 `ask_failed` 单行日志仍可取得，可据该请求 ID 做定点核对，不需要重发正式问答或提供原始模型输出。RN-FE-01 的真实问答人工验收仍有此独立失败，BASE 慢语义分支仍待人工触发观察。

## 2026-09-24 新截图：BASE 超时与顶部提示

后续同日截图：用户报告关闭并重开后，相同问题的 Agent 回答成功，BASE 新增 5 条合法证据，最终引用/生成后校验通过；但 BASE 外工具仍有 1 次失败，Planner 以 `no_progress` 提前结束，靠已有证据完成。当前后端/前端进程均始于 17:50 左右，与上一轮观测的 PID 不同；不能将这次成功解释为同一 Python 进程中的模型已被前次请求预热，也不能单凭 PID 确认进程更换由用户关闭操作引起。前次 RAG 超时与后次成功共同说明“持续零命中/索引永久损坏”缺乏支持；还不足以区分模型加载、编码、CPU/GPU 负载或检索其他环节的延迟。

当前判定：**间歇性触发的系统可靠性风险，具体瓶颈未定位**。`EmbeddingService` 按需加载并在进程内复用；V1 `HybridRetriever` 先做词法检索，再进入语义检索。如果语义阶段触及工具/请求时限，活跃检查会终止整次 BASE，已有词法候选也无法走完 Candidate→Evidence 晋升。此路径需要独立、可控的冷/热加载与慢语义替身回归，再选择启动预热与就绪门禁，或在预算内安全保留词法结果的方案；不能仅凭截图调大预算或关闭校验。生产请求缺少可归属的模型加载、编码与检索分段时长，故本轮不宣称根因已修复。用户即时操作：仅在收到**明确终态**且摘要显示 BASE 超时时，手动重问一次；无需删除、重索引或反复关闭项目。若再次超时，保留请求 ID、BASE 状态和当时服务端安全诊断，不自动重试；连接中断且最终状态未知时先核对记录。

用户补充：第一次失败后，立即以同一问题重问，主观观察约五秒内得到带 E1–E5 引用的 RAG 回答；新截图显示 BASE 完成并新增 5 条 Evidence，随后三个具名引用校验阶段及回答完成。它证明这次重问在同一项目/revision 下找到了可验证源码证据，进一步排除“该问题在仓库中必然没有答案”的解释；它不能反推前次超时的具体耗时点，也不能证明是模型冷启动。源码中 `EmbeddingService` 确实按需加载并在服务实例内复用模型，所以首次加载是可能机制；其他瞬时负载或检索阶段延迟仍可能，缺少前次请求的细分安全诊断时保持未知。本补充不触发正式重跑、不调整预算或检索合同。

新截图请求 ID `309402e5-d667-47cc-a2a2-443439e1c1ee` 的**截图内**安全摘要显示“基础检索：超时”，耗时约 33 秒，终态为 `retrieval · evidence_insufficient`。现存仓库和启动器目录没有找到该 ID 的可归属日志；截图证明 BASE 报告了超时，不能证明具体慢在模型首次加载、检索、设备或其他环节，也不能由后续成功问答倒推原因。后端现有分类在未形成足够证据时返回 `evidence_insufficient`，前端旧标题只看 `agent_status`，因此把超时显示为笼统“证据不足”。本轮只在“retrieval + evidence_insufficient + BASE 确实 timed_out”的受限诊断组合下，将回答卡片标题改为“基础检索超时”，说明未在时限内形成足够的可验证证据，并保留原始阶段、原因码、请求 ID 和折叠摘要；不宣称仓库没有相关源码，不改变后端检索、超时预算或失败合同。

旧顶部“请查看安全诊断卡片”是结构化失败的通用 `ApiError` 文案，工作台的自动消失规则不把“未生成”视作需持续显示的关键词，因此约五秒后消失；普通页面点击没有删除请求卡片。现在结构化失败只在该回答原位显示持久的失败卡片，不再同时弹出这个无定位能力的顶部提示。前端集成测试覆盖普通标签切换后失败卡片仍在同一项目上下文，单独渲染测试覆盖 BASE 超时说明；完整前端测试 163 项和 build 通过。此处没有正式 `/ask` 重跑、Provider 调用、后端修改或数据库变更。

## 2026-09-24 小范围验收收尾

三张截图拍摄于上一轮差量修复之前。当前实现已能在零 Evidence 时抑制空集 CitationValidator 的“通过”进度，失败详情已保留安全 BASE 状态与计数，失败进度会保留 BASE 结束事件；本轮未重复改动这些规则。截图中的失败请求 ID 为 `dbf6b65c-048e-45d4-ae45-f0d7ee45cf67`。按该 ID 前缀查现存仓库、`output/` 和桌面启动器临时目录，没有可归属的日志或记录；启动器直接启动后端，无文件日志重定向，失败请求按合同不写 `chat_answers`。**历史失败原因无法恢复**：不能区分 BASE 异常、零命中、候选未晋升或后续不足；后来的成功请求和隔离零命中样例都不用于归因。

重复“证据引用预检完成”不是同一流帧重复：finalization 在初始证据、回答后各调用一次权威 CitationValidator，`answer_from_evidence` 在生成输入处又调用一次。旧 recorder 仅按是否开始生成区分前后，前两个真实检查共用文案；旧前端还按事件类型/阶段/结果压缩，可能合并真实检查。现在事件增加有限 `checkpoint=initial_evidence|generation_input|post_answer_evidence`，前端分别标为“初始证据身份校验”“生成输入证据校验”“回答后证据复核”，保留三个真实事件；传输重复仍按请求身份及递增 `sequence` 忽略，绝不按文案去重。零证据“通过”继续抑制；失败进度标题明确“请求失败”。旧事件没有 checkpoint 时沿用阶段文案。

`execution_summary.agent_elapsed_ms` 取自有界问答编排的 `budget_usage.elapsed_ms`，起点在 `_run_bounded_agent_stages` 内，终点在形成回答；它不覆盖此前的路由预处理，也不覆盖随后响应校验、保存，故界面称“服务端问答编排耗时”。`tool_duration_ms` 是 ToolRegistry 的工具执行耗时累计，**包含固定 BASE 检索**；`tool_calls_attempted`、成功/失败次数及预算使用量只计 BASE 外的补充/回退工具。因此 RAG 的 Planner 与 BASE 外工具计数为 0 而工具耗时非 0 并不矛盾。界面明确“BASE 外工具执行尝试/结果”和“工具执行（含基础检索）”，不重算、不拆分无法精确区分的耗时。

隔离流式路由回归使用临时数据库、可控检索/Planner/模型替身，正式 Provider 调用为 0：RAG 成功保留三个 checkpoint、Agent 补充阶段提前结束后成功、RAG 零命中证据不足，三次请求分别终结，成功保存两次、失败保存零次。前端测试覆盖同一序号重复帧被忽略、不同 checkpoint 真实事件均显示、失败标题与零证据旧事件过滤。直接受影响后端回归 85 项、前端完整测试 162 项与 build 通过；本轮未向正式 `/ask` 或流式入口提交问题。真实 Provider UAT 仍待人工确认，最高状态保持 `READY_FOR_MANUAL_UAT`。人工复验：在当前 5173 服务中先看旧失败卡片是否显示“请求失败”和 BASE 诊断；随后仅在已授权条件下各提一次 RAG、Agent 问题，观察三个具名校验点、摘要计数与含 BASE 的工具耗时。不要用新请求解释旧失败。

## 2026-09-24 用户实测回报与差量修复

用户提供同一 Click 项目、同一 revision、同一问题的三张浏览器截图：一次 Agent 成功，摘要显示 Planner 请求 2 次、工具执行尝试 2 次及可恢复的 `no_progress` 终止；一次 RAG 成功，摘要显示 Planner 与补充工具均为 0；更早一次 RAG 返回 `retrieval · evidence_insufficient`。这是已发生的真实环境部分人工验收，不能推广为整个 Provider UAT 通过。失败截图只保留最后三条引用预检事件，缺少 BASE 状态，不能据此认定是零命中、候选拒绝、瞬时错误或索引缺失；本轮没有向正式 `/ask` 或流式入口再发送问题。

核查发现失败详情后端已携带受限 `base_retrieval` 状态及命中、候选、新增证据计数，前端安全投影却丢弃，完成后进度又截取最后三条，因此用户看不到失败前 BASE 发生了什么。零 Evidence 的 CitationValidator 空集校验会得到真值，旧进度将其重复显示为“通过”，容易误解为真实证据已校验。差量修复保留后端原有诊断与校验合同：零 Evidence 时不发送面向用户的 `citation_checked` 通过事件；失败回答若诊断证据数为 0，也过滤旧流中的空集“通过”。失败摘要现在投影受限的 BASE 状态和计数；失败进度保留 BASE 结束事件、按实际状态表述。只有 BASE 确实完成并有可靠计数的状态才显示新增证据数字。Agent 成功且结构化终止码为 `no_progress` 时，将截图中的固定英文告警解释为“补充检索连续未增加证据，已基于现有证据完成回答”，避免误读为整个请求失败。成功和失败仍由原业务终态决定，没有自动重试或 RAG→Agent 升级。

可控零命中生产编排测试证明 RAG 不调用回答模型、返回证据不足、失败投影保留 `zero_hit` 和新增证据 0，且不发空集引用通过事件；前端测试覆盖安全失败帧投影、失败进度与摘要。该受控测试无法追溯截图中那一次失败的确切 BASE 状态；后续若再次出现，应直接查看该条失败回答的新“本次执行摘要”和 BASE 进度，记录状态及计数，再决定是否修复检索。当前批次状态为**部分人工验收，RAG 间歇性证据不足原因未确认**；此前隔离浏览器验证与地图人工验收记录仍有效，`READY_FOR_MANUAL_UAT` 仅是此前实施时的最高结论，不作为本次实测后的全面通过结论。

本轮差量回归使用项目 Python 环境运行 `tests.test_ask_execution_modes tests.test_ask_observability tests.test_m1_ask tests.test_product_api`，84 项通过；前端完整测试 159 项与 build 通过，`git diff --check` 通过。浏览器仅装载不含 `App` 和 API 调用的临时隔离 React 页面，在既有 5173 Vite 服务验证失败卡片，不触发任何 `/ask`：1440×900 与 390×844 的明暗主题均检查，宽度无横向溢出；失败 BASE 状态、折叠摘要展开及空集“通过”过滤均可见。截图保存在仓库外 `C:\Users\Anon\.codex\visualizations\2026\09\24\01a0d222-3959-70a3-9cc5-ecc919f74441\rnfe-feedback`。临时页面和本轮 Playwright 会话已清理，原有 5173/8000 人工验收服务保留。此浏览器核对只验证展示修复，不证明截图中的真实失败由零命中造成，也不新增真实 Provider UAT。

## 2026-09-24 模式与进度实施前合同核查

当前 `POST /ask` 为同步最终 JSON，未提供流式事件或幂等重试。旧客户端未传模式时执行有界 Agent：请求级 BASE 固定 `search_code`，经 Candidate 校验、数据库按项目/revision/块 ID 权威重读、EvidenceBuilder 和 EvidenceStore 晋升；其后 Planner 可按预算选择补充工具，最终由共用 finalization 生成回答并做 CitationValidator、适用的 RelationValidator 与正文校验。诊断 recorder 已在 BASE、Planner、工具、生成、校验实际调用点记录，但此前仅供最终投影，不会在执行中发送。路由先校验响应，成功 `save_chat_answer` 一次；任何此前失败不保存。前端 requestGate 防本地重复提交及旧上下文回调，不构成服务端幂等。同步执行若直接放入异步流生成器会阻塞发送，因此流式入口须把同一执行核心放在线程中，事件用请求私有队列转给异步响应。地图拖拽和鼠标中心滚轮缩放已获本批人工验收通过；真实 Provider UAT 与旧请求影响仍未确认。

## 2026-09-24 RAG／Agent 与实时进度实施

`AskRequest.execution_mode` 仅接受 `rag|agent`，缺省 `agent`，保留旧客户端的有界 Agent 行为；非法值由 Pydantic 返回 422。RAG 执行固定 BASE `search_code` 后直接进入现有 finalization，不构造或调用 Planner，也没有自主工具循环。Agent 保留 BASE 后的有界 Planner 补充阶段及预算、权限、终止规则。两者均使用同一请求的 Candidate→权威 Evidence 晋升、EvidenceStore、生成、引用与关系校验、响应校验和一次成功保存合同；RAG 的 ToolContext 绑定失败会封闭失败，不进入旧的平行 fallback 检索。证据不足和其他失败不静默升级模式。

`POST /api/projects/{project_id}/ask` 保留最终 JSON，新增可选 `execution_mode`、`client_request_id`、`repository_revision` 请求字段以及响应 `execution_mode`。`POST /api/projects/{project_id}/ask/stream` 使用同一 `_execute_ask` 核心，一次提交只有一次执行和最多一次保存。流为逐行 NDJSON，每帧包含服务端 `request_id`、客户端 `client_request_id`、`project_id`、`repository_revision`、递增 `sequence`、有限 `type` 和该类型的安全字段。事件类型为 `request_received`、`base_started`、`base_completed`、`planner_started`、`planner_stopped`、`tool_completed`、`answer_started`、`citation_checked`、`relation_checked`，终态为 `completed`（带已经校验并保存的完整 `/ask` 响应）或 `failed`（带安全失败详情或空详情）。关系校验没有实际运行时不产生其通过事件；可恢复 Planner 终止是阶段事件，之后仍可成功终结。事件来自请求私有 recorder 的实际调用点，异步响应由工作线程经事件队列发送；不含原始 prompt、模型响应、密钥、异常正文或工具参数。

`citation_checked` 与 `relation_checked` 带 `phase=before_answer|after_answer`：生成前的 Evidence/关系预检只称“预检”，生成后的校验才称“生成后校验”。最终可操作的正文及引用仍仅在 `completed` 终态到达后展示；单次预检通过不代表最终请求成功。

底层旧 recorder 对空关系链的 `RelationValidator` 调用保留原有诊断语义；新进度仅在存在待校验关系链时发送 `relation_checked`，新执行摘要在没有 `evidence_chains` 时将 `relation_validation_passed` 留为 `null`，界面不把空集合校验描述为真实关系证据通过。

前端在提交时固定模式、项目、revision 和本地请求 ID，流帧必须匹配这些身份及同一服务端请求 ID，序号必须递增；重复和未知类型忽略，终态只处理一次。响应模式与提交模式不一致会拒绝展示答案。每条 pending 原位积累真实进度，终态后保留最近若干真实事件，失败时保留 BASE 结束事件，摘要仍默认折叠；事件不会触发自动滚底。等待计时是前端状态，不代表业务完成。流结束无终态或读流中断显示“最终状态未知”，不自动重发或宣称后端已取消；切换上下文或卸载会终止浏览器连接并隔离旧回调。历史响应缺模式显示“未记录”，缺摘要仍保持原有说明。没有可靠进度分母，不显示百分比。

隔离验证：使用项目 Python 环境的生产编排/路由替身，RAG 测得 Planner 0 次且通过最终引用校验；Agent 的确定性场景确实调用 Planner 与补充 `search_code`；可恢复终止在最终生成前出现。ASGI 流测试人为阻塞执行核心，在 `base_completed` 已被 `send` 收到时确认 `completed` 尚未发送。生成失败场景保留实际发生的生成前引用预检事件，但无生成后引用校验成功事件。前端流解析测试覆盖分片、未知/重复事件、重复终态、模式与 revision 不匹配、失败终态和无终态；问答历史引用的稀疏 E5、重复 E1 与乱序片段回归保留。最终相关后端回归 142 项、前端完整测试 155 项与 build 均通过，`git diff --check` 通过。

浏览器使用本轮独立 `127.0.0.1:8765` 隔离服务及 5175 Vite 实例。前端实际编译的 API Base 指向隔离服务；普通 `/ask` 和未匹配的流式请求在该服务均实测 403，匹配问题只有 `rag one`、`agent one`、`recover one`、`disconnect one`；服务端显式 Provider 计数始终为 0。真实分片间隔 350ms：RAG pending 时已观察到接收和 BASE 开始，Agent 成功后记录补充工具；恢复场景的提前终止与完成摘要一致；断线无终态呈现未知且不再 pending。1440×900、390×844 的亮暗主题均检查，两个尺寸下各提交了 RAG 和 Agent，横向溢出为 0。引用正文、源码抽屉、放大阅读、折叠摘要、上翻保位、项目切换隔离、地图拖拽及滚轮缩放通过隔离浏览器检查；截图在仓库外独立目录。浏览器只切换了项目，revision 变化由 requestGate/解析测试覆盖，未做真实 Provider UAT。人工验收时用已授权真实项目各问一次 RAG 和 Agent，核对模式、实时 BASE/补充阶段、最终引用及摘要，再在请求中切换项目或 revision，观察旧回答不复现。

剩余边界：断线后后端可能仍完成并保存，当前无服务端幂等/恢复协议，故没有自动重试、停止按钮或断线续传；真实 Provider UAT 未确认。没有新增依赖或数据库迁移。既有 npm audit 待办保留；隔离测试不构成正式 Provider 验收，最高状态为 `READY_FOR_MANUAL_UAT`。

## 目标与设计基准

将已验收的明亮阅读型桌面原型接入现有 React/Vite 前端：左侧真实项目导航，中间真实 `/ask` 问答和右侧按当前回答 citation 打开的片段栏。深色模式是同一个界面的本地主题偏好；宽屏内容轴最多 1080px，窄屏使用项目栏和源码栏抽屉。

## API 与证据边界

`POST /api/projects/{project_id}/ask` 仍由 `frontend/src/lib/api.ts` 统一调用。回答、warning 与 `citations` 仅采用该响应的数据；citation 编号只在单条回答内使用。源码栏只显示响应所给 `snippet/path/qualified_name/range`，不推断片段行号、不增加任意本地路径读取或完整文件入口。回答使用受限的文本 Markdown 投影：React 文本节点转义内容，不解析原始 HTML、不加载远程图片，也不把回答中的 URL 自动变成链接。

## 实施范围

- 新增 `workbench/Workbench.tsx` 和其样式，保留原有概览、地图、学习路线、报告，以及旧项目导入/更新/删除入口。
- 复用 `requestGate`：每次 `/ask` 绑定提交时 workspace、project、revision 和本地请求号；切换 context 会失效旧请求、清空回答和已打开引用。
- 支持主题切换、引用抽屉、复制、折行、Escape 焦点恢复、IME 组合输入保护、Enter 发送与 Shift+Enter 换行。不会对 `/ask` 自动重试，也不宣称浏览器 AbortController 能取消后端任务。

## 已验证结果

- `npm.cmd test`：131 项通过；涵盖 workspace request gate 的 context 失效、重复提交释放、错误投影，新增工作台受限 Markdown / 多回答 citation 投影，以及统一外壳下的默认入口、空项目库、跨页主题和问答草稿保留检查。
- `npm.cmd run build`：TypeScript 与 Vite 生产构建通过。
- 真实浏览器：已在 1280×800、1440×900、1920×1080、390×844 打开已索引的 Click 项目；明暗主题、长回答、宽屏源码栏、窄屏引用抽屉、Escape 关闭和焦点恢复均已实际检查，控制台为 0 errors / 0 warnings（React DevTools 提示除外）。
- 一次真实 `/ask`：问题为 `CommandCollection 如何从多个 Group 中查找和聚合子命令？`，成功返回真实回答与 5 条 citations；打开的 `CommandCollection` 片段为 `src/click/core.py:2147–2202`，revision `6aabf099bfdd…`。

真实中文输入法会话和操作系统级 `prefers-reduced-motion` 未做自动化验证；前者由组合输入事件保护，后者由 CSS 媒体查询保护，但都不应表述为已实际 UAT。

## 统一应用外壳（RN-FE-01 补完）

`Workbench` 现在是整个正式应用唯一的常驻外壳：项目选择栏、工作台导航、顶部项目上下文和本地主题状态不会随页面切换卸载。`App.tsx` 只负责既有 API 状态、workspace/revision 请求隔离和内容页选择；项目管理、概览、地图、学习路线、报告均作为 `Workbench` 的内容区渲染，不再返回旧 `app-shell`。项目管理复用已有项目库、导入、打开、revision 检查/更新、删除与学习连续性操作；其来源字段仅根据已记录的 `source_type` 显示“本地来源”“URL 来源”或“来源信息未提供”。

启动时优先恢复有效 URL workspace，再恢复最近项目；两者均不存在时，自动打开项目库中第一个 `openable` 项目并进入源码问答。项目库为空时仍显示统一外壳、禁用问答输入，并从“项目管理”提供原有导入入口。页面切换不会触发 `/ask`；问答草稿、答案和请求 gate 状态保留在 `App`，同一项目内切换后不会无故丢失；切换项目或 revision 仍通过既有 context gate 失效旧请求并清空项目相关问答。

本轮实机从桌面启动器提供的 `http://127.0.0.1:5173/` 验证了 Click workspace 的默认问答、项目管理、概览、地图、学习路线、报告及回到问答；每一页均保留一套项目栏和顶部上下文，无旧版全局布局或重复栏。已检查明暗主题、390×844 窄屏项目抽屉及输入区，浏览器控制台为 0 errors / 0 warnings。未调用 `/ask`、Provider、导入、重索引、更新或删除项目。

## 人工验收缺陷修复：通知与滚动

状态提示不再使用固定右下角定位：`Notification` 位于主内容滚动区顶部，因而不会覆盖问答 composer、发送按钮或源码抽屉。普通成功/恢复提示可关闭且在 5 秒后自动消失；包含失败、无法、错误、超时、未完成、不可或证据不足的需处理提示保持可见直到用户关闭或新状态替换。Provider 摘要不再伪装成持续通知。

统一外壳明确滚动归属：`wb-frame` 与 `wb-main` 可收缩，`wb-scroll` 是普通内容页唯一的纵向滚动宿主；问答仍由同一消息区滚动和底部 composer 组合；侧栏自身滚动；citation 抽屉保持代码区独立滚动。2026-09-19 实机在启动器地址验证：1280×800 Click 地图的 `wb-scroll` 从 `0` 滚至 `1200`，再到精确底部 `3348 = 4088 − 740`，并回到 `0`；390×844 同一区域从 `0` 滚至 `3000`（总高 `5097`、可视高 `789`）。项目切换提示出现时已实际填入未发送草稿，发送按钮仍可点击；提示随后自动消失。未调用 `/ask` 或 Provider。

## 图形化地图数据核查（只读）

当前 `GET /api/projects/{project_id}/map` 提供树、模块、粗粒度 `dependency_edges` 和核心文件，但不提供 map 专用 revision、节点详情或源码定位接口。树节点有稳定 `path`、`type` 和 `children` 父子关系，可作为目录/文件节点 ID；它是包含关系，不是调用关系。`dependency_edges` 是按顶层目录归组后从有限 top-files 的 import 字符串推得的模块级依赖，且现有 UI 未消费它；空 `depends_on` 只能表示该启发式未找到边，不能标注为“独立”。

后端另有按 `project_id + repository_revision` 约束的 Python 静态 relation index（文件/代码块 node ID，含 resolved 或 ambiguous 的关系边），但它仅供受限问答 relation 工具使用，没有暴露给地图 API；它也不代表运行时调用。最小下一轮方案应只用现有 map API 做“目录/模块概览 → 按需展开文件 → 本地 path 搜索定位 → 点击展示现有路径、核心标记、重要度和模块归属”。依赖边开关必须等 map API 明确暴露有版本身份、边类型、解析状态、端点详情和分页/上限的真实 relation 数据后再加入；大仓库默认折叠目录、搜索后展开命中路径并限制可视节点，不能一次铺满图。

## 已知边界与后续

项目删除的既有 `project_delete_failed` / `cleanup_pending` 风险未在本轮修复，也没有使用正式项目试删。无 source API 时不存在完整文件浏览能力。后续可在后端提供有 revision、范围和证据身份约束的 source 合同后，再单独评估扩展。

当前开发分支：`feature/frontend-workbench`，基于 `b0094dcf943c67ec9415899d0da22d151d2f7f30`；`origin/main` 尚未包含 RN-LWB-03。本轮不 stage、commit 或 push。

## 体验修正第 1 轮：首次提问与 composer

问答页无回答、无请求、无错误时，内容区以左侧项目栏和可选源码栏之间的可用空间为基准，居中展示“从源码中找到答案”、用途说明、三个通用可编辑示例和同一个 composer。点击示例只写入草稿并聚焦输入框，不会调用 `/ask`。已有回答、提交中或失败后则保留消息区与底部 composer；失败不会清空问题，项目与 revision request gate 仍由 `App.tsx` 维护。

composer 的输入和工具栏现在是同一张卡片，发送按钮位于卡片内右侧，快捷键说明只保留一处。加载期只显示消息区内的实际等待计时及一个主要状态；顶部普通通知在加载期间不重复显示。计时器随完成、失败、context 切换或卸载清理；减少动画偏好沿用工作台的媒体查询。界面文案改为面向使用者的引用说明。`warnings` 目前仅是未带结构化终止状态的字符串数组，故未猜测性翻译原始英文 Agent 终止文字或将其误报为失败。

本轮执行 `npm.cmd test -- --run`（131 项通过）和 `npm.cmd run build`（通过）。通过桌面启动器的正式 `http://127.0.0.1:5173/`，以隔离的浏览器响应验证了首次提问、示例填草稿、桌面对话/源码栏/夜间主题、390×844 窄屏与长回答实际滚动；未向后端或 Provider 发出真实 `/ask`。

## 体验修正第 2 轮：回答排版与源码阅读

回答改用安全的 `react-markdown` 加 `remark-gfm`，支持 CommonMark/GFM 的标题、段落、列表、加粗、行内代码、围栏代码和表格；不启用 raw HTML，图片组件不渲染，链接只保留 `http`、`https` 与 `mailto` 协议。`[E1]` 等普通原文标记保持文本，尚未实现正文证据跳转。

`CodeViewer` 统一用于回答代码块、引用抽屉和只读放大阅读：本地 Prism token 树以 React 文本节点渲染 Python 等已加载语言，未知语言退回纯文本；支持复制原始代码、折行和放大阅读。引用抽屉保留服务端提供的路径、真实范围与 revision，代码区占用剩余高度独立滚动。放大阅读支持 Escape、Tab 焦点循环和关闭后焦点恢复；workspace、项目或 revision 变化会关闭抽屉与弹窗，避免显示旧源码。

本轮 `npm.cmd test -- --run` 为 134 项通过，`npm.cmd run build` 通过。隔离浏览器响应验证了亮/暗主题、长行横向滚动与折行、窄屏抽屉和放大阅读、关闭后继续操作、以及项目切换清除旧弹窗；未向正式数据写入测试答案。

## 体验修正第 2 轮差量：代码高度、真实排版核查与提交顺序

引用数据合同目前只提供 `citation.snippet/path/qualified_name/start_line/end_line`，没有 `truncated` 或省略字段。此前用户截图所对应的真实浏览器回答已不在当前客户端状态或 localStorage 中，且本次受限只读搜索未找到可归属该次请求的持久化回答记录；因此不能将该事件的 snippet 数据截断判定为已证实。前端链路本身将 `citation.snippet` 原样传给 `CodeViewer`，其 DOM `textContent` 与复制原文一致。后续若服务端截断片段，需要随 citation 明确返回截断身份后再显示“片段已截断”，不能由前端猜测或从本地文件补全。

已确认的高度根因位于旧 `frontend/src/styles.css` 全局 `pre`：`max-height:260px; overflow:auto`。它在 `CodeViewer` 的抽屉和放大变体中仍生效，尽管新的 `.wb-code-pane` 已是 flex 剩余空间的滚动宿主，导致 260px 的嵌套小滚动框与下方空白。新增组件作用域 `workbench/reading-fixes.css` 只为 `.wb-drawer/.wb-code-modal .wb-code-pane pre` 清除该遗留限高并让外层 pane 滚动；普通正文代码块继续受原有适度限高。

问答会话不再在响应完成时前插。`App.tsx` 在用户提交时追加带 workspace/project/revision/requestId 组合身份的 `pending` 条目，成功和失败只 map 更新相同 `id`；`Workbench` 按数组提交顺序渲染，引用按钮闭包保留该 citation 对象而不按回答数组下标取引用。提交时仅滚动至新问题；响应落地不会强制打断用户向上阅读。现有 request gate 仍会在项目或 revision 变更时清空该 context 的会话。

2026-09-23 隔离浏览器验证（`/ask` 路由拦截，未触达后端/Provider）：1440×900 下抽屉代码 pane 为 `clientHeight 717 / scrollHeight 1945`，滚轮后 `scrollTop 1228.7` 且末行 `return total` 可见；放大弹窗为 `676 / 1945`、`scrollTop 1269.3`，同样到达末行。两处 viewer 都填满工具栏以下空间，`pre` 计算值为 `max-height:none; overflow:visible`，没有内部 260px 小框。复制值与 DOM 代码均为 1510 字符且相同；折行后 `scrollWidth === clientWidth`。两条隔离问答按“第一组、第二组”自上而下，第二条的实际 Markdown 渲染为标题、段落和保留的 `[E5]` 文本；首条响应是早先错误转义的隔离样例，故不作为真实回答排版证据。390×844 下页面无横向溢出，composer 宽度 390px。

`npm audit --json` 当前为 4 项：`vitest`（直接开发依赖，中等）与其 `@vitest/mocker`（间接开发依赖，中等）；`vite` 的构建链 `postcss`（间接，运行/构建依赖，高）及其 `nanoid`（间接，运行/构建依赖，高）。建议修复分别要求升级 Vitest 至 5.0.1 或升级受影响构建链；本轮未执行 `npm audit fix` 或任何依赖升级。

## 正文证据与本次执行摘要（2026-09-24）

### 数据合同

`EvidenceStore` 按请求持有 canonical Evidence，按 `chunk_identity` 去重，并在加入时分配 E1、E2……；最终回答的模型只拿到服务端派生的 A 别名，服务端将别名渲染成 `[E#] path:start-end` 并校验正文引用。`qa_agent._m1_response` 从最终有效 Evidence 生成 citation 列表，列表顺序属于 Evidence 投影，不能拿 `[E5]` 对应 `citations[4]`，也不能按路径或符号猜。新增可选的 `citation.evidence_id` 直接取自同一个有效 Evidence；`AskResponse` 在响应校验时核对 ID、路径、符号和行范围。旧 citation 没有此字段仍能读取，但正文标记不激活。前端还要求该回答的 Evidence 明确属于当前 project/revision，并按稳定回答 id 隔离选择。若一个 ID 关联多个片段，正文打开片段选择栏；点击底部的某一片段则直接打开该片段。高亮只依据 citation 的实际行范围，不表示某句结论的精确行。

`execution_summary` 是新的可选、安全、请求内投影；缺失时显示“未提供结构化执行详情”。来源是当前请求已验证的 `agent_status`、`answer_mode`、最终 Evidence/citation 列表、`budget_usage` 的 Agent 耗时及步骤/工具预算占用、`SmokeDiagnosticsRecorder.snapshot()` 的 BASE 状态、Planner 请求与修复数、Provider 逻辑调用与 HTTP 尝试、工具执行统计、终止原因及明确完成的验证布尔值。它不含原始 Provider 内容、prompt、异常正文或诊断明细列表。`agent_elapsed_ms` 只指 Agent 内耗时，不是浏览器等待或包含持久化的请求总耗时。缺失字段不推成 0；`diagnostics_truncated` 表示明细不全，展示的聚合数只来自独立计数器。失败回答只展示该失败请求已有的安全诊断，绝不从 HTTP 200 推断阶段成功。该字段在保存前经响应 schema 校验；原有成功一次、失败零次持久化边界不变，无数据库迁移、额外检索或模型调用。

### 验证与限制

- 前端 `npm.cmd test -- --run`：142 项通过；`npm.cmd run build` 通过。相关后端 `D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest tests.test_m1_ask tests.test_agent_finalization tests.test_ask_observability`：85 项通过。首次误用系统 Python 缺少 `pydantic`/`fastapi`，改用项目可用环境后通过。`git diff --check` 通过。
- Playwright 浏览器使用本轮独立 Vite 端口 5174 和先注册的全 `/api/**` 拦截；提交前探测 `/api/projects/p/ask` 返回隔离哨兵，未知 `/ask` 被主动阻断。1440×900 的亮/暗主题验证了两条不同回答的重复 E1、乱序 E5、正文与底部同片段、切换引用、放大阅读和折叠摘要；旧响应无字段保持文本。390×844 的暗/亮主题验证抽屉、弹窗、输入框可用和页面横向溢出 0。长代码抽屉 `scrollTop=max=1577`（桌面）及 `1464`（窄屏），弹窗 `scrollTop=max=1618`（桌面），末行 `return total` 可达。唯一浏览器控制台错误是主动阻断未知 `/ask` 的预期 `ERR_FAILED`。截图位于桌面独立目录 `C:\Users\Anon\Desktop\RepoNoesis-RN-FE-01-Evidence-20260924`。
- 这是隔离响应浏览器验收，不是 Provider UAT。历史那次影响未知的真实请求仍为 `NOT_VERIFIED`，本轮未复查。若后端返回没有新绑定字段的历史响应，底部只显示“片段 N”且正文 E 标记为纯文本；摘要缺失也不补推执行状态。人工验收：在已授权的真实问答环境依次提交两个可产生重复 E 编号的问题，核对正文/底部/源码栏和摘要；再切换项目或 revision 确认旧引用消失，检查长代码末行、键盘与手机抽屉。真实 Provider 问答需另行授权。

## 图形化项目地图（2026-09-24）

前一批人工基础验收通过，暂未发现问题。未明确确认真实 Provider 调用，因此不记作完整真实 Provider UAT。前一批 `citation.evidence_id` 和 `execution_summary` 仍为可选字段，旧回答降级投影与既有未提交改动均保留；本轮未重开全面问答审查。

### 数据与身份合同

正式地图复用 `GET /api/projects/{project_id}/map`。树是分析时从 `RepositorySnapshot.files` 建立并保存在项目分析记录中的目录/文件包含树；`file_count` 来自同一项目持久化的 `repo_files` 记录。它表示**已分析文件**，不表示完整 Git 仓库，也不等于仅有 Python `code_chunks` 的文件。产品导入从 Git 跟踪文件中按文本类型、文件/总字节数与 `max_files` 预算筛选；旧 GitHub 导入最多选择 45 个候选文本文件。因此页面和搜索范围均明确写“已分析文件”。接口一次返回该分析树，无服务端分页；前端每个已展开目录只挂载 24 项一页，搜索在已返回的整棵树中进行，可直接跳到命中项所在页。若未来接口改成分页，不能沿用“整棵分析树可搜索”的描述。

后端给原响应增补可选兼容字段 `project_id`、`repository_revision`、`coverage: analyzed_files`、`file_count`；来源是按项目 ID 读取的持久化项目及文件记录，不扫描当前服务器工作目录，不读取历史 revision 的当前 checkout，不调用索引、Embedding 或 Provider。`core_files` 仍保留原有元数据，但地图响应不再携带数据库中的源码 `content`。不存在的项目按原路径返回 404；没有新增任意路径参数。历史 revision 只能通过其已存在的项目 ID 读取，工作台权威当前 revision 来自 workspace 的 `active_snapshot`；未提供独立的历史 revision 列表或切换器。

节点 ID 为 `[project_id, revision, type, exact_relative_path]` 的结构化字符串，路径比较保留原始大小写，同名文件不会合并。工作台切换上下文立即清除旧地图数据和详情；加载结果及重试响应都核对当前 generation、项目和 revision。旧接口缺新增字段时仍显示树，并标注“覆盖范围未声明”；若新版字段与 workspace 不符，则报错而不展示旧树。地图节点不分配 E 编号、不进入问答引用选择；没有 revision 一致的完整源码读取接口，详情只展示名称、类型、相对路径、revision、当前分析结构中的直接子项数及已有核心标记。

### 图形与交互

使用现有 React、SVG 连线与局部 CSS，没有新增依赖。连线只表达父目录包含子目录/文件；目录与文件图标、选中和搜索命中均有区分。目录选择与展开是独立按钮；空白画布可拖动，按钮可缩放与适应画布，Ctrl/⌘ 加滚轮缩放而普通滚动不被困住。搜索结果带相对路径；点击后展开祖先、切到所在 24 项窗口、居中定位并同步详情。关闭详情支持 Escape 和焦点恢复。390×844 使用下方详情，初始缩放 70%，可通过键盘操作搜索结果、节点和控件。组件样式仅在 `workbench/project-map.css` 内，并检查了旧全局 button/SVG/overflow 规则；浏览器验收中修复了画布指针捕获抢按钮点击和详情覆盖缩放按钮的问题。无调用或导入依赖边，现有启发式 `dependency_edges` 未绘制。

### 验证与限制

- 地图相关 Vitest 4 项通过；完整前端 `npm.cmd test` 146 项通过；`npm.cmd run build` 通过。后端项目 API 模块 `D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest tests.test_product_api` 12 项通过，包含真实数据库响应构造、revision/项目绑定、404 和源码正文不外露；`git diff --check` 通过。
- 正式只读接口与浏览器：现有 Click 项目 revision `6aabf099bfdd…` 返回 115 个已分析文件、22 个目录；1440×900 初始挂载 8 节点，展开 `src` 后 9 节点。实测目录折叠、`src/click/core.py` 搜索定位、画布平移、缩放、适应画布、详情、明暗主题；390×844 初始图和详情、Escape 焦点恢复均可用，两个视口整页横向溢出均为 0。浏览器请求守卫先行注册，`/ask` 探针得到隔离 451，正式请求仅放行指定只读 GET；预期的 451 探针是该次唯一控制台错误。现有 5173/8000 用户服务在本轮前已运行，本轮未终止或占用它们。
- 隔离响应：空结构、503 失败、文件数不一致、旧接口缺字段，以及从 Click 切到 revision `cc8b12395a3e…` 后旧详情失效均通过。延迟的旧地图重试晚于新项目返回后，当前地图仍为新 revision。隔离 3,202 文件树含同名文件；初始实际 DOM 为 3 节点，展开后 27 节点，搜索最后一项定位后 12 节点，未观察到明显交互阻塞。此为本机一次浏览器观察，非通用性能保证。隔离 `/ask` 响应还验证了切地图返回问答后两条回答顺序、正文 E1、源码栏、放大阅读、执行摘要和输入框保留；未调用真实 Provider。
- 截图保存于桌面独立目录 `C:\Users\Anon\Desktop\RepoNoesis-RN-FE-01-ProjectMap-20260924`。当前仍不提供完整仓库清单、源码全文入口、调用关系或历史 revision 切换；大树搜索覆盖本次完整返回的**分析文件**，并非全仓搜索。已有 npm audit 四项记录与此前真实请求影响 `NOT_VERIFIED` 维持原结论，本轮未处理、未重新调查。

人工验收：打开现有项目，进入“项目地图”；展开 `src`、搜索一个深层文件并确认相对路径与详情，拖动画布并使用缩放/适应按钮；切换明暗主题及手机宽度，关闭详情后返回“源码问答”检查草稿和既有回答。若测试跨项目，确认旧详情与旧 revision 不再显示。该批随后由用户人工验收发现拖拽白屏阻断，状态改为“人工验收发现阻断，待修复”；此前 `READY_FOR_MANUAL_UAT` 结论不再沿用。差量修复及重新验证见下。

## 项目地图拖拽白屏差量修复（2026-09-24）

用户约 17 秒的 2560×1600 录屏显示：调整浏览器和地图缩放、展开目录、选中文件且保持详情打开后，在地图拖拽时节点文字一度被选中，随即包括项目栏在内的应用区域持续白屏。发现时本批状态为**人工验收发现阻断，待修复**。录屏本身没有控制台堆栈，文字选中不能单独证明异常来源。

复现时先注册 `pageerror`、console error、导航与网络失败记录，并对 `/api/**` 只放行明确的只读 GET；`/ask` 探针被隔离 451 拦截。真实 Click 地图上的逐次等待鼠标移动没有触发崩溃；针对同一批次内 `pointerdown → pointermove → pointerup` 的 React 回归测试在修复前失败，首次异常为 `TypeError: Cannot read properties of null (reading 'initialX')`，指向修复前 `ProjectMapView.tsx:80` 的 `setView` updater。浏览器按同一批次派发指针事件后重现应用白屏：`.wb-frame` 和 `.pm` 均从 DOM 消失，URL 未变化，未见导航或网络失败；首次 `pageerror` 的栈同样从 `ProjectMapView.tsx` 中读取 `initialX` 开始，经 React `basicStateReducer → updateReducer → useCanvas → ProjectMapView`。这是**已证明的同类白屏触发路径**；没有用户原录屏的异常堆栈，无法断言该次真实操作的每一项时序细节完全相同。

根因是 `pointerMove` 把对可变 `drag.current` 的读取留在 React 状态 updater 中；`pointerup` 先清空该 ref，稍后提交的 updater 再读取 `initialX` 时抛错，使没有局部错误边界的工作台根视图卸载。修复仅在地图组件内：事件处理时将拖拽起点和本次坐标复制为局部数值，再交给 updater；用 pointer ID 关联拖拽；在 `pointerup`、`pointercancel`、`lostpointercapture`、窗口失焦和组件卸载时结束并释放已持有的指针捕获。仅在画布空白处启动拖拽并阻止该次默认文字选中；`user-select:none` 只作用于地图画布及其中节点，不影响画布外详情、源码与回答复制。没有刷新页面、吞异常、删除拖拽或改全站快捷键。

先前地图浏览器验收对 `page.mouse.move` 逐步等待，React 每步均有机会提交更新，未覆盖移动与释放在同一更新批次的情况；此前只检查了画布位移及普通释放，也未捕获 pageerror。新增回归先在旧代码上证实失败，再在修复后通过。修复后的同批次浏览器复测保留工作台与地图 DOM，位移 `(40,30)`，`pageerror` 和 console error 均为 0。

真实鼠标验收使用完整的按下、连续移动、松开序列：1440×900 下首次位移 `(80,42)`，节点位置同向移动；松开后再次移动没有继续平移。反向拖拽、画布外释放、重入后再次拖拽、缩放至 35% 后拖拽均通过；保持多层展开和 `core.py` 详情时，路径与选择未变，画布内没有意外文字选中。窗口失焦和 pointercancel 后即使继续移动也不再平移；切到另一项目 revision `cc8b12395a3e…` 后旧详情为 0，新地图仍可拖动。2560×1600 下以页面 CSS `zoom:125%` 近似非默认浏览器缩放，位移有限且工作台保持；这不是原用户浏览器的工具栏缩放设置。390×844 下搜索、详情与关闭正常，整页横向溢出 0。修复路径 `pageerror`、console error 与网络失败均为 0；请求守卫探针单独验证了隔离 451。问答界面以隔离 `/ask` 响应确认输入草稿、正文 E1 引用、源码栏和放大弹窗在地图切回后仍正常，未调用真实 Provider。

完整拖拽短录屏 `C:\Users\Anon\Desktop\RepoNoesis-RN-FE-01-MapDrag-20260924\fixed-drag-validation.webm`、首次异常完整堆栈 `pre-fix-first-pageerror.txt` 与前后截图保存在同一桌面目录。最终执行的前端相关测试、完整测试、build、`git diff --check` 结果见本批交付报告；纯前端修复未重跑后端。当前结论最多为 `READY_FOR_MANUAL_RETEST`，等待用户按“打开地图 → 展开两级目录 → 选文件保持详情 → 用缩放按钮调整比例 → 从画布空白处连续拖拽并松开 → 再移动鼠标、再次拖拽”复验，不进入后续功能批次。

用户已对上述拖拽修复进行人工复验，未发现异常。此结论仅对应拖拽修复；本次滚轮缩放增强仍需独立实现与验证。

## 项目地图鼠标滚轮缩放增强（2026-09-24）

拖拽白屏修复已经用户人工复验，未发现异常。本次仅增强地图视口：鼠标在画布上普通滚轮向上放大、向下缩小；以画布边框内的鼠标局部坐标为锚点，按现有 `translate(x, y) scale(zoom)` 顺序换算平移量。单次 wheel 按 `deltaMode` 折算并限制输入幅度，连续事件用函数式状态更新累计；与按钮共用 35%～220% 上下限和百分比，触及边界时不再改变平移。纯横向/横向占主导、Ctrl/Meta、拖拽进行中的 wheel 不由地图处理。只在地图画布上注册 `{ passive: false }` 原生 wheel 监听，卸载时移除；搜索、搜索结果、工具栏、详情和页面滚动继续由浏览器处理。拖拽的事件时快照及 updater 不读已清空 `drag.current` 的修复保持原样。提示更新为“拖动画布平移 · 滚轮缩放”。未新增依赖、后端改动或触摸捏合操作。

回归测试先在旧行为上失败（普通滚轮没有缩放），修改后地图测试 8 项通过；覆盖双向缩放、鼠标坐标不漂移、同批次快速滚轮、上下限、修饰键与非画布区域、滚轮后同批次拖拽及释放。完整前端测试 150 项、build 与 `git diff --check` 通过。真实 Chromium 1440×900 浏览器使用 `page.mouse.wheel`：画布两处锚点在滚轮前后换算的地图坐标差均小于 0.001，比例依次 100%→120%→143%；上下限为 35%/220%，继续滚动后平移和比例不变。按钮与滚轮交替后“适应画布”显示 102%；展开 `src/click`、搜索并选择 `src/click/core.py`、保持详情时，滚轮仍有效；详情内滚轮实测 `scrollTop=100`，地图比例不变。工具栏、搜索框/结果不触发地图缩放，画布外滚轮使工作台滚动；纯横向 wheel 不缩放。随后真实鼠标连续移动拖拽使地图平移 `(65,35)`，释放后继续移动不再平移，切走地图再返回单次缩放倍率正常。390×844 的按钮、搜索、详情和关闭正常，整页横向溢出为 0；Ctrl+滚轮未改地图视口。修复路径 `pageerror`、console error、网络失败均为 0，工作台和地图 DOM 保持。浏览器工具栏缩放比例未由测试环境直接控制，因此只确认页面/地图宽度与 `devicePixelRatio`、`visualViewport.scale` 在普通滚轮时不变，不把 CSS zoom 冒充浏览器缩放验证。

浏览器测试使用现有 Click 项目 revision `6aabf099bfdd…` 的只读地图结构；未调用 Provider、未导入/更新/删除/重索引项目。测试脚本首次试探请求拦截时，错误的无请求体 `POST /api/projects/probe/ask` 抵达后端并返回 422；`AskRequest` 校验发生在问答处理函数执行前，故没有进入 Provider 路径。修正后的安全假路径 POST 探针得到隔离 451，后续地图操作仅放行已列明的只读 GET；未以真实请求兜底。正式操作录像和截图位于仓库外 `C:\Users\Anon\Desktop\RepoNoesis-RN-FE-01-MapWheel-20260924`。已有 npm audit 四项及此前真实请求影响 `NOT_VERIFIED` 保持原记录。

人工验收：打开现有项目的“项目地图”，将鼠标放在画布两处分别向上、向下滚动，确认节点围绕鼠标位置变化且页面比例不变；展开目录并选中文件，检查详情滚动不缩放地图；交替用滚轮、放大/缩小、适应画布，再拖动并松开，确认视口停止移动；切到“源码问答”再返回，核对单次滚轮与手机宽度下按钮、搜索、详情。该地图批次已由用户人工验收通过；不代表整个问答系统已全面验收。

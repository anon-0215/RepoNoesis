# 百炼 qwen3-rerank 独立重排

重排序默认复用当前生成 API 的地址、密钥和基础模型（`LLM_MODEL`），无须重复输入。用户更换生成配置时，默认重排随之变化。选择独立服务后，仅重排改用北京地域的百炼 `qwen3-rerank`；回答、Planner、HyDE 和原重排开关各自保持。没有数据库迁移。

## 配置

| 变量 | 默认值/用途 |
| --- | --- |
| `RETRIEVAL_RERANK_ENABLED` | 原开关，默认 false |
| `RETRIEVAL_RERANK_MODE` | `generation`；可选 `independent` |
| `RETRIEVAL_RERANK_SERVICE` | 独立服务类型，本轮仅支持 `bailian` |
| `RETRIEVAL_RERANK_PROVIDER` | 旧变量；MODE未设置时，`deepseek`/`generation`映射默认方式，`bailian`映射独立百炼 |
| `DASHSCOPE_API_KEY` | 百炼独立密钥，不读取 `LLM_API_KEY` |
| `BAILIAN_RERANK_ENDPOINT` | `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank` |
| `BAILIAN_RERANK_MODEL` | 固定支持 `qwen3-rerank` |
| `BAILIAN_RERANK_TIMEOUT_SECONDS` | 7秒；0.1至15秒，仍受原共享增强时限约束 |

新版官方文档同时提供业务空间地址 `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`。适配器接受该北京官方地址与上表地址；本轮上表地址已实际返回 HTTP 200。请求为 `model`、`input.query`、`input.documents` 与 `parameters.top_n/return_documents`。不使用生成接口。

桌面设置中，先开启“检索结果重排序”，再选择“重排方式”：使用当前生成模型（默认），或使用独立重排服务。独立方式显示服务类型、API地址、模型名称、超时与API Key；本轮只适配百炼北京官方接口和qwen3-rerank，不承诺任意厂商兼容。

密钥沿用当前 Windows 用户 DPAPI；留空保留，明确勾选才清除界面密钥。优先级为界面保存、当前进程环境、Windows用户环境。清除界面密钥不删除环境变量，界面说明有效来源和是否配置，不返回密钥。保存不发收费检测请求；缺少必需字段会明确显示，未配置的重排保留基础排序。

旧设置缺少模式时继续复用生成API，旧配置明确选择百炼时迁移到独立方式；显式新模式优先。读取迁移不落盘、不重加密。关闭开关或切换方式保留独立配置。保存沿用重启租约，问答、同步导入、排队或执行中的更新均阻止重启，完成后才可保存。

## 输入、返回及失败

- 与旧 DeepSeek 重排共用最多20个真实候选。每个候选携带ID、路径、符号、原始行范围、前1600字符源码及截断标记；完整源码仍留在候选对象中，重排不改写仓库版本、身份、源码内容、原检索分数与名次。第一名的最终正文补全沿用旧规则。
- 裁剪策略名、被裁剪候选ID与百炼返回分数进入安全审计。最终证据只使用真实源码，不使用服务返回的文档正文。服务返回的零基 `index` 映射回原候选；有限的0至1相关性分数仅表示本次请求内相对相关性，不是正确概率。
- 百炼单条 Query/Document 上限4000 tokens，最多500个文档，请求上限120000 tokens，按 `query_tokens × 文档数 + 文档tokens总和` 计算。适配器以UTF-8字节数作保守上界预检，超限直接记录 `rerank_input_limit`，不继续截短问题或源码文档。该规则可能提前拒绝仍可被官方分词器接受的输入。
- 要求返回所有候选，每个编号恰好一次。缺项、重复、非法编号、非有限/越界分数、超时、认证或服务错误均使本次重排回退到原排序，并记录固定原因码。没有自动重试。原 DeepSeek 合法前缀补尾行为保留，并记录补项数。
- 用户选择哪种方式，只调用该方式。失败不会切换另一家收费服务；摘要记录所选服务、实际生成模型和简短失败原因。默认大模型重排保留原短任务参数（温度0、240tokens、7秒、单次HTTP），不读取最终回答模型覆盖或回答/Planner额度，也不修改共享客户端。普通兼容接口未配置厂商扩展时，不发送DeepSeek专用thinking字段。
- 复用现有隔离HTTP worker，以绝对截止时间覆盖连接和读取等待，超时终止本次自建worker；不阻断原本可继续的回答。7秒是本地配置，官方文档未提供此模型的固定响应时延保证。

## 上轮效果对照（历史记录，本轮不重复）

2026-11-13，从同一Click/HTTPX冻结12题与共同候选池比较，HyDE及Agent后续补查关闭。独立匿名盲评完整通过：DeepSeek 9/12、百炼10/12，原9题均保留；检索交付覆盖505/552与492/552，百炼有1题减少。回答另有1题完整性与实际证据支持相互取舍。百炼重排中位数0.4145秒，DeepSeek为1.336秒。因此保留百炼为可选项，继续保持现有默认。结果只适用于本轮固定诊断题。

上轮完整记录及独立试用入口位于 `output/qwen-rerank-20261113/`，不随源码提交。上轮后端1314项执行，仅既有README标题断言失败；前端314项、桌面相关40项、shell 7项通过。工程收尾、日常部署及恢复状态以本机 `output/qwen-rerank-closeout-20261113/` 记录为准；旧便携包和历史实验保留。

## 官方依据

- [重排序与北京模型限制](https://help.aliyun.com/zh/model-studio/rerank)
- [排序请求格式、零基编号与相对分数](https://help.aliyun.com/zh/model-studio/text-rerank-api)
- [北京旧地址的官方集成示例](https://help.aliyun.com/zh/es/user-guide/configuration-template-big-model-service-platform-refined)
- [百炼调用价格](https://help.aliyun.com/zh/model-studio/model-pricing)：本轮查询北京文本输入0.5元/百万tokens，输出不收费；活动、额度与实际账单另计。
- [DeepSeek模型与价格](https://api-docs.deepseek.com/quick_start/pricing/)：本轮请求与响应名为 `deepseek-flash`；正式实验在非高峰时段，缓存命中/未命中输入及输出分别为0.003/0.15/0.6美元每百万tokens。

# M5 真实系统评测协议

## 数据与获取

外部仓库只读 shallow clone 到操作者指定目录，不认证、不拉 LFS/submodule、不运行脚本、不
安装依赖。pilot-v1 固定：

| repo | revision | license | role |
| --- | --- | --- | --- |
| itsdangerous | `672971d66a2ef9f85151e53283113f33d642dabd` | BSD-3-Clause | small |
| click | `00e592cea702e0b2caa0dee42489fdb1c22cd845` | BSD-3-Clause | medium |
| httpx | `b5addb64f0161ff6bfe94c124ef76f6a1fba5254` | BSD-3-Clause | cross-file transports/client |

validator 必须先通过。annotation 由 AST 稳定身份与 Agent 辅助问题措辞生成，状态统一为待人工
复核。禁止称为 human verified gold。

## Smoke gates

按顺序执行 dataset validation、单仓库静态摄取、lexical、BGE-M3 load、dense、answer provider、
structured evaluator、单场景 E2E、invalid citation negative、预算/timeout。任一 live gate 失败，
停止后续 live pilot，但仍可运行完全离线 fake engineering benchmark。

真实调用必须同时使用 `--live` 和环境 gate；不在命令行或聊天粘贴 key。默认上限 250 次 LLM
调用、1,000,000 input tokens、250,000 output tokens；预计超过 5 美元时必须另行确认。

## 公平配置

同一 run 固定 repository revision、scenario、chunk corpus/filter、answer model、prompt version、
temperature、max output、Evidence byte cap、citation contract、timeout/retry、机器信息和 seed。
记录每模式检索法、Planner、relation、learning、tools、steps/calls/Evidence/relation budget。

正式比较包括 lexical-only、dense-only、hybrid、M2、M3、M4-profiled；adaptive sequence 单独报告。
validator-off 只做负向隔离测试，不进入正式回答结果。

## 指标公式

- Hit@k：前 k Evidence 是否命中任一 gold file/symbol/span。
- MRR@10：首个相关 Evidence rank 的倒数，无命中为 0。
- nDCG@10：binary relevance DCG 除以相同 gold count 的 ideal DCG。
- recall：gold identity 集与返回 identity 集的交集占 gold 数。
- Evidence precision/recall/F1：返回 Evidence 中匹配稳定 gold identity 的比例及其调和平均。
- citation pass：回答引用 ID 中属于最终 valid Evidence 的比例。
- relation recall：返回 validated chain edge 对 expected relation identity 的覆盖。
- correct abstention：unanswerable 且正式返回 insufficient Evidence。
- latency：成功样本 wall-clock p50/p95；失败/timeout 另计。
- paired delta：同一成功 scenario 的 mode B 减 mode A；固定 seed bootstrap 2,000 次给 95% CI。

失败、timeout、cancelled、degraded、invalid output 和 budget exhaustion 全部保留。失败不进入质量
均值但进入失败率；缺 usage/价格为 unknown。结果只反映当前固定 pilot，不代表大规模仓库泛化。

## 可复现命令

从 `backend` 执行，路径由操作者配置：

```powershell
python -B -m app.m5 validate --dataset ..\benchmarks\m5\datasets\pilot-v1 --repository-root <root>
python -B -m app.m5 dry-run --dataset ..\benchmarks\m5\datasets\pilot-v1 --repository-root <root> --artifacts ..\artifacts\m5 --mode fixed_lexical_rag
python -B -m app.m5 run --dataset ..\benchmarks\m5\datasets\pilot-v1 --repository-root <root> --artifacts ..\artifacts\m5 --mode fixed_lexical_rag
```

## Live dense-only 增量索引验收协议 v1

正式、机器可读且唯一的规则来源是：

```text
benchmarks/m5/protocols/live-dense-acceptance-v1.json
protocol_id = m5-live-dense-acceptance
protocol_version = 1
```

对应导出的 JSON Schema 是
`benchmarks/m5/schemas/live-dense-acceptance.schema.json`。Markdown 只解释该协议，不是第二份
可独立修改的阈值来源。协议文件缺失、版本未知、必需字段缺失、枚举未知、公式变化、`top_k`
不是 3，或 overall 标准不是全部 answerable 通过时，loader 和 dataset validator 都必须 fail
closed。该协议按 `all_validated_m5_python_repositories` 规则覆盖 Click、HTTPX 以及后续进入
M5 validator 的同类 Python 仓库；没有关联协议的仓库不能进入 live dense 验收。

### Stable identity 与 A/B/C

排序字段采用现有 standalone manifest 的 `chunk_identities` 元素。每个元素必须是现有
`app.m5.dense_artifact.build_chunk_inventory` 产生的 `chunk-sha256:<64-lowercase-hex>`；本协议
不改变生成算法。完整 inventory 在固定 repository、revision 和 content identity 下必须非空、
无重复、无跨仓库记录，并满足现有 persistent code-chunk identity 合同。

排序是对 `chunk_identity.encode("utf-8")` 的字节严格升序，不依赖文件遍历、SQLite、容器、
locale、模型或检索结果。令排序后的完整去重集合大小为 `N`：

```text
A count = (N + 1) // 2
B count = N - A
A = sorted_chunks[0:A]
B = sorted_chunks[A:N]
FULL = A union B
C target = FULL
```

A 从空状态只生成 A；B 在同一索引和模型实例上只生成 B 并复用 A；C 对 B 结束时的 FULL
再次执行。C 必须满足 `generated=0`、`cached=N`，document encode calls、batches、items 全部
为 0。C 前后的 manifest/checkpoint 字节长度、SHA-256、mtime 都必须相同；只有“向量命中缓存”
但 metadata 被改写不算物理 no-op。本协议不保存 Click 的实际 `N` 或实际 A/B identity 集合，
这些只能在另行授权的真实预检中计算。

### Dense-only 场景与通过条件

answerable 场景只允许 dense，`top_k=3`，query 必须恰好编码一次。BM25、Weighted RRF、
hybrid、relation expansion、Planner、Agent loop、LLM 和 evaluator 全部禁用。单场景只有在
Top 3 中至少一个完整合法 gold 被命中时才通过；多个合法 gold 命中任一个即可。

“完整合法”要求候选与 gold 的 repository id、固定 revision、content identity、规范 POSIX
path、qualified symbol、完整 start/end span、content hash、现有 chunk identity 全部一致，并且
候选可证明来自本轮 FULL 正式索引。只匹配文件名、仓库名、问题文字或部分 symbol/span 不能
通过，也不能用字符串硬编码制造命中。所有需要参与验收的 answerable 场景都通过时仓库才通过；
pass rate 固定为 100%，不是根据 fake 或真实结果事后选择的聚合阈值。

unanswerable 场景在 dense retrieval quality 验收中跳过 query encode，记录
`skipped_unanswerable_count`，不生成伪 gold、候选或 rank，也不进入 answerable pass rate。没有
正式拒答评分机制时，只能报告这一执行行为。

### 诊断报告

报告仍须包含协议列出的 Hit@1、Hit@3、Hit@5、MRR@10、nDCG@10、Evidence
precision/recall/F1、expected file recall 和逐场景 gold rank。正式通过判定只使用逐 answerable
场景 Top 3 完整 gold 条件。由于实际最多返回 3 条，Hit@5 必须同时标注：

```text
computed-from-at-most-top-3-not-five-retrieved
```

它不表示真实检索了 5 条。历史 fake-provider 指标只用于工程回归，不能设置或证明真实
BGE-M3 的验收阈值。

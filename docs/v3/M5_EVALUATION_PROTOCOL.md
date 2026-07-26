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

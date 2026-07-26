# M5 结果记录

## 当前结论

M5 infrastructure 与离线真实仓库 fake-provider pilot 已建立。真实 BGE-M3、真实 answer LLM、
真实 evaluator 和至少 18 条 live pilot 尚需通过显式 gate；在完成前不得宣称 M5 完整完成。

运行日期：2026-07-26。代码基线：
`f7ec8f16be3585053db4982a3d00ed4457c06dbc` 加当前未提交 M5 tree。dataset `pilot-v1`，
benchmark schema 1，metric schema 1。annotation 尚未人工复核。

## 已验证的离线工程结果

- dataset validator：3 repositories、36 scenarios、6 sequences；9 locate、9 explain、9 relation、
  6 impact、3 unanswerable；0 validation error。
- 离线全矩阵 run `m5-b145acaf85d7407080570d4a`：36 scenes × 6 answer modes =
  216/216 succeeded，失败、超时、provider error 均为 0；使用 fake answer/Embedding，
  不代表真实模型质量。
- adaptive sequence：6/6 sequence 执行完成，并实际经过 M4 goal/plan/task/attempt/state 路径。
- 全矩阵总体：Hit@5 0.6944、MRR@10 0.5112、nDCG@10 0.5617、Evidence F1
  0.4158、p50 218 ms、p95 1706.75 ms；这些只用于工程可重复性检查。
- 按仓库 Hit@5：Click 0.8194、HTTPX 0.5833、ItsDangerous 0.6806。
- 按类别 Hit@5：explain 0.9444、impact 0.8611、locate 0.4259、relation 0.8333、
  unanswerable 0.0000。unanswerable 的答案质量须看正确拒答指标，不能用检索命中率解释。
- fake usage accounting：216 calls、499816 tokens、$0.00；这是假 provider 的确定性记账，
  不是任何真实模型的 token 或价格测量。
- M5 单元测试：44/44；M4 32/32；M3 30/30；M2 36/36；M1 12/12；
  完整后端 262/262。以上各轮均为 0 failure、0 error、0 skipped。
- 目标仓库 execution/import/Shell tool：0/0/0。

fake provider identity 为 `fake-deterministic/fake-m5-v1/fixture-v1`；fake embedding 为
`fake-bge-m3/fixture-v1`、CPU、float32、normalized、16 dimensions。fake provider cost 为真实的
0；任何真实 provider 未给价格时必须记 unknown。

## Live gate 状态

| gate | status | note |
| --- | --- | --- |
| dataset validation | passed | fixed SHA and content fingerprint verified |
| static ingestion | passed | three local read-only checkouts available |
| lexical smoke | passed | formal BM25/Evidence/Citation path |
| BGE-M3 load | not run | requires explicit model-load gate and cached model |
| dense BGE-M3 | not run | blocked by prior gate |
| real answer LLM | not run | requires explicit network/real-LLM gate and local key config |
| real evaluator | not run | additionally requires paid-eval gate |
| live E2E/pilot | not run | blocked by live provider gates |

因此当前没有可诚实报告的 live retrieval、answer、token、cost、paired delta、confidence interval
或 judge repeatability 表。离线完整结果位于
`artifacts/m5/runs/m5-b145acaf85d7407080570d4a`，由结构化记录生成且不提交。

离线 paired delta 仅用于验证比较器：相对 lexical，hybrid Hit@5 delta -0.0278，95% bootstrap
CI [-0.1111, 0.0556]；相对 M2，M3 relation-edge recall delta +0.1111，95% CI
[0.0000, 0.3333]。由于 answer/evaluator/embedding 均为 fake、标注待人工复核，这些差值不能作为
模型优劣或教学效果结论。

## 解释边界

当前结果只支持：dataset/identity/validator、模式隔离、预算、恢复、指标公式和 M1—M4 正式链路
复用等工程结论。不能支持真实用户学习效果、mastery 准确性、M4 教学有效性、普遍优于 RAG、
LLM judge 等价人工教师、跨语言泛化或论文级 benchmark 结论。

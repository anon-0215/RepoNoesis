# 已保存项目文档的问答与引用

问答现在可以使用当前快照中已经保存的 README、overview.md、description.md。
不会下载文档、执行文档内容或重新构建源码向量。未保存的文件不能作为依据，
源码问答仍沿用现有检索、版本检查和引用校验。

首次检索与 Agent 补搜在原关键词／语义检索之外，对这些文本进行有界词法检索，
每次最多补充一个相关文档窗口，使用原有证据容量和上下文预算。
文档不会交给代码候选重排器，也不会假装成函数块。HyDE 与重排仍只增强首次代码检索，
两个开关均关闭时，文档发现与读取仍可用。

Agent 可以看到有界的已保存文档目录（真实路径和物理行数），用现有 read_source
读取最多 200 行。读取范围、字节数、仓库版本、文件身份、正文哈希均需校验。
短文件越界时返回实际行数；路径缺失时提示使用已发现的精确路径。
成功读取新文档或补齐正文后，真实正文进入后续 Planner 和最终回答；重复范围不算新进展。

文档证据具有 `source_kind=document`、真实 `document_file_id`、版本、路径、行号和内容哈希，
`code_chunk_id` 为空。代码证据原有完整校验保持不变。
最终提示区分“作者文档描述”和“源码实现事实”，文档中嵌入的指令没有执行权限。

引用在回答和历史中标为“文档”。当前活动快照可以打开引用及完整文件；
旧快照和身份不可靠的历史继续按原规则限制全文访问，仅展示保存的片段。
无需数据库结构迁移，旧记录缺少类型字段时仍按代码引用解释。

工具失败会写入独立的有界结构化日志：请求 ID、工具、调用 ID、执行序号、
实际请求路径与范围、稳定错误码和简短原因。执行摘要与历史保留前四条精简失败收据；
过长路径以哈希和显式裁剪状态表示，完整失败日志独立于普通摘要裁剪。
不保存密钥、模型思考过程或工具源码正文。

相关离线验证：

```powershell
cd backend
python -B -m unittest tests.test_document_evidence tests.test_evidence tests.test_m2_tools tests.test_chat_history tests.test_planner_delivery_progress tests.test_retrieval_enhancement tests.test_source_file_api tests.test_qa_agent tests.test_smoke_diagnostics tests.test_ask_observability
cd ../frontend
npm test -- src/workbench/documentEvidence.test.ts src/workbench/SourceFileReader.test.tsx src/App.history.test.tsx src/App.ask.test.tsx src/lib/api.stream.test.ts
npm run build
```

文档检索目前是有界词法匹配。问题不包含与已保存文档一致的文字时，RAG 可能没有文档命中；
Agent 仍可从目录继续读取。此功能只处理已保存的有限项目说明文本。

# 最终回答可选 Flash low

`LLM_ANSWER_MODEL` 只覆盖 `purpose == "final_answer"` 的模型；未设置或空白继承 `LLM_MODEL`。RAG 与 Agent 的初次最终回答、已有的一次协议修复及有限 HTTP 重试共用该选择。Planner、普通调用和其他任务继续使用原模型与参数，RAG/Agent 的模式含义不变。

完整启用组合为：

```text
LLM_ANSWER_MODEL=deepseek-flash
LLM_ANSWER_THINKING=enabled
LLM_ANSWER_REASONING_EFFORT=low
LLM_ANSWER_MAX_GENERATION_TOKENS=16384
```

这是自愿启用的候选配置，产品默认不变。16384 是 Provider 的总生成预算，包含思考与正文；可见回答限制、JSON Schema、引用身份/范围校验、证据容量、上下文字节预算、截止时间和重试次数不变。非法参数组合会在发送前失败，不会静默关闭思考、忽略 low 或自动切换其他模型。连接地址与凭据继续取原配置。

## Windows PowerShell 临时启用

以下是供用户以后操作的说明，本次交付没有启动产品。

1. 在原后端终端按 Ctrl+C 停止后端（包括开发重载进程）。确认所用端口不再由旧后端监听；不要结束来源不明的进程。默认端口的只读检查如下；若已有配置使用其他端口，请替换为实际端口：

   ```powershell
   Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
       Select-Object LocalAddress, LocalPort, OwningProcess
   ```

2. 打开一个专用 PowerShell 窗口，激活原来使用的 Python/Conda 环境。粘贴以下完整代码，按提示输入仓库根目录。它复用原有 `backend\run_backend.bat`，不写 `.env`，不修改用户级或系统级环境变量。进程变量仅影响这个启动会话和子进程；退出后端时还会恢复该窗口原有的四项进程变量。

   ```powershell
   Set-Location -LiteralPath (Read-Host '请输入 RepoNoesis 仓库根目录')
   $answerKeys = @(
       'LLM_ANSWER_MODEL', 'LLM_ANSWER_THINKING',
       'LLM_ANSWER_REASONING_EFFORT', 'LLM_ANSWER_MAX_GENERATION_TOKENS'
   )
   $answerPrevious = @{}
   foreach ($answerKey in $answerKeys) {
       $answerPrevious[$answerKey] = [Environment]::GetEnvironmentVariable($answerKey, 'Process')
   }
   try {
       $env:LLM_ANSWER_MODEL = 'deepseek-flash'
       $env:LLM_ANSWER_THINKING = 'enabled'
       $env:LLM_ANSWER_REASONING_EFFORT = 'low'
       $env:LLM_ANSWER_MAX_GENERATION_TOKENS = '16384'
       & .\backend\run_backend.bat
   } finally {
       foreach ($answerKey in $answerKeys) {
           [Environment]::SetEnvironmentVariable($answerKey, $answerPrevious[$answerKey], 'Process')
       }
   }
   ```

3. 后端在创建客户端时读取配置，因此切换或恢复都需要停止并重新启动后端；仅刷新网页不会更新配置。等待这个后端正常启动，确认没有端口占用错误。前端可继续沿用原启动路径 `frontend\run_frontend.bat`；已有前端不必重启，但浏览器使用的 API 地址必须指向本次后端，不能仍指向另一个端口或旧进程。

4. 在浏览器实际使用的后端地址查看 `/api/config/status`（默认示例 `http://127.0.0.1:8000/api/config/status`），核对 `llm.model` 仍为原模型、`answer_model` 和 `effective_answer_model` 为 `deepseek-flash`、`answer_thinking` 为 `enabled`、`answer_reasoning_effort` 为 `low`、`answer_max_generation_tokens` 为 `16384`。这只是配置核对，不会发出问答或证明远端模型身份。

## 恢复

在候选后端窗口按 Ctrl+C，等后端及重载子进程退出、`finally` 执行完成。关闭专用窗口。在原来的普通环境中用既有 `start_all.bat`，或分别用 `backend\run_backend.bat`、`frontend\run_frontend.bat` 启动；启动前同样确认旧后端已释放端口。再次核对浏览器的 API 地址与 `/api/config/status`。

恢复时以启动前的原配置为准：原来未设置覆盖值时，`answer_model` 为 null，`effective_answer_model` 等于 `llm.model`；原来已有回答阶段设置时会恢复那些设置。不要为恢复而改动全局 `LLM_MODEL`、删除原有配置或编辑真实 `.env`。如果手工中断导致 `finally` 未运行，关闭专用窗口即可丢弃其进程变量；已启动的后端子进程仍需停止。

## 模型名称与验收边界

配置摘要中的模型名是配置值；诊断的 `request_model`（错误诊断兼容字段 `model`）表示该次调用选择/请求的模型名；`response_model` 只记录 Provider 实际返回的合法模型字段，缺失时不以配置值补填。底层真实模型身份无法独立确认，仍为 UNKNOWN。诊断不保存密钥、完整凭据或原始思考内容，不改数据库结构或历史记录。

本选项仅完成模拟传输与临时数据库的离线配置验收。Flash low 的真实质量、可靠性、费用和时延尚未验收；既有两道产品集成题启用了定向补证原型，不能视为当前默认检索的直接验收。本选项不启用定向补证，不修复缺证推断，不改变 Prompt，也不自动启用候选。

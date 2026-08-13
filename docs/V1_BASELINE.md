# V1 Demo 基线

记录日期：2026-06-07

## 冻结信息

- 冻结提交 SHA：`d8a4d5646e10034b46739b6e79849b16d304aed1`
- 冻结标签：`v0.1-demo`
- 本地验证：`git rev-list -n 1 v0.1-demo` 解析到上述提交。
- 当前 V2 worktree 基线：`v2-development` 分支位于同一提交。

## 项目目录结构

```text
GitLearnAgent/
  backend/
    app/
      main.py
      config.py
      database.py
      models.py
      services/
        github_client.py
        analyzer.py
        ranker.py
        learning_agent.py
        qa_agent.py
        llm_client.py
        report.py
    tests/
      test_analyzer.py
      test_github_client.py
      test_qa_agent.py
    requirements.txt
    run_backend.bat
    run_backend.ps1
  frontend/
    src/
      App.tsx
      lib/api.ts
      main.tsx
      styles.css
      types.ts
    package.json
    package-lock.json
    run_frontend.bat
    run_frontend.ps1
  docs/
    experiment_plan.md
    project_proposal.md
    technical_route.md
  samples/
    demo_repositories.md
  .env.example
  .gitignore
  environment.yml
  README.md
  start_all.bat
```

## 当前已有功能

- 输入公开 GitHub 仓库 URL，解析 `owner/repo`。
- 使用 GitHub REST API 获取仓库元信息、默认分支、递归文件树和 blob 内容。
- 筛选文本文件，跳过依赖目录、构建产物、缓存、虚拟环境、隐藏目录和超大文件。
- 对候选文件做优先级排序，最多抓取 45 个文本文件，并发抓取文件内容。
- 对 Python、JavaScript、TypeScript、配置和 Markdown 文件做基础静态分析。
- 生成项目概览、技术栈线索、目录树、模块列表、依赖边和核心文件。
- 生成五阶段学习路线、任务和测验。
- 支持基于已抓取源码的本地检索式问答。
- 支持可选 OpenAI 兼容 LLM 增强学习路线和问答表达。
- 将分析结果、文件、模块、学习步骤和问答记录写入 SQLite。
- 导出 Markdown 项目学习报告。

## 当前前端页面

前端是 React/Vite 单页应用，主入口为 `frontend/src/App.tsx`。已存在以下标签页：

- 概览：展示仓库、语言、框架、文本文件数、核心文件数、模块数量、启动命令和核心文件。
- 项目地图：展示目录树和模块关系。
- 学习路线：展示五阶段学习步骤、推荐文件、任务和测验。
- 源码问答：向后端提交问题，展示回答和引用片段。
- 报告：展示 Markdown 报告，并支持复制和下载。

## 当前后端服务

FastAPI 入口为 `backend/app/main.py`，已存在接口：

- `GET /api/health`
- `POST /api/projects/analyze`
- `GET /api/projects/{project_id}`
- `GET /api/projects/{project_id}/map`
- `GET /api/projects/{project_id}/learning-path`
- `POST /api/projects/{project_id}/ask`
- `GET /api/projects/{project_id}/report`

后端使用 Pydantic 请求模型 `AnalyzeRequest` 和 `AskRequest`，响应主要由字典组装返回。

## 当前数据库情况

- 存储实现：`backend/app/database.py`
- 数据库：SQLite。
- 默认路径：`backend/data/gitlearn.sqlite`。
- 可通过环境变量 `GITLEARN_DB` 指定路径。
- 默认路径不可写时回退到系统临时目录下的 `gitlearnagent.sqlite`。
- 表结构包含 `projects`、`repo_files`、`modules`、`learning_steps`、`chat_answers`。
- 当前没有数据库迁移框架；Schema 通过 `CREATE TABLE IF NOT EXISTS` 初始化。

## 当前 LLM 接入方式

- 实现文件：`backend/app/services/llm_client.py`。
- 使用 OpenAI 兼容的 `/v1/chat/completions` 接口。
- 环境变量：`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`。
- 默认 base URL 为 `https://api.deepseek.com`，默认模型为 `deepseek-chat`。
- 未配置 `LLM_API_KEY` 时 LLM 不可用，系统使用本地规则兜底。
- 当前 LLM 输出没有统一结构化 Schema 校验。
- 当前错误处理会在网络、超时或 JSON 解析失败时返回 `None`。

## 当前静态分析方式

- 实现文件：`backend/app/services/analyzer.py`。
- Python 使用标准库 `ast` 提取 import、函数名和类名。
- JavaScript/TypeScript 使用正则提取 import、require、export、函数、类和箭头函数符号。
- 框架识别来自依赖文件和源码标记，覆盖 React、Vite、Next.js、Express、Vue、Svelte、FastAPI、Flask、Django、Streamlit、Pytest。
- 当前 Python AST 还没有函数/类代码块、起止行号、内容哈希、跨文件调用图或增量更新。

## 当前核心文件排序方式

- 实现文件：`backend/app/services/ranker.py`。
- 根据 README、依赖文件、入口文件、源码目录、测试路径、配置文件、入口特征、框架特征、import 数量、符号数量和内容长度评分。
- 默认选取得分靠前且分数大于 12 的最多 14 个核心文件。
- 排序是规则评分，不是学习得到的权重。

## 当前检索方式

- 实现文件：`backend/app/services/qa_agent.py`。
- 对问题做英文/路径 token 提取，并根据路径、内容词频、意图提示、核心文件标记和重要性分数排序。
- 意图提示覆盖启动、入口和核心模块问题。
- 返回最多 5 个引用片段，每个片段来自相关文件附近行文本。
- 当前没有向量检索、BGE-M3、Embedding 缓存、BM25 或行号级引用。

## 当前学习路线生成方式

- 实现文件：`backend/app/services/learning_agent.py`。
- 默认生成五个固定阶段：全局印象、依赖和启动、入口主流程、分模块阅读、小修改任务。
- 阶段文件来自 README、配置文件、入口文件、核心文件和模块文件。
- 配置 LLM 时，只尝试增强第一阶段 goal 文案。
- 当前没有初始诊断、用户历史表现、动态路径调整或评分闭环。

## 当前报告功能

- 实现文件：`backend/app/services/report.py`。
- 输出 Markdown 字符串。
- 内容包含仓库、默认分支、主语言、框架、项目概览、核心文件、模块地图、推荐学习路线和大创展示说明。
- 前端报告页支持复制和下载。
- 当前没有 PDF、Word、图表或学习档案导出。

## 真实启动命令

根目录一键启动：

```powershell
start_all.bat
```

分别启动：

```powershell
backend\run_backend.bat
frontend\run_frontend.bat
```

手动后端：

```powershell
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

手动前端：

```powershell
cd frontend
npm run dev
```

注意：当前 `start_all.bat` 在 V2 worktree 中仍硬编码调用 `D:\Project\GitLearnAgent\backend\run_backend.bat` 和 `D:\Project\GitLearnAgent\frontend\run_frontend.bat`，不是相对调用当前 worktree。

## 真实测试命令与执行条件

后端测试：

```powershell
cd backend
python -B -m unittest discover tests
```

当前环境中 `python` 不在 PATH，本次使用 `backend/run_backend.bat` 中约定的 Conda Python 路径执行等价命令：

```powershell
cd backend
D:\Programme\Anaconda\envs\gitlearnagent\python.exe -B -m unittest discover tests
```

结果：8 个测试通过。

前端构建命令来自 `frontend/package.json`：

```powershell
cd frontend
npm run build
```

执行条件：需要已安装 `node_modules`。本次未修改前端业务代码，也未安装依赖。

## 已知缺陷

- 根目录一键启动脚本硬编码 V1 稳定目录，V2 worktree 中运行时不会直接启动当前 worktree 的后端和前端。
- 后端和前端启动脚本存在本机 Conda 路径假设。
- GitHub Token 相关错误提示中硬编码了 V1 目录下的 `.env` 路径。
- `.env.example` 未列出 `GITLEARN_DB`。
- LLM 输出没有 Schema 校验、有限重试记录或结构化失败记录。
- 源码问答没有精确行号和向量检索。
- Python AST 分析只提取符号名，不保存函数/类代码块边界。
- 数据库缺少迁移机制和版本字段。

## V2 必须保留的旧功能

- 公开 GitHub 仓库 URL 解析和抓取。
- 路径过滤、大文件过滤和关键文件优先抓取。
- 基础项目概览、模块划分、核心文件排序和目录树。
- React/Vite 单页工作台的概览、地图、学习路线、问答和报告标签页。
- 未配置 LLM 时仍可运行的本地规则兜底。
- SQLite 保存分析结果和问答记录。
- Markdown 报告导出。
- 现有后端 unittest 测试覆盖的基础行为。

## 当前尚未验证的事项

- 真实 GitHub 网络访问在当前环境中的可用性和限流表现。
- 前端 `npm run build` 在当前机器上的结果。
- 一键启动脚本在 V2 worktree 中的实际运行效果。
- 使用真实 `LLM_API_KEY` 时的 LLM 增强效果。
- 分析大型仓库时的性能和超时表现。
- Windows 以外系统的启动流程。

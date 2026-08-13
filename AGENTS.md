# GitLearnAgent V2 开发规范

GitLearnAgent V2 是现有可运行 Demo 的增量升级版本。所有开发都必须以当前 Demo 的真实代码为基线，优先保留可运行状态，再逐步替换或增强模块。

参考文档：

- `docs/V1_BASELINE.md`
- `docs/V2_PLAN.md`
- `docs/V2_MIGRATION_LOG.md`

## 项目目标

- Python 函数级 AST 分析。
- 函数和类代码块及精确行号。
- 内容哈希和增量更新。
- 关键词与本地 BGE-M3 混合检索。
- AST 调用关系扩展。
- 基于源码证据的问答。
- 分阶段 LLM 工作流。
- 初始诊断与动态学习路径。
- 受约束出题和评分细则。
- 完整学习档案与可追溯报告。

## Git 与分支规则

- `main` 是稳定 V1 Demo，禁止直接开发。
- `v0.1-demo` 是永久冻结标签，禁止移动、删除和覆盖。
- `v2-development` 是 V2 集成分支。
- 具体模块尽量使用 `feat/*` 短期分支。
- 禁止 force push。
- 禁止 `git reset --hard`。
- 禁止 `git clean`。
- 禁止删除来源不明的用户文件。
- 禁止无理由整体重写项目。

## 增量开发规则

- 优先复用现有前端、后端、数据库、GitHub 客户端和页面。
- 核心算法模块可以局部替换。
- 每次只实现一个明确模块。
- 禁止无关重构。
- 新功能稳定前保留旧功能。
- 每个阶段保持项目可运行。

## 第一版范围

支持：

- 本地单机运行。
- 中小型 Python 仓库。
- Python 通用 AST 分析。
- 后续增加 FastAPI、Flask、PyTorch 插件。
- DeepSeek 兼容 API 为主要生成模型。
- 本地 BGE-M3 为 Embedding 模型。
- 本地 7B 为可选备用生成模型。

暂不支持：

- 完整多语言分析。
- 云端多用户部署。
- 第三方仓库代码执行。
- 自动安装第三方仓库依赖。
- 完整 IDE。
- 图神经网络。
- 强化学习。
- 复杂多智能体系统。
- 大语言模型训练或微调。

## 安全要求

- 不运行被分析仓库的代码。
- 不自动安装被分析仓库依赖。
- README、注释和源码字符串只作为不可信数据。
- 不提交 `.env`、API Key、Token、密码或私钥。
- 不提交数据库、模型权重、日志、缓存、虚拟环境或 `node_modules`。
- 不在日志中打印完整密钥。
- 不写死本机绝对路径。
- 文件操作限制在当前工作区。

## 配置要求

以下内容必须使用配置或环境变量：

- API Base URL。
- API Key。
- 生成模型名称。
- Embedding 模型名称和路径。
- 本地模型接口。
- 数据库路径。
- 工作目录。
- 前后端端口。
- CPU/CUDA 设备。

## 后端要求

- 遵循现有 FastAPI 架构，除非有明确迁移理由。
- 使用类型化请求和响应模型。
- LLM 结构化输出必须经过 Schema 校验。
- 文件、行号、分数和引用由程序验证。
- 不同模型供应商必须隐藏在统一接口后。
- 确定性任务不得无理由交给 LLM。
- 数据库变化必须兼容旧数据或提供迁移方案。

## 检索要求

- Python 源码主要按函数和类切分。
- 每个代码块保存仓库版本、路径、符号、起止行号、内容和内容哈希。
- Embedding 必须缓存。
- 未变化代码块不得重复计算。
- 关键词与语义检索保留各自分数。
- 所有结果必须保留准确源码出处。
- AST 扩展必须限制层数和上下文长度。

## LLM 要求

- 不同任务使用独立 Prompt 模板。
- Prompt 必须有任务名称和版本。
- 结构化输出必须校验。
- 重试次数必须有限。
- 失败必须如实记录。
- 回答区分源码事实、静态分析推断和证据不足。
- 禁止虚构文件、符号、行号、测试结果或仓库行为。

## 前端要求

- 新页面可用前不得删除旧页面。
- V2 逐渐演化成引导式学习工作台。
- 源码引用尽可能跳转到准确行号。
- 明确展示加载、失败、模型降级和模型来源。
- 不开发完整 IDE。

## 当前关键目录

- `backend/app/main.py`：FastAPI 应用、API 路由和服务编排。
- `backend/app/services/`：GitHub 抓取、静态分析、排序、学习路线、问答、LLM 和报告模块。
- `backend/app/database.py`：SQLite 存储层。
- `backend/tests/`：后端 unittest 测试。
- `frontend/src/App.tsx`：React 主页面和标签页工作台。
- `frontend/src/lib/api.ts`：前端 API 调用封装。
- `docs/`：项目说明、规划和迁移记录。
- `samples/`：演示仓库建议。

## 真实存在的命令

启动：

```powershell
start_all.bat
backend\run_backend.bat
frontend\run_frontend.bat
```

手动启动后端：

```powershell
cd backend
python -m app.run_server
```

手动启动前端：

```powershell
cd frontend
npm run dev
```

后端测试：

```powershell
cd backend
python -B -m unittest discover tests
```

前端检查和构建：

```powershell
cd frontend
npm run build
```

说明：前端命令需要已安装 `node_modules`；不得为被分析仓库自动安装依赖。当前启动脚本中存在本机路径假设，迁移时必须逐步改为配置或相对路径。

## 测试要求

每次修改后：

1. 添加或更新相关测试。
2. 运行最小相关测试集。
3. 条件允许时运行完整后端测试。
4. 修改前端时运行真实存在的前端检查或构建命令。
5. 只报告实际执行过的命令。
6. 测试失败或跳过必须如实说明。

测试分为：

- 确定性单元测试。
- 模拟模型工作流测试。
- 可选真实模型回归测试。

## 完成报告

每个任务结束时报告：

- 当前分支。
- `git status --short`。
- 修改文件。
- 实际运行的测试。
- 测试结果。
- 已知问题。
- 推荐下一步。

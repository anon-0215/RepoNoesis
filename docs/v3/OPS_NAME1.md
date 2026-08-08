# OPS·NAME1｜仓库与项目命名统一

## 命名决定

- 中文名：源鉴。
- 英文名：RepoNoesis。
- 标准显示名：源鉴 RepoNoesis。
- GitHub 仓库：`anon-0215/RepoNoesis`。
- V3 本地工作树目标路径：`D:\Project\RepoNoesis`。

GitLearnAgent 是 RepoNoesis 的早期 V1/V2 阶段名称。当前产品文本使用
“源鉴 RepoNoesis”；英文仓库、代码和工具语境使用 `RepoNoesis`。
V1/V2 文档中的 GitLearnAgent 名称、路径、tag、分支、commit 和基线记录是
历史事实，不重写。

## 已统一的当前名称

- README 主标题、项目介绍、仓库 URL 和项目结构。
- 浏览器标签页、前端显示品牌、报告下载文件名和前端包元数据。
- FastAPI OpenAPI title、Python 包说明、GitHub REST User-Agent 和面向用户的错误文本。
- 当前 V3 产品、架构、Schema/API、实验、申报和启动说明。

## 兼容性保留项

本轮不修改 Python import path、API endpoint、JSON 字段、数据库表/字段、
schema v11、workspace/learner/revision identity、learning event 类型、缓存键、
Provider/Embedding 配置名或环境变量名。`gitlearnagent` Conda 环境名和
`gitlearnagent.sqlite` fallback 文件名是已有兼容标识，继续保留。

## 边界与人工项

本轮只治理命名，没有新增产品功能，没有数据迁移，也不改变 API、
Prompt、Agent、Provider、Embedding 或学习连续性算法。缓存、构建输出、模型文件、
数据库业务数据、第三方依赖和用户未跟踪文件不在命名修改范围内。

Codex/ChatGPT 中的项目显示名不存储在本仓库。如界面仍显示旧名，需要用户
在 Codex 界面中手工改为 `RepoNoesis`；不应修改 Codex 内部系统文件。

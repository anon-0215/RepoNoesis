# 交付核验环境记录

记录日期：2026-10-22。用途：遇到安装差异时对照已用环境。以下为实际读取的版本，**不是完整依赖锁，也不要求其他硬件照搬 CUDA 构建**。

| 项目 | 本次已有环境 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.12.13 |
| Node.js | 24.19.0 |
| npm | 11.17.0 |
| Git | 2.54.0.windows.1 |
| fastapi | 0.136.3 |
| uvicorn | 0.49.0 |
| pydantic | 2.13.4 |
| sentence-transformers | 3.4.1 |
| torch | 2.12.1+cu130 |
| transformers | 4.57.6 |
| numpy | 2.5.1 |
| huggingface-hub | 0.36.2 |
| tokenizers | 0.22.2 |
| safetensors | 0.8.0 |

前端依赖按 `frontend/package-lock.json` 定义，首次安装使用 `npm ci`；后端按 `backend/requirements.txt` 的版本范围安装。开发依赖与模型不随源码包分发。PyTorch 与 BGE 相关版本由包元数据读取，本轮没有加载 BGE 模型。

## 核验方式

本轮从新生成的源码 ZIP 独立解压，复用本机已有 Python 与前端依赖，在独立端口和隔离数据副本中启动；不读取正式数据库或真实 `.env`。浏览器核对源码问答入口、报错到源码、用例到断言与实现。启动脚本测试及打包校验另有记录。

此核验与从全新电脑下载安装依赖不同。本轮未重新下载模型、构建首次向量、发送真实 RAG/Agent 问答，未声称所有模型配置、所有仓库或跨平台均通过。首次模型准备与 API 配置按使用说明进行。

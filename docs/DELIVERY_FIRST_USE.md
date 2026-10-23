# RepoNoesis：安装、配置与首次使用

适用范围：Windows 本地单用户、中小型 Python Git 仓库。请在**解压后的 RepoNoesis 根目录**打开 PowerShell；下面的相对路径均以此为起点。无需克隆本项目 GitHub，源码包已经包含当前交付实现。

## 1. 准备运行环境

需要 Python 3.12、Node.js/npm、Git，以及产品后端依赖。当前核验版本见 [DELIVERY_ENVIRONMENT.md](DELIVERY_ENVIRONMENT.md)。建议使用独立 Python 环境，避免影响其他项目。

```powershell
python --version
node --version
npm --version
git --version
```

如果已有可用产品环境，可以继续使用。首次准备可用下面的虚拟环境方式，或使用包内 `environment.yml` 创建 Conda 环境，两者选一即可。

```powershell
# 执行前确认 python --version 是 Python 3.12。
python -m venv .venv
$env:REPONOESIS_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
& $env:REPONOESIS_PYTHON -m pip install -r backend/requirements.txt
Push-Location frontend
npm ci
Pop-Location
```

后端依赖会涉及 PyTorch。若需 NVIDIA GPU，应按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)为设备安装兼容版本，不要照搬其他电脑的 CUDA 版本。CPU 可作为配置选项，但首次索引速度取决于设备。本项目的声明是版本范围；本机已装版本用于核对，不是跨硬件通用的完整环境锁。

依赖安装需要网络；只安装 RepoNoesis 自身依赖，**不安装被分析仓库的依赖**。

## 2. 获取 BGE-M3 模型

首次导入要建立本地向量索引，需要预先准备 [BAAI/bge-m3 官方模型](https://huggingface.co/BAAI/bge-m3)的完整 Sentence Transformers 文件，包括模型权重、配置、分词器及模块文件。不要只下载 README 或单个权重文件。

已有完整模型时直接使用其目录，无需重复下载。没有模型时，可在联网的准备阶段使用已安装的 huggingface_hub：

```powershell
& $env:REPONOESIS_PYTHON -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='BAAI/bge-m3', local_dir='models/bge-m3')"
```

下载会占用磁盘并依赖网络；完成后把模型目录的**绝对路径**写入下一步的 `EMBEDDING_MODEL`。此处只说明获取方法，交付包不包含模型。产品运行时保留 `EMBEDDING_OFFLINE=true`，避免启动时隐式下载。

## 3. 创建自己的配置

```powershell
if (Test-Path -LiteralPath .env) { throw '已有 .env，请编辑现有配置，不要覆盖。' }
Copy-Item -LiteralPath .env.example -Destination .env
notepad .env
```

在 `.env` 中核对以下内容；密钥只保存在自己电脑的私有配置中：

| 配置项 | 填写方式 |
| --- | --- |
| `EMBEDDING_ENABLED` | 首次导入使用 `true` |
| `EMBEDDING_MODEL` | 已下载完整 BGE-M3 目录的绝对路径，可使用正斜杠 |
| `EMBEDDING_OFFLINE` | 保持 `true` |
| `EMBEDDING_DEVICE` | `auto`；需要明确指定时用 `cpu` 或 `cuda` |
| `GITLEARN_DB`、`RUNTIME_DATA_DIR` | 新解压包可保留模板中的相对路径；它们存放项目、历史和源码材料，后续请保留 |
| `LLM_PROVIDER` | 生成问答使用 `openai_compatible` |
| `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` | 从所选服务商控制台与文档获取，分别填写接口地址、密钥和支持的模型名 |

报错定位、用例对照及源码阅读本身不调用生成 API；已有索引时可先不填 Provider。**全新数据仍要完成模型准备与首次导入。** RAG 和 Agent 都需要生成 API，题文和选中的源码上下文会发送给配置的服务商。

初次使用保留模板默认预算；`LLM_ANSWER_MODEL` 空白时继承主模型。思考开关和 reasoning effort 只在所选模型支持时填写，不要为不同供应商照抄历史实验配置。

## 4. 启动与停止

在刚才设置了 `REPONOESIS_PYTHON` 的同一终端执行：

```powershell
.\start_all.bat
```

也可以先激活自己的 Conda 环境再运行该命令。需要指定解释器时，`REPONOESIS_PYTHON` 指向其 `python.exe`；需要指定 Node 时使用 `REPONOESIS_NODE`。这些是当前进程变量，不要求修改本机全局环境。

启动器默认使用后端 8000、前端 5173，打印实际地址和本次进程 PID。端口被占用时会拒绝接管。可换一对空闲端口：

```powershell
.\start_all.bat -BackendPort 8041 -FrontendPort 5221
```

打开启动器打印的前端地址。后端默认健康页为 `http://127.0.0.1:8000/api/health`；更换端口后相应替换。HTTP 可达说明服务启动，`embedding_ready` 等模型状态及项目导入结果需另行核对。

用完后只停止本次打印且确认属于该包的进程，例如把下方占位符替换为本次 PID：`taskkill /PID <PID> /T /F`。不要按 Python/Node 名称批量结束进程。手动前台运行可按 Ctrl+C；关闭网页不会自动关闭服务。

## 5. 第一次导入仓库

1. 项目管理中选择“本地目录”，填写可信、干净、带 `.git` 的 Python 仓库绝对路径；ZIP 解压的纯源码目录没有 Git 身份，不能直接代替。
2. 提交一次分析，等待静态解析、代码块提取及 BGE 向量构建。此阶段不执行仓库代码，也不调用生成 API。
3. 导入完成后打开项目，核对仓库名和 revision。项目库显示已持久化的项目状态和向量数量。
4. 首次索引可能较久。界面暂未返回或项目列表尚未出现不等于导入失败；保持后端运行，检查日志与项目库，不要重复提交同一导入。

若要对齐示例版本，可在自己选定的材料目录准备以下仓库（命令只获取源码，不安装或执行它们）：

```powershell
git clone --no-checkout https://github.com/pallets/click.git click-demo
git -C click-demo checkout --detach 6aabf099bfdd4c1e75fe8d0e0d4241372b988ab1
git clone --no-checkout https://github.com/encode/httpx.git httpx-demo
git -C httpx-demo checkout --detach b5addb64f0161ff6bfe94c124ef76f6a1fba5254
```

仓库下载需要网络。用自己已有的同版本干净 Git 副本亦可。示例源码和测试属于各自作者，保留其许可证。

## 6. 体验三种能力

**问源码**：打开“源码问答”，可选 RAG 或 Agent。例如在 Click 中询问：`click.utils.make_str 怎样处理字节输入和非字节输入？按文件系统编码解码发生 UnicodeError 时如何处理？` RAG 使用检索依据回答；Agent 可自主继续搜索、读取源码，再回答。点击回答中的 E 引用，核对文件、行号和同版本全文。生成结果受模型和输入影响，不要求与视频逐字相同。

**查报错**：打开“源码探索 → 报错定位”。Click 可输入 `nargs=-1 is not supported for options.`，HTTPX 可输入 `Cannot send a request, as the client has been closed.`。查看可能位置、局部条件并打开源码。普通 Python 堆栈可额外打开调用行。上述文本是源码中的报错示例，并非对评审电脑实际故障的诊断。

**比用例**：切到“用例对照”，选择一组，查看两侧条件和作者测试预期，展开依据，依次打开原始断言与候选实现。页面展示静态提取结果，不表示作者测试已执行。缺少测试材料时，按提示补充同版本干净 Git 目录；材料不足、解析失败或不支持会分别说明。

完成后保留数据库与运行数据目录。项目和历史记录可以再次打开；源码包不携带开发者原有问答记录。

## 7. 常见问题

| 现象 | 处理方法 |
| --- | --- |
| 启动提示缺少依赖 | 核对实际 Python 路径；后端安装 requirements，前端执行 npm ci |
| 模型不可用 | 检查完整模型目录、离线配置及设备与 PyTorch 是否兼容 |
| 端口被占用 | 先确认原服务身份，或用启动参数选择空闲端口；不要结束未知服务 |
| API 不可用或回答超时 | 检查服务商地址、模型支持、额度和网络；源码探索可独立使用 |
| 导入长时间等待／浏览器连接中断 | 保留服务和日志，核查同次后台任务及项目库，避免重导入 |
| 用例材料缺失 | 补充包含同一精确 revision 的本地 Git 材料，不混入其他版本 |

交付核验使用本机已有依赖和隔离样例数据；干净机器安装、模型重新下载及首次导入未在本轮重做。详细核验范围见 [环境记录](DELIVERY_ENVIRONMENT.md)。

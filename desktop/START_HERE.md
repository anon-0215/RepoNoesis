# 源鉴桌面版源码

Windows x64 可选桌面入口。网页版仍使用原启动脚本，HyDE 和重排仍默认关闭。桌面沿用主仓库后端、React 工作台、检索和问答协议，没有单独维护一套检索实现。

## 检索增强设置

桌面设置中“生成 API”之后有“检索增强”区域，提供“HyDE 辅助检索”和“检索结果重排序”两个独立开关。显示发行默认与用户覆盖合并后的有效布尔值；旧配置缺字段时保持原默认，非法字符串不视为开启。普通网页仍默认关闭。

“保存并应用”后会重启本地服务；等待顶部“本地服务已就绪”后，下次问答使用新配置。正在生成回答时保存会被拒绝，提示“未保存”，不会中断问答。打开/保存设置本身不调用生成API；重启按原流程准备本地模型。读取失败时禁止保存，保存无变化时不重启；重开设置与退出重启均重新读取持久化值。

开关仅代表配置选择。每次回答沿用既有执行摘要，分别记录成功、关闭或失败回退，不会因开关开启就显示成功。只改开关时保留阶段预算、密钥、模型、资源路径及其他设置。桌面启动诊断logs/runtime-identity.json中的retrieval_enhancement_config记录后端实际解析值，作为配置记录，不是执行成功记录。

设置重启使用实例及请求ID绑定的短租约，问答worker退出前不重启（包括流连接已断开而后台仍在运行）。响应丢失时补偿释放，租约最多30秒；落盘后再次核对空闲状态，过期后有新问答则还原设置并拒绝重启。恢复文件也失败时明确提示“可能已保存但未应用”，请重开窗口确认。

相关离线测试：`python -B -m unittest desktop.test_retrieval_enhancement_config desktop.test_settings_lifecycle desktop.test_gui_settings desktop.test_agent_runtime_config desktop.test_desktop_host desktop.test_build`。界面脚本测试为`node --test desktop/test_shell.cjs`，复用前端已安装的jsdom（需Node可解析frontend/node_modules，例如在测试子进程中设置NODE_PATH）。shell.html由窗口直接加载，修改它无需重新构建未变化的React生产前端。

桌面壳页面按内容哈希写入用户目录desktop-ui/shell-<sha256>.html，以新URL载入更新后的设置，避免WebView缓存旧页面。只包含公开界面HTML，不复制设置或密钥，不清除WebView存储、历史和其他用户文件。

## 桌面 Agent 阶段参数

桌面在子进程白名单中传递阶段配置，不读取项目 `.env`。可在发行 `desktop-config.json` 或用户数据 `desktop-settings.json` 的 `agent_runtime` 对象中设置下列环境变量同名字段。用户字段按键覆盖发行默认，其他设置与 DPAPI 密钥保留；保存普通设置不会丢弃这些字段。

| 配置 | 缺省桌面值 |
| --- | --- |
| `LLM_PLANNER_MAX_GENERATION_TOKENS` / `LLM_ANSWER_MAX_GENERATION_TOKENS` | 各8192，包含思考与正文 |
| `LLM_TIMEOUT_SECONDS` | 单次90秒 |
| `AGENT_TOTAL_DEADLINE_MS` / `AGENT_FINAL_ANSWER_RESERVE_MS` | 总计300000 / 回答预留90000毫秒 |
| `AGENT_MAX_PLANNER_TOKENS_PER_STEP` / `AGENT_MAX_TOTAL_PLANNER_TOKENS` | 可见决策每步1024 / 累计8192 |
| `AGENT_MAX_FINAL_ANSWER_TOKENS` | 可见回答4096 |

桌面复用后端严格预算配置，仍为5步、8次工具调用；用 `AGENT_BASE_EVIDENCE_COUNT=auto` 保持既有首检策略（Agent最多5条，RAG按请求证据容量），不改基础召回和补查容量。quality模式未配置auto时仍沿用原默认5条。取消、有限重试和结构化校验保留。回答预留须覆盖一个API时限，总时限须大于预留。额度不足导致供应商截断仍报错，不接受空正文。

官方 `api.deepseek.com` 的 `deepseek-flash` / `deepseek-v4-pro` 缺省明确使用 `LLM_PLANNER_THINKING`、`LLM_ANSWER_THINKING` 为 `enabled`，两个 `REASONING_EFFORT` 为 `high`。这用于明确兼容参数，不代表模型效果最优。其他端点默认省略这四个扩展字段；有已验证兼容协议时才显式配置。`thinking=disabled` 时不发送 effort，已保存值保留。不要把按供应商推导的模式默认值固化为通用配置，以免切换供应商后遗留参数。普通网页默认不变。

`agent_runtime` 数字使用JSON整数，思考为 `enabled` / `disabled` / 空字符串，effort为 `low` / `high` / `max` / 空字符串（空字符串表示省略）。例如仅提高Planner生成额度：`"agent_runtime": {"LLM_PLANNER_MAX_GENERATION_TOKENS": 16384}`。不更改模型、资源目录、设备、缓存身份或增强开关。

后端 `provider_request_parameters` 日志仅记录每次实际请求的阶段、模型、额度、模式、强度及时限，不记录密钥、地址、提示词或思考正文。离线贯穿检查：`python -B -m unittest desktop.test_agent_runtime_config desktop.test_gui_settings desktop.test_launcher desktop.test_retrieval_enhancement_config desktop.test_desktop_host`；后端另有阶段额度、正文限制、截断、取消和绝对截止时间回归。

## 目录与入口

| 文件 | 职责 |
| --- | --- |
| `desktop_host.py`、`shell.html` | pywebview 窗口、设置桥接、加载状态、窗口恢复、启动退出 |
| `gui_settings.py` | 独立设置、Windows 当前用户 DPAPI 加密密钥 |
| `launcher.py` | 子进程环境白名单、端口检查和 Windows Job Object；其浏览器启动 main 是早期调试入口，当前 Start.cmd 不走它 |
| `server.py` | 同源托管生产前端及既有 FastAPI，控制管道关闭后清理后台 |
| `app_files.py` | 初始构建和刷新共用的窗口模块与启动文件复制清单 |
| `settings_lifecycle.py` | 桌面设置重启租约，问答未完成时拒绝保存，避免中断生成 |
| `build_frontend.py` | 在构建子进程中设置空API地址和桌面标记，不改父终端或.env |
| `build.py` | 从选定、已准备的 Conda Python 环境复制可重定位运行库及应用 |
| `refresh_app.py` | 显式刷新停止运行的开发副本，仅替换应用文件，保留运行库和已有发行配置 |
| `build_portable.py` | 从指定运行库和当前源码/前端组装独立便携 ZIP，排除个人数据 |
| `acceptance.py` | 仅 `--acceptance` 启用的受限验收账本；普通启动无两题限制 |

发行目录中的 `Start.cmd` 和 `源鉴.vbs` 均启动 `desktop/desktop_host.py`。便携布局为 `源鉴.vbs`、`排错启动.cmd` 与 `app/` 并列，程序文件位于 `app` 内。源码目录本身不含 Python 运行库，不能直接双击源码中的 Start.cmd 代替构建。

## 开发和构建

以下 PowerShell 命令从仓库根目录执行。先自行准备 Python 3.12 Conda 产品环境及主仓库后端依赖，窗口依赖见 `desktop/window-requirements.txt`；Node/npm 和 `frontend/node_modules` 只用于构建。仅在开发环境安装产品依赖，不为被分析仓库安装依赖。

```powershell
$BuildPython = (Get-Command python).Source # 指向已准备的产品 Conda 环境
$BuildTools = Join-Path $PWD 'output/desktop-build-tools'
$FrontendDist = Join-Path $PWD 'output/desktop-ui-new'
$Release = Join-Path $PWD 'output/desktop-app-new' # 使用新的目标，保留旧包
& $BuildPython -m pip install -r desktop/window-requirements.txt
& $BuildPython -m pip install --target $BuildTools -r desktop/build-tools-requirements.txt
```

构建桌面前端；脚本仅在子进程设置 `VITE_API_BASE_URL=''`、`VITE_DESKTOP='true'`，兼容Windows PowerShell 5.1及新版本：

```powershell
& $BuildPython -I -B desktop/build_frontend.py --output $FrontendDist
& $BuildPython -I -B desktop/build.py --tools $BuildTools --output $Release --frontend-dist $FrontendDist
```

可用 `--node <node可执行文件>`选择构建期Node。不要在Windows PowerShell 5.1中仅用 `$env:VITE_API_BASE_URL=''`，该版本会删除变量并使前端退回网页API地址。普通网页构建不设置 `VITE_DESKTOP`，继续使用原有 `npm.cmd run build`。构建不会写真实 `.env`。

`build.py` 先打包并重定位选定 Python 环境，再核对 Python/窗口组件并复制应用。构建环境必须已含窗口依赖；本机系统还需 WebView2 和 .NET Framework。`BUILD.json` 和 `DEPENDENCY_VERSIONS.json`记录实际构建基线与组件版本。运行库复制不是冻结可执行文件；不要用 PyInstaller 的 sys.executable 行为替代现有 Provider 子进程机制。

只有明确准备了另一份独立开发运行库时才使用 `--reuse-runtime`。它不复制或更新运行库。后续刷新也必须明确指定允许替换应用：

```powershell
& $BuildPython -I -B desktop/refresh_app.py --output $Release --frontend-dist $FrontendDist --replace-app
```

刷新会覆盖目标内的后端、前端和窗口应用文件，不清理旧文件，不覆盖已有 `desktop-config.json` 或 `runtime`。先停止该副本；不要以旧样机、分发包或个人日常目录为目标。

需要另组便携包时，使用明确的运行库来源和全新输出目录：

```powershell
& $BuildPython -I -B desktop/build_portable.py --sample-app $Release --frontend-dist $FrontendDist --output output/desktop-portable-new --version-suffix=-trial
```

该命令只从 sample-app 复用运行库与依赖清单；后端、窗口文件、前端来自当前源码/显式构建目录。便携发行配置使用 `desktop-config.portable.json`，两项检索增强默认开启。已有 ZIP 不覆盖；`--resume` 仅允许接续未产出 ZIP 的专用组装目录。本轮源码整合不重建运行库或便携 ZIP。

## 配置、数据和退出

双击新发行目录的 `源鉴.vbs`（静默窗口）或 `Start.cmd`（排错入口）。先检查 WebView2 Runtime 与 .NET Framework 4.6.2+；缺 BGE/Git 时可先进入设置。模型使用完整本地 BGE-M3 权重，Git 使用完整 Git for Windows；GPU 另需与运行库 Torch 兼容的驱动。详见 `portable-docs/依赖说明.md`。

用户数据默认为发行 `app`/程序根目录旁的 `user-data`；可通过 `Start.cmd --data-dir <独立目录>` 指定。设置、数据库、缓存、仓库工作副本、日志和 WebView 状态均在此目录。桌面子进程跳过 `.env`，不继承开发环境 API token 或 Python/Conda 路径。

设置页的 API Key 不回显。留空保留已保存密钥，勾选清除密钥才删除；密钥使用当前 Windows 用户 DPAPI 加密后存入 `desktop-settings.json`，仅在启动自有后端时于内存中解密传递。更换 Windows 用户需重新填写。不要分发自己的 user-data。

配置优先级为发行 `desktop-config.json` → 用户 `desktop-settings.json` 同名字段覆盖。`hyde_enabled` 和 `rerank_enabled` 独立；普通桌面示例默认关闭，增强便携配置默认开启。关闭程序后编辑实际生效的文件，保存再启动；普通设置保存会保留这些选项。`device`、`batch_size`、`max_length`、`port` 也沿用现有配置，缓存身份校验保持严格。

关闭窗口请求自有后端正常退出，超时由本次 Windows Job Object 清理子进程树；不会结束其他网页实例。相同数据目录只允许一个窗口实例。排错日志位于该数据目录 logs。

## 离线检查与验证边界

源码测试明确逐模块执行，禁止直接对整个 desktop 目录做测试发现：

```powershell
Push-Location desktop
& $BuildPython -B -m unittest test_launcher test_gui_settings test_desktop_host test_portable_first_run test_retrieval_enhancement_config test_portable_package test_build test_acceptance
Pop-Location
& $BuildPython -B desktop/offline_checks.py --without-timing
& $BuildPython -B desktop/offline_checks.py --transport-only
```

默认验证当前源码，输出报告实际 app/host 路径。`test_desktop_host.py`、`test_portable_first_run.py` 可用 `--package <发行根目录>` 显式检查新产物；`offline_checks.py --package <发行根目录>` 选择该后端，但使用当前源码中的测试。包内通过不能替代源码通过。

`manual_native_layout.py <发行根目录> --output <新记录路径>` 会创建真实窗口，仅供显式手工布局回归；不属于离线单元测试。旧浏览器阶段 `validate_runtime.py` 未纳入当前入口，其历史原件留在本机 output。

本轮整合验证设置/密钥语义、路径隔离、模拟后台退出、增强配置传递、构建复制清单，以及网页/桌面前端构建；不加载 BGE 或调用真实 Provider，不新增模型效果结论。历史桌面记录已验证过真实双模块问答、引用/完整源码和历史恢复，见主仓库 `docs/HYDE_RERANK_VALIDATION.md`，属于历史证据。安装器、完整运行库重建、换机/干净系统验证仍未完成；其他硬件和 CPU 适用范围需要另行验证。

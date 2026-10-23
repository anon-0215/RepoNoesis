# 源鉴 RepoNoesis

让每一次代码解释，都源于真实源码。

RepoNoesis 是面向陌生 Python 仓库的本地源码阅读工作台。用户导入仓库后，可以直接提问、从报错追查相关分支，或比较作者测试中的不同条件与预期，再点击回到同版本源码。

## 三种源码阅读入口

- **问源码**：关键词与本地 BGE-M3 混合检索支撑 RAG 回答；Agent 可根据已有依据继续搜索、阅读源码，受工具次数、上下文与时间预算约束。回答附带可打开的源码引用。
- **查报错**：整理可能的抛错位置、所属函数和局部条件。普通 Python 堆栈可关联当前快照中的调用行和抛错范围，多个可能位置同时保留。
- **比用例**：静态整理作者测试的条件与预期，并连接断言和候选实现。用于理解行为差异，不执行作者测试。

当前主要支持 Windows 本地单用户、中小型 Python Git 仓库。不执行被分析仓库代码，不需要安装其依赖。静态结果用于查读源码，实际运行绑定和故障根因需结合现场判断。

## 从这里开始

[使用入口](START_HERE.md) → [安装与首次使用](docs/DELIVERY_FIRST_USE.md) → [核验环境版本](docs/DELIVERY_ENVIRONMENT.md)。

使用收到的源码 ZIP，不必先克隆本项目远端。包内包含尚未提交到 Git 的产品修复；精确版本由 `DELIVERY_VERSION.json` 和逐文件哈希确定。

在解压后的 `RepoNoesis` 目录运行以下 PowerShell 命令，可核对文件完整性：

```powershell
Get-Content SHA256SUMS.txt | ForEach-Object {
    $expected, $relative = $_ -split '  ', 2
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $relative).Hash -ne $expected) {
        throw "文件哈希不符：$relative"
    }
}
```

## 目录

| 目录或文件 | 用途 |
| --- | --- |
| `backend/app` | FastAPI、索引、检索、RAG/Agent、源码探索和存储 |
| `backend/tests` | 产品相关自动化测试 |
| `frontend/src` | React 工作台和源码阅读器 |
| `frontend/dist` | 随包前端构建产物；正常开发启动仍需要 Node.js 与前端依赖 |
| `scripts`、`start_all.bat` | Windows 本地启动和运行时查找 |
| `.env.example` | 不含密钥的配置模板 |
| `backend/requirements.txt`、`frontend/package-lock.json` | 后端依赖声明和前端依赖锁文件 |
| `docs` | 部署、操作与第三方来源说明 |

## 开发检查

完成产品依赖准备后，在包根目录可运行启动脚本测试：

```powershell
python -B -m unittest discover -s backend/tests -p test_windows_startup_scripts.py -v
```

完整后端测试在 `backend` 中执行 `python -B -m unittest discover tests`；前端测试和构建在 `frontend` 中执行 `npm test`、`npm run build`。测试存在不表示所有环境均已通过，当前交付核验范围见环境记录。

包内不含真实 `.env`、API 密钥、数据库、模型权重、依赖安装目录或历史实验。保留的第三方片段及相关说明见 [分发事项](docs/DELIVERY_DISTRIBUTION_REVIEW.md)。收到源码包不代表取得未明确授予的对外再分发许可。

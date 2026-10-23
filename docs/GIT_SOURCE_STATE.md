# Git源码与本地成果的边界

此开发分支保存当前产品源码、配套离线测试、配置示例、启动脚本与使用/维护说明。已有可选实验路径若被产品引用，其模块一并保留，默认开关不作调整。

`output/`、`backend/output/`中的运行、审计、备份、源码交付包及录像留在本机；研究/参赛草稿与历史模型响应回放资料也不入库。相关本地文件没有删除。源码树不依赖这些目录才能正常启动。包含历史原始响应的回放夹具及其专用测试不随Git提供，其余产品离线测试保留。

通过Git取得源码后，按README、DELIVERY_FIRST_USE说明准备依赖并运行`npm run build`。`frontend/dist`为本机构建结果，不入库。README/START_HERE中关于DELIVERY_VERSION.json、SHA256SUMS.txt及随包构建的说明针对已有独立源码ZIP；它们不是Git检出的必备文件，Git版本使用提交身份核对。本次整理不重新生成或替换任何既有交付包。

必要Click静态夹具来自公开的`pallets/click`固定revision，供解析/范围/引用回归使用，不执行作者代码。main_method_click与main_method_regression_click仅保存验证绑定/AST范围所需的选定文件；source_read_click保存选定片段；auxiliary_source_cases保存对应测试条件。来源/hash信息保留在夹具中，[原始Click许可](licenses/Click-LICENSE.txt)一并保留。没有加入完整第三方仓库、权重、依赖或私人数据库。

首次导入仍沿用现有BGE流程；本次Git整理只做隔离源码检查，不证明干净机器安装、真实模型质量或学习提效。项目自身许可选择和分发边界继续以[已有说明](DELIVERY_DISTRIBUTION_REVIEW.md)为准，本次不变更许可或仓库可见性。

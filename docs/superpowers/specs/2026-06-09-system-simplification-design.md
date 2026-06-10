# System Simplification Design

## Verdict

方案 A 是第一轮整理的正确边界：先修复会阻断静态验证的代码错误，建立研究内容、数据、结果的证据账本，再更新 README 和忽略规则。第一轮不移动、不删除大型数据、模型权重或历史结果。

## Scope

本轮只处理 `/home/data/xzy/system`。父目录 `/home/data/xzy` 是 Git 顶层，但 `system/` 当前整体未跟踪，因此所有变更都必须限制在 `system/` 内。

本轮包含：
- 修复已定位的 Python 语法错误和非法导入。
- 移除源码中的硬编码外部 API key，改为环境变量占位。
- 新增 `PROJECT_AUDIT.md`、`DATA_INVENTORY.md`、`RESULTS_LEDGER.md`。
- 修正根 README 的真实技术栈、运行边界和 claim 边界。
- 扩展 `.gitignore`，排除环境目录、大型数据、模型权重、输出、缓存和本地密钥。
- 运行最小静态验证。

本轮不包含：
- 不移动 `data/`、`dataset/`、`projects/glucose/data/` 等大型目录。
- 不删除原始数据、派生数据、模型文件或历史结果。
- 不跑训练或完整评估。
- 不改公开 API 形状，除非是修复无法导入的包初始化文件。

## Evidence Rules

整理后的文档必须区分：
- observed result：已有文件中可直接读取的指标。
- inference：基于目录结构和结果文件的判断。
- hypothesis：未验证的研究假设。
- unsupported speculation：不能写入结果结论。

血糖、营养、推荐三个方向单独记录证据。推荐系统目前只能声明模块诊断存在，不能声明完整推荐效果已验证。

## Data Boundary

大型数据保留原位。第一轮只建立清单和风险说明。重复数据、镜像数据、派生结果先标记，不执行合并或删除。

## Verification

最小验证包括：
- `python3 -m compileall -q` 覆盖已修复文件和核心源码目录。
- `node --version`、`python3 --version`、`pnpm --version` 环境事实记录。
- 清理 `__pycache__`，不保留验证生成缓存。

若依赖缺失导致运行测试不可行，文档中记录未验证项。

# system: 智能健康监测研究工程

## 结论

`system/` 是一个旧研究工程快照，对应 GitHub 仓库 `ten-love` 的本地整理对象。当前项目包含营养健康建模、文化感知推荐、血糖预测、前端演示、后端服务、大型数据和历史实验产物。第一轮整理已经把项目定位为“研究整理阶段”，不再把历史目标、局部诊断或单次结果写成最终临床或顶会结论。

## 当前结构

| 路径 | 作用 | 当前状态 |
|---|---|---|
| `projects/nutrition/` | 营养健康多任务模型 | 有本地评估、多 seed 摘要和数据扩增验证 |
| `projects/recommendation/` | 文化感知推荐系统 | 目前主要有配置、预训练模型和约束门控诊断 |
| `projects/glucose/` | 血糖预测、GluFormer、LoRA 个性化 | 有多步预测、数据清洗和微调评估产物 |
| `backend/app/` | FastAPI 风格后端服务 | 与研究实验模块混合，仍需进一步分层 |
| `frontend/` | Vue 3 + Vite 前端演示 | 使用 Element Plus、Pinia、Vue Router |
| `data/`、`dataset/` | 大型数据集和派生数据 | 不适合直接纳入 Git |
| `docs/` | 项目文档 | 新增第一轮治理文档 |

## 真实技术栈

| 区域 | 技术 |
|---|---|
| Python | 当前环境 `python3` 为 3.12.3，`python` 命令不可用 |
| 前端 | Vue 3、Vite、Element Plus、Pinia、Vue Router |
| 后端 | FastAPI 风格应用，核心入口在 `backend/app/main.py` |
| 训练与实验 | PyTorch、NumPy、Pandas、scikit-learn 等，依赖需按子项目确认 |
| 包管理 | `node` 可用，当前环境未安装 `pnpm` |

## 快速环境检查

```bash
cd /home/data/xzy/system
python3 --version
node --version
pnpm --version
```

当前已观察结果：
- `python3 --version`：Python 3.12.3
- `node --version`：v20.18.2
- `pnpm --version`：当前环境未找到 `pnpm`

## 最小静态验证

```bash
cd /home/data/xzy/system
python3 -m compileall -q \
  projects/nutrition/src \
  projects/glucose/src/real_data_collector.py \
  backend/app/modules/__init__.py \
  backend/app/modules/glucose_prediction/__init__.py \
  backend/app/modules/image_recognition/__init__.py \
  backend/app/data_integration \
  utils \
  clinical_metrics.py
```

完整训练和完整评估暂未作为第一轮整理验证目标，因为数据体量大、依赖不完全、部分结果需要重新审计拆分方式。

## 结果边界

详见 `RESULTS_LEDGER.md`。

当前可保守表述：
- Nutrition 有本地评估结果和 3 seed 摘要。
- Glucose 有多步预测训练结果，包含 t+1 到 t+6 指标。
- Recommendation 有约束门控和负采样相关诊断。

当前不能保守表述：
- 不能声明已达到临床部署标准。
- 不能声明已满足顶会或顶刊完整实验标准。
- 不能声明已完成专家盲评或真实用户研究。
- 不能把单个用户个性化记录写作群体泛化结果。

## 数据边界

详见 `DATA_INVENTORY.md`。

大型数据、模型权重、派生训练集、历史输出和本地环境目录应保持在本地，不直接提交到 GitHub。后续如需清理大数据，必须先建立 hash 或采样 hash 清单，再确认 canonical source。

## 第一轮治理文档

- `PROJECT_AUDIT.md`：项目做了什么、有哪些模块、目前问题。
- `DATA_INVENTORY.md`：数据目录、体量、重复风险和 Git 管理规则。
- `RESULTS_LEDGER.md`：已有结果、证据等级和 claim 边界。
- `docs/superpowers/specs/2026-06-09-system-simplification-design.md`：方案 A 设计边界。
- `docs/superpowers/plans/2026-06-09-system-simplification.md`：方案 A 执行计划。

## 后续建议

1. 为 `projects/nutrition/` 和 `projects/glucose/` 各固定一个最小可复现命令。
2. 为大结果生成轻量 summary JSON，避免保存逐样本预测和模型权重字符串。
3. 审计 train、validation、test split，优先确认 user-level 或 patient-level 互斥。
4. 对 `data/`、`dataset/`、`projects/glucose/data/` 做 hash 级重复清单，再决定归档或去重。
5. 在 `ten-love` GitHub 仓库中只保留源码、轻量文档、配置模板和可复现入口。

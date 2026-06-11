# system 项目审计

## 结论

`system/` 是一个旧研究工程快照，包含智能健康监测应用、营养健康模型、文化感知推荐、血糖预测、后端服务、大量数据和历史实验产物。当前治理重点已转向顶刊导向的实验可信度：数据 provenance、可复现入口、结果证据等级、claim boundary 和最小维护面。前端演示已从 active scope 中移除。

## 当前仓库状态

| 项目项 | 观察结果 |
|---|---|
| 本地路径 | `/home/data/xzy/system` |
| Git 顶层 | `/home/data/xzy` |
| Git 状态 | `system/` 当前在父仓库中整体显示为未跟踪目录 |
| 前端状态 | `frontend/` 从 active scope 移除，历史代码可由 Git 历史追溯 |
| README 风险 | 已从 active 技术栈中移除前端表述 |
| Python 命令 | `python` 不存在，`python3` 为 3.12.3 |
| OpenSpec 命令 | `openspec` 为 1.3.1 |

## 研究主线

### Nutrition

位置：`projects/nutrition/`

观察结果：
- 面向营养健康多任务建模，包含健康分、口味分、临床指标、细粒度营养特征、数据增强和消融脚本。
- `projects/nutrition/outputs/evaluation_results.json` 记录单次评估结果。
- `projects/nutrition/outputs/multi_seed_evaluation_results.json` 记录 3 个 seed 的评估摘要。

边界：
- 可以声明已有本地评估产物。
- 不能只依据旧报告中的积极表述升级为发表级结论，因为数据采集、专家标注、外部验证仍有缺口。

### Recommendation

位置：`projects/recommendation/`

观察结果：
- 该目录主要包含文化感知推荐系统说明、配置、Food101 预训练模型和 `constraint_gate_diagnosis.json`。
- 当前可直接读取的结果更接近模块诊断，不是完整推荐系统离线评估。

边界：
- 可以声明存在文化约束门控诊断。
- 不能声明 Recall@K、NDCG@K、临床遵循率等完整推荐效果已经被该目录证实。

### Glucose

位置：`projects/glucose/`

观察结果：
- 面向血糖预测、GluFormer、LoRA 个性化、多步预测、数据清洗、真实数据接入和 API 数据采集。
- `projects/glucose/outputs/enhanced_glucose_prediction/training_results_20251030_000534.json` 记录多步预测训练和评估结果。
- `projects/glucose/outputs/final_evaluation/final_evaluation_report.json` 记录文化适配数据清洗后模型结果。

边界：
- 可以声明已有本地训练和评估产物。
- 需要继续审计数据来源、拆分方式、泄漏风险和样本独立性，才可升级为可靠科研结论。

## 应用与服务代码

| 区域 | 观察结果 | 初步处理建议 |
|---|---|---|
| `frontend/` | 旧 Vue/Vite 前端演示 | 从 active scope 移除，Git 历史保留 provenance |
| `backend/app/` | FastAPI 风格后端与大量实验模块混合 | 先修语法和导入，后续分离 service 与 research |
| `main.py` | Stage2 推荐训练入口，但路径仍引用旧 `stage2` 包名 | 第一轮仅记录，后续再决定兼容层或迁移 |
| `utils/` | Stage2 数据加载和指标工具 | 保留为共享研究工具，后续拆小文件 |

## 已定位问题

| 严重级别 | 问题 | 文件 |
|---|---|---|
| P0 | Python 语法错误，阻断编译 | `projects/glucose/src/real_data_collector.py` |
| P0 | 非法 `from app..` 导入，阻断编译 | `backend/app/modules/*.py` |
| P0 | 拼接错误或非法相对导入，阻断后续维护 | `backend/app/data_integration/workflow_integration.py` |
| P0 | 源码中存在硬编码外部 API key fallback | `projects/glucose/src/real_data_collector.py` |
| P0 | 测试脚本中存在硬编码外部 API key fallback | `projects/glucose/src/test_api_connection.py` |
| P1 | 历史文档中存在前端和企业级系统过度表述 | `README.md`、`backend/app/services/论文.md` |
| P1 | 数据和模型体量过大，包含重复镜像 | `data/`、`dataset/`、`projects/glucose/data/` |
| P1 | 文档中存在过度积极的效果表述 | 多个历史报告 |

## 第一轮整理策略

1. 修复会阻断静态验证的代码错误。
2. 移除源码硬编码密钥，改为本地环境变量。
3. 建立 `DATA_INVENTORY.md` 和 `RESULTS_LEDGER.md`，避免后续误删或过度声称。
4. 更新 README、OpenSpec 和 `.gitignore`，让 GitHub 仓库只承载研究代码、轻量文档、配置模板和可复现入口。
5. 大数据、模型权重、历史输出暂不移动，先以 manifest 方式治理。

## 第二轮 active scope 调整

已采用 `prune-frontend-research-focus` 变更：
- 删除 active `frontend/` 代码。
- 删除本地源目录中的 `frontend/` 与 `apps/frontend/`。
- 使用 Git 历史和 OpenSpec 记录作为 provenance。
- 将项目叙述从产品演示调整为研究实验、数据治理、结果证据和论文 claim boundary。

## 第三轮 experiment-readiness gate

已定义 `glucose-experiment-readiness` 变更：
- 将 `projects/glucose/` 设为第一条候选论文实验主线。
- 当前状态是 gate defined, not passed。
- preliminary leakage audit 已创建，但审计未通过：`unified_cleaned_glucose.json` 因大量空时间戳和重复键被阻断。
- `public_glucose_source_aware_split_manifest.json` 已保留为历史工程 artifact：基于 `public_glucose_preprocessed.json`，使用 `source + patient_id`，80/10/10 个 train/validation/test group，不含逐行血糖值或原始 patient ID；由于 `glucose_ml_collection` provenance closure，它不能作为 manuscript canonical split。
- `bigideas_glucose_source_report.json` 与 `bigideas_source_aware_split_manifest.json` 已生成：基于 PhysioNet BigIdeas v1.0.0 本地镜像，16 个 Dexcom source files，36898 条 EGV records，16 个 subject groups，13/2/1 个 train/validation/test groups；source report 不含逐行血糖值，保留公开 PhysioNet 文件路径，split manifest 不含原始 subject ID。
- baseline 和训练入口已在 smoke mode 消费 split artifact：baseline smoke 使用 512 windows per split 的 persistence 和 LinearRegression；training smoke 使用 32 windows per split、1 epoch、LSTM only。
- full baseline parity 已运行：persistence、LinearRegression、GBM、MLPRegressor 使用同一 split artifact，同一 input/output horizon 和同一 metric set，聚合结果写入 `projects/glucose/protocols/glucose_baseline_parity_result_summary.json`。
- GluFormer candidate pilot 已运行：full split，3 epochs，聚合结果写入 `projects/glucose/protocols/glucose_candidate_rerun_result_summary.json`；它没有超过 MLPRegressor baseline。
- GluFormer 10-epoch seed-42 triage 已运行：结果是 mixed，RMSE/R2 略优于 MLPRegressor，但 MAE 仍差，不能升级 claim。
- `metric_definitions.md` 已补齐本地 MAE、RMSE、R2 和 per-horizon MAE/RMSE 定义，但不覆盖临床范围或安全性指标。
- `data_availability_audit.md` 已创建且仍 blocking：OhioT1DM 是受控访问；`glucose_ml_collection` 已关闭为 unresolved；BigIdeas 路线有公开 PhysioNet access route 和 ODC-By licence，但仍需 final leakage pass 和 full baseline parity。
- BigIdeas baseline smoke 和 LSTM training smoke 已运行，训练入口现在导出 inverse-scaled mg/dL overall/per-horizon metrics；这些仍是 smoke evidence。
- 在 BigIdeas full baseline parity、leakage audit pass、multi-seed policy 和下一轮 candidate strategy 完成前，Glucose 结果保持本地工程证据，不升级 claim。
- 该 gate 不移动数据、不删除数据、不升级 claim。

# system: 智能健康监测研究工程

## 结论

`system/` 是一个旧研究工程快照，对应 GitHub 仓库 `ten-love` 的本地整理对象。当前 active scope 已调整为营养健康建模、文化感知推荐、血糖预测、后端研究服务、数据治理、结果账本和论文证据边界。前端演示已从 active scope 中移除，后续以 Git 历史作为 provenance。

## 当前结构

| 路径 | 作用 | 当前状态 |
|---|---|---|
| `projects/nutrition/` | 营养健康多任务模型 | 有本地评估、多 seed 摘要和数据扩增验证 |
| `projects/recommendation/` | 文化感知推荐系统 | 目前主要有配置、预训练模型和约束门控诊断 |
| `projects/glucose/` | 血糖预测、GluFormer、LoRA 个性化 | 有多步预测、数据清洗和微调评估产物 |
| `backend/app/` | FastAPI 风格后端服务 | 与研究实验模块混合，仍需进一步分层 |
| `data/`、`dataset/` | 大型数据集和派生数据 | 不适合直接纳入 Git |
| `docs/` | 项目文档 | 新增第一轮治理文档 |
| `openspec/` | 结构变更记录 | 记录 active scope 和验证 gate |

## 真实技术栈

| 区域 | 技术 |
|---|---|
| Python | 当前环境 `python3` 为 3.12.3，`python` 命令不可用 |
| 后端 | FastAPI 风格应用，核心入口在 `backend/app/main.py` |
| 训练与实验 | PyTorch、NumPy、Pandas、scikit-learn 等，依赖需按子项目确认 |
| 规范管理 | OpenSpec 1.3.1，Superpowers 计划文档 |

## 快速环境检查

```bash
cd /home/data/xzy/system
python3 --version
openspec --version
```

当前已观察结果：
- `python3 --version`：Python 3.12.3
- `openspec --version`：1.3.1

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

## 当前 active experiment gate

当前优先 gate 是 `glucose-experiment-readiness`。它把 `projects/glucose/`
作为第一条候选论文实验主线，但状态是未通过。该 gate 要求先固定
canonical dataset、patient/user-level split、seed list、baseline parity、
metric definitions、leakage audit 和轻量 result summary，再决定哪些结果
能进入论文主图和主 claim。

当前新增了 `public_glucose_preprocessed.json` 的 source-aware split artifact：
`projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`。
该 artifact 使用 `source + patient_id` 作为 group key，记录 80/10/10 个
train/validation/test group，不包含逐行血糖值或原始 patient ID。它是下一轮
重跑的输入协议。baseline 和训练入口已在 smoke mode 消费该 artifact，
full baseline parity 已完成并写入轻量 summary。3-epoch GluFormer candidate
pilot 也已完成，但没有超过 MLPRegressor baseline。failure analysis 已
记录为 `projects/glucose/protocols/gluformer_failure_analysis.md`：当前
证据更支持训练预算不足和单 seed 不足，同时要求把 MLPRegressor 作为当前
强 baseline。随后 10-epoch seed-42 triage 已完成，结果是 mixed result：
RMSE/R2 略优于 MLPRegressor，但 MAE 仍差，因此仍不能升级 claim。metric
definition、leakage audit pass、多 seed policy 和数据可用性审计仍未完成，
因此尚未让 gate 通过。

关键文件：
- `openspec/changes/glucose-experiment-readiness/`
- `projects/glucose/protocols/experiment_readiness_gate.md`
- `projects/glucose/protocols/baseline_parity_table.md`
- `projects/glucose/protocols/glucose_result_summary_schema.md`
- `projects/glucose/protocols/glucose_baseline_parity_result_summary.json`
- `projects/glucose/protocols/glucose_candidate_rerun_budget.md`
- `projects/glucose/protocols/glucose_candidate_rerun_result_summary.json`
- `projects/glucose/protocols/glucose_candidate_10epoch_triage_result_summary.json`
- `projects/glucose/protocols/gluformer_failure_analysis.md`
- `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`
- `docs/superpowers/specs/2026-06-10-glucose-experiment-readiness-design.md`
- `docs/superpowers/plans/2026-06-10-glucose-experiment-readiness.md`

## 数据边界

详见 `DATA_INVENTORY.md`。

大型数据、模型权重、派生训练集、历史输出和本地环境目录应保持在本地，不直接提交到 GitHub。后续如需清理大数据，必须先建立 hash 或采样 hash 清单，再确认 canonical source。

## 第一轮治理文档

- `PROJECT_AUDIT.md`：项目做了什么、有哪些模块、目前问题。
- `DATA_INVENTORY.md`：数据目录、体量、重复风险和 Git 管理规则。
- `RESULTS_LEDGER.md`：已有结果、证据等级和 claim 边界。
- `docs/superpowers/specs/2026-06-09-system-simplification-design.md`：方案 A 设计边界。
- `docs/superpowers/plans/2026-06-09-system-simplification.md`：方案 A 执行计划。
- `docs/superpowers/specs/2026-06-10-prune-frontend-research-focus-design.md`：前端移出 active scope 的设计边界。
- `docs/superpowers/plans/2026-06-10-prune-frontend-research-focus.md`：前端移出 active scope 的执行计划。
- `openspec/changes/prune-frontend-research-focus/`：OpenSpec 变更记录。
- `docs/superpowers/specs/2026-06-10-glucose-experiment-readiness-design.md`：Glucose 实验 readiness gate 设计边界。
- `docs/superpowers/plans/2026-06-10-glucose-experiment-readiness.md`：Glucose 实验 readiness gate 执行计划。
- `openspec/changes/glucose-experiment-readiness/`：Glucose 实验 readiness OpenSpec 变更记录。

## 后续建议

1. 按 `gluformer_failure_analysis.md` 跑 30 epoch、至少 3 seeds 的 GluFormer 正式比较。
2. 补齐 metric definitions、leakage audit 和数据 source/licence/access route 审计。
3. 若 30 epoch 多 seed 仍是 mixed result，则把 MLPRegressor 明确写成当前强 baseline。
4. 后续再为 `projects/nutrition/` 固定最小可复现命令。
5. 对 `data/`、`dataset/`、`projects/glucose/data/` 做 hash 级重复清单，再决定归档或去重。

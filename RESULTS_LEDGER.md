# 结果证据账本

## 结论

当前结果证据显示项目已经跑过多个本地实验，但不同子方向的证据强度不一致。Nutrition 有 3 seed 摘要；Glucose 有多步预测和文化适配模型结果；Recommendation 目前主要是模块诊断。任何对外 README 或论文材料都应按证据等级分别表述。

## 证据等级

| 等级 | 含义 |
|---|---|
| A | 有明确数据拆分、多个随机种子、指标摘要和可复现命令 |
| B | 有本地训练或评估产物，但缺少完整复现实验链 |
| C | 有模块诊断或 smoke test，不足以支持完整效果结论 |
| D | 仅有计划、目标或历史叙述 |

## Nutrition 结果

| 证据 | 文件 | 观察指标 | 等级 | 边界 |
|---|---|---|---|---|
| 单次集成模型评估 | `projects/nutrition/outputs/evaluation_results.json` | test_samples 7119；taste MAE 0.01376、R2 0.77410；health MAE 0.01490、R2 0.77257 | B | 有指标和测试样本数，但需确认数据拆分和标签来源 |
| 多 seed 评估 | `projects/nutrition/outputs/multi_seed_evaluation_results.json` | 3 runs；taste MAE mean 0.01371；health MAE mean 0.01483 | B | 只有 3 个 seed，未见外部测试集 |
| 改进验证报告 | `projects/nutrition/VERIFICATION_RESULTS.md` | 35,596 样本，8 个模块可导入，36 维特征 | C | 报告明确写了性能提升仍待实际训练验证 |
| 数据充足性报告 | `projects/nutrition/COMPREHENSIVE_EVALUATION_REPORT.md` | 总训练样本 2994；高健康价值样本和专家标注不足 | C | 支持数据缺口判断，不支持发表级效果结论 |

可写结论：
- 已有本地营养健康多任务模型评估和 3 seed 摘要。
- 数据扩增和模块实现已通过基础检查。

不可写结论：
- 不能写作已完成专家验证。
- 不能写作已满足顶会或临床部署证据链。

## Glucose 结果

| 证据 | 文件 | 观察指标 | 等级 | 边界 |
|---|---|---|---|---|
| 多步血糖预测训练 | `projects/glucose/outputs/enhanced_glucose_prediction/training_results_20251030_000534.json` | overall MAE 4.844152 mg/dL；RMSE 5.9735966；R2 0.855993；t+1 到 t+6 均有指标 | B | 需继续确认数据来源、时间拆分和 patient-level 独立性 |
| 个性化训练记录 | 同上 | initial_loss 0.768688；final_loss 0.295523；samples_used 983 | B | 只是一位 test_user 的本地记录，不能泛化到群体结论 |
| 文化适配最终评估 | `projects/glucose/outputs/final_evaluation/final_evaluation_report.json` | MAE 0.005194689；RMSE 0.0101306075；R2 0.997292689 | B | 指标很高，需要审计标签、拆分和泄漏风险 |
| 数据清洗报告 | `projects/glucose/data/cleaned_dataset/cleaning_report.json` | 原始 3644，保留 3005，去重 416，异常值 159，无效数据 64 | B | 支持数据清洗过程，不单独支持模型泛化 |
| 公共血糖预处理报告 | `projects/glucose/data/cleaned_dataset/public_glucose_preprocess_report.json` | sources: ohio_t1dm、glucose_ml_collection；patients_kept 100；total_samples 201600 | B | 需确认是否存在重复患者或模拟复制 |

当前 gate：
- `glucose-experiment-readiness` 已定义，但未通过。
- gate 文件：`projects/glucose/protocols/experiment_readiness_gate.md`。
- preliminary dataset manifest：`projects/glucose/protocols/canonical_dataset_manifest.md`。
- preliminary split manifest：`projects/glucose/protocols/split_manifest.md`。
- source-aware split artifact：`projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`，基于 `public_glucose_preprocessed.json`，80/10/10 个 train/validation/test group，不含逐行血糖值或原始 patient ID。
- preliminary leakage audit：`projects/glucose/protocols/leakage_audit.md`，审计未通过，`unified_cleaned_glucose.json` 因大量空时间戳和重复键被阻断。
- source-aware smoke baseline：`outputs/glucose_baselines_source_aware_smoke/split_manifest_baseline_report.json`，512 windows per split，persistence test MAE 11.6648、RMSE 17.7028、R2 0.4079；LinearRegression test MAE 15.3053、RMSE 19.2317、R2 0.3012。该输出被 Git 忽略，只能证明入口可运行。
- source-aware LSTM training smoke：`TRAIN/outputs/exp_20260610_154849/split_manifest_training_results.json`，32 windows per split，1 epoch，LSTM only。该输出被 Git 忽略，只能证明训练入口可运行。
- OpenSpec：`openspec/changes/glucose-experiment-readiness/`。
- 在 full baseline parity、主模型训练预算、metric definition、leakage audit pass、数据可用性审计和 result summary 完成前，Glucose 结果保持 B 级本地证据。

可写结论：
- 已有多步血糖预测本地训练结果，t+1 到 t+6 有指标。
- 已有文化适配数据清洗和微调评估产物。

不可写结论：
- 不能直接写作临床可部署。
- 不能把单个 test_user 个性化提升写作群体效果。

## Recommendation 结果

| 证据 | 文件 | 观察指标 | 等级 | 边界 |
|---|---|---|---|---|
| 约束门控诊断 | `projects/recommendation/results/constraint_gate_diagnosis.json` | constraint_gate mean 0.71487；violation rate 0.0 | C | 只是模块诊断，不是完整推荐评估 |
| 根快速验证 | `experiments/quick_validation_report.json` | DynamicConstraintGate、CulturalNegativeSampler、SoftConstraintLoss passed | C | smoke test，不支持最终效果 |
| Stage2 说明 | `README_stage2.md` | 写有 Recall@10、NDCG@10 等目标 | D | 多数是目标或计划，需要结果文件支撑 |

可写结论：
- 已有文化约束门控和负采样模块的本地诊断。

不可写结论：
- 不能写作完整推荐系统达到 Recall@10 或 NDCG@10 目标。
- 不能写作完成临床指南遵循率验证。

## 总体 claim 边界

可对外表述：
- 这是一个智能健康监测研究工程，包含营养、推荐、血糖三个子方向。
- 已有多个本地实验结果和诊断结果。
- 代码和数据仍处在研究整理阶段。

不应对外表述：
- 已达到临床部署标准。
- 已满足顶会顶刊实验标准。
- 已完成专家盲评或真实用户研究。
- 所有大模型和数据都可直接 GitHub 复现。

## 下一步证据 gate

1. 让训练和 baseline 入口消费 `public_glucose_source_aware_split_manifest.json`。
2. 将 scaler 和 feature normalization 限定为 train-only，并记录时间序列窗口边界。
3. 为大结果生成轻量摘要 JSON，避免提交逐样本预测和模型权重字符串。
4. Glucose gate 通过后，再评估是否把证据等级从 B 升为 A。
5. 后续再对 Nutrition 补泄漏审计，对 Recommendation 跑完整 Recall@K、NDCG@K、Precision@K 评估后更新 claim。

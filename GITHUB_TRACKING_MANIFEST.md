# GitHub Tracking Manifest

## 结论

不要直接运行 `git add system`。当前 `system/` 是一个包含大数据、环境目录、历史报告、实验输出和源码的旧工程快照。推荐先按本清单显式纳入轻量源码、OpenSpec 变更和治理文档，再分阶段处理历史材料。前端演示已从 active scope 中移除，不再作为后续源码批次纳入。

## 第一批建议纳入

### 项目治理文档

```bash
git add \
  system/.gitignore \
  system/.env.example \
  system/README.md \
  system/PROJECT_AUDIT.md \
  system/DATA_INVENTORY.md \
  system/RESULTS_LEDGER.md \
  system/GITHUB_TRACKING_MANIFEST.md \
  system/openspec/changes/prune-frontend-research-focus \
  system/openspec/changes/glucose-experiment-readiness \
  system/docs/superpowers/specs/2026-06-09-system-simplification-design.md \
  system/docs/superpowers/plans/2026-06-09-system-simplification.md \
  system/docs/superpowers/specs/2026-06-10-prune-frontend-research-focus-design.md \
  system/docs/superpowers/plans/2026-06-10-prune-frontend-research-focus.md \
  system/docs/superpowers/specs/2026-06-10-glucose-experiment-readiness-design.md \
  system/docs/superpowers/plans/2026-06-10-glucose-experiment-readiness.md \
  system/projects/glucose/protocols/README.md \
  system/projects/glucose/protocols/experiment_readiness_gate.md \
  system/projects/glucose/protocols/canonical_dataset_manifest.md \
  system/projects/glucose/protocols/split_manifest.md \
  system/projects/glucose/protocols/leakage_audit.md \
  system/projects/glucose/protocols/baseline_parity_table.md \
  system/projects/glucose/protocols/metric_definitions.md \
  system/projects/glucose/protocols/data_availability_audit.md \
  system/projects/glucose/protocols/glucose_ml_collection_provenance_closure.md \
  system/projects/glucose/protocols/bigideas_glucose_source_report.json \
  system/projects/glucose/protocols/bigideas_source_aware_split_manifest.json \
  system/projects/glucose/protocols/glucose_result_summary_schema.md \
  system/projects/glucose/protocols/glucose_baseline_parity_result_summary.json \
  system/projects/glucose/protocols/glucose_candidate_rerun_budget.md \
  system/projects/glucose/protocols/glucose_candidate_rerun_result_summary.json \
  system/projects/glucose/protocols/glucose_candidate_10epoch_triage_result_summary.json \
  system/projects/glucose/protocols/gluformer_failure_analysis.md \
  system/projects/glucose/protocols/public_glucose_source_aware_split_manifest.json \
  system/projects/glucose/src/analysis/source_aware_split_manifest.py \
  system/projects/glucose/src/analysis/source_aware_split_dataset.py \
  system/projects/glucose/src/analysis/bigideas_dataset_builder.py \
  system/projects/glucose/src/test_source_aware_split_manifest.py \
  system/projects/glucose/src/test_source_aware_split_dataset.py \
  system/projects/glucose/src/test_bigideas_dataset_builder.py \
  system/projects/glucose/src/test_split_manifest_baselines.py \
  system/projects/glucose/src/test_run_glucose_training_cli.py \
  system/projects/glucose/src/external_validation_and_baselines.py \
  system/projects/glucose/src/run_glucose_training.py \
  system/projects/glucose/src/enhanced_glucose_system.py \
  system/projects/glucose/src/data_processing/__init__.py
```

### 本轮修复过的源码

```bash
git add \
  system/projects/glucose/src/real_data_collector.py \
  system/projects/glucose/src/test_api_connection.py \
  system/backend/app/modules/__init__.py \
  system/backend/app/modules/glucose_prediction/__init__.py \
  system/backend/app/modules/image_recognition/__init__.py \
  system/backend/app/data_integration/workflow_integration.py
```

### 可选源码批次

这些目录可以作为后续批次纳入，但需要先确认是否要保留完整历史接口：

```bash
git add \
  system/projects/nutrition/README.md \
  system/projects/nutrition/src \
  system/projects/glucose/README.md \
  system/projects/glucose/protocols \
  system/projects/glucose/src \
  system/projects/recommendation/README.md \
  system/projects/recommendation/configs \
  system/backend/app \
  system/utils \
  system/models/*.py
```

## 暂不纳入

| 路径或模式 | 原因 |
|---|---|
| `system/data/` | 约 182G，混合原始与派生数据 |
| `system/dataset/` | 约 49G，公开数据集镜像 |
| `system/projects/*/data/` | 子项目本地数据，含大文件 |
| `system/data_nutrition/` | 约 2.4G 数据 |
| `system/XZY/`、`system/XZY2/`、`system/academic_env/`、`system/recbole310/` | 本地环境目录 |
| `system/models/cache/`、`system/models/pretrained/` | 权重和缓存 |
| `system/projects/*/models/pretrained/` | 预训练权重 |
| `system/projects/*/outputs/`、`system/**/results/` | 历史输出和大结果 |
| `system/frontend/`、`system/apps/frontend/` | 旧前端演示，已移出 active research scope |
| `*.pt`、`*.pth`、`*.ckpt`、`*.bin`、`*.onnx`、`*.pkl`、`*.h5` | 模型权重或二进制产物 |
| `*.db`、`*.sqlite`、`*.sqlite3` | 本地数据库 |
| `.env`、`configs/api_keys.yaml` | 本地密钥配置 |

## 当前 dry-run 发现

`git add -n system` 当前显示 778 个候选文件。`.gitignore` 已确认会排除：
- `system/data`
- `system/dataset`
- `system/projects/glucose/data`
- `system/XZY`
- `system/XZY2`
- `system/recbole310`
- `system/.env`
- `system/models/pretrained`
- `system/projects/recommendation/models/pretrained/vit-base-food101/pytorch_model.bin`

敏感路径扫描没有命中本地数据库、大数据目录、`.env`、环境目录或预训练权重。仍需通过显式 `git add` 避免把历史报告、旧演示应用和非当前目标文件一次性纳入。

## 后续 gate

1. 对新增 `glucose-experiment-readiness` 轻量文档先运行 `git add -n`。
2. 确认 dry-run 只包含 OpenSpec、Superpowers plan/spec 和 Glucose protocol gate 后，再提交。
3. 第二批再决定是否纳入完整 `backend/app`、`projects/*/src`。
4. 大数据去重必须走 `DATA_INVENTORY.md` 中的 hash 清单 gate。
5. Glucose 实验结果升级必须先通过 `projects/glucose/protocols/experiment_readiness_gate.md`。

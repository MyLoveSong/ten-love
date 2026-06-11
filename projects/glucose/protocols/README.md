# Glucose Protocol Index

Status: active gate index, not a passed manuscript gate.

## Verdict

This directory contains the lightweight, Git-trackable evidence chain for the
current Glucose experiment-readiness gate. It is an audit index only. Raw
glucose rows, row-level predictions, model checkpoints, and full training
outputs stay outside Git.

## Workflow Order

| Step | File | Current status | Boundary |
|---|---|---|---|
| 1 | `canonical_dataset_manifest.md` | preliminary | canonical dataset not frozen |
| 2 | `leakage_audit.md` | preliminary, blocking | `unified_cleaned_glucose.json` blocked; public preprocessed candidate still needs source/licence audit |
| 3 | `split_manifest.md` | preliminary | source-aware group split artifact exists, gate still not passed |
| 4 | `public_glucose_source_aware_split_manifest.json` | generated split artifact | no raw patient IDs or row-level glucose values |
| 5 | `baseline_parity_table.md` | full same-split baseline parity completed | local claim only |
| 6 | `glucose_baseline_parity_result_summary.json` | lightweight baseline summary | persistence, linear, GBM, MLPRegressor |
| 7 | `glucose_candidate_rerun_budget.md` | 3-epoch pilot budget executed | pilot only |
| 8 | `glucose_candidate_rerun_result_summary.json` | lightweight 3-epoch summary | GluFormer did not beat MLPRegressor |
| 9 | `gluformer_failure_analysis.md` | failure analysis plus 10-epoch triage completed | no superiority claim |
| 10 | `glucose_candidate_10epoch_triage_result_summary.json` | lightweight 10-epoch summary | mixed result versus MLPRegressor |
| 11 | `metric_definitions.md` | active local metric definition | not a clinical metric charter |
| 12 | `data_availability_audit.md` | preliminary, blocking | source/licence/access route not Nature-ready |
| 13 | `glucose_result_summary_schema.md` | active schema | local evidence only |
| 14 | `experiment_readiness_gate.md` | gate status document | gate not passed |

## Current Decision

The 10-epoch seed-42 GluFormer triage is mixed: RMSE and R2 are slightly better
than MLPRegressor, but MAE is worse. The next model step is a 30-epoch,
multi-seed rerun. Until then, MLPRegressor remains the current strongest
same-split baseline.

## Remaining Gate Blockers

- Source, licence, and access-route resolution for `glucose_ml_collection`.
- Final data availability statement after source reconciliation.
- Multi-seed policy and 30-epoch GluFormer rerun.
- Final leakage pass for the chosen canonical dataset.
- Claim-boundary decision after the above artifacts exist.

## Artifact Policy

Track only lightweight protocol files, hashes, aggregate metrics, and code
needed to reproduce the audit. Do not commit `TRAIN/outputs/`, `outputs/`,
`projects/glucose/data/`, checkpoints, row-level predictions, raw patient IDs,
or private source data.

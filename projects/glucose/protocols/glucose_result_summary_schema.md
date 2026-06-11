# Glucose Result Summary Schema

Status: active schema, not a passed manuscript result.

## Verdict

This schema defines the minimum lightweight result artifact required before any
Glucose result can be promoted beyond local observation. It is not a result
summary and does not upgrade any claim level.

## Required Fields

| Field | Requirement |
|---|---|
| `run_id` | Stable run identifier |
| `git_commit` | Commit SHA for the code used in the run |
| `canonical_dataset_manifest` | Path to the frozen dataset manifest |
| `split_manifest` | Path to the frozen split artifact |
| `dataset_sha256` | Hash recorded in the split or dataset manifest |
| `seed` | Random seed used by the run |
| `model_name` | Model or ensemble name |
| `baseline_name` | Baseline name, or `null` for the candidate model |
| `training_budget` | Epochs, max windows, early-stopping policy, and selected models |
| `input_horizon` | Input steps |
| `output_horizon` | Forecast steps |
| `normalization_scope` | Must be train-only |
| `metric_definitions` | Path to the active metric definition artifact |
| `mae` | Overall MAE with unit, defined by `metric_definitions.md` |
| `rmse` | Overall RMSE with unit, defined by `metric_definitions.md` |
| `r2` | Overall R2, defined by `metric_definitions.md` |
| `per_horizon_metrics` | Metrics for `t+1` through `t+6` when output horizon is 6 |
| `leakage_audit` | Path and pass/fail status |
| `data_availability_status` | Source, licence, and access-route status, linked to `data_availability_audit.md` |
| `claim_level` | `smoke`, `local`, `local-pilot`, `local-triage`, `dataset-level`, `external-validation`, or `clinical` |

## Current Smoke Runs

| Run | Output path | Scope | Commit status | Claim level |
|---|---|---|---|---|
| source-aware baseline smoke | `outputs/glucose_baselines_source_aware_smoke/split_manifest_baseline_report.json` | 512 windows per split, persistence and LinearRegression | ignored output, not committed | smoke |
| source-aware LSTM training smoke | `TRAIN/outputs/exp_20260610_154849/split_manifest_training_results.json` | 32 windows per split, 1 epoch, LSTM only | ignored output, not committed | smoke |
| source-aware full baseline parity | `projects/glucose/protocols/glucose_baseline_parity_result_summary.json` | full split, persistence, LinearRegression, GBM, MLPRegressor | committed lightweight summary | local |
| source-aware GluFormer candidate pilot | `projects/glucose/protocols/glucose_candidate_rerun_result_summary.json` | full split, 3 epochs, GluFormer only | committed lightweight summary | local-pilot |
| source-aware GluFormer 10-epoch triage | `projects/glucose/protocols/glucose_candidate_10epoch_triage_result_summary.json` | full split, 10 epochs, seed 42, GluFormer only | committed lightweight summary | local-triage |
| source-aware GluFormer failure analysis | `projects/glucose/protocols/gluformer_failure_analysis.md` | full split, 3-epoch pilot versus MLPRegressor | committed protocol analysis | local-pilot |
| BigIdeas-only baseline smoke | `outputs/glucose_baselines_bigideas_source_aware_smoke/split_manifest_baseline_report.json` | 512 windows per split, persistence and LinearRegression | ignored output, not committed | smoke |
| BigIdeas-only LSTM training smoke | `TRAIN/outputs/exp_20260611_194606/split_manifest_training_results.json` | 32 windows per split, 1 epoch, LSTM only, direct inverse-scaled metrics | ignored output, not committed | smoke |
| BigIdeas-only full baseline parity | `projects/glucose/protocols/glucose_bigideas_baseline_parity_result_summary.json` | full split, persistence, LinearRegression, GBM, MLPRegressor | committed lightweight summary | local |
| BigIdeas-only final leakage pass | `projects/glucose/protocols/bigideas_final_leakage_audit.md` | duplicate keys, null fields, group overlap, train-only scaling, row-level artifact boundary | committed protocol audit | local |

Rows labelled `source-aware` above use the old public-preprocessed candidate.
After `glucose_ml_collection_provenance_closure.md`, they are retained only as
historical engineering evidence. Future manuscript-facing summaries must use
the BigIdeas-only or another verified-source split.

## Definition Artifacts

| Artifact | Scope | Status |
|---|---|---|
| `metric_definitions.md` | MAE, RMSE, R2, per-horizon MAE/RMSE, unit and selection rules | active local definition |
| `data_availability_audit.md` | reused source audit, access route, redistribution boundary, required citations | preliminary, blocking |
| `glucose_ml_collection_provenance_closure.md` | old public-preprocessed source closure | active blocker closure |
| `bigideas_glucose_source_report.json` | BigIdeas source file hashes and subject counts | active draft source report |
| `bigideas_source_aware_split_manifest.json` | BigIdeas-only group-disjoint split | active draft split |
| `glucose_bigideas_baseline_parity_result_summary.json` | BigIdeas-only full baseline parity aggregate metrics | active local baseline summary |
| `bigideas_final_leakage_audit.md` | BigIdeas-only baseline-parity leakage pass | active local audit with limitations |

## Non-Requirements

This schema must not contain raw glucose values, row-level predictions, raw
patient IDs, model checkpoints, or private data.

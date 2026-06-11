# Glucose Experiment Readiness Gate

## Verdict

Status: not passed.

`projects/glucose/` is the first candidate manuscript experiment line, but existing results remain local observations until this gate is executed and audited. The gate must pass before reported Glucose metrics are used as main manuscript claims or figures.

## Current Evidence

| Evidence | Source artifact | Current level | Boundary |
|---|---|---:|---|
| Multi-step glucose prediction | `projects/glucose/outputs/enhanced_glucose_prediction/training_results_20251030_000534.json` | B | Local result; split and leakage audit required |
| Personalization record | same artifact | B | Single `test_user` style record; not group evidence |
| Cultural-adaptation evaluation | `projects/glucose/outputs/final_evaluation/final_evaluation_report.json` | B | Very high metrics require leakage audit |
| Cleaned dataset report | `projects/glucose/data/cleaned_dataset/cleaning_report.json` | B | Supports cleaning process, not model generalization |
| GluFormer protocol | `experiments/protocol/gluformer_research_protocol.md` | D to C | Protocol input; not completed evidence |

## Required Gate Artifacts

| Artifact | Required content | Status |
|---|---|---|
| canonical dataset manifest | path, source, access route, raw/derived status, sample count, patient/user count, timestamp coverage, hash strategy | preliminary manifest created: `projects/glucose/protocols/canonical_dataset_manifest.md`; canonical dataset not frozen |
| split manifest | train/validation/test policy, patient/user exclusivity, time-order rule, seed list | preliminary group split artifact created: `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`; baseline and training smoke runs consumed it |
| seed record | all random seeds and deterministic settings used for each run | missing |
| baseline parity table | same split, same input horizon, same output horizon, same metric definitions | full baseline parity completed for persistence, LinearRegression, GBM, and MLPRegressor; see `projects/glucose/protocols/baseline_parity_table.md` |
| metric definition table | MAE, RMSE, R2, per-horizon t+1 through t+6 metrics, unit definitions | missing |
| leakage audit | duplicates, overlapping windows, patient overlap, generated IDs, scaler leakage, target leakage, test reuse | preliminary audit created: `projects/glucose/protocols/leakage_audit.md`; audit does not pass |
| result summary | lightweight JSON or Markdown table, no raw data, no checkpoints, no row-level predictions | baseline summary, GluFormer pilot summary, and failure analysis created; claim remains local-pilot |
| claim boundary | local, dataset-level, external-validation, or clinical claim level | missing |

## Default Protocol

The current prior result used:

| Field | Value |
|---|---|
| input horizon | 12 |
| output horizon | 6 |
| train ratio | 0.8 |
| validation ratio | 0.1 |
| test ratio | 0.1 |
| prior epochs | 30 for multi-step prediction; 60 for final cultural-adaptation report |

The future gate should keep these values only if the canonical dataset and split audit confirm they are valid for the target manuscript claim.

## Claim Rules

Allowed before this gate passes:
- "Glucose has local multi-step prediction and cultural-adaptation evaluation artifacts."
- "Prior local Glucose outputs show promising metrics that require split and leakage audit."
- "The repository is preparing a glucose experiment-readiness gate."

Not allowed before this gate passes:
- "The model is clinically deployable."
- "The model generalizes across patients."
- "The model outperforms baselines under audited conditions."
- "The reported high cultural-adaptation metrics are manuscript-ready."

## Execution Order

1. Freeze canonical dataset manifest.
2. Freeze split manifest.
3. Audit leakage.
4. Run same-split baselines.
5. Export lightweight summary.
6. Update `RESULTS_LEDGER.md`.
7. Decide manuscript claim level.

## Data Safety

Do not commit raw glucose data, processed large datasets, model checkpoints, row-level predictions, `.env` files, API keys, or private patient-level records. Commit only manifests, summaries, protocols, and code required to reproduce the analysis.

## Current Blocking Findings

- `canonical_dataset_manifest.md` initially identified `unified_cleaned_glucose.json` as the strongest training-input candidate, but `leakage_audit.md` found 243107 null timestamps, 4529 duplicate `patient_id + timestamp` groups, and 243012 rows inside duplicate groups.
- `public_glucose_source_aware_split_manifest.json` records a deterministic group-disjoint split for `public_glucose_preprocessed.json`: 80 train groups, 10 validation groups, 10 test groups, seed 42, input horizon 12, output horizon 6.
- `run_glucose_training.py` and `external_validation_and_baselines.py` now have source-aware split-manifest modes that build windows after group assignment and use train-sequence-only scaling.
- Smoke execution has verified the split-manifest baseline path on `persistence` and `LinearRegression` with 512 windows per split, and the training path on one LSTM model for 1 epoch with 32 windows per split.
- Full baseline parity has now been run on the same split for persistence, LinearRegression, GBM, and MLPRegressor. Test MAE/RMSE/R2 are recorded in `glucose_baseline_parity_result_summary.json`.
- A 3-epoch full-split GluFormer pilot run has now been executed under `glucose_candidate_rerun_budget.md`, with aggregate metrics recorded in `glucose_candidate_rerun_result_summary.json`.
- The GluFormer pilot improves over persistence, LinearRegression, and GBM, but does not outperform MLPRegressor under the same split. Therefore, no superiority claim is allowed.
- `gluformer_failure_analysis.md` records the current cause analysis: validation loss and MAE were still improving at epoch 3, best epoch was the final epoch, learning-rate warmup reached 0.001 only at epoch 3, feature engineering was disabled in the split-manifest path, and the result is single-seed only.
- The gate remains not passed until a seed policy, metric definitions for final comparison, source/licence audit, leakage audit, and a stronger candidate strategy all pass.

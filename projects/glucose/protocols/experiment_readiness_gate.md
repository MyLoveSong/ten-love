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
| canonical dataset manifest | path, source, access route, raw/derived status, sample count, patient/user count, timestamp coverage, hash strategy | preliminary manifest created: `projects/glucose/protocols/canonical_dataset_manifest.md`; BigIdeas-only is the current verified-source draft candidate, not yet frozen |
| split manifest | train/validation/test policy, patient/user exclusivity, time-order rule, seed list | preliminary BigIdeas-only group split artifact created: `projects/glucose/protocols/bigideas_source_aware_split_manifest.json`; smoke baseline and training runs consumed it |
| seed record | all random seeds and deterministic settings used for each run | BigIdeas GluFormer multi-seed summary records seeds 42, 123, and 456 plus source-output hashes; final claim policy still pending |
| baseline parity table | same split, same input horizon, same output horizon, same metric definitions | full baseline parity completed for persistence, LinearRegression, GBM, and MLPRegressor on the old public-preprocessed candidate and on the BigIdeas-only split; see `projects/glucose/protocols/baseline_parity_table.md` and `projects/glucose/protocols/glucose_bigideas_baseline_parity_result_summary.json` |
| metric definition table | MAE, RMSE, R2, per-horizon t+1 through t+6 metrics, unit definitions | active local definition created: `projects/glucose/protocols/metric_definitions.md`; training and baseline split paths now emit inverse-scaled metrics directly |
| data availability audit | reused data sources, licence or DUA terms, access route, redistribution boundary, citation actions | preliminary blocking audit created: `projects/glucose/protocols/data_availability_audit.md`; old public candidate rejected, BigIdeas route verified but not final |
| leakage audit | duplicates, overlapping windows, patient overlap, generated IDs, scaler leakage, target leakage, test reuse | preliminary audit created: `projects/glucose/protocols/leakage_audit.md`; BigIdeas final leakage pass created: `projects/glucose/protocols/bigideas_final_leakage_audit.md`; gate still not passed for manuscript claim |
| result summary | lightweight JSON or Markdown table, no raw data, no checkpoints, no row-level predictions | old-candidate summaries, BigIdeas baseline summary, and BigIdeas GluFormer multi-seed summary created; claim remains local |
| claim boundary | local, dataset-level, external-validation, or clinical claim level | partial: current summaries are local, local-pilot, or local-triage; final manuscript claim level missing |

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
- `run_glucose_training.py` now records explicit split-manifest seed settings for Python, NumPy, Torch, CUDA, CuBLAS, and cuDNN.
- A 10-epoch seed-42 GluFormer triage is recorded in `glucose_candidate_10epoch_triage_result_summary.json`. It is mixed versus MLPRegressor: RMSE and R2 improve slightly, but MAE remains worse.
- `metric_definitions.md` now defines local MAE, RMSE, R2, and per-horizon MAE/RMSE rules for source-aware comparisons. It does not define clinical-range or safety metrics.
- `data_availability_audit.md` records that OhioT1DM is controlled access and that `glucose_ml_collection` is unresolved against an authoritative release, commit, and source licence chain.
- `glucose_ml_collection_provenance_closure.md` closes the old `public_glucose_preprocessed.json` candidate as unresolved for manuscript canonical use because the local Glucose-ML processing path generates example records and does not prove upstream row provenance.
- `bigideas_glucose_source_report.json` and `bigideas_source_aware_split_manifest.json` now define the BigIdeas-only draft candidate: 36898 Dexcom EGV records from 16 source files, 16 subject groups, 13/2/1 train/validation/test groups, and 29783/4696/2147 train/validation/test windows.
- BigIdeas smoke baseline execution consumed the new split with persistence and LinearRegression at 512 windows per split and wrote `outputs/glucose_baselines_bigideas_source_aware_smoke/split_manifest_baseline_report.json`.
- BigIdeas training smoke execution consumed the new split with LSTM, 1 epoch, 32 windows per split, seed 42, and wrote `TRAIN/outputs/exp_20260611_194606/split_manifest_training_results.json`; this report includes inverse-scaled mg/dL overall and per-horizon metrics.
- BigIdeas full baseline parity has now been run on the same BigIdeas split for persistence, LinearRegression, GBM, and MLPRegressor. Test metrics are recorded in `glucose_bigideas_baseline_parity_result_summary.json`; MLPRegressor is the strongest BigIdeas-only baseline by aggregate test MAE, RMSE, and R2.
- `bigideas_final_leakage_audit.md` records a pass-with-limitations for BigIdeas baseline parity: no duplicate `source + patient_id + timestamp` groups, no null source/patient/timestamp/glucose fields, no group hash crossing partitions, and train-sequence-only scaling in the baseline path.
- BigIdeas GluFormer 30-epoch multi-seed comparison has now been run for seeds 42, 123, and 456. `glucose_bigideas_gluformer_30epoch_multiseed_result_summary.json` records mean test MAE 5.3346 mg/dL, RMSE 8.3588 mg/dL, and R2 0.7852. This is mixed versus MLPRegressor because RMSE/R2 improve but MAE is worse.
- The gate remains not passed until final claim-specific Data Availability wording, claim-boundary review, and any required external validation pass.

# Glucose Split Manifest

Status: preliminary artifact created, blocked-unverified.

## Verdict

The first verified-source draft train, validation, and test group split
artifact has been generated for BigIdeas-only
`bigideas_glucose_records.json`.

This manifest records the required split policy and the split-related risks
observed from current metadata and code. The generated artifact does not make
any Glucose result manuscript-ready. Baseline and training entrypoints have
consumed the old public-preprocessed artifact in smoke mode and later full
same-split runs. That old artifact is now historical engineering evidence only
because `glucose_ml_collection` provenance is closed as unresolved. The overall
gate still remains blocked by BigIdeas multi-seed model evidence and final
claim-boundary review. BigIdeas full baseline parity and the baseline-specific
final leakage pass are now recorded.

The first leakage audit is recorded at
`projects/glucose/protocols/leakage_audit.md`. It blocks freezing
`unified_cleaned_glucose.json`. `glucose_ml_collection_provenance_closure.md`
then rejects `public_glucose_preprocessed.json` for manuscript canonical use.

Generated split artifact:
`projects/glucose/protocols/bigideas_source_aware_split_manifest.json`.

Historical split artifact:
`projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`.

| Field | Value |
|---|---|
| dataset | `projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json` |
| dataset SHA-256 | `00ba30752fa04748aa99b7dc1997102b4946d9525379fb86df56497be3a899e8` |
| split type | group-disjoint |
| group key | `source + patient_id` |
| seed | 42 |
| input horizon | 12 |
| output horizon | 6 |
| train groups | 13 |
| validation groups | 2 |
| test groups | 1 |
| train records | 30004 |
| validation records | 4730 |
| test records | 2164 |
| train windows | 29783 |
| validation windows | 4696 |
| test windows | 2147 |
| privacy boundary | no row-level glucose values or raw patient IDs |

## Entry Point Smoke Evidence

| Entry point | Command scope | Observed status | Claim level |
|---|---|---|---|
| `external_validation_and_baselines.py` | `--models persistence,linear --max-windows-per-split 512` | wrote `outputs/glucose_baselines_source_aware_smoke/split_manifest_baseline_report.json` | smoke only |
| `run_glucose_training.py` | `--models lstm --epochs 1 --max_windows_per_split 32` | wrote `TRAIN/outputs/exp_20260610_154849/split_manifest_training_results.json` | smoke only |
| `external_validation_and_baselines.py` on BigIdeas-only | `--models persistence,linear --max-windows-per-split 512` | wrote `outputs/glucose_baselines_bigideas_source_aware_smoke/split_manifest_baseline_report.json` | smoke only |
| `run_glucose_training.py` on BigIdeas-only | `--models lstm --epochs 1 --max_windows_per_split 32` | wrote `TRAIN/outputs/exp_20260611_194606/split_manifest_training_results.json`; includes inverse-scaled mg/dL metrics | smoke only |
| `external_validation_and_baselines.py` on BigIdeas-only | `--models persistence,linear,gbm,mlp` on full split | wrote `outputs/glucose_baselines_bigideas_source_aware_full/split_manifest_baseline_report.json`; lightweight summary committed as `glucose_bigideas_baseline_parity_result_summary.json` | local baseline only |

## Full Same-Split Evidence

| Run | Summary | Claim level |
|---|---|---|
| baseline parity | `glucose_baseline_parity_result_summary.json` | local |
| GluFormer 3-epoch pilot | `glucose_candidate_rerun_result_summary.json` | local-pilot |
| GluFormer 10-epoch triage | `glucose_candidate_10epoch_triage_result_summary.json` | local-triage |
| BigIdeas baseline parity | `glucose_bigideas_baseline_parity_result_summary.json` | local |
| BigIdeas final leakage pass | `bigideas_final_leakage_audit.md` | local baseline pass with limitations |
| BigIdeas GluFormer 30-epoch multi-seed | `glucose_bigideas_gluformer_30epoch_multiseed_result_summary.json` | local mixed result |

These runs used the old public-preprocessed candidate. After provenance
closure, they are not comparable to future BigIdeas-only reruns.

The BigIdeas rows use the BigIdeas-only split artifact and are the active
verified-source baseline evidence for the next candidate comparison.

## Default Horizon And Ratio Targets

| Field | Current target | Evidence |
|---|---|---|
| input horizon | 12 steps | prior result config and Glucose README command use `--in_len 12` |
| output horizon | 6 steps | prior result config and Glucose README command use `--out_len 6` |
| sampling interval | approximately 5 minutes for BigIdeas Dexcom EGV records | observed Dexcom timestamps and PhysioNet dataset description |
| train ratio | 0.8 | prior enhanced prediction config |
| validation ratio | 0.1 for prior enhanced prediction; 0.2 in `EnhancedGlucosePredictionSystem` | conflicting active code paths |
| test ratio | 0.1 for prior enhanced prediction | not present in `EnhancedGlucosePredictionSystem.train_complete_system` path |
| seed list | 42, 123, 456 | tied to the BigIdeas GluFormer 30-epoch multi-seed summary; older candidate seeds 789 and 101112 remain unexecuted |

## Required Split Policy

| Requirement | Policy |
|---|---|
| patient or user independence | use disjoint groups when reliable subject identifiers exist |
| source-aware grouping | use `source + patient_id` for BigIdeas-only so subject groups stay disjoint and future multi-source candidates remain namespaced |
| temporal boundary | sort by timestamp within each patient or source-patient group before building windows |
| window boundary | no input or target window may cross train, validation, or test boundaries |
| normalization | fit scalers and feature normalization parameters on training data only |
| model selection | validation data may be used for early stopping and tuning; test data cannot be used for model selection |
| claim level | patient-disjoint split supports stronger cross-patient claims; within-patient temporal split supports personalization or forecasting claims only |

## Candidate Split Units

| Dataset candidate | Split unit | Status | Risk |
|---|---|---|---|
| `unified_cleaned_glucose.json` | `patient_id` | blocked | leakage audit found 243107 null timestamps and 4529 duplicate `patient_id + timestamp` groups |
| `public_glucose_preprocessed.json` | `source + patient_id` | rejected for manuscript canonical use | `glucose_ml_collection` provenance closure blocks source identity |
| `bigideas_glucose_records.json` | `source + patient_id` | verified-source draft split artifact, baseline parity and final leakage pass completed | only 16 subjects; validation/test group counts are small |
| `unified_cleaned_glucose.csv` | `patient_id` | blocked-unverified | must verify equivalence with JSON before use |
| BigIdeas raw mirror | subject folder ID plus source label | draft split artifact created, baseline leakage pass recorded | only 16 subjects; external validation still missing |

## Observed Code Paths

| Code path | Observed behavior | Split risk |
|---|---|---|
| `projects/glucose/src/run_glucose_training.py` | builds sliding windows from full series; `--use_patient_ids` aggregates by patient for personalization | no frozen train/val/test partition; default path ignores patient grouping |
| `projects/glucose/src/enhanced_glucose_system.py` | shuffles all windows with `np.random.permutation`, then applies `val_split=0.2` | patient independence and temporal independence are not guaranteed |
| `projects/glucose/src/enhanced_glucose_prediction_trainer.py` | uses `train_ratio=0.8`, `val_ratio=0.1`, `test_ratio=0.1` with index slices | scaler is fitted before split; patient grouping is not represented |
| `projects/glucose/src/fine_tuning_pipeline.py` | groups by `patient_id`, generates IDs when missing, and can prefix generated IDs by source | generated IDs are weak identifiers and cannot support patient-level claims |
| `projects/glucose/src/optimized_lora_trainer.py` | fits scalers before `random_split`; uses KFold with `shuffle=True, random_state=42` | scaler leakage and non-grouped folds are possible |

## Known Leakage Risks

| Risk | Current status | Blocking reason |
|---|---|---|
| scaler leakage | observed risk | at least one trainer calls `fit_transform` before train/validation/test split |
| feature-normalization leakage | observed risk | engineered feature normalization is computed before shuffle and split in one path |
| overlapping sliding windows | not audited | adjacent windows can share most input values if split after windowing |
| patient overlap | resolved for BigIdeas draft candidate only | generated split artifact uses disjoint `source + patient_id` groups |
| source-level ID collision | resolved for BigIdeas-only candidate | single source label `physionet_big_ideas` |
| generated patient IDs | observed risk | some code creates IDs from row order when missing |
| test-set reuse | not audited | no run-level selection rule or frozen test artifact exists |

## Required Future Split Artifact

The generated split artifact is a lightweight JSON manifest with these fields:

```text
run_id
canonical_dataset_sha256_or_sampling_hash
split_policy
group_key
seed
train_group_count
val_group_count
test_group_count
train_sample_count
val_sample_count
test_sample_count
timestamp_min_train
timestamp_max_train
timestamp_min_val
timestamp_max_val
timestamp_min_test
timestamp_max_test
groups with hashed IDs, partition, source, record count, window count, timestamp min and max
```

## Minimum Acceptable Split For Next Rerun

For the next Glucose rerun, use the BigIdeas-only group-disjoint split artifact
and keep the BigIdeas MLPRegressor result as the strong baseline. Do not use the
old public-preprocessed split for manuscript claims.

| Policy | Use case | Minimum condition |
|---|---|---|
| group-disjoint split | cross-patient or cross-user generalization | reliable group key exists and no group appears in more than one partition |
| within-group temporal split | personalization or within-patient forecasting | each group is split by timestamp, with a gap of at least `input_horizon + output_horizon` steps between partitions |

For approximately 5-minute BigIdeas Dexcom data with 12 input steps and 6
output steps, the minimum temporal gap is 18 steps, equal to about 90 minutes.

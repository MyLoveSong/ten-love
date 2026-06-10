# Glucose Split Manifest

Status: preliminary artifact created, blocked-unverified.

## Verdict

The first train, validation, and test group split artifact has been generated
for `public_glucose_preprocessed.json`.

This manifest records the required split policy and the split-related risks
observed from current metadata and code. The generated artifact does not make
any Glucose result manuscript-ready. Baseline and training entrypoints have
consumed it in smoke mode only; full same-split runs are still missing.

The first leakage audit is recorded at
`projects/glucose/protocols/leakage_audit.md`. It blocks freezing
`unified_cleaned_glucose.json` and identifies `public_glucose_preprocessed.json`
as the next source-aware split-audit candidate.

Generated split artifact:
`projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`.

| Field | Value |
|---|---|
| dataset | `projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json` |
| dataset SHA-256 | `c40ff621d3ff3a82e45bb69981a5df8b365407170a3e27057d232170a3cefd36` |
| split type | group-disjoint |
| group key | `source + patient_id` |
| seed | 42 |
| input horizon | 12 |
| output horizon | 6 |
| train groups | 80 |
| validation groups | 10 |
| test groups | 10 |
| train records | 161280 |
| validation records | 20160 |
| test records | 20160 |
| train windows | 159920 |
| validation windows | 19990 |
| test windows | 19990 |
| privacy boundary | no row-level glucose values or raw patient IDs |

## Entry Point Smoke Evidence

| Entry point | Command scope | Observed status | Claim level |
|---|---|---|---|
| `external_validation_and_baselines.py` | `--models persistence,linear --max-windows-per-split 512` | wrote `outputs/glucose_baselines_source_aware_smoke/split_manifest_baseline_report.json` | smoke only |
| `run_glucose_training.py` | `--models lstm --epochs 1 --max_windows_per_split 32` | wrote `TRAIN/outputs/exp_20260610_154849/split_manifest_training_results.json` | smoke only |

## Default Horizon And Ratio Targets

| Field | Current target | Evidence |
|---|---|---|
| input horizon | 12 steps | prior result config and Glucose README command use `--in_len 12` |
| output horizon | 6 steps | prior result config and Glucose README command use `--out_len 6` |
| sampling interval | 5 minutes for `public_glucose_preprocessed` | preprocess report `window_frequency_minutes=5` |
| train ratio | 0.8 | prior enhanced prediction config |
| validation ratio | 0.1 for prior enhanced prediction; 0.2 in `EnhancedGlucosePredictionSystem` | conflicting active code paths |
| test ratio | 0.1 for prior enhanced prediction | not present in `EnhancedGlucosePredictionSystem.train_complete_system` path |
| seed list | 42, 123, 456, 789, 101112 | inherited from earlier protocol; not yet tied to a frozen run |

## Required Split Policy

| Requirement | Policy |
|---|---|
| patient or user independence | use disjoint groups when reliable subject identifiers exist |
| source-aware grouping | use `source + patient_id` for `public_glucose_preprocessed` because `patient_id` repeats across sources |
| temporal boundary | sort by timestamp within each patient or source-patient group before building windows |
| window boundary | no input or target window may cross train, validation, or test boundaries |
| normalization | fit scalers and feature normalization parameters on training data only |
| model selection | validation data may be used for early stopping and tuning; test data cannot be used for model selection |
| claim level | patient-disjoint split supports stronger cross-patient claims; within-patient temporal split supports personalization or forecasting claims only |

## Candidate Split Units

| Dataset candidate | Split unit | Status | Risk |
|---|---|---|---|
| `unified_cleaned_glucose.json` | `patient_id` | blocked | leakage audit found 243107 null timestamps and 4529 duplicate `patient_id + timestamp` groups |
| `public_glucose_preprocessed.json` | `source + patient_id` | preliminary split artifact created | report shows 100 per-patient entries but only 50 unique `patient_id` values; generated artifact uses hashed source-patient groups |
| `unified_cleaned_glucose.csv` | `patient_id` | blocked-unverified | must verify equivalence with JSON before use |
| BigIdeas raw mirror | subject folder ID | blocked-unverified | raw mirror and working-copy relationship not reconciled |

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
| patient overlap | resolved for public-preprocessed candidate only | generated split artifact uses disjoint `source + patient_id` groups; other candidates remain unaudited |
| source-level ID collision | observed risk | public preprocess report repeats patient IDs across two sources |
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

For the next Glucose rerun, use the generated group-disjoint split artifact
unless a later source/licence audit rejects the public-preprocessed candidate.
If a within-group temporal split is needed for personalization, create a
separate artifact and do not reuse this one.

| Policy | Use case | Minimum condition |
|---|---|---|
| group-disjoint split | cross-patient or cross-user generalization | reliable group key exists and no group appears in more than one partition |
| within-group temporal split | personalization or within-patient forecasting | each group is split by timestamp, with a gap of at least `input_horizon + output_horizon` steps between partitions |

For 5-minute public-preprocessed data with 12 input steps and 6 output steps,
the minimum temporal gap is 18 steps, equal to 90 minutes.

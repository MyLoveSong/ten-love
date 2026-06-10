# Glucose Leakage Audit

Status: preliminary, blocking.

## Verdict

The Glucose leakage audit does not pass.

Prior Glucose metrics remain local observations. The current evidence is enough
to block claim upgrade, because the strongest derived training-input candidate
has severe timestamp and duplicate-key problems, and active training paths can
fit normalization parameters before split or split already-windowed sequences.

Do not freeze `projects/glucose/data/cleaned_dataset/unified_cleaned_glucose.json`
as the canonical manuscript dataset in its current form.

For the next split-aware rerun, prefer the source-aware audit candidate
`projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json` with
the generated artifact
`projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`.
This is a candidate split decision, not a passed gate.

## Audit Scope

This audit used lightweight aggregate checks, source-code inspection, and a
lightweight source-aware split artifact. It did not read large raw mirrors in
full, run training, move data, delete data, or create model outputs.

## Duplicate And Identifier Checks

| Dataset | Check | Observed result | Status |
|---|---|---:|---|
| `public_glucose_preprocessed.json` | duplicate `source + patient_id + timestamp` groups | 0 groups, 0 duplicate rows | passes this key-level check |
| `public_glucose_preprocessed.json` | duplicate `patient_id + timestamp` groups | 0 groups, 0 duplicate rows | passes this file-level check |
| `public_glucose_preprocessed.json` | unique source-patient groups | 100 | supports `source + patient_id` grouping |
| `public_glucose_preprocessed.json` | unique patient IDs | 50 | `patient_id` alone is not source-safe |
| `public_glucose_preprocessed.json` | unique timestamps | 201600 | observed, not a split proof |
| `public_glucose_source_aware_split_manifest.json` | source-patient group split | 80 train, 10 validation, 10 test groups | preliminary artifact created |
| `unified_cleaned_glucose.json` | duplicate `patient_id + timestamp` groups | 4529 groups, 243012 rows inside duplicate groups | blocking |
| `unified_cleaned_glucose.json` | null timestamps | 243107 of 254445 records | blocking |
| `unified_cleaned_glucose.json` | largest duplicate key group | `patient_id=unknown`, `timestamp=null`, 230940 records | blocking |

Interpretation:

- `public_glucose_preprocessed.json` is currently the cleaner source-aware
  candidate for split construction, because it keeps the `source` field and
  passes the duplicate key checks above.
- `unified_cleaned_glucose.json` cannot support a frozen CGM forecasting split
  until its missing timestamps, `unknown` patient group, and duplicate-key
  provenance are repaired or explained.
- The public preprocess report still shows repeated `patient_id` values across
  sources, so any group split must namespace IDs as `source + patient_id`.

## Code-Level Leakage Checks

| Risk | Evidence | Status |
|---|---|---|
| scaler leakage | `enhanced_glucose_prediction_trainer.py` calls `self.scaler.fit_transform(...)` on all sequences before computing train, validation, and test slices. See lines 289-308. | observed blocking risk |
| feature-normalization leakage | `enhanced_glucose_system.py` computes feature mean/std over the passed feature tensor before the train/validation split. See lines 333-340. | observed blocking risk |
| window split leakage | `enhanced_glucose_system.py` shuffles all windows, then splits by `val_split`. See lines 247-260. | observed blocking risk |
| no frozen split path | `run_glucose_training.py` loads windows, then calls `train_complete_system(...)` without a persisted split artifact. See lines 344-373. | observed blocking risk |
| default non-grouped path | `run_glucose_training.py` uses `aggregate_datasets(...)` unless `--use_patient_ids` is provided. See lines 357-361. | observed risk |
| generated weak IDs | `fine_tuning_pipeline.py` creates row-order IDs when `patient_id` is missing. See lines 236-240. | observed risk |
| LoRA utility scaler leakage | `optimized_lora_trainer.py` fits scalers before `random_split` or shuffled KFold. See lines 428-463. | observed risk in adjacent glucose module |
| target leakage | not proven safe for all engineered fields such as `rolling_mean`, `delta`, and `glucose_normalized` | blocked-unverified |
| test reuse | no run-level test selection rule or frozen test artifact exists | blocked-unverified |

## Canonical Dataset Decision

| Candidate | Decision | Reason |
|---|---|---|
| `unified_cleaned_glucose.json` | do not freeze | severe missing timestamp and duplicate-key findings; lacks source field in observed schema |
| `unified_cleaned_glucose.csv` | do not freeze | equivalent content to JSON is not audited, and JSON candidate is blocked |
| `public_glucose_preprocessed.json` | next split-audit candidate | source-aware schema, 100 source-patient groups, no duplicate source-patient-timestamp groups in this audit |
| BigIdeas raw mirror | hold for source reconciliation | raw mirror relationship across `dataset/` and `projects/glucose/data/` is not resolved |

## Requirements Before Claim Upgrade

The gate can move forward only after these artifacts exist:

1. Frozen canonical dataset selection with source, licence, access route, and
   hash strategy.
2. Training and baseline entrypoints consume the frozen train/validation/test
   split artifact using `source + patient_id` for public-preprocessed data.
3. Window construction that does not allow input or target windows to cross
   split boundaries.
4. Train-only normalization and scaler fitting.
5. Same-split baseline parity table.
6. Result summary that links each metric to dataset manifest, split manifest,
   seed, code revision, and leakage audit.

## Commands Used

```bash
jq -r '.records | sort_by(.source, .patient_id, .timestamp) | group_by([.source, .patient_id, .timestamp]) | map(select(length > 1)) | {duplicate_source_patient_timestamp_groups:length, duplicate_rows:(map(length) | add // 0)}' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json
jq -r '.records | sort_by(.patient_id, .timestamp) | group_by([.patient_id, .timestamp]) | map(select(length > 1)) | {duplicate_patient_timestamp_groups:length, duplicate_rows:(map(length) | add // 0)}' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json
jq -r '.records | sort_by(.patient_id, .timestamp) | group_by([.patient_id, .timestamp]) | map(select(length > 1)) | {duplicate_patient_timestamp_groups:length, duplicate_rows:(map(length) | add // 0)}' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/unified_cleaned_glucose.json
jq -r '.records | {record_count:length, unique_patients:(map(.patient_id)|unique|length), unique_timestamps:(map(.timestamp)|unique|length), null_patient_ids:(map(select(.patient_id == null))|length), null_timestamps:(map(select(.timestamp == null))|length)}' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/unified_cleaned_glucose.json
rg -n "fit_transform|scaler|random_split|KFold|permutation|normalize|patient_id|generated|prefix|train_ratio|val_ratio|test_ratio" /home/data/xzy/ten-love/projects/glucose/src -g '*.py'
```

## Next Minimal Step

Update the Glucose training and baseline entrypoints to consume
`public_glucose_source_aware_split_manifest.json`, build windows after group
assignment, and fit all normalization parameters on the training partition only.

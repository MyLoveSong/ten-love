# BigIdeas Final Leakage Audit

Status: pass-with-limitations-for-baseline-parity.

## Verdict

The BigIdeas-only split path passes the final leakage checks needed for the
full baseline parity summary in
`glucose_bigideas_baseline_parity_result_summary.json`.

This is a local baseline-parity pass only. It does not make a manuscript result
claim pass, because the BigIdeas-only split contains 16 subject groups and the
test partition contains 1 subject group. Candidate-model evidence, multi-seed
policy, and final claim-specific Data Availability wording are still required.

## Scope

| Field | Value |
|---|---|
| derived dataset | `projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json` |
| dataset SHA-256 | `00ba30752fa04748aa99b7dc1997102b4946d9525379fb86df56497be3a899e8` |
| source report | `projects/glucose/protocols/bigideas_glucose_source_report.json` |
| split manifest | `projects/glucose/protocols/bigideas_source_aware_split_manifest.json` |
| baseline summary | `projects/glucose/protocols/glucose_bigideas_baseline_parity_result_summary.json` |
| full local report | `outputs/glucose_baselines_bigideas_source_aware_full/split_manifest_baseline_report.json` |
| full local report SHA-256 | `9e681d58ceae03338dc56ed0734440837c16df29058bad41b02407c29009ac1b` |
| input horizon | 12 |
| output horizon | 6 |
| seed | 42 |
| metric unit | mg/dL after inverse scaling |

## Checks

| Risk | Evidence | Status |
|---|---|---|
| missing required fields | 36898 records; 0 null source, 0 null patient_id, 0 null timestamp, 0 null glucose | pass |
| duplicate keys | 0 duplicate `source + patient_id + timestamp` groups; 0 duplicate rows | pass |
| patient overlap across partitions | 0 group hashes appear in more than one partition | pass |
| split construction order | `bigideas_source_aware_split_manifest.json` partitions source-patient groups before window construction | pass |
| window boundary | baseline entrypoint consumes the split manifest and builds windows inside each partition | pass |
| scaler leakage | full baseline report records `normalization_scope=train_sequences_only` | pass |
| feature-normalization leakage | split-manifest baseline path uses glucose sequences only for the declared baselines | pass for this run |
| target leakage | no future target field is recorded in the split manifest or lightweight summary | pass for manifest artifacts |
| test-set reuse | this run is baseline parity only; no model selection claim is made from test results | pass for baseline summary |
| row-level data exposure | committed artifacts contain aggregate metrics, hashes, counts, public source paths, and hashed group IDs only | pass |

## Split Evidence

| Partition | Groups | Records | Windows |
|---|---:|---:|---:|
| train | 13 | 30004 | 29783 |
| validation | 2 | 4730 | 4696 |
| test | 1 | 2164 | 2147 |

The split is group-disjoint by `source + patient_id`. Within each group, records
are sorted by timestamp before window generation.

## Baseline Evidence

| Baseline | Test MAE | Test RMSE | Test R2 | Boundary |
|---|---:|---:|---:|---|
| persistence | 6.4117 | 10.2676 | 0.6759 | local baseline |
| LinearRegression | 5.3239 | 8.8584 | 0.7588 | local baseline |
| GBM | 5.6735 | 8.9541 | 0.7535 | local baseline |
| MLPRegressor | 5.2368 | 8.6506 | 0.7700 | local strongest baseline |

MLPRegressor is the strongest baseline by aggregate test MAE, RMSE, and R2 in
this BigIdeas-only full-split run. This does not compare GluFormer or any
candidate model.

## Data Availability Boundary

BigIdeas is a reused public PhysioNet source with an ODC Attribution License
recorded in `data_availability_audit.md`. The repository should publish source
reports, split manifests, protocol files, code, and aggregate summaries. It
should not publish row-level CGM data, derived row-level processed datasets,
model checkpoints, or row-level predictions unless a separate data-sharing plan
is finalized.

## Remaining Blockers

- Only 16 subject groups are available, and the test partition has 1 subject
  group. This limits any cross-subject generalization claim.
- No GluFormer 30-epoch multi-seed BigIdeas comparison has been run yet.
- No external validation dataset has been added.
- Final Nature-style Data Availability text still depends on the exact
  manuscript figure and claim scope.

## Commands Used

```bash
/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python projects/glucose/src/external_validation_and_baselines.py --split-dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json --split-manifest projects/glucose/protocols/bigideas_source_aware_split_manifest.json --output outputs/glucose_baselines_bigideas_source_aware_full --input-horizon 12 --output-horizon 6 --models persistence,linear,gbm,mlp
sha256sum outputs/glucose_baselines_bigideas_source_aware_full/split_manifest_baseline_report.json
jq -r '.records | sort_by(.source, .patient_id, .timestamp) | group_by([.source, .patient_id, .timestamp]) | map(select(length > 1)) | {duplicate_source_patient_timestamp_groups:length, duplicate_rows:(map(length) | add // 0)}' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json
jq -r '.records | {record_count:length, null_source:(map(select(.source == null or .source == ""))|length), null_patient_id:(map(select(.patient_id == null or .patient_id == ""))|length), null_timestamp:(map(select(.timestamp == null or .timestamp == ""))|length), null_glucose:(map(select(.glucose_mg_dl == null))|length), unique_source_patient:(map([.source,.patient_id])|unique|length)}' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json
jq -r '[.groups[] | {group_hash, partition}] | group_by(.group_hash) | map(select((map(.partition)|unique|length) > 1)) | length' projects/glucose/protocols/bigideas_source_aware_split_manifest.json
```

## Next Minimal Step

Run GluFormer on the same BigIdeas split for 30 epochs with seeds 42, 123, and
456, then compare it against MLPRegressor under the same metric definitions and
claim boundary.

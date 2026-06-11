# Glucose Candidate Rerun Budget

Status: candidate pilot budget executed, gate not passed.

## Verdict

This budget defines the first source-aware candidate-model rerun after full
baseline parity. It is a pilot rerun for the GluFormer candidate under the
frozen split artifact. It does not upgrade any manuscript claim.

After `glucose_ml_collection_provenance_closure.md`, this pilot is historical
engineering evidence only. Future candidate budgets must use
`bigideas_source_aware_split_manifest.json` or another verified-source split.

## Fixed Budget

| Field | Value |
|---|---|
| dataset | `projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json` |
| dataset SHA-256 | `c40ff621d3ff3a82e45bb69981a5df8b365407170a3e27057d232170a3cefd36` |
| split artifact | `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json` |
| group key | `source + patient_id` |
| input horizon | 12 |
| output horizon | 6 |
| train windows | 159920 |
| validation windows | 19990 |
| test windows | 19990 |
| model | `gluformer` |
| epochs | 3 |
| batch size | 128 |
| learning rate | 0.001 |
| early stopping patience | 3 |
| feature engineering | disabled in split-manifest path |
| augmentation | disabled in split-manifest path |
| monitoring/anomaly detection | disabled in split-manifest path |
| normalization | train sequences only |
| output policy | full run output under ignored `TRAIN/outputs/`; commit only lightweight summary |

## Run Command

```bash
/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python \
  projects/glucose/src/run_glucose_training.py \
  --split_dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json \
  --split_manifest projects/glucose/protocols/public_glucose_source_aware_split_manifest.json \
  --in_len 12 \
  --out_len 6 \
  --epochs 3 \
  --models gluformer
```

## Interpretation Boundary

The run can support only a historical local candidate-vs-baseline comparison
under the old source-aware split. It cannot support clinical deployment,
external validation, population-generalization claims, or BigIdeas-only
manuscript claims.

## Observed Pilot Result

Output:
`TRAIN/outputs/exp_20260611_004421/split_manifest_training_results.json`.

Lightweight summary:
`projects/glucose/protocols/glucose_candidate_rerun_result_summary.json`.

The GluFormer candidate completed 3 epochs on the full split. Its inverse-scale
test estimate is MAE 9.1689 mg/dL, RMSE 13.5598 mg/dL, and R2 0.7417. This
improves over persistence, LinearRegression, and GBM in this run, but it does
not outperform MLPRegressor, which recorded test MAE 9.1583 mg/dL, RMSE
13.4614 mg/dL, and R2 0.7454 under the same split.

Failure analysis is recorded in
`projects/glucose/protocols/gluformer_failure_analysis.md`. The pilot ended at
the requested 3 epochs with best epoch 3, validation loss and validation MAE
still decreasing, and learning-rate warmup reaching 0.001 only at epoch 3.
At the time of this 3-epoch pilot, the next candidate strategy required
explicit seed control and a longer budget before any manuscript-facing model
claim could be considered.

Follow-up status: explicit split-manifest seed control has since been added, and
a 10-epoch seed-42 triage is recorded in
`projects/glucose/protocols/glucose_candidate_10epoch_triage_result_summary.json`.
That triage is mixed versus MLPRegressor. After provenance closure, the next
serious comparison must be a 30-epoch, multi-seed rerun on the BigIdeas-only
split after BigIdeas full baseline parity and final leakage checks.

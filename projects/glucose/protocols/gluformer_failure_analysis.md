# GluFormer Failure Analysis

Status: local failure analysis plus BigIdeas 30-epoch multi-seed comparison completed, gate not passed.

## Verdict

The 3-epoch full-split GluFormer pilot does not support a superiority claim
over MLPRegressor. The strongest observed explanation is insufficient training
budget under a single seed: validation loss and validation MAE were still
improving at the final requested epoch, and the learning-rate warmup reached
0.001 only at epoch 3. This is a negative candidate result against the current
MLP baseline, not evidence that the GluFormer architecture is intrinsically
invalid.

A follow-up 10-epoch single-seed triage was run after adding explicit seed
control to the split-manifest training path. It remains a mixed result: R2 and
RMSE are slightly better than MLPRegressor, but MAE remains worse. It still
does not support a superiority claim.

After `glucose_ml_collection_provenance_closure.md`, the compared runs below
are historical engineering evidence on the old public-preprocessed candidate.
The optimization lessons still inform the next budget, but future manuscript
candidate runs must use `bigideas_source_aware_split_manifest.json` or another
verified-source split.

The BigIdeas-only 30-epoch multi-seed rerun has now been completed for seeds
42, 123, and 456. It remains mixed versus the BigIdeas MLPRegressor strong
baseline: mean RMSE and R2 improve, but mean MAE is worse.

## Compared Runs

| Field | GluFormer candidate | MLPRegressor baseline |
|---|---:|---:|
| split artifact | `public_glucose_source_aware_split_manifest.json` | `public_glucose_source_aware_split_manifest.json` |
| input horizon | 12 | 12 |
| output horizon | 6 | 6 |
| train windows | 159920 | 159920 |
| validation windows | 19990 | 19990 |
| test windows | 19990 | 19990 |
| normalization | train sequences only | train sequences only |
| seed | 42 | 42 |
| test MAE, mg/dL | 9.1689 | 9.1583 |
| test RMSE, mg/dL | 13.5598 | 13.4614 |
| test R2 | 0.7417 | 0.7454 |

Delta versus MLPRegressor:

| Metric | GluFormer minus MLPRegressor |
|---|---:|
| MAE, mg/dL | +0.010571 |
| RMSE, mg/dL | +0.098365 |
| R2 | -0.003734 |

Interpretation: the gap is small, but every aggregate test metric favors
MLPRegressor under the current same-split protocol. The candidate cannot be
reported as outperforming all baselines.

## Root-Cause Evidence

| Finding | Evidence | Interpretation |
|---|---|---|
| The run ended at the budget limit, not at convergence | total epochs = 3, best epoch = 3 | early stopping did not explain the gap |
| Validation metrics were still improving | val loss: 0.272813 -> 0.260225 -> 0.254544; val MAE: 0.353722 -> 0.347321 -> 0.342732 | the model was still learning at the final epoch |
| Learning-rate warmup consumed the full pilot | learning rate: 0.0003 -> 0.00065 -> 0.001 | only the final epoch used the target learning rate |
| Feature path was intentionally disabled | split-manifest path disables augmentation, feature engineering, monitoring, and anomaly detection | this was a fair protocol decision, but it leaves GluFormer with only the raw one-channel glucose sequence |
| Model capacity is much larger than MLP | GluFormer 159494 trainable parameters; MLP formula 10310 trainable parameters | the candidate has a larger optimization burden for a short 12-step input |
| Evidence is single-seed only | seed = 42 | no stability claim is possible |

## Non-Primary Explanations

- Early stopping is not the primary cause for this pilot, because the observed
  total epoch count equals the requested epoch count and best epoch is the last
  epoch.
- A split or scaler mismatch is not the primary explanation for the GluFormer
  versus MLP comparison, because both summaries use the same split artifact and
  train-sequence-only normalization.
- Wavelet or engineered feature modules are not part of this observed failure,
  because the executed candidate was plain `gluformer` and the split-manifest
  path disabled feature engineering.

## Decision

- Do not claim GluFormer superiority.
- Treat MLPRegressor as the current strongest same-split baseline.
- Keep the Glucose claim level at local engineering evidence until final Data
  Availability wording, claim-boundary review, and any required external
  validation are complete.
- Treat the BigIdeas MLPRegressor as the strong baseline unless a later
  candidate improves the selected metric set under the same split.

## 10-Epoch Triage Update

Lightweight summary:
`projects/glucose/protocols/glucose_candidate_10epoch_triage_result_summary.json`.

Source output:
`TRAIN/outputs/exp_20260611_124005/split_manifest_training_results.json`.

| Field | Value |
|---|---:|
| seed | 42 |
| epochs | 10 |
| best validation epoch | 8 |
| test MAE, mg/dL | 9.1781 |
| test RMSE, mg/dL | 13.4444 |
| test R2 | 0.7461 |
| MAE delta versus MLPRegressor, mg/dL | +0.0198 |
| RMSE delta versus MLPRegressor, mg/dL | -0.0170 |
| R2 delta versus MLPRegressor | +0.0006 |

Interpretation: the 10-epoch triage improves RMSE and R2 relative to MLP, but
does not improve MAE. The result is too small and mixed to justify a model
claim, especially because it is still single-seed and the leakage/data
availability gates remain blocked.

## BigIdeas 30-Epoch Multi-Seed Update

Lightweight summary:
`projects/glucose/protocols/glucose_bigideas_gluformer_30epoch_multiseed_result_summary.json`.

Source outputs:

- `TRAIN/outputs/exp_20260612_165133/split_manifest_training_results.json`
- `TRAIN/outputs/exp_20260612_165345/split_manifest_training_results.json`
- `TRAIN/outputs/exp_20260612_165515/split_manifest_training_results.json`

| Field | Value |
|---|---:|
| seeds | 42, 123, 456 |
| requested epochs | 30 |
| observed total epochs | 10, 8, 12 |
| best epochs | 7, 4, 8 |
| mean test MAE, mg/dL | 5.3346 |
| sample std test MAE, mg/dL | 0.0932 |
| mean test RMSE, mg/dL | 8.3588 |
| sample std test RMSE, mg/dL | 0.0898 |
| mean test R2 | 0.7852 |
| sample std test R2 | 0.0046 |
| MAE delta versus MLPRegressor, mg/dL | +0.0979 |
| RMSE delta versus MLPRegressor, mg/dL | -0.2918 |
| R2 delta versus MLPRegressor | +0.0152 |

Interpretation: the 30-epoch multi-seed BigIdeas run improves mean RMSE and
R2 relative to MLPRegressor, but it does not improve mean MAE. Under the active
selection rule, this does not support a GluFormer superiority claim.

## Required Next Review

Before another manuscript-facing candidate claim:

1. Decide whether the paper's main selection rule prioritizes all aggregate
   metrics or a primary metric with secondary metrics.
2. Keep MLPRegressor as the comparison baseline unless a stronger published or
   domain-standard baseline is added under the same split.
3. If GluFormer is retained, report it as mixed rather than superior.

Candidate single-seed triage command:

```bash
/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python \
  projects/glucose/src/run_glucose_training.py \
  --split_dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json \
  --split_manifest projects/glucose/protocols/bigideas_source_aware_split_manifest.json \
  --in_len 12 \
  --out_len 6 \
  --epochs 30 \
  --models gluformer \
  --seed 42
```

The old 10-epoch command has already been executed once for triage on the old
public-preprocessed candidate. The BigIdeas-only 30-epoch multi-seed command
has now also been executed and summarized.

## Artifact Boundary

Source candidate output remains ignored by Git:
`TRAIN/outputs/exp_20260611_004421/split_manifest_training_results.json`.
`TRAIN/outputs/exp_20260611_124005/split_manifest_training_results.json`.
`TRAIN/outputs/exp_20260612_165133/split_manifest_training_results.json`.
`TRAIN/outputs/exp_20260612_165345/split_manifest_training_results.json`.
`TRAIN/outputs/exp_20260612_165515/split_manifest_training_results.json`.

Committed lightweight summaries:

- `projects/glucose/protocols/glucose_candidate_rerun_result_summary.json`
- `projects/glucose/protocols/glucose_candidate_10epoch_triage_result_summary.json`
- `projects/glucose/protocols/glucose_baseline_parity_result_summary.json`
- `projects/glucose/protocols/glucose_bigideas_gluformer_30epoch_multiseed_result_summary.json`

No raw glucose rows, row-level predictions, model checkpoints, or patient IDs
are included in this document.

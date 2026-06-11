# GluFormer Failure Analysis

Status: local failure analysis completed, gate not passed.

## Verdict

The 3-epoch full-split GluFormer pilot does not support a superiority claim
over MLPRegressor. The strongest observed explanation is insufficient training
budget under a single seed: validation loss and validation MAE were still
improving at the final requested epoch, and the learning-rate warmup reached
0.001 only at epoch 3. This is a negative candidate result against the current
MLP baseline, not evidence that the GluFormer architecture is intrinsically
invalid.

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
- Keep the Glucose claim level at local B-level evidence until leakage audit,
  source/licence audit, metric definitions, and seed policy are complete.
- The next candidate run should test whether GluFormer catches or exceeds MLP
  under a budget that can actually reach convergence.

## Required Next Experiment

Before another manuscript-facing candidate claim:

1. Add explicit seed control to `run_glucose_training.py` for split-manifest
   training.
2. Run GluFormer on the same split with an extended budget, at minimum 10
   epochs for triage and 30 epochs for a serious comparison.
3. Run at least three seeds, for example 42, 123, and 456.
4. Report mean and standard deviation for MAE, RMSE, R2, and per-horizon MAE.
5. Keep MLPRegressor as the comparison baseline unless a stronger published or
   domain-standard baseline is added under the same split.

Candidate single-seed triage command:

```bash
/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python \
  projects/glucose/src/run_glucose_training.py \
  --split_dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json \
  --split_manifest projects/glucose/protocols/public_glucose_source_aware_split_manifest.json \
  --in_len 12 \
  --out_len 6 \
  --epochs 10 \
  --models gluformer
```

This command is proposed, not executed in this failure analysis.

## Artifact Boundary

Source candidate output remains ignored by Git:
`TRAIN/outputs/exp_20260611_004421/split_manifest_training_results.json`.

Committed lightweight summaries:

- `projects/glucose/protocols/glucose_candidate_rerun_result_summary.json`
- `projects/glucose/protocols/glucose_baseline_parity_result_summary.json`

No raw glucose rows, row-level predictions, model checkpoints, or patient IDs
are included in this document.

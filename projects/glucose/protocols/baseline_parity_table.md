# Glucose Baseline Parity Table

Status: full baseline parity runs completed, gate not passed.

## Verdict

Baseline parity for the four declared baselines is complete under both the
historical public-preprocessed split and the active BigIdeas-only split.

After `glucose_ml_collection_provenance_closure.md`, this table is historical
engineering evidence only. It must not be used as manuscript baseline parity
for the BigIdeas-only candidate.

The historical baseline entrypoint consumed
`projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`
on the full split for persistence, LinearRegression, GBM, and MLPRegressor.
This satisfied the baseline-parity execution requirement for these four
baselines under the old candidate, but it does not pass the overall Glucose
experiment-readiness gate.
Canonical data availability, leakage audit, seed policy, and main-model rerun
requirements remain unresolved.

The active BigIdeas-only baseline entrypoint consumed
`projects/glucose/protocols/bigideas_source_aware_split_manifest.json` on the
full split for persistence, LinearRegression, GBM, and MLPRegressor. The
strongest BigIdeas-only baseline is MLPRegressor by aggregate test MAE, RMSE,
and R2.

## Current Split Contract

| Field | Value |
|---|---|
| dataset | `projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json` |
| split artifact | `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json` |
| group key | `source + patient_id` |
| input horizon | 12 |
| output horizon | 6 |
| seed | 42 |
| normalization | train sequences only |
| train windows | 159920 |
| validation windows | 19990 |
| test windows | 19990 |

## Smoke Baseline Run

| Field | Value |
|---|---|
| command | `/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python projects/glucose/src/external_validation_and_baselines.py --split-dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json --split-manifest projects/glucose/protocols/public_glucose_source_aware_split_manifest.json --output outputs/glucose_baselines_source_aware_smoke --input-horizon 12 --output-horizon 6 --models persistence,linear --max-windows-per-split 512` |
| output | `outputs/glucose_baselines_source_aware_smoke/split_manifest_baseline_report.json` |
| artifact status | ignored by Git via `**/outputs/`; not a committed manuscript artifact |
| evaluation scope | `smoke_subset` |
| max windows per split | 512 |
| models | `persistence`, `linear` |
| normalization | train sequences only |

Smoke test metrics on the first 512 test windows:

| Baseline | Test MAE | Test RMSE | Test R2 | Claim level |
|---|---:|---:|---:|---|
| persistence | 11.6648 | 17.7028 | 0.4079 | smoke only |
| LinearRegression | 15.3053 | 19.2317 | 0.3012 | smoke only |

## Full Baseline Parity Run

| Field | Value |
|---|---|
| command | `/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python projects/glucose/src/external_validation_and_baselines.py --split-dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json --split-manifest projects/glucose/protocols/public_glucose_source_aware_split_manifest.json --output outputs/glucose_baselines_source_aware_full --input-horizon 12 --output-horizon 6 --models persistence,linear,gbm,mlp` |
| output | `outputs/glucose_baselines_source_aware_full/split_manifest_baseline_report.json` |
| output SHA-256 | `e5447e5ed76a8fd2403755c31cd3ace55d532b0dac0ecea5e70624dadb276bfb` |
| lightweight summary | `projects/glucose/protocols/glucose_baseline_parity_result_summary.json` |
| artifact status | full output ignored by Git via `**/outputs/`; lightweight summary committed |
| evaluation scope | `full_split` |
| train windows | 159920 |
| validation windows | 19990 |
| test windows | 19990 |
| normalization | train sequences only |
| GBM backend | sklearn `GradientBoostingRegressor` wrapped by `MultiOutputRegressor` |

Full-split test metrics:

| Baseline | Test MAE | Test RMSE | Test R2 | Claim level |
|---|---:|---:|---:|---|
| persistence | 12.6028 | 18.3948 | 0.5246 | local |
| LinearRegression | 11.4788 | 16.4071 | 0.6218 | local |
| GBM | 9.8045 | 14.2221 | 0.7158 | local |
| MLPRegressor | 9.1583 | 13.4614 | 0.7454 | local |

## BigIdeas Full Baseline Parity Run

| Field | Value |
|---|---|
| command | `/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python projects/glucose/src/external_validation_and_baselines.py --split-dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json --split-manifest projects/glucose/protocols/bigideas_source_aware_split_manifest.json --output outputs/glucose_baselines_bigideas_source_aware_full --input-horizon 12 --output-horizon 6 --models persistence,linear,gbm,mlp` |
| output | `outputs/glucose_baselines_bigideas_source_aware_full/split_manifest_baseline_report.json` |
| output SHA-256 | `9e681d58ceae03338dc56ed0734440837c16df29058bad41b02407c29009ac1b` |
| lightweight summary | `projects/glucose/protocols/glucose_bigideas_baseline_parity_result_summary.json` |
| final leakage audit | `projects/glucose/protocols/bigideas_final_leakage_audit.md` |
| artifact status | full output ignored by Git via `**/outputs/`; lightweight summary committed |
| evaluation scope | `full_split` |
| train windows | 29783 |
| validation windows | 4696 |
| test windows | 2147 |
| normalization | train sequences only |
| GBM backend | sklearn `GradientBoostingRegressor` wrapped by `MultiOutputRegressor` |

BigIdeas full-split test metrics:

| Baseline | Test MAE | Test RMSE | Test R2 | Claim level |
|---|---:|---:|---:|---|
| persistence | 6.4117 | 10.2676 | 0.6759 | local |
| LinearRegression | 5.3239 | 8.8584 | 0.7588 | local |
| GBM | 5.6735 | 8.9541 | 0.7535 | local |
| MLPRegressor | 5.2368 | 8.6506 | 0.7700 | local strong baseline |

## Baseline Matrix

| Baseline | Same split | Same input horizon | Same output horizon | Same metrics | Status |
|---|---|---|---|---|---|
| naive persistence | full split | yes | yes | MAE, RMSE, R2, per horizon | completed, local claim only |
| LinearRegression | full split | yes | yes | MAE, RMSE, R2, per horizon | completed, local claim only |
| GBM | full split | yes | yes | MAE, RMSE, R2, per horizon | completed, local claim only |
| MLPRegressor | full split | yes | yes | MAE, RMSE, R2, per horizon | completed, local claim only |
| Enhanced Glucose ensemble | smoke-run on LSTM subset | yes | yes | ensemble test metrics | full run required |
| GluFormer candidate | full split pilot | yes | yes | MAE, RMSE, R2, per horizon | local-pilot; did not outperform MLPRegressor; failure analysis completed |
| GluFormer 10-epoch triage | full split | yes | yes | MAE, RMSE, R2, per horizon | local-triage; mixed versus MLPRegressor |
| BigIdeas MLPRegressor strong baseline | full split | yes | yes | MAE, RMSE, R2, per horizon | local; strongest BigIdeas-only baseline so far |
| BigIdeas GluFormer 30-epoch multi-seed | full split | yes | yes | MAE, RMSE, R2, per horizon | local; mixed versus MLPRegressor, no superiority claim |

## Commands To Run In A Full Environment

```bash
python3 projects/glucose/src/external_validation_and_baselines.py \
  --split-dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json \
  --split-manifest projects/glucose/protocols/public_glucose_source_aware_split_manifest.json \
  --output outputs/glucose_baselines_source_aware \
  --input-horizon 12 \
  --output-horizon 6

python3 projects/glucose/src/run_glucose_training.py \
  --split_dataset /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json \
  --split_manifest projects/glucose/protocols/public_glucose_source_aware_split_manifest.json \
  --in_len 12 \
  --out_len 6
```

## Remaining Blockers

- A stronger predefined candidate strategy is still required after BigIdeas
  baseline parity.
- A claim-boundary decision is required because the BigIdeas GluFormer
  multi-seed result improves RMSE/R2 but not MAE versus MLPRegressor.

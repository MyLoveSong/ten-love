# Glucose Baseline Parity Table

Status: smoke-run completed, full parity not passed.

## Verdict

Baseline parity is not complete.

The baseline entrypoint can now consume
`projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`,
and a source-aware smoke baseline has been executed with a dependency-complete
local venv. The smoke run verifies the split, train-only scaler, metric export,
and JSON report path. It does not replace a full same-split baseline rerun.

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

## Baseline Matrix

| Baseline | Same split | Same input horizon | Same output horizon | Same metrics | Status |
|---|---|---|---|---|---|
| naive persistence | smoke-run on subset | yes | yes | MAE, RMSE, R2, per horizon | full run required |
| LinearRegression | smoke-run on subset | yes | yes | MAE, RMSE, R2, per horizon | full run required |
| GBM | wired to split artifact | yes | yes | MAE, RMSE, R2, per horizon | full run required |
| MLPRegressor | wired to split artifact | yes | yes | MAE, RMSE, R2, per horizon | full run required |
| Enhanced Glucose ensemble | smoke-run on LSTM subset | yes | yes | ensemble test metrics | full run required |
| GluFormer candidate | not separately audited | not yet audited | not yet audited | not yet audited | pending |

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

- Full same-split baseline reruns are still required for persistence,
  LinearRegression, GBM, and MLPRegressor.
- A full or predefined budget Enhanced Glucose rerun is still required after
  the smoke training path.
- The public-preprocessed dataset source, licence, and access route still need
  a Nature-ready data availability audit.
- Generated metrics must be exported as a lightweight result summary before
  any claim upgrade.

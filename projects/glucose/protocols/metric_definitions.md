# Glucose Metric Definitions

Status: active local definition, not a final manuscript metric charter.

## Verdict

The current source-aware Glucose gate uses regression metrics only:
MAE, RMSE, R2, and per-horizon MAE/RMSE for t+1 through t+6.

These metrics are now defined for local baseline parity and GluFormer triage.
They do not establish clinical utility, safety, calibration, hypoglycemia
detection, or treatment-decision performance.

## Metric Scope

| Field | Current value |
|---|---|
| prediction target | future CGM glucose values |
| input horizon | 12 steps |
| output horizon | 6 steps |
| sampling interval | 5 minutes for `public_glucose_preprocessed.json` |
| forecast horizons | t+1, t+2, t+3, t+4, t+5, t+6 |
| unit for reported local summaries | mg/dL after inverse scaling |
| primary comparison split | `public_glucose_source_aware_split_manifest.json` |
| primary comparison models | persistence, LinearRegression, GBM, MLPRegressor, GluFormer |

## Definitions

Let `y_i,h` be the true glucose value for sample `i` and horizon `h`, and
`yhat_i,h` be the predicted value. Current six-step runs use `h = 1..6`.

| Metric | Definition | Direction | Current implementation |
|---|---|---|---|
| MAE | `mean(abs(y - yhat))` over all test samples and horizons | lower is better | `evaluate_glucose_predictions()` in `external_validation_and_baselines.py`; tensor mean absolute error in `EnhancedGlucosePredictionSystem.evaluate_ensemble()` |
| MSE | `mean((y - yhat)^2)` over all test samples and horizons | lower is better | intermediate value for RMSE |
| RMSE | `sqrt(MSE)` over all test samples and horizons | lower is better | `sqrt(mean_squared_error(...))` or tensor equivalent |
| R2 | `1 - sum((y - yhat)^2) / sum((y - mean(y))^2)` | higher is better | local manual implementation in baseline and ensemble evaluation paths |
| per-horizon MAE | MAE computed separately for each horizon `t+h` | lower is better | `per_horizon` / `step_metrics` dictionaries |
| per-horizon RMSE | RMSE computed separately for each horizon `t+h` | lower is better | `per_horizon` / `step_metrics` dictionaries |

## Aggregation Rule

For overall MAE, MSE, RMSE, and R2, the current gate flattens the
`n_windows x output_horizon` target and prediction arrays. Each horizon
therefore contributes equally at the element level.

For per-horizon metrics, each horizon is evaluated independently over all test
windows. Current baseline summaries include per-horizon MAE/RMSE. Current
GluFormer summaries include per-horizon MAE/RMSE after converting normalized
metrics back to estimated mg/dL.

## Normalization Rule

Metrics used for manuscript-facing summaries must be reported in mg/dL after
inverse scaling. Normalized-space metrics may be retained for diagnostics, but
they must not be mixed with mg/dL metrics in model comparison tables.

For the source-aware split path, scaling parameters are fit from the training
partition only by `load_source_aware_split_windows(...)`. The baseline path then
uses `inverse_scale(...)` before reporting metrics. The training path reports
normalized metrics from `evaluate_ensemble(...)`; lightweight summaries convert
those values with the train-split scaler and label them as mg/dL estimates.

## Selection Rule

The current local selection rule is conservative:

1. A candidate model cannot be claimed to outperform the strongest baseline
   unless it improves MAE, RMSE, and R2 under the same split, horizon, and seed
   policy.
2. Single-seed improvements are local diagnostics only.
3. If MAE, RMSE, and R2 disagree, the result is mixed and cannot support a
   superiority claim.
4. Test metrics cannot be used for iterative model selection. Tuning must use
   validation metrics and a predeclared budget.

## Non-Metrics For This Gate

The following are out of scope until separately audited:

- clinical-range accuracy;
- hypoglycemia or hyperglycemia event detection;
- Clarke Error Grid, Parkes Error Grid, or treatment-zone metrics;
- calibration, uncertainty, or conformal coverage;
- patient-safety endpoints;
- real-world intervention outcomes.

Any future clinical-range metric must declare thresholds, units, missing-data
policy, event-label construction, and whether the metric is computed on raw
CGM values or model outputs.

## Source Basis

| Source | Use in this document |
|---|---|
| scikit-learn documentation, `mean_absolute_error`, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html | MAE convention and multi-output behavior |
| scikit-learn documentation, `mean_squared_error`, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html | MSE convention used to derive RMSE |
| scikit-learn documentation, `r2_score`, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html | R2 convention, including negative scores and non-constant target assumption |
| local code: `projects/glucose/src/external_validation_and_baselines.py` | current baseline metric implementation |
| local code: `projects/glucose/src/enhanced_glucose_system.py` | current GluFormer ensemble metric implementation |

## Remaining Risks

| Risk | Status |
|---|---|
| R2 implementation differs from scikit-learn `force_finite=True` behavior when targets are constant | low risk for current summaries, but should be unit-tested before a final manuscript table |
| GluFormer summaries convert normalized metrics to mg/dL estimates after the fact | acceptable for local triage, but final runs should emit inverse-scaled metrics directly from the training entrypoint |
| no confidence intervals or multi-seed mean/std yet | blocking for manuscript-level comparison |
| clinical glucose metrics are not defined | blocks clinical or safety claims |

## Next Minimal Step

Update `run_glucose_training.py` so final multi-seed runs export inverse-scaled
overall and per-horizon metrics directly, then compute mean and standard
deviation across seeds 42, 123, and 456 under this definition table.

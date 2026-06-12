## Context

The current result ledger shows `projects/glucose/` has the strongest local result evidence in the repository. Observed artifacts include:

- `projects/glucose/outputs/enhanced_glucose_prediction/training_results_20251030_000534.json`: multi-step glucose prediction with 12-step input, 6-step output, 80/10/10 split, 30 epochs, and local MAE/RMSE/R2 metrics.
- `projects/glucose/outputs/final_evaluation/final_evaluation_report.json`: cultural-adaptation evaluation with very high reported metrics.
- `projects/glucose/data/cleaned_dataset/cleaning_report.json`: cleaned cultural-adaptation dataset report.
- `experiments/protocol/gluformer_research_protocol.md`: earlier protocol text with population, baselines, seeds, and metrics.

These are useful starting evidence, but they do not yet satisfy manuscript-level reproducibility. The gate therefore treats existing outputs as prior local observations, not as final claims.

## Goals / Non-Goals

**Goals:**
- Define the exact conditions required before Glucose becomes the primary manuscript experiment line.
- Require canonical dataset and split manifests before any claim upgrade.
- Require patient-level or user-level independence checks where patient identifiers exist.
- Require time-order and scaler-fit leakage checks for CGM prediction.
- Require same-split baseline parity and per-horizon metrics.
- Require a lightweight summary artifact for manuscript tables and figures.

**Non-Goals:**
- No claim-upgrading full training run in this change.
- No deletion, movement, or de-duplication of large data.
- No acceptance of current high metrics as publication-ready.
- No rewrite of Glucose model code.
- No clinical-deployment claim.

## Readiness Gate

The Glucose line is not ready for manuscript-level claims until all of the following gates pass:

1. **Canonical data gate:** a manifest identifies the exact dataset path, source, access route, sample counts, patient or user counts, timestamp coverage, raw versus derived status, and hash or sampling-hash method.
2. **Split gate:** train, validation, and test splits are frozen with seed list, patient or user exclusivity where possible, and time-order policy for CGM windows.
3. **Baseline gate:** all baselines use the same canonical split, input horizon, prediction horizon, training budget, and metric definitions.
4. **Metric gate:** reports include MAE, RMSE, R2, and per-horizon t+1 through t+6 metrics. Clinical range metrics should be reported only if labels and thresholds are audited.
5. **Leakage gate:** audit covers duplicate rows, overlapping windows, patient overlap, generated patient IDs, normalization fitted outside training data, target leakage, and reuse of test data for model or hyperparameter selection.
6. **Data availability gate:** reused data sources, licence or DUA terms, access route, redistribution boundary, and dataset citations are audited before manuscript use.
7. **Result-summary gate:** manuscript-facing results are exported as lightweight JSON or Markdown tables, excluding raw data, model weights, and row-level predictions.
8. **Claim-boundary gate:** the final claim states local algorithmic performance only unless external validation, clinical review, and data-use permissions support stronger claims.

## Decisions

1. Use `glucose-experiment-readiness` as a gate definition with smoke execution records, not as a completed manuscript experiment.
2. Mark the current gate status as not passed until final data availability statement, claim-boundary review, and any required external validation are resolved.
3. Reuse prior protocol material only as input. It is not treated as evidence of completed validation.
4. Prefer conservative evidence promotion: B-level local results can become A-level only after split, seed, baseline, metric, and leakage artifacts are present.
5. Keep large data and outputs local. Git should receive only protocol files, summaries, and reproducibility metadata.

## Risks / Trade-offs

- Existing high metrics may drop after stricter split and leakage audit. That is acceptable and should be recorded as a finding.
- Patient identifiers may be missing or generated from row order in some sources. Those cases must be classified as weaker evidence.
- Public glucose datasets may have repeated subjects or mirrored files across `dataset/` and `projects/glucose/data/`. The canonical data gate must resolve this before training.
- Reused human-participant CGM data may have controlled-access or third-party redistribution limits. The data availability gate must resolve these before manuscript submission.
- The source-aware split artifact for `public_glucose_preprocessed.json` is now historical engineering evidence only because `glucose_ml_collection` provenance is unresolved.
- The BigIdeas-only split artifact has a clearer public source route, but only 16 subject groups and 1 test subject group. Its full baseline parity, final baseline leakage pass, and GluFormer multi-seed comparison support local evidence only, and cannot support broad generalization without additional data or external validation.
- The BigIdeas GluFormer comparison is mixed versus MLPRegressor: RMSE/R2 improve, but MAE is worse. A claim-boundary review must decide whether this can be reported as a limited result or should remain a negative/mixed finding.
- A strict gate delays manuscript writing, but reduces rejection risk from unsupported claims and unclear reproducibility.

## Verification

This change is verified by:

- `openspec validate glucose-experiment-readiness --type change --strict --no-interactive`
- `python3 -m compileall -q projects/glucose/src/run_glucose_training.py projects/glucose/src/enhanced_glucose_prediction_trainer.py projects/glucose/src/external_validation_and_baselines.py`
- `/home/data/xzy/MyProject-Guochuang/gluformer_plus/.venv/bin/python -m unittest test_bigideas_dataset_builder.py test_source_aware_split_manifest.py test_source_aware_split_dataset.py test_split_manifest_baselines.py test_run_glucose_training_cli.py` from `projects/glucose/src`
- `git diff --check`
- `rg -n "glucose-experiment-readiness|experiment_readiness_gate" README.md RESULTS_LEDGER.md PROJECT_AUDIT.md GITHUB_TRACKING_MANIFEST.md openspec docs projects/glucose/protocols`

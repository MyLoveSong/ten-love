# Glucose Experiment Readiness Design

## Verdict

`projects/glucose/` should become the first candidate manuscript experiment line, but only through a gate-first process. Current Glucose outputs are useful local observations; they are not yet publication-grade evidence until data provenance, split independence, baseline parity, metrics, and leakage risks are audited.

## Scope

This design covers:
- `projects/glucose/` model and experiment readiness.
- Existing local results listed in `RESULTS_LEDGER.md`.
- Canonical data, split, baseline, metric, leakage, and claim-boundary artifacts.
- Lightweight metadata suitable for GitHub tracking.

This design does not cover:
- Running claim-upgrading full training in this change.
- Moving or deleting large data.
- Publishing raw data, checkpoints, or row-level predictions.
- Making a clinical deployment claim.
- Rewriting the Glucose model architecture.

## Observed Evidence

The current local evidence is B-level:
- Multi-step glucose prediction result exists with 12 input steps, 6 output steps, 80/10/10 split, 30 epochs, MAE/RMSE/R2, and per-horizon metrics.
- Cultural-adaptation final evaluation exists with very high metrics, which makes leakage audit mandatory before claim promotion.
- Cleaned dataset report exists for 3005 retained samples after duplicate, outlier, and invalid-data filtering.
- Earlier GluFormer protocol text lists target population, candidate baselines, seed list, and primary metrics, but it remains protocol input rather than completed evidence.

## Gate Design

The gate has seven required artifacts:

1. Canonical dataset manifest.
2. Frozen split manifest.
3. Seed and deterministic setting record.
4. Baseline parity table.
5. Metric definition table.
6. Leakage audit.
7. Lightweight result summary and claim-boundary statement.

The gate passes only when these artifacts exist and no blocking leakage or provenance issue remains unresolved.

Current smoke execution verifies that the source-aware split artifact can be
consumed by baseline and training entrypoints. It does not satisfy baseline
parity, main-model evidence, or manuscript-level result summary requirements.

## Claim Policy

Before the gate passes, permitted claims are limited to:
- local Glucose training and evaluation artifacts exist;
- multi-step prediction has prior local metrics;
- data cleaning and cultural-adaptation reports exist.

Before the gate passes, forbidden claims include:
- clinical deployment readiness;
- population-level generalization;
- superiority over baselines under audited conditions;
- top-journal-ready evidence;
- direct use of the reported cultural-adaptation metrics as main figure evidence.

## Verification

Definition verification:
- `openspec validate glucose-experiment-readiness --type change --strict --no-interactive`
- `python3 -m compileall -q projects/glucose/src/run_glucose_training.py projects/glucose/src/enhanced_glucose_prediction_trainer.py projects/glucose/src/external_validation_and_baselines.py`
- `python3 -m unittest test_source_aware_split_manifest.py test_source_aware_split_dataset.py test_split_manifest_baselines.py test_run_glucose_training_cli.py`
- `git diff --check`

Execution verification, for a future gate run:
- dataset manifest exists and identifies canonical data;
- split manifest proves patient or user disjointness where identifiers are reliable;
- baseline parity table references the same split;
- leakage audit records pass, fail, or blocked for each risk item;
- result summary excludes raw data, checkpoints, secrets, and row-level predictions.

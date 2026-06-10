## Why

`projects/glucose/` contains the strongest current local result evidence: multi-step blood glucose prediction, personalization records, cleaned-dataset reports, and a final cultural-adaptation evaluation. The observed metrics are promising, but the result ledger still grades them as B-level evidence because dataset provenance, patient-level independence, time split policy, baseline parity, and leakage risks have not been audited end to end.

Before using these results as a top-journal manuscript basis, the repository needs a hard experiment-readiness gate.

## What Changes

- Add an OpenSpec change named `glucose-experiment-readiness`.
- Define readiness requirements for canonical data, split protocol, seed policy, baseline parity, metric reporting, leakage audit, and claim boundaries.
- Add a glucose protocol checklist at `projects/glucose/protocols/experiment_readiness_gate.md`.
- Add a source-aware split manifest generator for the public-preprocessed glucose candidate.
- Add a lightweight split artifact at `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`.
- Add source-aware split consumers for baseline and training entrypoints, with smoke-run limits.
- Add a lightweight result summary schema for future manuscript-facing result artifacts.
- Add Superpowers design and implementation-plan records for executing the gate later.
- Update README, result ledger, project audit, and GitHub tracking manifest to point to the active gate.

## Capabilities

### New Capabilities

- `experiment-readiness`: Defines the conditions required before a local experiment result can be promoted into manuscript-level evidence.

### Modified Capabilities

- `research-repo-governance`: The active research boundary now includes a glucose readiness gate as the next manuscript-preparation step.

## Impact

Affected areas:
- `openspec/changes/glucose-experiment-readiness/`
- `docs/superpowers/specs/2026-06-10-glucose-experiment-readiness-design.md`
- `docs/superpowers/plans/2026-06-10-glucose-experiment-readiness.md`
- `projects/glucose/protocols/experiment_readiness_gate.md`
- `projects/glucose/protocols/baseline_parity_table.md`
- `projects/glucose/protocols/glucose_result_summary_schema.md`
- `projects/glucose/protocols/glucose_baseline_parity_result_summary.json`
- `projects/glucose/protocols/glucose_candidate_rerun_budget.md`
- `projects/glucose/protocols/glucose_candidate_rerun_result_summary.json`
- `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`
- `projects/glucose/src/analysis/source_aware_split_manifest.py`
- `projects/glucose/src/analysis/source_aware_split_dataset.py`
- `projects/glucose/src/test_source_aware_split_manifest.py`
- `projects/glucose/src/test_source_aware_split_dataset.py`
- `projects/glucose/src/test_split_manifest_baselines.py`
- `projects/glucose/src/test_run_glucose_training_cli.py`
- `projects/glucose/src/external_validation_and_baselines.py`
- `projects/glucose/src/run_glucose_training.py`
- `projects/glucose/src/enhanced_glucose_system.py`
- `projects/glucose/src/data_processing/__init__.py`
- README, project audit, result ledger, and GitHub tracking manifest

No raw data, processed datasets, model checkpoints, generated predictions, or historical result files are moved, deleted, or committed. Smoke outputs remain ignored local artifacts.

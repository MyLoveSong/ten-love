## 1. Change Records

- [x] 1.1 Create OpenSpec proposal, design, task, and requirement delta files.
- [x] 1.2 Create Superpowers design and implementation plan files.
- [x] 1.3 Create `projects/glucose/protocols/experiment_readiness_gate.md`.

## 2. Documentation Alignment

- [x] 2.1 Update README to identify `glucose-experiment-readiness` as the active experiment gate.
- [x] 2.2 Update `RESULTS_LEDGER.md` so current Glucose evidence remains B-level until this gate passes.
- [x] 2.3 Update `PROJECT_AUDIT.md` with the third-round glucose readiness scope.
- [x] 2.4 Update `GITHUB_TRACKING_MANIFEST.md` to include the new lightweight gate documents.

## 3. Future Gate Execution

- [x] 3.1 Create preliminary canonical glucose dataset manifest from lightweight metadata.
- [x] 3.2 Create preliminary split manifest with observed split and leakage risks.
- [ ] 3.3 Freeze the canonical dataset selection and hash or sampling-hash strategy.
- [x] 3.4 Generate preliminary source-aware group split artifact for `public_glucose_preprocessed.json`, with hashed `source + patient_id` groups and time-order metadata.
- [x] 3.5 Run baseline parity checks on the same split and prediction horizon.
  - [x] 3.5.a Run source-aware smoke baseline on `persistence` and `LinearRegression` with 512 windows per split.
  - [x] 3.5.b Run full same-split baseline parity for persistence, LinearRegression, GBM, and MLPRegressor.
- [x] 3.6 Run preliminary leakage audit for duplicate rows, overlapping windows, patient overlap, generated patient IDs, normalization scope, target leakage, and test-set reuse; audit does not pass.
- [ ] 3.7 Export lightweight manuscript-facing result summaries.
  - [x] 3.7.a Scaffold `projects/glucose/protocols/glucose_result_summary_schema.md`.
  - [x] 3.7.b Populate a full baseline-parity summary.
  - [x] 3.7.c Populate a candidate-model pilot summary after fixed-budget GluFormer execution.
  - [x] 3.7.d Record failure analysis for the GluFormer pilot that did not outperform MLPRegressor.
  - [x] 3.7.e Populate a 10-epoch single-seed GluFormer triage summary without claim upgrade.
  - [ ] 3.7.f Populate a final candidate strategy summary after multi-seed policy and leakage audit pass.
- [ ] 3.8 Update `RESULTS_LEDGER.md` only after the gate artifacts exist.
  - [x] 3.8.a Record smoke-only outputs in `RESULTS_LEDGER.md` without claim upgrade.
  - [x] 3.8.b Record full baseline parity results without claim upgrade.
  - [x] 3.8.c Record candidate-model pilot results without claim upgrade.
  - [x] 3.8.d Record candidate-model failure analysis without claim upgrade.
  - [x] 3.8.e Record 10-epoch single-seed triage without claim upgrade.
  - [ ] 3.8.f Record final candidate results only after all required gate artifacts exist.
- [x] 3.9 Add explicit seed control to split-manifest training and record seed metadata.
- [x] 3.10 Create local metric definitions for MAE, RMSE, R2, and per-horizon metrics without claim upgrade.
- [x] 3.11 Create preliminary data availability/source audit; audit remains blocking because `glucose_ml_collection` provenance is unresolved.
- [x] 3.12 Close `glucose_ml_collection` provenance as unresolved for the current local derived dataset and reject `public_glucose_preprocessed.json` as manuscript canonical data.
- [x] 3.13 Generate BigIdeas-only source report and source-aware split manifest from the verified PhysioNet BigIdeas local mirror.
- [x] 3.14 Add direct inverse-scaled mg/dL overall/per-horizon metric export to the split-manifest training entrypoint.
- [x] 3.15 Run BigIdeas-only baseline and training smoke checks without claim upgrade.

## 4. Verification

- [x] 4.1 Run OpenSpec strict validation for this change.
- [x] 4.2 Run targeted Python compile checks for Glucose entrypoints.
- [x] 4.3 Run stale reference search for the new gate.
- [x] 4.4 Run `git diff --check`.
- [x] 4.5 Run split-manifest seed-control and baseline regression tests.
- [x] 4.6 Run BigIdeas builder and inverse-metric regression tests.

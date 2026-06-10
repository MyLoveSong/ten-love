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
- [ ] 3.5 Run baseline parity checks on the same split and prediction horizon.
  - [x] 3.5.a Run source-aware smoke baseline on `persistence` and `LinearRegression` with 512 windows per split.
  - [ ] 3.5.b Run full same-split baseline parity for persistence, LinearRegression, GBM, and MLPRegressor.
- [x] 3.6 Run preliminary leakage audit for duplicate rows, overlapping windows, patient overlap, generated patient IDs, normalization scope, target leakage, and test-set reuse; audit does not pass.
- [ ] 3.7 Export lightweight manuscript-facing result summaries.
  - [x] 3.7.a Scaffold `projects/glucose/protocols/glucose_result_summary_schema.md`.
  - [ ] 3.7.b Populate a full rerun summary after baseline parity and main-model budget execution.
- [ ] 3.8 Update `RESULTS_LEDGER.md` only after the gate artifacts exist.
  - [x] 3.8.a Record smoke-only outputs in `RESULTS_LEDGER.md` without claim upgrade.
  - [ ] 3.8.b Record full rerun results only after all required gate artifacts exist.

## 4. Verification

- [x] 4.1 Run OpenSpec strict validation for this change.
- [x] 4.2 Run targeted Python compile checks for Glucose entrypoints.
- [x] 4.3 Run stale reference search for the new gate.
- [x] 4.4 Run `git diff --check`.

# Glucose Experiment Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the existing Glucose local results into an auditable experiment package before any manuscript claim upgrade.

**Architecture:** Treat `projects/glucose/` as a gate-driven research line. Freeze data and split metadata first, audit leakage second, run same-split baselines third, then export lightweight result summaries and update `RESULTS_LEDGER.md`.

**Tech Stack:** Python 3.12, PyTorch-based Glucose scripts, Markdown manifests, JSON summaries, OpenSpec 1.3.1.

---

### Task 1: Freeze Canonical Dataset Manifest

**Files:**
- Create: `projects/glucose/protocols/canonical_dataset_manifest.md`
- Read: `DATA_INVENTORY.md`
- Read: `projects/glucose/README.md`

- [x] **Step 1: Create the manifest skeleton**

```markdown
# Canonical Glucose Dataset Manifest

Status: blocked-unverified

## Canonical Dataset

| Field | Value |
|---|---|
| canonical_path | not yet audited |
| source_dataset | not yet audited |
| access_route | local only until repository and licence are confirmed |
| raw_or_derived | not yet audited |
| sample_count | not yet audited |
| patient_or_user_count | not yet audited |
| timestamp_range | not yet audited |
| hash_strategy | sampling hash required before training rerun |

## Duplicate And Mirror Policy

`dataset/` and `projects/glucose/data/` must be classified as source, working copy, derived output, cache, or excluded duplicate before any rerun.
```

- [x] **Step 2: Populate only observed fields**

Run:

```bash
cd /home/data/xzy/system
du -sh projects/glucose/data dataset 2>/dev/null
find projects/glucose/data -maxdepth 3 -type f -name '*.json' -o -name '*.csv' | sed -n '1,80p'
```

Expected: enough path-level evidence to fill dataset candidates without reading large files in full.

- [x] **Step 3: Keep unresolved fields explicit**

If a field cannot be verified from light metadata, keep `not yet audited` and record it as a blocker in the manifest. Do not infer patient counts from filenames.

### Task 2: Freeze Split And Seed Protocol

**Files:**
- Create: `projects/glucose/protocols/split_manifest.md`
- Read: `projects/glucose/src/run_glucose_training.py`
- Read: `projects/glucose/src/enhanced_glucose_prediction_trainer.py`

- [x] **Step 1: Create split manifest**

```markdown
# Glucose Split Manifest

Status: blocked-unverified

| Field | Value |
|---|---|
| input_horizon | 12 |
| output_horizon | 6 |
| train_ratio | 0.8 |
| val_ratio | 0.1 |
| test_ratio | 0.1 |
| seed_list | 42, 123, 456, 789, 101112 |
| split_unit | patient_or_user_id required when reliable identifiers exist |
| time_order_policy | sliding windows must not cross train/val/test boundaries |
| normalization_scope | fit on train only |
```

- [x] **Step 2: Audit split code**

Run:

```bash
cd /home/data/xzy/system
rg -n "train_ratio|val_ratio|test_ratio|patient_id|random_split|train_test_split|KFold|seed|normaliz|scaler" projects/glucose/src experiments -g '*.py'
```

Expected: list of split and scaling code paths that must be reviewed before rerun.

### Task 3: Baseline Parity Matrix

**Files:**
- Create: `projects/glucose/protocols/baseline_parity_table.md`
- Read: `projects/glucose/src/external_validation_and_baselines.py`
- Read: `experiments/protocol/gluformer_research_protocol.md`

- [ ] **Step 1: Create baseline table**

```markdown
# Glucose Baseline Parity Table

Status: blocked-unverified

| Baseline | Same split | Same input horizon | Same output horizon | Same metrics | Status |
|---|---|---|---|---|---|
| naive persistence | not yet audited | not yet audited | not yet audited | not yet audited | required |
| LSTM | not yet audited | not yet audited | not yet audited | not yet audited | required |
| GRU | not yet audited | not yet audited | not yet audited | not yet audited | required |
| Transformer | not yet audited | not yet audited | not yet audited | not yet audited | required |
| GluFormer candidate | not yet audited | not yet audited | not yet audited | not yet audited | required |
```

- [ ] **Step 2: Verify baseline entrypoints compile**

Run:

```bash
cd /home/data/xzy/ten-love
python3 -m compileall -q projects/glucose/src/external_validation_and_baselines.py projects/glucose/src/run_glucose_training.py
```

Expected: exit code 0.

### Task 4: Leakage Audit

**Files:**
- Create: `projects/glucose/protocols/leakage_audit.md`

- [x] **Step 1: Create leakage audit checklist**

```markdown
# Glucose Leakage Audit

Status: blocked-unverified

| Risk | Required check | Status |
|---|---|---|
| duplicate rows | hash or key-level duplicate check before split | blocked-unverified |
| overlapping windows | no train/val/test boundary crossing | blocked-unverified |
| patient overlap | disjoint patient or user IDs where identifiers are reliable | blocked-unverified |
| generated IDs | classify generated row-order IDs as weak identifiers | blocked-unverified |
| scaler leakage | fit scalers on train only | blocked-unverified |
| target leakage | no future glucose values in input features | blocked-unverified |
| test reuse | no test set used for model selection or early stopping | blocked-unverified |
```

- [x] **Step 2: Do not promote high metrics until audit passes**

Expected: any unresolved item keeps the result at local-observation level in `RESULTS_LEDGER.md`.

Observed: `projects/glucose/protocols/leakage_audit.md` was created as a
preliminary, blocking audit. It does not pass. `unified_cleaned_glucose.json`
is not frozen because of severe null timestamp and duplicate-key findings.
`public_glucose_preprocessed.json` is the next source-aware split-audit
candidate, using `source + patient_id` as the required group key.

### Task 4.5: Source-Aware Split Artifact

**Files:**
- Create: `projects/glucose/src/analysis/source_aware_split_manifest.py`
- Create: `projects/glucose/src/test_source_aware_split_manifest.py`
- Create: `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json`

- [x] **Step 1: Add TDD coverage for source-aware split manifest generation**

Expected: tests prove the generated manifest uses hashed `source + patient_id`
groups, avoids raw patient IDs, avoids row-level glucose values, and blocks
duplicate `source + patient_id + timestamp` keys.

- [x] **Step 2: Generate the public-preprocessed split artifact**

Observed: `public_glucose_source_aware_split_manifest.json` records 100
source-patient groups, split into 80 train, 10 validation, and 10 test groups
with seed 42, input horizon 12, and output horizon 6. Training and baseline
entrypoints now consume this artifact in smoke mode only. Full baseline parity,
main-model rerun budget, and result summary artifacts are still required before
any result upgrade.

### Task 5: Result Summary And Ledger Update

**Files:**
- Create: `projects/glucose/protocols/glucose_result_summary_schema.md`
- Modify: `RESULTS_LEDGER.md`

- [x] **Step 1: Define summary schema**

```markdown
# Glucose Result Summary Schema

Required fields:
- run_id
- git_commit
- canonical_dataset_manifest
- split_manifest
- seed
- model_name
- baseline_name
- input_horizon
- output_horizon
- mae
- rmse
- r2
- per_horizon_metrics
- leakage_audit
- claim_level
```

- [ ] **Step 2: Update ledger only after artifacts exist**

Run:

```bash
cd /home/data/xzy/ten-love
rg -n "Glucose 结果|下一步证据 gate|glucose-experiment-readiness" RESULTS_LEDGER.md
```

Expected: ledger references the gate and keeps current evidence at B until all future artifacts exist.

Observed: `glucose_result_summary_schema.md` is scaffolded. The ledger records
smoke baseline and training outputs as local, ignored artifacts only. Current
Glucose evidence remains B-level.

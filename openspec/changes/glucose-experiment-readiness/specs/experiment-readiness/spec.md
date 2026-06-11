## ADDED Requirements

### Requirement: Glucose claim upgrades require an experiment-readiness gate

Glucose results SHALL NOT be promoted from local observation to manuscript-level evidence until the experiment-readiness gate records canonical data, split, baseline, metric, data-availability, leakage, and claim-boundary artifacts.

#### Scenario: Existing Glucose result is cited
- **WHEN** documentation cites a prior Glucose training or evaluation result
- **THEN** it identifies the result as a local observation unless the readiness gate has passed
- **AND** it links the result to its source artifact
- **AND** it avoids clinical deployment or population-generalization claims

#### Scenario: Glucose result is promoted to manuscript evidence
- **WHEN** a maintainer upgrades Glucose evidence to manuscript-level support
- **THEN** the gate contains a canonical data manifest
- **AND** the gate contains a frozen split manifest
- **AND** the gate contains a baseline parity table
- **AND** the gate contains metric definitions and per-horizon results
- **AND** the gate contains a data availability and source-access audit
- **AND** the gate contains a leakage audit
- **AND** the gate contains a claim-boundary statement

### Requirement: Canonical glucose data must be frozen before reruns

The Glucose experiment line SHALL identify a canonical dataset before any new result is treated as comparable to prior results.

#### Scenario: Canonical data is declared
- **WHEN** a canonical Glucose dataset is selected
- **THEN** its manifest records path, source, access route, raw versus derived status, sample count, patient or user count when available, timestamp range when available, and hash or sampling-hash strategy
- **AND** mirrored or duplicate dataset locations are classified as source, working copy, derived output, cache, or excluded duplicate

#### Scenario: A source label cannot be reconciled
- **WHEN** a source label in a derived Glucose dataset cannot be tied to an authoritative release, repository file list, or controlled-access route
- **THEN** the derived dataset is rejected for manuscript canonical use
- **AND** any existing results on that derived dataset remain local engineering evidence only
- **AND** a replacement candidate uses only sources with recorded access routes and licence or DUA terms

### Requirement: Glucose splits must protect patient and temporal independence

The Glucose experiment line SHALL use split rules that minimize leakage for CGM prediction.

#### Scenario: Split manifest is frozen
- **WHEN** train, validation, and test partitions are created
- **THEN** patient or user identifiers are disjoint across partitions when identifiers are reliable
- **AND** the time-order policy for sliding windows is documented
- **AND** raw patient identifiers and row-level glucose values are excluded from committed split artifacts
- **AND** normalization and feature-scaling parameters are fitted on training data only
- **AND** the seed list is recorded

### Requirement: Glucose metrics and baselines must be comparable

Glucose experiments SHALL report metrics and baselines under the same split and prediction horizon.

#### Scenario: Baseline results are reported
- **WHEN** a baseline is compared with GluFormer or an enhanced Glucose model
- **THEN** both models use the same input horizon, output horizon, split, and metric definitions
- **AND** MAE, RMSE, R2, and per-horizon t+1 through t+6 metrics are reported when six-step prediction is used
- **AND** training-entrypoint metrics intended for result comparison are inverse-scaled to the declared glucose unit
- **AND** clinical-range metrics are reported only when thresholds, labels, and units are audited

### Requirement: Glucose leakage audit blocks unsupported claims

The Glucose experiment line SHALL complete a leakage audit before any high-performing result is used in a main claim or figure.

#### Scenario: Leakage audit is reviewed
- **WHEN** a result shows unusually high performance or is selected for manuscript use
- **THEN** the audit checks duplicate rows, overlapping windows, patient overlap, generated patient IDs, target leakage, scaler leakage, and test-set reuse
- **AND** unresolved leakage risks are recorded as blockers
- **AND** the claim level remains local or exploratory until blockers are resolved

### Requirement: Reused glucose data must have audited availability before manuscript use

The Glucose experiment line SHALL NOT rely on reused human-participant CGM data
for a manuscript claim unless source identity, access route, licence or DUA
terms, redistribution boundary, and dataset citations are recorded.

#### Scenario: Reused dataset source is included
- **WHEN** a source-labelled derived glucose dataset is selected as canonical
- **THEN** every source label maps to an authoritative dataset release, repository, or controlled-access route
- **AND** source-specific licence, DUA, or use restrictions are recorded
- **AND** row-level data are excluded from Git unless redistribution is explicitly permitted
- **AND** unresolved source-access findings block claim upgrade

## ADDED Requirements

### Requirement: Active repository excludes frontend demo code

The active research repository SHALL exclude frontend demo source code unless a future OpenSpec change demonstrates that the frontend directly supports experiment execution, evidence review, or manuscript reproduction.

#### Scenario: Frontend pruning is applied
- **WHEN** the active repository is prepared for research-focused publication work
- **THEN** `frontend/` source code is absent from the active tracked tree
- **AND** documentation describes the repository as research code, data governance, result evidence, and reproducibility infrastructure

#### Scenario: Historical frontend provenance is needed
- **WHEN** a maintainer needs to inspect the removed frontend demo
- **THEN** the maintainer can recover it from Git history rather than from an active archive directory

### Requirement: Research claims remain evidence-bounded

Documentation that supports manuscript preparation SHALL distinguish implemented research infrastructure from unsupported product, deployment, or clinical claims.

#### Scenario: Manuscript-facing documentation is updated
- **WHEN** a document describes the system architecture
- **THEN** it avoids presenting Vue/Vite frontend code as an active contribution
- **AND** it does not upgrade demo or service scaffolding into validated clinical or top-journal evidence

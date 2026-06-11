# Glucose Protocol Index

Status: active gate index, not a passed manuscript gate.

## Verdict

This directory contains the lightweight, Git-trackable evidence chain for the
current Glucose experiment-readiness gate. It is an audit index only. Raw
glucose rows, row-level predictions, model checkpoints, and full training
outputs stay outside Git.

## Workflow Order

| Step | File | Current status | Boundary |
|---|---|---|---|
| 1 | `canonical_dataset_manifest.md` | preliminary | canonical dataset not frozen |
| 2 | `glucose_ml_collection_provenance_closure.md` | closed blocker | old public-preprocessed candidate rejected for manuscript canonical use |
| 3 | `bigideas_glucose_source_report.json` | generated source report | BigIdeas-only verified-source draft, no row-level values |
| 4 | `bigideas_source_aware_split_manifest.json` | generated split artifact | 13/2/1 subject-group train/val/test split, no raw patient IDs or row-level glucose values |
| 5 | `leakage_audit.md` | preliminary, blocking | old public-preprocessed and unified candidates blocked or historical |
| 6 | `split_manifest.md` | preliminary | BigIdeas-only split exists, gate still not passed |
| 7 | `public_glucose_source_aware_split_manifest.json` | historical engineering artifact | old public-preprocessed split only, not canonical |
| 8 | `baseline_parity_table.md` | full same-split baseline parity completed on old public candidate | local engineering claim only |
| 9 | `glucose_baseline_parity_result_summary.json` | lightweight baseline summary on old public candidate | not manuscript evidence after provenance closure |
| 10 | `glucose_candidate_rerun_budget.md` | 3-epoch pilot budget executed on old public candidate | pilot only |
| 11 | `glucose_candidate_rerun_result_summary.json` | lightweight 3-epoch summary on old public candidate | GluFormer did not beat MLPRegressor |
| 12 | `gluformer_failure_analysis.md` | failure analysis plus 10-epoch triage completed | no superiority claim |
| 13 | `glucose_candidate_10epoch_triage_result_summary.json` | lightweight 10-epoch summary on old public candidate | mixed result versus MLPRegressor |
| 14 | `metric_definitions.md` | active local metric definition | training entry now exports inverse-scaled split metrics |
| 15 | `data_availability_audit.md` | preliminary, blocking | BigIdeas route is verified; old public candidate remains rejected |
| 16 | `glucose_result_summary_schema.md` | active schema | local evidence only |
| 17 | `experiment_readiness_gate.md` | gate status document | gate not passed |
| 18 | `glucose_bigideas_baseline_parity_result_summary.json` | BigIdeas full baseline parity completed | local baseline evidence only |
| 19 | `bigideas_final_leakage_audit.md` | BigIdeas leakage pass completed with limitations | baseline parity pass only |

## Current Decision

`glucose_ml_collection` provenance has been closed as unresolved for the
current local derived dataset. The old `public_glucose_preprocessed` split and
results remain engineering evidence only. The next candidate line is
BigIdeas-only, with `bigideas_glucose_source_report.json` and
`bigideas_source_aware_split_manifest.json` as the draft verified-source
artifacts.

## Remaining Gate Blockers

- GluFormer 30-epoch rerun on the BigIdeas-only split for seeds 42, 123, and
  456.
- Final data availability statement for the selected BigIdeas-only claim level.
- Multi-seed policy summary and claim-boundary decision after the above
  artifacts exist.
- Additional data or external validation if the manuscript needs a
  generalization claim beyond the 16-subject BigIdeas cohort.

## Artifact Policy

Track only lightweight protocol files, hashes, aggregate metrics, and code
needed to reproduce the audit. Do not commit `TRAIN/outputs/`, `outputs/`,
`projects/glucose/data/`, checkpoints, row-level predictions, raw patient IDs,
or private source data.

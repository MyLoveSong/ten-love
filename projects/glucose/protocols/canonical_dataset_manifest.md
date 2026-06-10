# Canonical Glucose Dataset Manifest

Status: preliminary, blocked-unverified.

## Verdict

No canonical Glucose dataset is frozen yet.

The strongest current training-input candidate is
`projects/glucose/data/cleaned_dataset/unified_cleaned_glucose.json`, because
the Glucose README uses `data/cleaned_dataset/unified_cleaned_glucose.json` as
the local training input. This file is a derived local working dataset, not a
verified raw source. It can be used for the first audit pass only after split,
identifier, duplicate, and leakage checks are completed.

The first leakage audit blocks freezing this file: it contains 243107 null
timestamps, 4529 duplicate `patient_id + timestamp` groups, and 243012 rows
inside duplicate-key groups.

The strongest source-aware audit candidate is
`projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json`,
because it retains a `source` field. Its report shows duplicate patient IDs
across sources, so `patient_id` alone is not reliable as a split key.
It passed the current duplicate `source + patient_id + timestamp` check and is
the next split-audit candidate. A preliminary group-disjoint split artifact now
exists, but the dataset is not a manuscript-ready canonical source until source,
licence, and access-route checks are complete.

## Metadata Collection Scope

This manifest was created from lightweight metadata only:

- directory sizes;
- file paths and byte sizes;
- CSV headers;
- JSON top-level schema and aggregate counts;
- existing small report JSON summaries;
- SHA-256 for small candidate files under `cleaned_dataset/`.

No raw large CSV, large dataset mirror, checkpoint, or row-level prediction file
was read in full for this manifest.

## Top-Level Data Volumes

| Path | Observed size | Role | Status |
|---|---:|---|---|
| `/home/data/xzy/system/projects/glucose/data` | 69G | Glucose working data tree | local only, not Git |
| `/home/data/xzy/system/dataset` | 49G | public dataset mirror tree | local only, not Git |

## Candidate Dataset Files

| Candidate | Bytes | Observed schema or summary | Current role | Boundary |
|---|---:|---|---|---|
| `projects/glucose/data/cleaned_dataset/unified_cleaned_glucose.json` | 26482566 | JSON object with `records`; 254445 records; fields `glucose`, `patient_id`, `timestamp`; 4980 unique `patient_id`; 243107 null timestamps; 4529 duplicate `patient_id + timestamp` groups | blocked training-input candidate | derived local working file; source provenance not encoded per record; timestamp and duplicate-key findings block freezing |
| `projects/glucose/data/cleaned_dataset/unified_cleaned_glucose.csv` | 5895203 | CSV header `patient_id,timestamp,glucose` | compact export candidate | equivalence with JSON not audited |
| `projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json` | 48612811 | JSON object with `records`; 201600 records; fields `delta`, `glucose_mg_dl`, `glucose_normalized`, `patient_id`, `rolling_mean`, `source`, `timestamp`; 50 unique `patient_id`; 100 unique `source + patient_id` groups; 0 duplicate `source + patient_id + timestamp` groups in current audit | source-aware split-audit candidate with preliminary split artifact | `patient_id` is duplicated across sources; split key must include source; source/licence/access route still unresolved |
| `projects/glucose/protocols/public_glucose_source_aware_split_manifest.json` | 33906 | lightweight JSON split artifact; 80 train groups, 10 validation groups, 10 test groups; no row-level glucose values or raw patient IDs | split artifact for next rerun | not a model result and not a passed gate |
| `projects/glucose/data/cleaned_dataset/public_glucose_preprocess_report.json` | 30645 | sources `ohio_t1dm`, `glucose_ml_collection`; `patients_kept=100`; `total_samples=201600`; 5-minute window frequency; per-source count 50 each; 50 duplicate patient-id groups across sources | metadata support for public preprocess | report-level evidence only, not a split manifest |
| `projects/glucose/data/cleaned_dataset/cleaning_report.json` | 1899 | cultural-adaptation cleaning report; original 3644, final 3005, feature dimensions 21 | cultural-adaptation data report | supports cleaning provenance, not CGM forecasting generalization |
| `projects/glucose/data/physionet_big_ideas/raw_data/` | small local mirror metadata only observed: `LICENSE.txt`, `SHA256SUMS.txt`, `Demographics.csv` | project-local BigIdeas mirror stub | raw-source candidate metadata | `Demographics.csv` is detected as Microsoft Excel 2007+ despite `.csv`; format needs audit |
| `dataset/big-ideas-lab-glycemic-variability-and-wearable-device-data-1.0.0/...` | large mirrored raw tree | large wearable data files observed by path and size | raw-source mirror candidate | must be classified against `projects/glucose/data/physionet_big_ideas/` before use |
| `projects/glucose/data/world_food_facts/world_food_facts.csv` | 1021405882 | food nutrition table by path | excluded from CGM canonical candidate | nutrition source, not blood-glucose forecasting target data |

## SHA-256 For Lightweight Candidate Files

| File | SHA-256 |
|---|---|
| `cleaned_dataset/unified_cleaned_glucose.json` | `10b57cffdbae9a2d22269d875eec8067c44a68024f7372480ec0ed2e8a49590d` |
| `cleaned_dataset/unified_cleaned_glucose.csv` | `a184f7a1bf5d3a6aff9cc0099856ec1e713ad84445ac39041f89d6a0fb80dfb8` |
| `cleaned_dataset/public_glucose_preprocessed.json` | `c40ff621d3ff3a82e45bb69981a5df8b365407170a3e27057d232170a3cefd36` |
| `cleaned_dataset/public_glucose_preprocess_report.json` | `ded935d3c2a2bf313ed3e8da3c880eb5ad661dc852a151f18593ad3abdc1723d` |
| `cleaned_dataset/cleaning_report.json` | `4b1b6a545e164f4b1545b441d5f189a48c840d081c9cccf4315c5473533fe967` |

## Identifier Reliability

| Dataset | Observed identifier | Reliability status | Required split key |
|---|---|---|---|
| `unified_cleaned_glucose.json` | `patient_id`; 4980 unique IDs | weak until provenance audit confirms IDs are real subject IDs and not generated row-order IDs | `patient_id` only after provenance audit |
| `public_glucose_preprocessed.json` | `patient_id` plus `source`; report shows 100 per-patient entries but only 50 unique `patient_id` values | `patient_id` alone is invalid across sources | `source + patient_id` |
| `unified_cleaned_glucose.csv` | `patient_id` | weak until equivalence with JSON and provenance are checked | `patient_id` only after audit |
| BigIdeas raw mirror | subject-folder IDs by path | promising raw grouping signal | subject folder ID after raw mirror classification |

## Canonical Selection Rule

Use the following rule before any rerun:

1. Select exactly one canonical dataset for the experiment line.
2. Classify every other glucose data path as raw source, working copy, derived output, cache, or excluded duplicate.
3. Freeze a hash or sampling-hash strategy for the canonical file or directory.
4. Prove split keys are reliable for the claim level.
5. Write the selected canonical path into `split_manifest.md`.

## Current Blockers

| Blocker | Why it matters |
|---|---|
| canonical dataset not selected | prior results cannot be reproduced against a frozen data identity |
| `unified_cleaned_glucose` has severe timestamp and duplicate-key findings | this blocks freezing it as the manuscript canonical dataset |
| `public_glucose_preprocessed` duplicates patient IDs across sources | patient-level split leaks if `patient_id` is used without source namespace |
| `unified_cleaned_glucose` lacks source field in observed schema | source-level duplicate and provenance audit cannot be done from schema alone |
| BigIdeas mirror not reconciled across `dataset/` and `projects/glucose/data/` | raw versus working-copy relationship is not fixed |
| training and baseline entrypoints do not consume the split artifact | current result remains B-level local observation |

## Next Minimal Step

Update training and baseline entrypoints to consume
`public_glucose_source_aware_split_manifest.json`, then rerun metrics with
train-only normalization and a lightweight result summary.

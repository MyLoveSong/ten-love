# Glucose-ML Collection Provenance Closure

Status: closed as unresolved for the current local derived dataset.

## Verdict

The local `glucose_ml_collection` source label cannot be used as manuscript
dataset provenance for the current `public_glucose_preprocessed.json` file.

The local code path records a Glucose-ML URL, but the implementation used by
`CGMDataSource` generates example records for `glucose_ml_collection` rather
than parsing authoritative upstream Glucose-ML files. Therefore, the existing
`public_glucose_preprocessed.json` and its source-aware split artifact are
kept only as local engineering smoke and parity evidence. They are rejected as
canonical manuscript data.

## Local Code Evidence

| Evidence | Finding | Decision |
|---|---|---|
| `projects/glucose/src/data_sources/web_data_collector.py` | `_process_glucose_ml_data(...)` logs Glucose-ML processing, then creates example records in a local loop with generated `GML_...` IDs and random glucose values | source label is not row provenance |
| `projects/glucose/src/data_sources/web_data_collector.py` | `_fetch_real_dataset(...)` falls back to `_generate_simulated_cgm_data(...)` after download/authentication failures | downloaded-source claims require explicit artifact evidence |
| `public_glucose_preprocess_report.json` | source labels are `ohio_t1dm` and `glucose_ml_collection`, with matched 50-patient/100800-record source counts | consistent with local preprocessing, not enough to prove external dataset identity |
| `public_glucose_source_aware_split_manifest.json` | group-disjoint split is valid as a local artifact, but inherits source uncertainty | keep for engineering regression only |

## External Source Reconciliation

The public Glucose-ML project now appears under
`Augmented-Health-Lab/Glucose-ML-Project`, with scripts and collection folders
for open-access and controlled-access datasets. That repository context does
not prove that the rows in the local derived file came from that release,
because the local code points to `irinagain/Glucose-ML` and does not record an
upstream commit, file list, or per-dataset licence chain for the derived rows.

Glucose-ML itself is a useful source catalogue, but a manuscript dataset must
name the specific underlying datasets actually used. A collection-level label
is not sufficient provenance for row-level CGM training data.

## Replacement Candidate

The next verified-source candidate is BigIdeas-only:

| Artifact | Status |
|---|---|
| local derived dataset | `projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json`, local only, Git-ignored |
| source report | `projects/glucose/protocols/bigideas_glucose_source_report.json` |
| split artifact | `projects/glucose/protocols/bigideas_source_aware_split_manifest.json` |
| raw source | PhysioNet BigIdeas v1.0.0 local mirror |
| licence | ODC Attribution License, with attribution required |
| split unit | `source + subject folder ID`, committed only as hashed group IDs |

## Claim Boundary

Old `public_glucose_preprocessed` baseline and GluFormer summaries remain
useful for debugging the training path and same-split comparison code. They do
not support manuscript claims after this closure.

BigIdeas-only baseline and candidate results now remain local evidence after
full baseline parity, the baseline-specific final leakage pass, and the
GluFormer multi-seed summary. They cannot support a manuscript claim until
final data availability and claim-boundary review are completed on the
BigIdeas split.

## Source Basis

| Source | Use |
|---|---|
| Augmented-Health-Lab, `Glucose-ML-Project`, https://github.com/Augmented-Health-Lab/Glucose-ML-Project | identifies current public Glucose-ML project and its collection structure |
| Prioleau, Lu, and Cui 2025, "Glucose-ML: A collection of longitudinal diabetes datasets for development of robust AI solutions", arXiv:2507.14077 | supports that Glucose-ML is a collection of multiple public datasets and includes both open and controlled access sources |
| PhysioNet, "BIG IDEAs Lab Glycemic Variability and Wearable Device Data v1.0.0", https://physionet.org/content/big-ideas-glycemic-wearable/1.0.0/ | identifies the BigIdeas source dataset and study context |
| Open Data Commons Attribution License, https://physionet.org/content/big-ideas-glycemic-wearable/1.0.0/LICENSE.txt | records attribution requirements for BigIdeas database use |

## Next Minimal Step

Complete claim-boundary review for the BigIdeas mixed result, using the
BigIdeas MLPRegressor result as the current strong baseline.

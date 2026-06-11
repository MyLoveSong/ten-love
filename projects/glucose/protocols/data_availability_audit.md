# Glucose Data Availability And Source Audit

Status: preliminary, blocking.

## Verdict

The current `public_glucose_preprocessed.json` candidate is not Nature-ready
for data availability.

The file is a local derived dataset built from two recorded source labels:
`ohio_t1dm` and `glucose_ml_collection`. OhioT1DM is controlled access and
should not be redistributed from this repository. The `glucose_ml_collection`
route in the local downloader points to an unresolved GitHub location and has
not yet been reconciled with an authoritative release, commit, dataset list, or
license for the exact rows represented in the derived file.

This audit supports local experiment triage only. It does not freeze the
canonical dataset or upgrade claim level.

## Local Dataset Evidence

| Field | Observed value |
|---|---|
| candidate file | `projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json` |
| candidate SHA-256 | `c40ff621d3ff3a82e45bb69981a5df8b365407170a3e27057d232170a3cefd36` |
| local report | `projects/glucose/data/cleaned_dataset/public_glucose_preprocess_report.json` |
| source labels | `ohio_t1dm`, `glucose_ml_collection` |
| record count | 201600 |
| `ohio_t1dm` records | 100800 |
| `glucose_ml_collection` records | 100800 |
| unique `source + patient_id` groups | 100 |
| unique `patient_id` values | 50 |
| current split key | `source + patient_id` |
| Git policy | do not commit raw rows, derived rows, checkpoints, or row-level predictions |

## Source Classification

| Source label | Current access route | Licence or use terms | Current decision |
|---|---|---|---|
| `ohio_t1dm` | controlled request route through the OhioT1DM dataset page; institutional researcher information required | local repo does not contain the DUA or redistribution permission | may be referenced as reused controlled-access data, but rows and derived row-level data must stay out of Git |
| `glucose_ml_collection` | local code points to `https://api.github.com/repos/irinagain/Glucose-ML/contents/data`; the current web audit identifies `Augmented-Health-Lab/Glucose-ML-Project` as a relevant Glucose-ML project | project software is MIT-licensed, but the exact dataset rows and underlying dataset licenses are not reconciled | blocked until exact source release, commit, file list, and per-dataset terms are verified |
| `public_glucose_preprocessed.json` | local derived working file only | inherits third-party source restrictions; no independent redistribution grant found | do not publish or upload row-level version |
| split and protocol files | Git-tracked lightweight artifacts | no row-level glucose values or raw patient IDs | safe to publish with code if secrets and raw identifiers remain absent |

## Nature-Readiness Requirements

Springer Nature and Nature Portfolio guidance requires the data availability
statement to say what data support the results, where those data can be found,
and the applicable access terms for both original and reused data. If data
cannot be openly shared, the statement must explain why and describe access
conditions.

For controlled-access data, the statement should precisely describe the reason
for restricted access, contact route, response timeframe, use restrictions, and
data-use agreement conditions where applicable.

## Draft Data Availability Statement

Not ready for submission. The current safe draft is:

```text
The processed split manifests, protocol files, and aggregate result summaries
supporting the local glucose-prediction analyses are available in the project
GitHub repository. Row-level CGM data, derived row-level processed datasets,
model checkpoints, and row-level predictions are not distributed with the
repository because the analyses reuse third-party human-participant CGM data.
The OhioT1DM data are available from the dataset maintainers through the
OhioT1DM request process. The source route and reuse terms for the records
labelled `glucose_ml_collection` in the local derived dataset remain under
audit and must be resolved before any manuscript claim relies on this dataset.
```

This wording is intentionally conservative. It should not be used as the final
Nature submission statement until `glucose_ml_collection` provenance and all
third-party use terms are resolved.

## Required Citation Actions

| Dataset or resource | Required action |
|---|---|
| OhioT1DM | cite Marling and Bunescu 2020, "The OhioT1DM dataset for blood glucose level prediction: Update 2020", KDH workshop / PMC record, and include the official request page in Data Availability |
| Glucose-ML | if the authoritative source is `Augmented-Health-Lab/Glucose-ML-Project`, cite Prioleau, Lu, and Cui 2025, "Glucose-ML: A collection of longitudinal diabetes datasets for development of robust AI solutions", arXiv:2507.14077, and record the exact Git commit or release |
| underlying Glucose-ML component datasets | cite each dataset actually used, not only the collection, once source reconciliation identifies them |
| local derived dataset | do not cite as public data until a reproducible regeneration route and redistribution status are written |

## FAIR Metadata Checklist

| Item | Current status | Required fix |
|---|---|---|
| dataset title | missing for derived file | define a local non-public working title |
| creators | missing for derived file | identify local preprocessing author or script provenance |
| source creators | partial | record source dataset authors and dataset-specific citations |
| publication year | partial | record per source |
| publisher or repository | missing for `glucose_ml_collection` rows | reconcile source route |
| persistent identifier | missing for derived file | do not mint until sharing policy is clear |
| version or commit | missing for `glucose_ml_collection` rows | record exact upstream release or commit |
| license or access terms | blocking | collect DUA/access terms and per-source licenses |
| file inventory | partial | create a regeneration manifest without row-level data |
| processing pipeline | partial | link exact preprocessing scripts and parameters |
| privacy review | blocking | confirm no row-level public release is allowed without source permissions |

## Source Basis

| Source | Use in this audit |
|---|---|
| Springer Nature, "Research data policy", https://www.springernature.com/gp/authors/research-data-policy | data availability statements must cover original and reused data, locations, and access terms |
| Springer Nature, "Data Availability Statements", https://www.springernature.com/gp/authors/research-data-policy/data-availability-statements | statement content and dataset citation expectations |
| Nature Portfolio, "Reporting standards and availability of data, materials, code and protocols", https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards | controlled-access and third-party data availability requirements |
| Nature Support, "Write a data availability statement for a paper", https://support.nature.com/en/support/solutions/articles/6000237611-write-a-data-availability-statement-for-a-paper | primary and secondary data must be described, including access conditions |
| OhioT1DM dataset page, https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html | controlled request route and OhioT1DM citation |
| Augmented-Health-Lab, `Glucose-ML-Project`, https://github.com/Augmented-Health-Lab/Glucose-ML-Project | current public Glucose-ML project context and MIT software license |
| Prioleau, Lu, Cui 2025, "Glucose-ML: A collection of longitudinal diabetes datasets for development of robust AI solutions", arXiv:2507.14077 | Glucose-ML collection scope and source-access context |

## Commands Used

```bash
jq -r '.sources, .patients_kept, .total_samples, .source_counts, .window_frequency_minutes, .max_gap_minutes, .min_coverage, .min_samples' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocess_report.json
jq -r '.records | {count:length, sources:(map(.source)|unique), source_counts:(group_by(.source)|map({source:.[0].source,count:length})), unique_source_patient:(map([.source,.patient_id])|unique|length), unique_patient:(map(.patient_id)|unique|length)}' /home/data/xzy/system/projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json
find /home/data/xzy/system/projects/glucose/data -maxdepth 4 -iname '*license*' -o -iname '*readme*' -o -iname '*dua*'
find /home/data/xzy/system/dataset -maxdepth 4 -iname '*license*' -o -iname '*readme*' -o -iname '*dua*'
rg -n "ohio_t1dm|glucose_ml_collection|url|license|licence|access" projects/glucose/src projects/glucose/protocols
```

## Next Minimal Step

Trace `glucose_ml_collection` to its exact upstream release, commit, file list,
underlying dataset components, and license or DUA terms. If this cannot be
resolved, remove that source from the canonical manuscript dataset and rebuild
the split artifact from a fully verified source set.

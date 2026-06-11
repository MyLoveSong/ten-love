import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SOURCE_LABEL = "physionet_big_ideas"
SOURCE_METADATA = {
    "title": "BIG IDEAs Lab Glycemic Variability and Wearable Device Data",
    "physionet_version": "1.0.0",
    "access_route": "public PhysioNet dataset",
    "source_url": "https://physionet.org/content/big-ideas-glycemic-wearable/1.0.0/",
    "license": "ODC-By",
    "license_url": "https://physionet.org/content/big-ideas-glycemic-wearable/1.0.0/LICENSE.txt",
}
TIMESTAMP_COLUMN = "Timestamp (YYYY-MM-DDThh:mm:ss)"
EVENT_TYPE_COLUMN = "Event Type"
GLUCOSE_COLUMN = "Glucose Value (mg/dL)"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_timestamp(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("missing timestamp")
    return datetime.fromisoformat(text).isoformat()


def extract_subject_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("Dexcom_"):
        return stem.removeprefix("Dexcom_")
    return path.parent.name


def iter_dexcom_files(raw_root: Path) -> List[Path]:
    return sorted(Path(raw_root).glob("*/Dexcom_*.csv"))


def load_dexcom_records(path: Path) -> List[Dict[str, Any]]:
    subject_id = extract_subject_id(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get(EVENT_TYPE_COLUMN, "")).strip() != "EGV":
                continue
            timestamp = str(row.get(TIMESTAMP_COLUMN, "")).strip()
            glucose = str(row.get(GLUCOSE_COLUMN, "")).strip()
            if not timestamp or not glucose:
                continue
            try:
                glucose_value = float(glucose)
                parsed_timestamp = parse_timestamp(timestamp)
            except ValueError:
                continue
            records.append(
                {
                    "source": SOURCE_LABEL,
                    "patient_id": subject_id,
                    "timestamp": parsed_timestamp,
                    "glucose_mg_dl": glucose_value,
                }
            )
    return records


def summarize_records(records: Iterable[Dict[str, Any]], dexcom_files: Iterable[Path]) -> Dict[str, Any]:
    record_list = list(records)
    subject_counts = Counter(record["patient_id"] for record in record_list)
    file_entries = [
        {
            "relative_path": str(path.parent.name + "/" + path.name),
            "sha256": sha256_file(path),
        }
        for path in dexcom_files
    ]
    timestamps = [record["timestamp"] for record in record_list]
    return {
        "schema_version": "bigideas-glucose-dataset-report/v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source": SOURCE_LABEL,
        "source_metadata": SOURCE_METADATA,
        "raw_file_count": len(file_entries),
        "subject_count": len(subject_counts),
        "record_count": len(record_list),
        "timestamp_min": min(timestamps) if timestamps else None,
        "timestamp_max": max(timestamps) if timestamps else None,
        "subjects": [
            {
                "subject_hash": hashlib.sha256(f"{SOURCE_LABEL}|{subject_id}".encode("utf-8")).hexdigest(),
                "record_count": count,
            }
            for subject_id, count in sorted(subject_counts.items())
        ],
        "raw_files": file_entries,
        "privacy_boundary": "no row-level glucose values are included; raw file paths are public PhysioNet source paths; subject entries use hashes",
    }


def build_bigideas_dataset(
    raw_root: Path,
    output_path: Path,
    report_path: Optional[Path] = None,
    *,
    dataset_label: Optional[str] = None,
) -> Dict[str, Any]:
    dexcom_files = iter_dexcom_files(raw_root)
    if not dexcom_files:
        raise ValueError(f"no Dexcom_*.csv files found under {raw_root}")

    records: List[Dict[str, Any]] = []
    for dexcom_file in dexcom_files:
        records.extend(load_dexcom_records(dexcom_file))
    records.sort(key=lambda record: (record["patient_id"], record["timestamp"]))
    if not records:
        raise ValueError("no usable EGV glucose records found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "bigideas-glucose-records/v1",
                "source": SOURCE_LABEL,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = summarize_records(records, dexcom_files)
    report["dataset_path"] = dataset_label or str(output_path)
    report["dataset_sha256"] = sha256_file(output_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a BigIdeas-only glucose JSON dataset from local PhysioNet Dexcom CSV files.")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dataset-label", default=None)
    args = parser.parse_args()
    build_bigideas_dataset(args.raw_root, args.output, args.report, dataset_label=args.dataset_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

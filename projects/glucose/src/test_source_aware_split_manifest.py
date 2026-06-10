import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from analysis.source_aware_split_manifest import build_split_manifest


def make_records(source, patient_id, count, start):
    rows = []
    for idx in range(count):
        rows.append(
            {
                "source": source,
                "patient_id": patient_id,
                "timestamp": (start + timedelta(minutes=5 * idx)).isoformat(),
                "glucose_mg_dl": 100 + idx,
            }
        )
    return rows


def collect_string_values(value):
    if isinstance(value, dict):
        values = []
        for key, item in value.items():
            values.append(str(key))
            values.extend(collect_string_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(collect_string_values(item))
        return values
    if isinstance(value, str):
        return [value]
    return []


class SourceAwareSplitManifestTest(unittest.TestCase):
    def write_dataset(self, records):
        tmpdir = tempfile.TemporaryDirectory()
        path = Path(tmpdir.name) / "records.json"
        path.write_text(json.dumps({"records": records}), encoding="utf-8")
        self.addCleanup(tmpdir.cleanup)
        return path

    def test_builds_group_disjoint_manifest_without_glucose_values(self):
        start = datetime(2026, 1, 1)
        records = []
        records += make_records("ohio", "001", 20, start)
        records += make_records("ohio", "002", 19, start)
        records += make_records("ml", "001", 18, start)
        records += make_records("ml", "002", 17, start)
        records += make_records("ml", "003", 20, start)

        manifest = build_split_manifest(
            self.write_dataset(records),
            seed=7,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            input_horizon=12,
            output_horizon=6,
            generated_at="2026-01-02T00:00:00Z",
        )

        self.assertEqual(manifest["dataset"]["record_count"], 94)
        self.assertEqual(manifest["split_policy"]["group_key"], "source + patient_id")
        self.assertEqual(manifest["normalization_scope"], "train_only_required")
        self.assertEqual(manifest["partitions"]["train"]["group_count"], 3)
        self.assertEqual(manifest["partitions"]["val"]["group_count"], 1)
        self.assertEqual(manifest["partitions"]["test"]["group_count"], 1)
        self.assertEqual(len(manifest["groups"]), 5)

        manifest_text = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("glucose_mg_dl", manifest_text)
        self.assertNotIn('"100"', manifest_text)
        self.assertNotIn(": 100", manifest_text)
        manifest_values = collect_string_values(manifest)
        self.assertNotIn("patient_id", manifest_values)
        self.assertNotIn("001", manifest_values)
        self.assertNotIn("002", manifest_values)
        self.assertNotIn("003", manifest_values)

    def test_duplicate_source_patient_timestamp_is_blocking(self):
        start = datetime(2026, 1, 1)
        records = make_records("ohio", "001", 20, start)
        records.append(records[0].copy())

        with self.assertRaisesRegex(ValueError, "duplicate source\\+patient\\+timestamp"):
            build_split_manifest(self.write_dataset(records))

    def test_records_canonical_dataset_label_when_provided(self):
        records = make_records("ohio", "001", 20, datetime(2026, 1, 1))

        manifest = build_split_manifest(
            self.write_dataset(records),
            dataset_label="projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json",
        )

        self.assertEqual(
            manifest["dataset"]["path"],
            "projects/glucose/data/cleaned_dataset/public_glucose_preprocessed.json",
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from analysis.source_aware_split_dataset import load_source_aware_split_windows
from analysis.source_aware_split_manifest import build_split_manifest


def make_records(source, patient_id, values, start):
    return [
        {
            "source": source,
            "patient_id": patient_id,
            "timestamp": (start + timedelta(minutes=5 * idx)).isoformat(),
            "glucose_mg_dl": value,
        }
        for idx, value in enumerate(values)
    ]


def flatten(values):
    out = []
    for item in values:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


class SourceAwareSplitDatasetTest(unittest.TestCase):
    def write_dataset_and_manifest(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        dataset_path = root / "records.json"
        manifest_path = root / "split.json"
        start = datetime(2026, 1, 1)

        records = []
        records += make_records("train", "001", [1, 2, 3, 4], start)
        records += make_records("val", "001", [101, 102, 103, 104], start)
        records += make_records("test", "001", [201, 202, 203, 204], start)
        dataset_path.write_text(json.dumps({"records": records}), encoding="utf-8")

        manifest = build_split_manifest(
            dataset_path,
            manifest_path,
            seed=1,
            train_ratio=1,
            val_ratio=1,
            test_ratio=1,
            input_horizon=2,
            output_horizon=1,
            generated_at="2026-01-02T00:00:00Z",
        )
        for group in manifest["groups"]:
            if group["source"] == "train":
                group["partition"] = "train"
            elif group["source"] == "val":
                group["partition"] = "val"
            else:
                group["partition"] = "test"
        manifest["partitions"] = {
            "train": {"group_count": 1, "record_count": 4, "window_count": 2},
            "val": {"group_count": 1, "record_count": 4, "window_count": 2},
            "test": {"group_count": 1, "record_count": 4, "window_count": 2},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return dataset_path, manifest_path

    def test_builds_windows_by_manifest_partition_and_train_only_scaler(self):
        dataset_path, manifest_path = self.write_dataset_and_manifest()

        split = load_source_aware_split_windows(
            dataset_path,
            manifest_path,
            input_horizon=2,
            output_horizon=1,
        )

        self.assertEqual(split["metadata"]["partitions"]["train"]["window_count"], 2)
        self.assertEqual(split["metadata"]["partitions"]["val"]["window_count"], 2)
        self.assertEqual(split["metadata"]["partitions"]["test"]["window_count"], 2)
        self.assertEqual(len(split["sequences"]["train"]), 2)
        self.assertEqual(len(split["sequences"]["train"][0]), 2)
        self.assertEqual(len(split["sequences"]["train"][0][0]), 1)
        self.assertEqual(len(split["targets"]["train"][0]), 1)

        raw_train_inputs = [1, 2, 2, 3]
        raw_mean = sum(raw_train_inputs) / len(raw_train_inputs)
        raw_std = (sum((value - raw_mean) ** 2 for value in raw_train_inputs) / len(raw_train_inputs)) ** 0.5
        train_scaled = flatten(split["sequences"]["train"])
        val_scaled = flatten(split["sequences"]["val"])
        self.assertAlmostEqual(split["scaler"]["mean"], raw_mean)
        self.assertAlmostEqual(split["scaler"]["std"], raw_std)
        self.assertAlmostEqual(sum(train_scaled) / len(train_scaled), 0.0, places=6)
        self.assertGreater(sum(val_scaled) / len(val_scaled), 80.0)
        self.assertEqual(split["metadata"]["normalization_scope"], "train_sequences_only")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from analysis.source_aware_split_manifest import build_split_manifest
from external_validation_and_baselines import run_split_manifest_baselines


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


class SplitManifestBaselineSmokeTest(unittest.TestCase):
    def write_dataset_and_manifest(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        dataset_path = root / "records.json"
        manifest_path = root / "split.json"
        start = datetime(2026, 1, 1)

        records = []
        records += make_records("train", "001", [100, 101, 102, 103, 104, 105], start)
        records += make_records("val", "001", [110, 111, 112, 113, 114, 115], start)
        records += make_records("test", "001", [120, 121, 122, 123, 124, 125], start)
        dataset_path.write_text(json.dumps({"records": records}), encoding="utf-8")

        manifest = build_split_manifest(
            dataset_path,
            seed=1,
            train_ratio=1,
            val_ratio=1,
            test_ratio=1,
            input_horizon=3,
            output_horizon=1,
            generated_at="2026-01-02T00:00:00Z",
        )
        for group in manifest["groups"]:
            group["partition"] = group["source"]
        manifest["partitions"] = {
            "train": {"group_count": 1, "record_count": 6, "window_count": 3},
            "val": {"group_count": 1, "record_count": 6, "window_count": 3},
            "test": {"group_count": 1, "record_count": 6, "window_count": 3},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return dataset_path, manifest_path, root / "outputs"

    def test_runs_selected_smoke_baselines_and_records_subset_scope(self):
        dataset_path, manifest_path, output_dir = self.write_dataset_and_manifest()

        report = run_split_manifest_baselines(
            dataset_path,
            manifest_path,
            output_dir,
            input_horizon=3,
            output_horizon=1,
            models=["persistence", "linear"],
            max_windows_per_split=2,
        )

        self.assertEqual(report["evaluation_scope"], "smoke_subset")
        self.assertEqual(report["max_windows_per_split"], 2)
        self.assertEqual(set(report["test"].keys()), {"persistence", "linear"})
        self.assertIn("mae", report["test"]["persistence"])
        self.assertTrue((output_dir / "split_manifest_baseline_report.json").exists())


if __name__ == "__main__":
    unittest.main()

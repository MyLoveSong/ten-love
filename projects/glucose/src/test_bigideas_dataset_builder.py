import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.bigideas_dataset_builder import build_bigideas_dataset


class BigIdeasDatasetBuilderTest(unittest.TestCase):
    def test_builds_records_and_privacy_preserving_report_from_dexcom_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subject_dir = root / "001"
            subject_dir.mkdir()
            dexcom_path = subject_dir / "Dexcom_001.csv"
            with dexcom_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Index",
                        "Timestamp (YYYY-MM-DDThh:mm:ss)",
                        "Event Type",
                        "Event Subtype",
                        "Patient Info",
                        "Device Info",
                        "Source Device ID",
                        "Glucose Value (mg/dL)",
                    ]
                )
                writer.writerow([1, "", "FirstName", "", "2019", "", "", ""])
                writer.writerow([2, "2020-02-13 17:23:32", "EGV", "", "", "", "iPhone G6", "61.0"])
                writer.writerow([3, "2020-02-13 17:28:32", "EGV", "", "", "", "iPhone G6", "62.0"])
                writer.writerow([4, "2020-02-13 17:33:32", "Calibration", "", "", "", "iPhone G6", "63.0"])

            output_path = root / "cleaned" / "bigideas.json"
            report_path = root / "protocols" / "bigideas_report.json"

            report = build_bigideas_dataset(root, output_path, report_path)
            dataset = json.loads(output_path.read_text(encoding="utf-8"))
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["record_count"], 2)
        self.assertEqual(report["subject_count"], 1)
        self.assertEqual(report["source"], "physionet_big_ideas")
        self.assertEqual(report["source_metadata"]["physionet_version"], "1.0.0")
        self.assertEqual(report["source_metadata"]["license"], "ODC-By")
        self.assertEqual(len(dataset["records"]), 2)
        self.assertEqual(dataset["records"][0]["source"], "physionet_big_ideas")
        self.assertEqual(dataset["records"][0]["patient_id"], "001")
        self.assertEqual(dataset["records"][0]["timestamp"], "2020-02-13T17:23:32")
        self.assertEqual(dataset["records"][0]["glucose_mg_dl"], 61.0)

        report_text = json.dumps(saved_report, sort_keys=True)
        self.assertNotIn("61.0", report_text)
        self.assertNotIn("62.0", report_text)
        self.assertNotIn("patient_id", report_text)

    def test_records_portable_dataset_label_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subject_dir = root / "001"
            subject_dir.mkdir()
            dexcom_path = subject_dir / "Dexcom_001.csv"
            with dexcom_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "Index",
                        "Timestamp (YYYY-MM-DDThh:mm:ss)",
                        "Event Type",
                        "Event Subtype",
                        "Patient Info",
                        "Device Info",
                        "Source Device ID",
                        "Glucose Value (mg/dL)",
                    ]
                )
                writer.writerow([1, "2020-02-13 17:23:32", "EGV", "", "", "", "iPhone G6", "61.0"])

            report = build_bigideas_dataset(
                root,
                root / "bigideas.json",
                dataset_label="projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json",
            )

        self.assertEqual(
            report["dataset_path"],
            "projects/glucose/data/cleaned_dataset/bigideas_glucose_records.json",
        )


if __name__ == "__main__":
    unittest.main()

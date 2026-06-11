import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_glucose_training import train_with_split_manifest


class RunGlucoseTrainingCliTest(unittest.TestCase):
    def test_direct_script_help_loads_split_manifest_options(self):
        result = subprocess.run(
            [sys.executable, "run_glucose_training.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--split_manifest", result.stdout)
        self.assertIn("--split_dataset", result.stdout)
        self.assertIn("--max_windows_per_split", result.stdout)
        self.assertIn("--epochs", result.stdout)
        self.assertIn("--models", result.stdout)
        self.assertIn("--seed", result.stdout)


class DummyEnsemble:
    def get_model_weights(self):
        return {"gluformer": 1.0}


class DummySystem:
    observed_config = None

    def __init__(self, config):
        self.config = config
        self.ensemble = DummyEnsemble()
        DummySystem.observed_config = config

    def _update_model_input_dims(self, input_dim):
        self.input_dim = input_dim

    def create_models(self):
        return None

    def train_models(self, data_dict):
        return {"gluformer": {"total_epochs": self.config["training"]["epochs"]}}

    def create_ensemble(self, data_dict):
        return None

    def setup_monitoring(self, train_sequences):
        return None

    def evaluate_ensemble(self, test_sequences, test_targets):
        return {"overall_mae": 0.0}

    def save_system_state(self):
        return None


class SplitManifestSeedTest(unittest.TestCase):
    def test_split_manifest_training_records_explicit_seed(self):
        split_data = {
            "sequences": {
                "train": np.ones((2, 3, 1), dtype=np.float32),
                "val": np.ones((2, 3, 1), dtype=np.float32),
                "test": np.ones((2, 3, 1), dtype=np.float32),
            },
            "targets": {
                "train": np.ones((2, 1), dtype=np.float32),
                "val": np.ones((2, 1), dtype=np.float32),
                "test": np.ones((2, 1), dtype=np.float32),
            },
            "metadata": {"normalization_scope": "train_sequences_only"},
            "scaler": {"mean": 100.0, "std": 10.0, "scope": "train_sequences_only"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with patch("run_glucose_training.load_source_aware_split_windows", return_value=split_data):
                with patch("run_glucose_training.EnhancedGlucosePredictionSystem", DummySystem):
                    result = train_with_split_manifest(
                        Path("dataset.json"),
                        Path("split.json"),
                        in_len=3,
                        out_len=1,
                        out_dir=out_dir,
                        epochs=1,
                        model_names=["gluformer"],
                        seed=123,
                    )
            saved = json.loads((out_dir / "split_manifest_training_results.json").read_text())

        self.assertEqual(result["seed"]["value"], 123)
        self.assertEqual(result["seed"]["python_random"], 123)
        self.assertEqual(result["seed"]["numpy"], 123)
        self.assertEqual(result["seed"]["torch"], 123)
        self.assertEqual(result["seed"]["python_hash_seed_env"], "123")
        self.assertEqual(result["seed"]["python_hash_seed_scope"], "runtime_environment_record")
        self.assertEqual(result["seed"]["cublas_workspace_config"], ":4096:8")
        self.assertTrue(result["seed"]["deterministic_algorithms"])
        self.assertTrue(result["seed"]["deterministic_warn_only"])
        self.assertEqual(DummySystem.observed_config["seed"]["value"], 123)
        self.assertEqual(saved["seed"]["value"], 123)


if __name__ == "__main__":
    unittest.main()

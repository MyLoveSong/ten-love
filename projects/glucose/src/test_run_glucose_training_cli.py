import subprocess
import sys
import unittest


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


if __name__ == "__main__":
    unittest.main()

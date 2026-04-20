from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder import save_metadata_json, save_spectral_kernel_csv


class OutputTypeTests(unittest.TestCase):
    def test_csv_and_metadata_are_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            json_path = Path(tmpdir) / "sample.json"
            frame = pd.DataFrame({"energy": [0.0, 1.0], "Tz": [0.1, 0.2]})
            save_spectral_kernel_csv(csv_path, frame)
            save_metadata_json(json_path, {"case": "unit_test"})
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())


if __name__ == "__main__":
    unittest.main()

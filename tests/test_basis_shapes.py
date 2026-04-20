from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder import basis_index, describe_basis, make_legacy_source_vector


class BasisShapeTests(unittest.TestCase):
    def test_source_vector_has_expected_shape(self):
        source = make_legacy_source_vector(10)
        self.assertEqual(source.shape, (80,))

    def test_basis_index_stays_inside_flattened_range(self):
        index = basis_index(chain=2, sector=-1, spin="down", site=9, num_sites=10)
        self.assertEqual(index, 79)

    def test_basis_description_mentions_8n(self):
        self.assertIn("8N", describe_basis())


if __name__ == "__main__":
    unittest.main()

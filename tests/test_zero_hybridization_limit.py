from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder import spin_transmission_z_kernel


class ZeroHybridizationLimitTests(unittest.TestCase):
    def test_zero_components_give_zero_spin_kernel(self):
        zeros = np.zeros(5, dtype=complex)
        group_1 = {
            "c1_xi_plus_up": zeros,
            "c1_xi_plus_down": zeros,
            "c1_xi_minus_up": zeros,
            "c1_xi_minus_down": zeros,
        }
        group_2 = {
            "c2_xi_plus_up": zeros,
            "c2_xi_plus_down": zeros,
            "c2_xi_minus_up": zeros,
            "c2_xi_minus_down": zeros,
        }
        kernel = spin_transmission_z_kernel(group_1, group_2)
        self.assertTrue(np.allclose(kernel, 0.0))


if __name__ == "__main__":
    unittest.main()

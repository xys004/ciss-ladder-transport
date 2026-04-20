from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder import LadderParameters, build_effective_operator
from ciss_ladder_transport.config import make_coherent_leads, make_uniform_sample


class GreenDimensionTests(unittest.TestCase):
    def test_effective_operator_is_8n_by_8n(self):
        parameters = LadderParameters(
            num_sites=4,
            gamma_in_chain_1=1.0,
            gamma_in_chain_2=1.0,
            gamma_out_parallel=0.0,
            gamma_out_spin_mixing=1.0,
            lambda_soc_chain_1=0.1,
            lambda_soc_chain_2=0.1,
        )
        leads = make_coherent_leads(4, p=0.0)
        sample = make_uniform_sample(4, eta=0.0)
        operator = build_effective_operator(0.0, parameters, leads, sample, advanced=False)
        self.assertEqual(operator.shape, (32, 32))


if __name__ == "__main__":
    unittest.main()

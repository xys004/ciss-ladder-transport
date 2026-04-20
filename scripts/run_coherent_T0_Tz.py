"""Compute the coherent spectral kernels T^z(E) and T^0(E).

This script reproduces the referee-facing coherent datasets used to discuss:

- Fig. 2: spin-transmission kernel T^z(E)
- Fig. 3: charge-transmission kernel T^0(E)

Physical regime: coherent.
Primary outputs: raw spectral kernels plus metadata.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder import (
    LadderParameters,
    SweepParameters,
    charge_transmission_kernel,
    make_default_coherent_parameters,
    make_legacy_source_vector,
    save_metadata_json,
    save_spectral_kernel_csv,
    spin_transmission_z_kernel,
    sweep_channel_components,
)
from ciss_ladder_transport.config import make_coherent_leads, make_uniform_sample


def _compute_kernel(parameters: LadderParameters, sweep: SweepParameters, kernel_name: str):
    source_vector = make_legacy_source_vector(parameters.num_sites)
    leads = make_coherent_leads(parameters.num_sites, p=0.0)
    sample = make_uniform_sample(parameters.num_sites, eta=0.0)

    group_1_parameters = replace(parameters, gamma_out_parallel=0.0, gamma_out_spin_mixing=1.0)
    energies, group_1 = sweep_channel_components(group_1_parameters, leads, sample, sweep, source_vector)

    if kernel_name == "T0":
        group_2_parameters = replace(parameters, gamma_out_parallel=-0.0, gamma_out_spin_mixing=1.0)
        _, group_2 = sweep_channel_components(group_2_parameters, leads, sample, sweep, source_vector)
        values = charge_transmission_kernel(group_1, group_2).real
    else:
        group_2_parameters = replace(parameters, gamma_out_parallel=-0.0, gamma_out_spin_mixing=-1.0)
        _, group_2 = sweep_channel_components(group_2_parameters, leads, sample, sweep, source_vector)
        values = spin_transmission_z_kernel(group_1, group_2).real

    return pd.DataFrame(
        {
            "energy": energies,
            kernel_name: values,
            "N": parameters.num_sites,
            "eta_d": 0.0,
            "W": 0.0,
            "lambda_soc": parameters.lambda_soc_chain_1,
            "gamma_in": parameters.gamma_in_chain_1,
            "gamma_out": parameters.gamma_out_spin_mixing,
        }
    )


def main() -> int:
    output_dir = REPO_ROOT / "data" / "raw"
    parameters = make_default_coherent_parameters(num_sites=10)
    sweep = SweepParameters(-3.0, 3.0, 901)

    tz_frame = _compute_kernel(parameters, sweep, "Tz")
    t0_frame = _compute_kernel(parameters, sweep, "T0")

    tz_csv = save_spectral_kernel_csv(output_dir / "fig2_Tz_coherent_N10_eta0.csv", tz_frame)
    t0_csv = save_spectral_kernel_csv(output_dir / "fig3_T0_coherent_N10_eta0.csv", t0_frame)

    metadata = {
        "script": "scripts/run_coherent_T0_Tz.py",
        "regime": "coherent",
        "num_sites": parameters.num_sites,
        "sweep": {"energy_min": sweep.energy_min, "energy_max": sweep.energy_max, "num_points": sweep.num_points},
        "outputs": [str(tz_csv), str(t0_csv)],
    }
    save_metadata_json(output_dir / "fig2_fig3_coherent_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

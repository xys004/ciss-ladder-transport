"""Compute a disorder-averaged spectral T^z(E) dataset.

This script corresponds to the Anderson-disorder workflow used to generate
spin-z transport kernels before any later integration step.

Physical regime: disorder averaged.
Primary output: raw T^z(E) dataset plus metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder import (
    SweepParameters,
    average_channel_components,
    build_disorder_realizations,
    make_default_disorder_parameters,
    make_legacy_source_vector,
    save_metadata_json,
    save_spectral_kernel_csv,
    spin_transmission_z_kernel,
)
from ciss_ladder_transport.config import make_spin_resolved_leads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=91)
    parser.add_argument("--realizations", type=int, default=100)
    parser.add_argument("--points", type=int, default=901)
    parser.add_argument("--energy-min", type=float, default=-4.0)
    parser.add_argument("--energy-max", type=float, default=4.0)
    parser.add_argument("--eta", type=float, default=1e-5, help="Small numerical broadening kept from the legacy workflow.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = REPO_ROOT / "data" / "raw"
    parameters = make_default_disorder_parameters(num_sites=args.sites)
    sweep = SweepParameters(args.energy_min, args.energy_max, args.points)
    source_vector = make_legacy_source_vector(args.sites)
    leads = make_spin_resolved_leads(args.sites, p=0.0)
    samples = build_disorder_realizations(args.sites, args.realizations, eta=args.eta, seed=args.seed)

    group_1_parameters = replace(parameters, gamma_out_parallel=0.0, gamma_out_spin_mixing=1.0)
    energies, group_1 = average_channel_components(group_1_parameters, leads, samples, sweep, source_vector)

    group_2_parameters = replace(parameters, gamma_out_parallel=0.0, gamma_out_spin_mixing=-1.0)
    _, group_2 = average_channel_components(group_2_parameters, leads, samples, sweep, source_vector)

    tz_frame = pd.DataFrame(
        {
            "energy": energies,
            "Tz": spin_transmission_z_kernel(group_1, group_2).real,
            "N": args.sites,
            "eta_d": 0.0,
            "W": 1.0,
            "lambda_soc": parameters.lambda_soc_chain_1,
            "gamma_in": parameters.gamma_in_chain_1,
            "gamma_out": parameters.gamma_out_spin_mixing,
        }
    )

    csv_path = save_spectral_kernel_csv(output_dir / f"fig4_Tz_disorder_N{args.sites}.csv", tz_frame)
    metadata = {
        "script": "scripts/run_disorder_scan.py",
        "regime": "disorder_averaged",
        "sites": args.sites,
        "realizations": args.realizations,
        "eta": args.eta,
        "seed": args.seed,
        "output": str(csv_path),
    }
    save_metadata_json(output_dir / f"fig4_Tz_disorder_N{args.sites}.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

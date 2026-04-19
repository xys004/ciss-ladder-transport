"""Run the cleaned Anderson-disorder calculation for the legacy Gz workflow."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder_transport import make_source_vector
from ciss_ladder_transport.config import LadderModel, SweepSpec, make_spin_resolved_leads
from ciss_ladder_transport.greens import average_channels
from ciss_ladder_transport.observables import gz_kernel
from ciss_ladder_transport.randomness import build_disorder_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, default=91, help="Number of sites per chain.")
    parser.add_argument("--realizations", type=int, default=10000, help="Number of disorder realizations.")
    parser.add_argument("--points", type=int, default=901, help="Number of energy points.")
    parser.add_argument("--energy-min", type=float, default=-4.0, help="Energy sweep lower bound.")
    parser.add_argument("--energy-max", type=float, default=4.0, help="Energy sweep upper bound.")
    parser.add_argument("--eta", type=float, default=0.00001, help="Numerical broadening kept from the legacy disorder script.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "disorder",
        help="Directory where the cleaned disorder outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = SweepSpec(args.energy_min, args.energy_max, args.points)
    source_vector = make_source_vector(args.sites)
    leads = make_spin_resolved_leads(args.sites, p=0.0)
    samples = build_disorder_samples(
        num_sites=args.sites,
        count=args.realizations,
        eta=args.eta,
        seed=args.seed,
    )

    base_model = LadderModel(
        num_sites=args.sites,
        gamma_chain_1=1.0,
        gamma_chain_2=1.0,
        gamma_per=0.0,
        gamma_per1=1.0,
        rashba_chain_1=0.1,
        rashba_chain_2=0.1,
        dresselhaus=0.0,
    )

    start_time = time.time()

    group_1_model = replace(base_model, gamma_per=0.0, gamma_per1=1.0)
    energies, group_1_channels = average_channels(
        model=group_1_model,
        leads=leads,
        samples=samples,
        sweep=sweep,
        source_vector=source_vector,
    )

    group_2_model = replace(base_model, gamma_per=0.0, gamma_per1=-1.0)
    _, group_2_channels = average_channels(
        model=group_2_model,
        leads=leads,
        samples=samples,
        sweep=sweep,
        source_vector=source_vector,
    )

    observable = gz_kernel(group_1_channels, group_2_channels)
    runtime_seconds = time.time() - start_time

    dat_path = output_dir / f"trans_SO_disorder_Z_N={args.sites}.dat"
    csv_path = output_dir / f"data_disorder_N{args.sites}.csv"
    runtime_path = output_dir / f"runtime_disorder_N{args.sites}.txt"

    np.savetxt(dat_path, np.column_stack((energies, observable.real)), fmt="%.6e")
    pd.DataFrame({"E": energies, "Gz": observable.real}).to_csv(csv_path, index=False)
    runtime_path.write_text(f"Execution time: {runtime_seconds}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

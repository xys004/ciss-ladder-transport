"""Run the cleaned coherent calculations that were bundled in one legacy script."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder_transport import make_source_vector
from ciss_ladder_transport.config import LadderModel, SweepSpec, make_coherent_leads, make_uniform_sample
from ciss_ladder_transport.greens import sweep_channels
from ciss_ladder_transport.observables import charge_kernel, gx_kernel, gy_kernel, gz_kernel


ObservableFn = Callable[[Dict[str, np.ndarray], Dict[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class CoherentCase:
    name: str
    value_label: str
    output_stem: str
    runtime_stem: str
    num_sites: int
    gamma_chain_1: float
    gamma_chain_2: float
    rashba_chain_1: float
    rashba_chain_2: float
    dresselhaus: float
    eta: float
    sweep: SweepSpec
    group_1_gamma_per_sign: float
    group_1_gamma_per1_sign: float
    group_2_gamma_per_sign: float
    group_2_gamma_per1_sign: float
    observable_fn: ObservableFn


CASES = (
    CoherentCase(
        name="charge",
        value_label="G0",
        output_stem="Salida_cadena_bi_W_SO",
        runtime_stem="archivo_tiempo_cadena_bi_W_SO",
        num_sites=10,
        gamma_chain_1=1.0,
        gamma_chain_2=1.0,
        rashba_chain_1=0.0,
        rashba_chain_2=0.0,
        dresselhaus=0.0,
        eta=0.001,
        sweep=SweepSpec(-3.0, 3.0, 901),
        group_1_gamma_per_sign=1.0,
        group_1_gamma_per1_sign=1.0,
        group_2_gamma_per_sign=-1.0,
        group_2_gamma_per1_sign=1.0,
        observable_fn=charge_kernel,
    ),
    CoherentCase(
        name="gz",
        value_label="Gz",
        output_stem="Salida_cadena_bi_W_SO_Z",
        runtime_stem="archivo_tiempo_cadena_bi_W_SO_Z",
        num_sites=10,
        gamma_chain_1=1.2,
        gamma_chain_2=1.2,
        rashba_chain_1=0.1,
        rashba_chain_2=0.1,
        dresselhaus=0.0,
        eta=0.001,
        sweep=SweepSpec(-3.0, 3.0, 901),
        group_1_gamma_per_sign=1.0,
        group_1_gamma_per1_sign=1.0,
        group_2_gamma_per_sign=-1.0,
        group_2_gamma_per1_sign=-1.0,
        observable_fn=gz_kernel,
    ),
    CoherentCase(
        name="gx",
        value_label="Gx",
        output_stem="Salida_cadena_bi_W_SO_X",
        runtime_stem="archivo_tiempo_cadena_bi_W_SO_X",
        num_sites=28,
        gamma_chain_1=1.0,
        gamma_chain_2=1.0,
        rashba_chain_1=0.1,
        rashba_chain_2=0.1,
        dresselhaus=0.0,
        eta=0.000001,
        sweep=SweepSpec(-3.0, 3.0, 901),
        group_1_gamma_per_sign=1.0,
        group_1_gamma_per1_sign=1.0,
        group_2_gamma_per_sign=-1.0,
        group_2_gamma_per1_sign=-1.0,
        observable_fn=gx_kernel,
    ),
    CoherentCase(
        name="gy",
        value_label="Gy",
        output_stem="Salida_cadena_bi_W_SO_Y",
        runtime_stem="archivo_tiempo_cadena_bi_W_SO_Y",
        num_sites=28,
        gamma_chain_1=1.0,
        gamma_chain_2=1.0,
        rashba_chain_1=0.1,
        rashba_chain_2=0.1,
        dresselhaus=0.0,
        eta=0.000001,
        sweep=SweepSpec(-3.0, 3.0, 901),
        group_1_gamma_per_sign=1.0,
        group_1_gamma_per1_sign=1.0,
        group_2_gamma_per_sign=-1.0,
        group_2_gamma_per1_sign=-1.0,
        observable_fn=gy_kernel,
    ),
)


def save_outputs(energies: np.ndarray, values: np.ndarray, value_label: str, dat_path: Path, csv_path: Path, runtime_path: Path, runtime_seconds: float) -> None:
    """Persist one observable with legacy-compatible filenames."""

    stacked = np.column_stack((energies, values.real))
    np.savetxt(dat_path, stacked, fmt="%.6e")
    pd.DataFrame({"E": energies, value_label: values.real}).to_csv(csv_path, index=False)
    runtime_path.write_text(f"Execution time: {runtime_seconds}\n", encoding="utf-8")


def run_case(case: CoherentCase, output_dir: Path) -> None:
    """Run one coherent observable block."""

    output_dir.mkdir(parents=True, exist_ok=True)
    source_vector = make_source_vector(case.num_sites)
    leads = make_coherent_leads(case.num_sites, p=0.0)
    sample = make_uniform_sample(case.num_sites, eta=case.eta)

    base_model = LadderModel(
        num_sites=case.num_sites,
        gamma_chain_1=case.gamma_chain_1,
        gamma_chain_2=case.gamma_chain_2,
        gamma_per=0.0,
        gamma_per1=1.0,
        rashba_chain_1=case.rashba_chain_1,
        rashba_chain_2=case.rashba_chain_2,
        dresselhaus=case.dresselhaus,
        use_legacy_advanced_chain2_backward_phase=True,
    )

    start_time = time.time()

    group_1_model = replace(
        base_model,
        gamma_per=case.group_1_gamma_per_sign * 0.0,
        gamma_per1=case.group_1_gamma_per1_sign * 1.0,
    )
    energies, group_1_channels = sweep_channels(
        model=group_1_model,
        leads=leads,
        sample=sample,
        sweep=case.sweep,
        source_vector=source_vector,
    )

    group_2_model = replace(
        base_model,
        gamma_per=case.group_2_gamma_per_sign * 0.0,
        gamma_per1=case.group_2_gamma_per1_sign * 1.0,
    )
    _, group_2_channels = sweep_channels(
        model=group_2_model,
        leads=leads,
        sample=sample,
        sweep=case.sweep,
        source_vector=source_vector,
    )

    observable = case.observable_fn(group_1_channels, group_2_channels)
    runtime_seconds = time.time() - start_time

    save_outputs(
        energies=energies,
        values=observable,
        value_label=case.value_label,
        dat_path=output_dir / f"{case.output_stem}.dat",
        csv_path=output_dir / f"{case.output_stem}.csv",
        runtime_path=output_dir / f"{case.runtime_stem}.txt",
        runtime_seconds=runtime_seconds,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[case.name for case in CASES] + ["all"],
        default="all",
        help="Select one observable block or run every coherent block.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "coherent",
        help="Directory where the cleaned coherent outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_cases = CASES if args.case == "all" else tuple(case for case in CASES if case.name == args.case)
    for case in selected_cases:
        run_case(case, args.output_dir)


if __name__ == "__main__":
    main()

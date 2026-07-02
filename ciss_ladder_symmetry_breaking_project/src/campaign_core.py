from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import simpson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ciss_ladder import (  # noqa: E402
    LadderParameters,
    SweepParameters,
    charge_transmission_kernel,
    make_legacy_source_vector,
    spin_transmission_z_kernel,
    sweep_channel_components,
)
from ciss_ladder_transport.config import (  # noqa: E402
    LeadProfiles,
    _edge_profile,
    make_spin_resolved_leads,
    make_uniform_sample,
)


DEFAULT_E_MIN = -4.0
DEFAULT_E_MAX = 4.0
DEFAULT_EF = 0.0
DEFAULT_BIAS = 4.0
DISCOVERY_POINTS = 51
FIGURE_POINTS = 201
CONTROL_POINTS = 101
TINY = 1e-12


def default_workers() -> int:
    cpu_count = os.cpu_count() or 2
    return min(8, max(2, cpu_count // 2))


def ensure_layout() -> None:
    for rel_path in [
        "data",
        "tables",
        "reports",
        "figs/paper",
        "figs/presentation",
        "logs",
        "notebooks",
    ]:
        (PROJECT_ROOT / rel_path).mkdir(parents=True, exist_ok=True)


def format_float(value: float) -> str:
    return f"{value:g}"


def make_scalar_contact_asymmetry_leads(num_sites: int, alpha: float) -> LeadProfiles:
    left_value = 1.0j
    right_value = alpha * 1.0j
    gamma_profile = _edge_profile(num_sites, left_value, right_value)
    return LeadProfiles(
        chain1_block_gammas=(gamma_profile,) * 4,
        chain2_block_gammas=(gamma_profile,) * 4,
    )


def choose_leads(num_sites: int, alpha: float = 1.0, p: float = 0.0) -> LeadProfiles:
    if abs(p) > 0.0:
        return make_spin_resolved_leads(num_sites, p=p)
    if not math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-14):
        return make_scalar_contact_asymmetry_leads(num_sites, alpha=alpha)
    return make_spin_resolved_leads(num_sites, p=0.0)


def compute_spectra(
    N: int,
    eta_d: float,
    lambda_soc: float,
    gamma_hyb: float,
    Delta: float = 0.0,
    alpha: float = 1.0,
    p: float = 0.0,
    E_min: float = DEFAULT_E_MIN,
    E_max: float = DEFAULT_E_MAX,
    num_points: int = DISCOVERY_POINTS,
) -> pd.DataFrame:
    parameters = LadderParameters(
        num_sites=N,
        gamma_in_chain_1=1.0,
        gamma_in_chain_2=1.0,
        gamma_out_parallel=0.0,
        gamma_out_spin_mixing=gamma_hyb,
        lambda_soc_chain_1=lambda_soc,
        lambda_soc_chain_2=lambda_soc,
        dresselhaus=0.0,
        beta=np.pi,
        phase_site_count=10,
        use_legacy_advanced_chain2_backward_phase=False,
    )
    sweep = SweepParameters(E_min, E_max, num_points)
    sample = make_uniform_sample(
        N,
        onsite_1=+Delta / 2.0,
        onsite_2=-Delta / 2.0,
        eta=eta_d,
    )
    leads = choose_leads(N, alpha=alpha, p=p)
    source_vector = make_legacy_source_vector(N)

    group_1_parameters = replace(parameters, gamma_out_parallel=0.0, gamma_out_spin_mixing=gamma_hyb)
    energies, group_1 = sweep_channel_components(group_1_parameters, leads, sample, sweep, source_vector)

    group_2_charge_parameters = replace(parameters, gamma_out_parallel=0.0, gamma_out_spin_mixing=gamma_hyb)
    _, group_2_charge = sweep_channel_components(group_2_charge_parameters, leads, sample, sweep, source_vector)

    group_2_spin_parameters = replace(parameters, gamma_out_parallel=0.0, gamma_out_spin_mixing=-gamma_hyb)
    _, group_2_spin = sweep_channel_components(group_2_spin_parameters, leads, sample, sweep, source_vector)

    return pd.DataFrame(
        {
            "E": energies,
            "T0": charge_transmission_kernel(group_1, group_2_charge).real,
            "Tz": spin_transmission_z_kernel(group_1, group_2_spin).real,
        }
    )


def extract_window(energy: np.ndarray, values: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    if upper <= lower:
        return np.array([], dtype=float), np.array([], dtype=float)

    interior_mask = (energy > lower) & (energy < upper)
    x_values = np.concatenate(([lower], energy[interior_mask], [upper]))
    y_values = np.interp(x_values, energy, values)
    return x_values, y_values


def integrate_landauer(energy: np.ndarray, values: np.ndarray, EF: float, V: float) -> tuple[float, float]:
    mu_left = EF + V / 2.0
    mu_right = EF - V / 2.0
    window_energy, window_values = extract_window(energy, values, mu_right, mu_left)
    if len(window_energy) < 2:
        return 0.0, 0.0

    simpson_value = float(simpson(window_values, x=window_energy))
    trapz_value = float(np.trapz(window_values, x=window_energy))
    return simpson_value, abs(simpson_value - trapz_value)


def even_odd_decomposition(energy: np.ndarray, values: np.ndarray, EF: float, num_samples: int = 801) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_eps = min(float(energy.max() - EF), float(EF - energy.min()))
    if max_eps <= 0.0:
        max_eps = float((energy.max() - energy.min()) / 2.0)

    eps_values = np.linspace(0.0, max_eps, num_samples)
    plus_values = np.interp(EF + eps_values, energy, values)
    minus_values = np.interp(EF - eps_values, energy, values)
    even_values = 0.5 * (plus_values + minus_values)
    odd_values = 0.5 * (plus_values - minus_values)
    return eps_values, even_values, odd_values


def count_sign_changes(values: np.ndarray, tolerance: float = 1e-12) -> int:
    cleaned = np.where(np.abs(values) < tolerance, 0.0, values)
    signs = np.sign(cleaned)
    nonzero_signs = signs[signs != 0.0]
    if len(nonzero_signs) < 2:
        return 0
    return int(np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1]))


def compute_metrics(frame: pd.DataFrame, EF: float = DEFAULT_EF, V: float = DEFAULT_BIAS) -> dict[str, float]:
    energy = frame["E"].to_numpy(dtype=float)
    transmission_charge = frame["T0"].to_numpy(dtype=float)
    transmission_spin = frame["Tz"].to_numpy(dtype=float)

    eps_values, even_values, odd_values = even_odd_decomposition(energy, transmission_spin, EF=EF)
    charge_current, charge_error = integrate_landauer(energy, transmission_charge, EF=EF, V=V)
    spin_current, spin_error = integrate_landauer(energy, transmission_spin, EF=EF, V=V)

    even_weight = float(simpson(np.abs(even_values), x=eps_values))
    odd_weight = float(simpson(np.abs(odd_values), x=eps_values))
    polarization = spin_current / charge_current if abs(charge_current) > 1e-15 else 0.0

    return {
        "Wz": float(simpson(np.abs(transmission_spin), x=energy)),
        "max_abs_Tz": float(np.max(np.abs(transmission_spin))),
        "sign_changes_Tz": count_sign_changes(transmission_spin),
        "even_weight": even_weight,
        "odd_weight": odd_weight,
        "R_even": even_weight / max(odd_weight, TINY),
        "I_charge": charge_current,
        "I_spin": spin_current,
        "polarization": polarization,
        "integration_error_charge": charge_error,
        "integration_error_spin": spin_error,
    }


def task_output_dir(task: dict[str, float | int | str]) -> Path:
    mechanism = str(task["mechanism"])
    stage = str(task.get("stage", "survey"))
    N = int(task["N"])
    eta_d = format_float(float(task["eta_d"]))
    lambda_soc = format_float(float(task["lambda_soc"]))
    gamma_hyb = format_float(float(task["gamma_hyb"]))
    Delta = format_float(float(task["Delta"]))
    alpha = format_float(float(task["alpha"]))
    p = format_float(float(task["p"]))

    if mechanism == "baseline":
        return PROJECT_ROOT / "data" / "baseline" / "spectra" / f"N_{N}" / f"eta_{eta_d}" / f"lambda_{lambda_soc}" / f"ghyb_{gamma_hyb}"
    if mechanism == "detuning":
        root = "detuning_refined" if stage == "refined" else "detuning_discovery"
        return PROJECT_ROOT / "data" / root / f"N_{N}" / f"eta_{eta_d}" / f"Delta_{Delta}"
    if mechanism == "contact_asymmetry":
        return PROJECT_ROOT / "data" / "contact_asymmetry" / "spectra" / f"N_{N}" / f"eta_{eta_d}" / f"alpha_{alpha}"
    if mechanism == "spin_active_contact":
        return PROJECT_ROOT / "data" / "spin_active" / f"N_{N}" / f"eta_{eta_d}" / f"p_{p}"
    if mechanism == "bias_gate":
        parent = "detuning" if abs(float(task["Delta"])) > 0.0 else "contact_asymmetry"
        knob = f"Delta_{Delta}" if parent == "detuning" else f"alpha_{alpha}"
        return PROJECT_ROOT / "data" / "bias_gate" / parent / f"N_{N}" / knob / f"eta_{eta_d}" / f"EF_{format_float(float(task['EF']))}" / f"V_{format_float(float(task['V']))}"
    if mechanism == "control":
        control_name = str(task["control_name"])
        return PROJECT_ROOT / "data" / "controls" / control_name / f"N_{N}"
    return PROJECT_ROOT / "data" / "misc" / mechanism / f"N_{N}"


def task_row(task: dict[str, float | int | str], metrics: dict[str, float], spectra_path: Path, integrated_path: Path) -> dict[str, float | int | str]:
    row = {
        "mechanism": task["mechanism"],
        "stage": task.get("stage", "survey"),
        "control_name": task.get("control_name", ""),
        "N": int(task["N"]),
        "eta_d": float(task["eta_d"]),
        "lambda_soc": float(task["lambda_soc"]),
        "gamma_hyb": float(task["gamma_hyb"]),
        "Delta": float(task["Delta"]),
        "alpha": float(task["alpha"]),
        "p": float(task["p"]),
        "EF": float(task["EF"]),
        "V": float(task["V"]),
        "num_points": int(task["num_points"]),
        "spectra_path": str(spectra_path),
        "integrated_path": str(integrated_path),
    }
    row.update(metrics)
    return row


def run_case(task: dict[str, float | int | str]) -> dict[str, float | int | str]:
    frame = compute_spectra(
        N=int(task["N"]),
        eta_d=float(task["eta_d"]),
        lambda_soc=float(task["lambda_soc"]),
        gamma_hyb=float(task["gamma_hyb"]),
        Delta=float(task["Delta"]),
        alpha=float(task["alpha"]),
        p=float(task["p"]),
        E_min=float(task.get("E_min", DEFAULT_E_MIN)),
        E_max=float(task.get("E_max", DEFAULT_E_MAX)),
        num_points=int(task["num_points"]),
    )
    metrics = compute_metrics(frame, EF=float(task["EF"]), V=float(task["V"]))

    output_dir = task_output_dir(task)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = output_dir / "spectra.csv"
    integrated_path = output_dir / "integrated.csv"

    frame.to_csv(spectra_path, index=False)
    pd.DataFrame([task_row(task, metrics, spectra_path, integrated_path)]).to_csv(integrated_path, index=False)
    return task_row(task, metrics, spectra_path, integrated_path)


def run_task_grid(tasks: Iterable[dict[str, float | int | str]], max_workers: int | None = None, label: str = "tasks") -> pd.DataFrame:
    task_list = list(tasks)
    if not task_list:
        return pd.DataFrame()

    workers = max_workers or default_workers()
    print(f"Running {len(task_list)} {label} with {workers} workers...")
    rows: list[dict[str, float | int | str]] = []
    completed = 0

    executor_cls = ProcessPoolExecutor
    executor_note = "processes"
    try:
        executor = executor_cls(max_workers=workers)
    except PermissionError:
        executor_cls = ThreadPoolExecutor
        executor_note = "threads"
        executor = executor_cls(max_workers=workers)

    print(f"Executor mode for {label}: {executor_note}")
    with executor:
        futures = [executor.submit(run_case, task) for task in task_list]
        for future in as_completed(futures):
            rows.append(future.result())
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(task_list):
                print(f"Completed {completed}/{len(task_list)} {label}")

    return pd.DataFrame(rows)


def build_task(
    *,
    mechanism: str,
    N: int,
    eta_d: float,
    lambda_soc: float,
    gamma_hyb: float,
    Delta: float = 0.0,
    alpha: float = 1.0,
    p: float = 0.0,
    EF: float = DEFAULT_EF,
    V: float = DEFAULT_BIAS,
    num_points: int = DISCOVERY_POINTS,
    stage: str = "survey",
    control_name: str = "",
) -> dict[str, float | int | str]:
    return {
        "mechanism": mechanism,
        "stage": stage,
        "control_name": control_name,
        "N": N,
        "eta_d": eta_d,
        "lambda_soc": lambda_soc,
        "gamma_hyb": gamma_hyb,
        "Delta": Delta,
        "alpha": alpha,
        "p": p,
        "EF": EF,
        "V": V,
        "E_min": DEFAULT_E_MIN,
        "E_max": DEFAULT_E_MAX,
        "num_points": num_points,
    }


def baseline_tasks(num_points: int = DISCOVERY_POINTS) -> list[dict[str, float | int | str]]:
    tasks = []
    for N in [10, 19, 28, 37, 46, 55, 64, 73, 82, 91]:
        for eta_d in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
            for lambda_soc in [0.05, 0.1, 0.2]:
                for gamma_hyb in [0.25, 0.5, 0.75, 1.0]:
                    tasks.append(
                        build_task(
                            mechanism="baseline",
                            N=N,
                            eta_d=eta_d,
                            lambda_soc=lambda_soc,
                            gamma_hyb=gamma_hyb,
                            num_points=num_points,
                        )
                    )
    return tasks


def baseline_discovery_tasks(num_points: int = DISCOVERY_POINTS) -> list[dict[str, float | int | str]]:
    return [task for task in baseline_tasks(num_points=num_points) if int(task["N"]) in {10, 37, 91}]


def detuning_discovery_tasks(num_points: int = DISCOVERY_POINTS) -> list[dict[str, float | int | str]]:
    tasks = []
    for N in [10, 37, 91]:
        for eta_d in [0.0, 0.1, 0.25, 0.5, 1.0]:
            for Delta in [0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
                tasks.append(
                    build_task(
                        mechanism="detuning",
                        stage="discovery",
                        N=N,
                        eta_d=eta_d,
                        lambda_soc=0.1,
                        gamma_hyb=1.0,
                        Delta=Delta,
                        num_points=num_points,
                    )
                )
    return tasks


def contact_asymmetry_tasks(num_points: int = DISCOVERY_POINTS) -> list[dict[str, float | int | str]]:
    tasks = []
    for N in [10, 37, 91]:
        for eta_d in [0.0, 0.1, 0.25, 0.5, 1.0]:
            for alpha in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]:
                tasks.append(
                    build_task(
                        mechanism="contact_asymmetry",
                        stage="discovery",
                        N=N,
                        eta_d=eta_d,
                        lambda_soc=0.1,
                        gamma_hyb=1.0,
                        alpha=alpha,
                        num_points=num_points,
                    )
                )
    return tasks


def spin_active_tasks(num_points: int = DISCOVERY_POINTS) -> list[dict[str, float | int | str]]:
    tasks = []
    for N in [10, 37, 91]:
        for eta_d in [0.0, 0.1, 0.25, 0.5, 1.0]:
            for p in [0.0, 0.1, 0.25, 0.5]:
                tasks.append(
                    build_task(
                        mechanism="spin_active_contact",
                        stage="benchmark",
                        N=N,
                        eta_d=eta_d,
                        lambda_soc=0.1,
                        gamma_hyb=1.0,
                        p=p,
                        num_points=num_points,
                    )
                )
    return tasks


def controls_tasks(num_points: int = CONTROL_POINTS) -> list[dict[str, float | int | str]]:
    tasks = []
    for N in [10, 37]:
        tasks.append(
            build_task(
                mechanism="control",
                control_name="lambda_soc_zero",
                N=N,
                eta_d=0.5,
                lambda_soc=0.0,
                gamma_hyb=1.0,
                num_points=num_points,
            )
        )
        tasks.append(
            build_task(
                mechanism="control",
                control_name="gamma_hyb_zero",
                N=N,
                eta_d=0.5,
                lambda_soc=0.1,
                gamma_hyb=0.0,
                num_points=num_points,
            )
        )
        tasks.append(
            build_task(
                mechanism="control",
                control_name="symmetric_eta_zero",
                N=N,
                eta_d=0.0,
                lambda_soc=0.1,
                gamma_hyb=1.0,
                num_points=num_points,
            )
        )
    return tasks


def nearest_window(grid: Iterable[float], center: float, half_width: int = 2) -> list[float]:
    ordered = sorted(set(float(value) for value in grid))
    best_index = min(range(len(ordered)), key=lambda idx: abs(ordered[idx] - center))
    start = max(0, best_index - half_width)
    end = min(len(ordered), best_index + half_width + 1)
    return ordered[start:end]


def detuning_refined_tasks(best_row: pd.Series, num_points: int = DISCOVERY_POINTS) -> list[dict[str, float | int | str]]:
    eta_grid = nearest_window([0.0, 0.05, 0.1, 0.175, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5], float(best_row["eta_d"]))
    delta_grid = nearest_window([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0], float(best_row["Delta"]))
    tasks = []
    for N in [10, 19, 28, 37, 46, 55, 64, 73, 82, 91]:
        for eta_d in eta_grid:
            for Delta in delta_grid:
                tasks.append(
                    build_task(
                        mechanism="detuning",
                        stage="refined",
                        N=N,
                        eta_d=eta_d,
                        lambda_soc=0.1,
                        gamma_hyb=1.0,
                        Delta=Delta,
                        num_points=num_points,
                    )
                )
    return tasks


def bias_gate_tasks(best_cases: list[dict[str, float | int | str]], num_points: int = CONTROL_POINTS) -> list[dict[str, float | int | str]]:
    eta_master = [0.0, 0.05, 0.1, 0.175, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5]
    tasks = []
    for case in best_cases:
        eta_grid = nearest_window(eta_master, float(case["eta_d"]))
        for eta_d in eta_grid:
            for EF in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                for V in [0.5, 1.0, 2.0, 4.0]:
                    tasks.append(
                        build_task(
                            mechanism="bias_gate",
                            stage=str(case["mechanism"]),
                            N=int(case["N"]),
                            eta_d=eta_d,
                            lambda_soc=float(case["lambda_soc"]),
                            gamma_hyb=float(case["gamma_hyb"]),
                            Delta=float(case.get("Delta", 0.0)),
                            alpha=float(case.get("alpha", 1.0)),
                            p=float(case.get("p", 0.0)),
                            EF=EF,
                            V=V,
                            num_points=num_points,
                        )
                    )
    return tasks


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_contact_audit() -> tuple[Path, Path]:
    ensure_layout()
    report_path = PROJECT_ROOT / "reports" / "00_contact_audit.md"
    csv_path = PROJECT_ROOT / "tables" / "contact_audit_summary.csv"

    content = """# Contact Audit

This short audit verifies the baseline contact implementation before any new numerical campaign is run.

## Verified statements

1. In the legacy stochastic scripts, the physical contact broadening enters through `Gamma1`, `Gamma2`, `Gamma3`, and `Gamma4`.
2. With `p = 0`, the code sets `val = i (1 + p)` and `val1 = i (1 - p)`, so `val = val1 = i`; therefore no effective spin-dependent contact asymmetry is active in the baseline case.
3. The quantities later called `gamma_out_parallel` / `gamma_out_spin_mixing` map onto `gamma_per` / `gamma_per1`, i.e. inter-chain hybridization knobs, not lead asymmetry knobs.
4. The new paper campaign must therefore introduce symmetry breaking explicitly and separately from the baseline lead definition.

## Implementation choice for this workspace

- Baseline and detuning runs use the explicit `Gamma1..Gamma4` contact representation with `p = 0`.
- Scalar contact asymmetry is implemented as a separate left-right lead broadening ratio `alpha = Gamma_R / Gamma_L`, equal for all four channel blocks.
- Spin-active contacts are benchmarked separately through `p`, without mixing them into the scalar asymmetry phase.
- All integrated observables use the corrected Landauer window `mu_L = EF + V/2`, `mu_R = EF - V/2`.
"""
    report_path.write_text(content, encoding="utf-8")

    pd.DataFrame(
        [
            {
                "claim": "Contacts enter through Gamma1..Gamma4",
                "status": "verified",
                "evidence": "legacy transport scripts build four explicit Gamma arrays at the contacts",
            },
            {
                "claim": "p=0 implies val=val1 and no active contact asymmetry",
                "status": "verified",
                "evidence": "val=i(1+p), val1=i(1-p), so both equal i at p=0",
            },
            {
                "claim": "gamma_out knobs are inter-chain hybridization, not lead asymmetry",
                "status": "verified",
                "evidence": "referee-facing wrapper maps gamma_out_parallel/spin_mixing to gamma_per/gamma_per1",
            },
            {
                "claim": "New paper needs explicit symmetry breaking",
                "status": "actioned",
                "evidence": "workspace separates baseline, detuning, scalar contact asymmetry, and spin-active contacts",
            },
        ]
    ).to_csv(csv_path, index=False)
    return report_path, csv_path

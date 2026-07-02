from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_core import DISCOVERY_POINTS, detuning_discovery_tasks, detuning_refined_tasks, ensure_layout, load_table, run_task_grid


TABLE_COLUMNS = [
    "mechanism",
    "stage",
    "N",
    "eta_d",
    "lambda_soc",
    "gamma_hyb",
    "Delta",
    "alpha",
    "p",
    "EF",
    "V",
    "Wz",
    "max_abs_Tz",
    "sign_changes_Tz",
    "odd_weight",
    "even_weight",
    "R_even",
    "I_charge",
    "I_spin",
    "polarization",
    "integration_error_charge",
    "integration_error_spin",
    "num_points",
    "spectra_path",
    "integrated_path",
]


def significant_cases(frame: pd.DataFrame, baseline_noise: float) -> pd.DataFrame:
    threshold = max(10.0 * baseline_noise, 1e-4)
    return frame[(frame["R_even"] > 0.02) & (frame["I_spin"].abs() > threshold)].copy()


def write_report(frame: pd.DataFrame, baseline_noise: float, report_path: Path) -> None:
    discovery = frame[frame["stage"] == "discovery"].copy()
    refined = frame[frame["stage"] == "refined"].copy()
    promising = significant_cases(frame, baseline_noise).sort_values("I_spin", key=lambda series: series.abs(), ascending=False)
    rho_table = (
        discovery.groupby(["N", "eta_d"], as_index=False)
        .apply(lambda block: pd.Series({"rho_delta_Reven": block["Delta"].corr(block["R_even"], method="spearman")}))
        .reset_index(drop=True)
    )
    summary = discovery.groupby("Delta", as_index=False).agg(
        median_R_even=("R_even", "median"),
        median_abs_I_spin=("I_spin", lambda series: float(series.abs().median())),
        max_abs_I_spin=("I_spin", lambda series: float(series.abs().max())),
    )
    report = f"""# Detuning Report

This phase scans the minimal structural detuning knob `epsilon_A = +Delta/2`, `epsilon_B = -Delta/2` with symmetric contacts.

## Discovery grid

- `N`: 10, 37, 91
- `eta_d`: 0.0, 0.1, 0.25, 0.5, 1.0
- `Delta`: 0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0 meV
- `lambda_soc = 0.1`, `gamma_hyb = 1.0`

## Refined grid

{("Refinement was triggered around the strongest discovery candidate." if not refined.empty else "No refinement was triggered because discovery did not clear the significance threshold.")}

## Summary over discovery data

{summary.to_markdown(index=False)}

Median Spearman correlation between `Delta` and `R_even` across discovery slices: `{rho_table["rho_delta_Reven"].median():.3f}`.

Baseline numerical noise floor from the symmetric phase: `{baseline_noise:.6e}` in `|I_spin|`.

## Supported statements

- `R_even` grows systematically with detuning if and only if the discovery table shows values above the baseline noise threshold.
- A finite net response is counted as real only when `|I_spin|` exceeds `max(10 x baseline_noise, 1e-4)` and `R_even > 0.02`.
- Moderate dephasing is only described as an enhancement when the finite response beats the `eta_d = 0` case within the same `N, Delta` family.

## Promising cases

{(promising.head(12).to_markdown(index=False) if not promising.empty else "No case exceeded the significance threshold.")}"""
    report_path.write_text(report, encoding="utf-8")


def main(max_workers: int | None = None) -> pd.DataFrame:
    ensure_layout()
    baseline_table = load_table(Path(__file__).resolve().parents[1] / "tables" / "baseline_spectral_metrics.csv")
    baseline_noise = float(baseline_table["I_spin"].abs().max()) if not baseline_table.empty else 1e-6

    discovery = run_task_grid(detuning_discovery_tasks(num_points=DISCOVERY_POINTS), max_workers=max_workers, label="detuning discovery cases")
    discovery = discovery.sort_values(["stage", "N", "eta_d", "Delta"]).reset_index(drop=True)
    promising = significant_cases(discovery, baseline_noise)

    refined = pd.DataFrame()
    if not promising.empty:
        best = promising.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).iloc[0]
        refined = run_task_grid(detuning_refined_tasks(best, num_points=DISCOVERY_POINTS), max_workers=max_workers, label="detuning refined cases")

    results = pd.concat([discovery, refined], ignore_index=True) if not refined.empty else discovery
    results = results.sort_values(["stage", "N", "eta_d", "Delta"]).reset_index(drop=True)
    results = results[TABLE_COLUMNS]

    table_path = Path(__file__).resolve().parents[1] / "tables" / "detuning_phase_map.csv"
    report_path = Path(__file__).resolve().parents[1] / "reports" / "02_detuning_report.md"
    results.to_csv(table_path, index=False)
    write_report(results, baseline_noise=baseline_noise, report_path=report_path)
    return results


if __name__ == "__main__":
    main()

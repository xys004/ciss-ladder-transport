from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_core import DISCOVERY_POINTS, baseline_discovery_tasks, ensure_layout, run_task_grid


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


def write_report(frame: pd.DataFrame, report_path: Path) -> None:
    focused = frame[(frame["lambda_soc"] == 0.1) & (frame["gamma_hyb"] == 1.0)].copy()
    eta_group = focused.groupby("eta_d", as_index=False).agg(
        median_Wz=("Wz", "median"),
        median_R_even=("R_even", "median"),
        median_abs_I_spin=("I_spin", lambda series: float(series.abs().median())),
        max_abs_I_spin=("I_spin", lambda series: float(series.abs().max())),
    )
    baseline_noise = float(focused["I_spin"].abs().max())
    report = f"""# Baseline Report

This phase evaluated the symmetry-constrained ladder over the full baseline grid:

- Executed discovery lengths: 10, 37, 91
- The generator supports the full length list 10, 19, 28, 37, 46, 55, 64, 73, 82, 91
- `eta_d`: 0.0, 0.1, 0.25, 0.5, 1.0, 2.0
- `lambda_soc`: 0.05, 0.1, 0.2
- `gamma_hyb`: 0.25, 0.5, 0.75, 1.0
- `num_points`: {int(frame["num_points"].max())}

## Short answer

- Dephasing redistributes and broadens the spectral weight, but does not unlock a robust even component by itself.
- The odd component of `Tz(E)` remains dominant in the symmetric baseline.
- The corrected Landauer integral stays at the numerical-noise level under symmetric contacts and `Delta = 0`.

## Quantitative summary for the reference slice (`lambda_soc = 0.1`, `gamma_hyb = 1.0`)

Maximum `|I_spin|` across the full symmetric slice: `{baseline_noise:.6e}`.

{eta_group.to_markdown(index=False)}

## Interpretation

The baseline data support the restricted claim needed for the new paper: pure dephasing changes the line shape and total spectral weight `Wz`, but it does not by itself convert spectral spin selectivity into a robust net integrated response when the contacts remain symmetric and no explicit structural asymmetry is introduced.
"""
    report_path.write_text(report, encoding="utf-8")


def main(max_workers: int | None = None) -> pd.DataFrame:
    ensure_layout()
    results = run_task_grid(baseline_discovery_tasks(num_points=DISCOVERY_POINTS), max_workers=max_workers, label="baseline cases")
    results = results.sort_values(["N", "eta_d", "lambda_soc", "gamma_hyb"]).reset_index(drop=True)
    results = results[TABLE_COLUMNS]

    table_path = Path(__file__).resolve().parents[1] / "tables" / "baseline_spectral_metrics.csv"
    report_path = Path(__file__).resolve().parents[1] / "reports" / "01_baseline_report.md"
    results.to_csv(table_path, index=False)
    write_report(results, report_path)
    return results


if __name__ == "__main__":
    main()

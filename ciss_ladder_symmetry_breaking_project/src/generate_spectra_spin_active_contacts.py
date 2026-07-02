from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_core import DISCOVERY_POINTS, ensure_layout, run_task_grid, spin_active_tasks


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
    summary = frame.groupby("p", as_index=False).agg(
        median_R_even=("R_even", "median"),
        median_abs_I_spin=("I_spin", lambda series: float(series.abs().median())),
        max_abs_I_spin=("I_spin", lambda series: float(series.abs().max())),
    )
    report = f"""# Spin-Active Contacts Benchmark Report

This optional benchmark keeps the scalar contact asymmetry phase separate and activates the legacy spin-selective lead parameter `p`.

## How `p` enters

- `val = i (1 + p)`
- `val1 = i (1 - p)`
- `Gamma1 = (val, val)`, `Gamma2 = (val, val1)`, `Gamma3 = (val1, val)`, `Gamma4 = (val1, val1)` at the two ends

## Survey summary

{summary.to_markdown(index=False)}

This phase is a benchmark only. Any finite response here is expected because the interface itself becomes spin active, so it should not be confused with the cleaner scalar contact asymmetry mechanism.
"""
    report_path.write_text(report, encoding="utf-8")


def main(max_workers: int | None = None) -> pd.DataFrame:
    ensure_layout()
    results = run_task_grid(spin_active_tasks(num_points=DISCOVERY_POINTS), max_workers=max_workers, label="spin-active contact cases")
    results = results.sort_values(["N", "eta_d", "p"]).reset_index(drop=True)
    results = results[TABLE_COLUMNS]

    table_path = Path(__file__).resolve().parents[1] / "tables" / "spin_active_contacts_map.csv"
    report_path = Path(__file__).resolve().parents[1] / "reports" / "04_spin_active_contacts_report.md"
    results.to_csv(table_path, index=False)
    write_report(results, report_path)
    return results


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_core import CONTROL_POINTS, controls_tasks, ensure_layout, load_table, run_task_grid


TABLE_COLUMNS = [
    "mechanism",
    "stage",
    "control_name",
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


def write_report(frame: pd.DataFrame, baseline: pd.DataFrame, report_path: Path) -> None:
    reference = baseline[(baseline["lambda_soc"] == 0.1) & (baseline["gamma_hyb"] == 1.0) & (baseline["eta_d"] == 0.5)][["N", "Wz", "I_spin"]]
    merged = frame.merge(reference, on="N", suffixes=("", "_reference"), how="left")
    merged["Wz_ratio_to_reference"] = merged["Wz"] / merged["Wz_reference"].replace(0.0, pd.NA)
    merged["abs_I_spin_ratio_to_reference"] = merged["I_spin"].abs() / merged["I_spin_reference"].abs().replace(0.0, pd.NA)

    content = f"""# Controls Report

This report collects the mandatory numerical controls at `N = 10` and `N = 37`.

## What was checked

1. `lambda_soc = 0` should collapse or strongly suppress `Tz` and `I_spin`.
2. `gamma_hyb = 0` should collapse or strongly suppress `Tz` and `I_spin`.
3. `eta_d = 0`, symmetric contacts, and `Delta = 0` should give `I_spin ~ 0` with the corrected Landauer window.
4. The integrated observable uses the signed kernel directly; no `abs(Tz)` or `abs(I_spin)` enters the physical current.
5. The even/odd decomposition is performed by symmetric interpolation around `EF`.
6. The integration error estimate is the Simpson-vs-trapezoid difference on the same window.

## Control summary

{merged.to_markdown(index=False)}
"""
    report_path.write_text(content, encoding="utf-8")


def main(max_workers: int | None = None) -> pd.DataFrame:
    ensure_layout()
    baseline = load_table(Path(__file__).resolve().parents[1] / "tables" / "baseline_spectral_metrics.csv")
    results = run_task_grid(controls_tasks(num_points=CONTROL_POINTS), max_workers=max_workers, label="control cases")
    results = results.sort_values(["control_name", "N"]).reset_index(drop=True)
    results = results[TABLE_COLUMNS]

    table_path = Path(__file__).resolve().parents[1] / "tables" / "controls_summary.csv"
    report_path = Path(__file__).resolve().parents[1] / "reports" / "06_controls_report.md"
    results.to_csv(table_path, index=False)
    write_report(results, baseline, report_path)
    return results


if __name__ == "__main__":
    main()

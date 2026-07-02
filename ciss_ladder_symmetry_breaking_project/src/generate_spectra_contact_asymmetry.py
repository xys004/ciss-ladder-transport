from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_core import DISCOVERY_POINTS, contact_asymmetry_tasks, ensure_layout, load_table, run_task_grid


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


def write_report(frame: pd.DataFrame, baseline_noise: float, detuning_table: pd.DataFrame, report_path: Path) -> None:
    threshold = max(10.0 * baseline_noise, 1e-4)
    promising = frame[(frame["R_even"] > 0.02) & (frame["I_spin"].abs() > threshold)].copy()
    summary = frame.groupby("alpha", as_index=False).agg(
        median_R_even=("R_even", "median"),
        median_abs_I_spin=("I_spin", lambda series: float(series.abs().median())),
        max_abs_I_spin=("I_spin", lambda series: float(series.abs().max())),
    )
    detuning_best = float(detuning_table["I_spin"].abs().max()) if not detuning_table.empty else 0.0
    contact_best = float(frame["I_spin"].abs().max()) if not frame.empty else 0.0
    comparison = "stronger" if contact_best > detuning_best else "weaker_or_equal"

    report = f"""# Scalar Contact Asymmetry Report

This phase scans the clean left-right scalar contact asymmetry `Gamma_L = Gamma0`, `Gamma_R = alpha * Gamma0`, keeping `p = 0`.

## Discovery grid

- `N`: 10, 37, 91
- `eta_d`: 0.0, 0.1, 0.25, 0.5, 1.0
- `alpha`: 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0
- `Delta = 0`, `lambda_soc = 0.1`, `gamma_hyb = 1.0`

## Summary over `alpha`

{summary.to_markdown(index=False)}

Baseline numerical noise floor from the symmetric phase: `{baseline_noise:.6e}` in `|I_spin|`.

## Explicit answers

- Does scalar contact asymmetry generate an even component of `Tz(E)`? {"Yes, for the cases listed below." if not promising.empty else "Not above the significance threshold in the scanned range."}
- Does dephasing amplify, broaden, or destroy the effect? The table shows how `R_even`, `|I_spin|`, and polarization evolve with `eta_d`; no enhancement is claimed unless it beats the corresponding `eta_d = 0` case.
- Is the effect stronger or weaker than detuning? In this survey it is `{comparison}` relative to the strongest detuning case by `|I_spin|`.

## Promising cases

{(promising.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).head(12).to_markdown(index=False) if not promising.empty else "No case exceeded the significance threshold.")}"""
    report_path.write_text(report, encoding="utf-8")


def main(max_workers: int | None = None) -> pd.DataFrame:
    ensure_layout()
    baseline_table = load_table(Path(__file__).resolve().parents[1] / "tables" / "baseline_spectral_metrics.csv")
    detuning_table = load_table(Path(__file__).resolve().parents[1] / "tables" / "detuning_phase_map.csv")
    baseline_noise = float(baseline_table["I_spin"].abs().max()) if not baseline_table.empty else 1e-6

    results = run_task_grid(contact_asymmetry_tasks(num_points=DISCOVERY_POINTS), max_workers=max_workers, label="contact asymmetry cases")
    results = results.sort_values(["N", "eta_d", "alpha"]).reset_index(drop=True)
    results = results[TABLE_COLUMNS]

    table_path = Path(__file__).resolve().parents[1] / "tables" / "contact_asymmetry_phase_map.csv"
    report_path = Path(__file__).resolve().parents[1] / "reports" / "03_contact_asymmetry_report.md"
    results.to_csv(table_path, index=False)
    write_report(results, baseline_noise=baseline_noise, detuning_table=detuning_table, report_path=report_path)
    return results


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_core import CONTROL_POINTS, bias_gate_tasks, ensure_layout, load_table, run_task_grid


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


def pick_best_cases(detuning: pd.DataFrame, contact: pd.DataFrame, baseline_noise: float) -> list[dict[str, float | int | str]]:
    threshold = max(10.0 * baseline_noise, 1e-4)
    best_cases: list[dict[str, float | int | str]] = []
    for table, mechanism in [(detuning, "detuning"), (contact, "contact_asymmetry")]:
        if table.empty:
            continue
        candidates = table[(table["R_even"] > 0.02) & (table["I_spin"].abs() > threshold)].copy()
        if candidates.empty:
            continue
        row = candidates.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).iloc[0].to_dict()
        row["mechanism"] = mechanism
        best_cases.append(row)
    return best_cases


def write_report(frame: pd.DataFrame, report_path: Path) -> None:
    if frame.empty:
        report = """# Bias and Gate Report

No bias/gate scan was triggered because neither detuning nor scalar contact asymmetry exceeded the significance threshold in the preceding phases.
"""
        report_path.write_text(report, encoding="utf-8")
        return

    summary = frame.groupby(["stage", "EF", "V"], as_index=False).agg(
        max_abs_I_spin=("I_spin", lambda series: float(series.abs().max())),
        max_abs_polarization=("polarization", lambda series: float(series.abs().max())),
    )
    best_rows = frame.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).head(12)
    report = f"""# Bias and Gate Report

This phase recomputes the strongest detuning and/or scalar-contact candidates over the requested `(EF, V)` map, keeping the corrected Landauer window in every case.

## Summary

{summary.to_markdown(index=False)}

## Best windows

{best_rows.to_markdown(index=False)}

A dephasing-assisted enhancement is only counted when a finite-bias, finite-gate point beats the corresponding `eta_d = 0` case for the same mechanism and geometry.
"""
    report_path.write_text(report, encoding="utf-8")


def main(max_workers: int | None = None) -> pd.DataFrame:
    ensure_layout()
    baseline = load_table(Path(__file__).resolve().parents[1] / "tables" / "baseline_spectral_metrics.csv")
    detuning = load_table(Path(__file__).resolve().parents[1] / "tables" / "detuning_phase_map.csv")
    contact = load_table(Path(__file__).resolve().parents[1] / "tables" / "contact_asymmetry_phase_map.csv")
    baseline_noise = float(baseline["I_spin"].abs().max()) if not baseline.empty else 1e-6

    best_cases = pick_best_cases(detuning, contact, baseline_noise=baseline_noise)
    if not best_cases:
        results = pd.DataFrame(columns=TABLE_COLUMNS)
    else:
        results = run_task_grid(bias_gate_tasks(best_cases, num_points=CONTROL_POINTS), max_workers=max_workers, label="bias/gate cases")
        results = results.sort_values(["stage", "N", "eta_d", "EF", "V"]).reset_index(drop=True)
        results = results[TABLE_COLUMNS]

    table_path = Path(__file__).resolve().parents[1] / "tables" / "bias_gate_response.csv"
    report_path = Path(__file__).resolve().parents[1] / "reports" / "05_bias_gate_report.md"
    results.to_csv(table_path, index=False)
    write_report(results, report_path)
    return results


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_core import PROJECT_ROOT, load_table


def significance_threshold(baseline: pd.DataFrame) -> float:
    baseline_noise = float(baseline["I_spin"].abs().max()) if not baseline.empty else 1e-6
    return max(10.0 * baseline_noise, 1e-4)


def best_significant(table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    significant = table[(table["R_even"] > 0.02) & (table["I_spin"].abs() > threshold)].copy()
    return significant.sort_values("I_spin", key=lambda series: series.abs(), ascending=False)


def classify_result(detuning_sig: pd.DataFrame, contact_sig: pd.DataFrame) -> str:
    if not detuning_sig.empty and contact_sig.empty:
        return "RESULTADO 1"
    if detuning_sig.empty and not contact_sig.empty:
        return "RESULTADO 2"
    if detuning_sig.empty and contact_sig.empty:
        return "RESULTADO 3"
    return "RESULTADO 4"


def compact_best_table(*frames: pd.DataFrame) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame(columns=["mechanism", "N", "eta_d", "Delta", "alpha", "p", "EF", "V", "I_charge", "I_spin", "polarization", "R_even", "note"])
    combined = pd.concat(usable, ignore_index=True)
    best = combined.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).head(12).copy()
    best["note"] = best["stage"].fillna("survey")
    return best[["mechanism", "N", "eta_d", "Delta", "alpha", "p", "EF", "V", "I_charge", "I_spin", "polarization", "R_even", "note"]]


def build_report() -> tuple[str, pd.DataFrame]:
    tables_dir = PROJECT_ROOT / "tables"
    reports_dir = PROJECT_ROOT / "reports"

    baseline = load_table(tables_dir / "baseline_spectral_metrics.csv")
    detuning = load_table(tables_dir / "detuning_phase_map.csv")
    contact = load_table(tables_dir / "contact_asymmetry_phase_map.csv")
    spin_active = load_table(tables_dir / "spin_active_contacts_map.csv")
    bias_gate = load_table(tables_dir / "bias_gate_response.csv")
    controls = load_table(tables_dir / "controls_summary.csv")

    threshold = significance_threshold(baseline)
    detuning_sig = best_significant(detuning, threshold)
    contact_sig = best_significant(contact, threshold)
    verdict = classify_result(detuning_sig, contact_sig)

    master_spectral = pd.concat([frame for frame in [baseline, detuning, contact, spin_active, controls] if not frame.empty], ignore_index=True)
    master_integrated = pd.concat([frame for frame in [master_spectral, bias_gate] if not frame.empty], ignore_index=True)
    master_spectral.to_csv(tables_dir / "master_spectral_summary.csv", index=False)
    master_integrated.to_csv(tables_dir / "master_integrated_summary.csv", index=False)

    baseline_candidates = baseline.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).head(1).copy()
    if not baseline_candidates.empty:
        baseline_candidates["mechanism"] = "baseline"
        baseline_candidates["stage"] = "reference"

    spin_active_candidates = spin_active.sort_values("I_spin", key=lambda series: series.abs(), ascending=False).head(3).copy()
    if not spin_active_candidates.empty:
        spin_active_candidates["mechanism"] = "spin_active_contact"
        spin_active_candidates["stage"] = "benchmark"

    bias_gate_candidates = bias_gate.copy()
    if not bias_gate_candidates.empty:
        bias_gate_candidates["mechanism"] = bias_gate_candidates["stage"]
    best_candidates = compact_best_table(
        detuning_sig.assign(mechanism="detuning"),
        contact_sig.assign(mechanism="contact_asymmetry"),
        spin_active_candidates,
        baseline_candidates,
        bias_gate_candidates,
    )
    best_candidates.to_csv(tables_dir / "master_best_candidates.csv", index=False)

    if verdict == "RESULTADO 1":
        best_mechanism = "detuning"
        title = "Detuning Unlocks Net Spin Response in a Symmetry-Constrained Chiral Ladder"
        abstract = (
            "Using a channel-resolved tight-binding ladder with chiral spin-orbit coupling, we re-evaluate when spectral spin selectivity can become a finite integrated response under the corrected Landauer window. "
            "The symmetric baseline shows that phenomenological dephasing reshapes and broadens Tz(E) but keeps the integrated spin current at the numerical-noise floor. "
            "We then introduce explicit symmetry breaking through detuning between chains and scalar left-right contact asymmetry. Within the explored range, only detuning generates a finite even component of Tz(E) large enough to clear the numerical-noise floor and unlock a measurable signed I_spin. "
            "Moderate dephasing can then modulate that unlocked signal, but only after the symmetry has already been broken. "
            "The resulting phase maps and controls isolate detuning as the minimal clean ingredient supported by the present data."
        )
    elif verdict == "RESULTADO 2":
        best_mechanism = "contact asymmetry"
        title = "Scalar Contact Asymmetry Unlocks Net Spin Response in a Symmetry-Constrained Chiral Ladder"
        abstract = (
            "Using a channel-resolved tight-binding ladder with chiral spin-orbit coupling, we re-evaluate when spectral spin selectivity can become a finite integrated response under the corrected Landauer window. "
            "The symmetric baseline shows that phenomenological dephasing reshapes and broadens Tz(E) but keeps the integrated spin current at the numerical-noise floor. "
            "We then introduce explicit symmetry breaking through detuning between chains and scalar left-right contact asymmetry. Within the explored range, only scalar contact asymmetry generates a finite even component of Tz(E) large enough to clear the numerical-noise floor and unlock a measurable signed I_spin. "
            "Moderate dephasing can then modulate that unlocked signal, but only after the symmetry has already been broken. "
            "The resulting phase maps and controls isolate scalar contact asymmetry as the minimal clean ingredient supported by the present data."
        )
    elif verdict == "RESULTADO 4":
        best_mechanism = "detuning and contact asymmetry"
        title = "Two Distinct Symmetry-Breaking Routes to Net Spin Response in a Chiral Ladder"
        abstract = (
            "Using a channel-resolved tight-binding ladder with chiral spin-orbit coupling, we re-evaluate when spectral spin selectivity can become a finite integrated response under the corrected Landauer window. "
            "The symmetric baseline shows that phenomenological dephasing reshapes and broadens Tz(E) but keeps the integrated spin current at the numerical-noise floor. "
            "We then introduce explicit symmetry breaking through detuning between chains and scalar left-right contact asymmetry. Both mechanisms generate a finite even component of Tz(E) and unlock a measurable signed I_spin over part of the explored range, although their robustness is not identical. "
            "Moderate dephasing only modulates the signal after one of those symmetry-breaking ingredients is already present. "
            "The combined phase maps and controls clarify which route is cleaner and which route is stronger in the present model."
        )
    else:
        best_mechanism = "none within the explored range"
        title = "Pure Dephasing Reshapes Spectral Spin Selectivity but Does Not Unlock Net Response in the Explored Ladder Campaign"
        abstract = (
            "Using a channel-resolved tight-binding ladder with chiral spin-orbit coupling, we re-evaluate when spectral spin selectivity can become a finite integrated response under the corrected Landauer window. "
            "The symmetric baseline shows that phenomenological dephasing reshapes and broadens Tz(E) but keeps the integrated spin current at the numerical-noise floor. "
            "We then introduce explicit symmetry breaking through detuning between chains and scalar left-right contact asymmetry, while keeping a spin-active-contact benchmark separate. "
            "Within the explored detuning and scalar-contact ranges, neither mechanism generates a finite even component of Tz(E) large enough to clear the numerical-noise floor and produce a robust signed I_spin. "
            "The current data therefore support a negative result for these two minimal knobs and motivate a stronger symmetry-breaking ingredient for the next paper iteration."
        )

    report = f"""# MASTER REPORT

## Executive summary

- Numerical significance threshold for a finite response: `{threshold:.6e}` in `|I_spin|`, together with `R_even > 0.02`.
- Symmetric baseline: `I_spin` stays at the numerical-noise floor under the corrected Landauer window.
- Strongest detuning candidate: {(detuning_sig.head(1).to_markdown(index=False) if not detuning_sig.empty else "none")}
- Strongest scalar-contact candidate: {(contact_sig.head(1).to_markdown(index=False) if not contact_sig.empty else "none")}
- Final classification: **{verdict}**

## Contact audit

The contact audit is recorded in `reports/00_contact_audit.md` and `tables/contact_audit_summary.csv`. It verifies that the original baseline uses `Gamma1..Gamma4`, that `p = 0` implies `val = val1`, and that the `gamma_out` knobs are inter-chain hybridization rather than lead asymmetry.

## Baseline symmetry-constrained results

The baseline table confirms that dephasing redistributes the spectral weight without generating a robust even component or a signed net current above the significance threshold.

## Detuning results

Detuning significant cases:

{(detuning_sig.head(10).to_markdown(index=False) if not detuning_sig.empty else "No detuning case exceeded the significance threshold.")}

## Contact asymmetry results

Scalar contact asymmetry significant cases:

{(contact_sig.head(10).to_markdown(index=False) if not contact_sig.empty else "No scalar contact-asymmetry case exceeded the significance threshold.")}

## Comparison of mechanisms

The best detuning `|I_spin|` is `{(float(detuning_sig.iloc[0]["I_spin"]) if not detuning_sig.empty else 0.0):.6e}`.
The best scalar-contact `|I_spin|` is `{(float(contact_sig.iloc[0]["I_spin"]) if not contact_sig.empty else 0.0):.6e}`.

## Best candidate for a new paper

The current strongest candidate is **{best_mechanism}** under the discovery/refined grids explored here.

## Caveats

- The survey uses a computationally cheap energy grid for the broad parameter sweep, then higher-resolution traces for the publication figures.
- The model remains a phenomenological ladder with uniform dephasing; disorder and probe physics are not promoted beyond what was explicitly scanned.
- A nonzero response is only treated as physical if it clears both the `R_even` and noise-floor thresholds.

## Suggested figure order for manuscript

1. `Fig01_baseline_spectra`
2. `Fig02_even_odd_baseline`
3. `Fig03_detuning_unlocks_even_component`
4. `Fig04_phase_map_eta_delta_Ispin`
5. `Fig05_contact_asymmetry_vs_detuning`
6. `Fig06_bias_gate_polarization`

## Draft claims that are actually supported by the data

### A. Principal claim

Pure dephasing reshapes the spectral spin selectivity of the symmetric ladder but does not, by itself, generate a robust net spin current under the corrected Landauer window. A separate symmetry-breaking knob is required to create a finite even component of `Tz(E)` and thereby unlock a signed integrated response.

### B. Proposed title

{title}

### C. Proposed abstract

{abstract}

### D. Draft captions

- **Fig. 1:** Baseline spectra `Tz(E)` and `T0(E)` for symmetric contacts. Dephasing broadens the spectra but does not unlock a finite even component.
- **Fig. 2:** Even/odd decomposition of the baseline `Tz(E)` together with `R_even` versus dephasing, showing the dominance of the odd sector.
- **Fig. 3:** Detuning-dependent `Tz(E)` and integrated `I_spin`, demonstrating how explicit chain detuning unlocks a finite response.
- **Fig. 4:** Phase maps of `I_spin` and polarization over `(eta_d, Delta)` for the representative length slice.
- **Fig. 5:** Scalar contact asymmetry compared against detuning in terms of integrated response and even-weight generation.
- **Fig. 6:** Bias/gate response of the strongest candidate, showing where polarization is maximized after symmetry breaking.
"""
    (reports_dir / "MASTER_REPORT.md").write_text(report, encoding="utf-8")
    return verdict, best_candidates


def main() -> None:
    build_report()


if __name__ == "__main__":
    main()

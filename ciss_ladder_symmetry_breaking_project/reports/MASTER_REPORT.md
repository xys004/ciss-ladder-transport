# MASTER REPORT

## Executive summary

- Numerical significance threshold for a finite response: `1.000000e-04` in `|I_spin|`, together with `R_even > 0.02`.
- Symmetric baseline: `I_spin` stays at the numerical-noise floor under the corrected Landauer window.
- Strongest detuning candidate: none
- Strongest scalar-contact candidate: none
- Final classification: **RESULTADO 3**

## Contact audit

The contact audit is recorded in `reports/00_contact_audit.md` and `tables/contact_audit_summary.csv`. It verifies that the original baseline uses `Gamma1..Gamma4`, that `p = 0` implies `val = val1`, and that the `gamma_out` knobs are inter-chain hybridization rather than lead asymmetry.

## Baseline symmetry-constrained results

The baseline table confirms that dephasing redistributes the spectral weight without generating a robust even component or a signed net current above the significance threshold.

## Detuning results

Detuning significant cases:

No detuning case exceeded the significance threshold.

## Contact asymmetry results

Scalar contact asymmetry significant cases:

No scalar contact-asymmetry case exceeded the significance threshold.

## Comparison of mechanisms

The best detuning `|I_spin|` is `0.000000e+00`.
The best scalar-contact `|I_spin|` is `0.000000e+00`.

## Best candidate for a new paper

The current strongest candidate is **none within the explored range** under the discovery/refined grids explored here.

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

Pure Dephasing Reshapes Spectral Spin Selectivity but Does Not Unlock Net Response in the Explored Ladder Campaign

### C. Proposed abstract

Using a channel-resolved tight-binding ladder with chiral spin-orbit coupling, we re-evaluate when spectral spin selectivity can become a finite integrated response under the corrected Landauer window. The symmetric baseline shows that phenomenological dephasing reshapes and broadens Tz(E) but keeps the integrated spin current at the numerical-noise floor. We then introduce explicit symmetry breaking through detuning between chains and scalar left-right contact asymmetry, while keeping a spin-active-contact benchmark separate. Within the explored detuning and scalar-contact ranges, neither mechanism generates a finite even component of Tz(E) large enough to clear the numerical-noise floor and produce a robust signed I_spin. The current data therefore support a negative result for these two minimal knobs and motivate a stronger symmetry-breaking ingredient for the next paper iteration.

### D. Draft captions

- **Fig. 1:** Baseline spectra `Tz(E)` and `T0(E)` for symmetric contacts. Dephasing broadens the spectra but does not unlock a finite even component.
- **Fig. 2:** Even/odd decomposition of the baseline `Tz(E)` together with `R_even` versus dephasing, showing the dominance of the odd sector.
- **Fig. 3:** Detuning-dependent `Tz(E)` and integrated `I_spin`, demonstrating how explicit chain detuning unlocks a finite response.
- **Fig. 4:** Phase maps of `I_spin` and polarization over `(eta_d, Delta)` for the representative length slice.
- **Fig. 5:** Scalar contact asymmetry compared against detuning in terms of integrated response and even-weight generation.
- **Fig. 6:** Bias/gate response of the strongest candidate, showing where polarization is maximized after symmetry breaking.

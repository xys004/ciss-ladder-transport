# Scalar Contact Asymmetry Report

This phase scans the clean left-right scalar contact asymmetry `Gamma_L = Gamma0`, `Gamma_R = alpha * Gamma0`, keeping `p = 0`.

## Discovery grid

- `N`: 10, 37, 91
- `eta_d`: 0.0, 0.1, 0.25, 0.5, 1.0
- `alpha`: 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0
- `Delta = 0`, `lambda_soc = 0.1`, `gamma_hyb = 1.0`

## Summary over `alpha`

|   alpha |   median_R_even |   median_abs_I_spin |   max_abs_I_spin |
|--------:|----------------:|--------------------:|-----------------:|
|    0.25 |     5.52364e-15 |         1.94354e-20 |      1.11087e-15 |
|    0.5  |     4.91403e-15 |         2.0835e-20  |      4.8312e-16  |
|    0.75 |     4.92935e-15 |         2.23074e-20 |      2.58311e-16 |
|    1    |     5.64895e-15 |         1.20735e-20 |      1.08854e-16 |
|    1.5  |     4.70421e-15 |         5.73732e-21 |      6.0119e-17  |
|    2    |     4.40422e-15 |         5.66371e-21 |      1.04395e-16 |
|    4    |     4.49649e-15 |         2.30535e-21 |      1.04571e-16 |

Baseline numerical noise floor from the symmetric phase: `9.380517e-16` in `|I_spin|`.

## Explicit answers

- Does scalar contact asymmetry generate an even component of `Tz(E)`? Not above the significance threshold in the scanned range.
- Does dephasing amplify, broaden, or destroy the effect? The table shows how `R_even`, `|I_spin|`, and polarization evolve with `eta_d`; no enhancement is claimed unless it beats the corresponding `eta_d = 0` case.
- Is the effect stronger or weaker than detuning? In this survey it is `weaker_or_equal` relative to the strongest detuning case by `|I_spin|`.

## Promising cases

No case exceeded the significance threshold.
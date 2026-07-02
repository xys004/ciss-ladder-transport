# Detuning Report

This phase scans the minimal structural detuning knob `epsilon_A = +Delta/2`, `epsilon_B = -Delta/2` with symmetric contacts.

## Discovery grid

- `N`: 10, 37, 91
- `eta_d`: 0.0, 0.1, 0.25, 0.5, 1.0
- `Delta`: 0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0 meV
- `lambda_soc = 0.1`, `gamma_hyb = 1.0`

## Refined grid

No refinement was triggered because discovery did not clear the significance threshold.

## Summary over discovery data

|   Delta |   median_R_even |   median_abs_I_spin |   max_abs_I_spin |
|--------:|----------------:|--------------------:|-----------------:|
|    0    |     5.64895e-15 |         1.20735e-20 |      1.08854e-16 |
|    0.02 |     8.6849e-15  |         2.09442e-20 |      6.73127e-16 |
|    0.05 |     7.21307e-15 |         4.50317e-21 |      1.15863e-15 |
|    0.1  |     8.18908e-15 |         5.03588e-21 |      6.5334e-16  |
|    0.25 |     8.94747e-15 |         2.09426e-20 |      2.15431e-16 |
|    0.5  |     1.3532e-14  |         1.76785e-20 |      2.09712e-16 |
|    1    |     2.06647e-14 |         1.31555e-20 |      8.25132e-16 |

Median Spearman correlation between `Delta` and `R_even` across discovery slices: `0.750`.

Baseline numerical noise floor from the symmetric phase: `9.380517e-16` in `|I_spin|`.

## Supported statements

- `R_even` grows systematically with detuning if and only if the discovery table shows values above the baseline noise threshold.
- A finite net response is counted as real only when `|I_spin|` exceeds `max(10 x baseline_noise, 1e-4)` and `R_even > 0.02`.
- Moderate dephasing is only described as an enhancement when the finite response beats the `eta_d = 0` case within the same `N, Delta` family.

## Promising cases

No case exceeded the significance threshold.
# Baseline Report

This phase evaluated the symmetry-constrained ladder over the full baseline grid:

- Executed discovery lengths: 10, 37, 91
- The generator supports the full length list 10, 19, 28, 37, 46, 55, 64, 73, 82, 91
- `eta_d`: 0.0, 0.1, 0.25, 0.5, 1.0, 2.0
- `lambda_soc`: 0.05, 0.1, 0.2
- `gamma_hyb`: 0.25, 0.5, 0.75, 1.0
- `num_points`: 51

## Short answer

- Dephasing redistributes and broadens the spectral weight, but does not unlock a robust even component by itself.
- The odd component of `Tz(E)` remains dominant in the symmetric baseline.
- The corrected Landauer integral stays at the numerical-noise level under symmetric contacts and `Delta = 0`.

## Quantitative summary for the reference slice (`lambda_soc = 0.1`, `gamma_hyb = 1.0`)

Maximum `|I_spin|` across the full symmetric slice: `1.088539e-16`.

|   eta_d |   median_Wz |   median_R_even |   median_abs_I_spin |   max_abs_I_spin |
|--------:|------------:|----------------:|--------------------:|-----------------:|
|    0    | 0.0343932   |     1.02893e-14 |         7.9987e-17  |      1.08854e-16 |
|    0.1  | 0.000853154 |     6.47792e-15 |         3.86332e-18 |      1.58836e-17 |
|    0.25 | 3.33153e-06 |     5.64895e-15 |         1.20735e-20 |      2.52755e-18 |
|    0.5  | 4.69483e-10 |     1.68725e-15 |         4.89927e-25 |      1.40607e-19 |
|    1    | 1.93702e-17 |     1.16046e-20 |         1.41375e-32 |      3.77194e-22 |
|    2    | 1.51637e-30 |     2.81308e-33 |         7.40334e-46 |      2.76265e-25 |

## Interpretation

The baseline data support the restricted claim needed for the new paper: pure dephasing changes the line shape and total spectral weight `Wz`, but it does not by itself convert spectral spin selectivity into a robust net integrated response when the contacts remain symmetric and no explicit structural asymmetry is introduced.

# Search for near-90 observables

Parameter sets scanned including duplicate literal/eV modes: 28800
Candidates within |abs(value)-0.90|<=0.20 before refinement: 701620

## Top 20 closest to 0.90
| rank | abs value | observable | family | context | Lambda | t | Gamma | alpha_factor | a0 | unit | flags |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 0.899997 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.2 | 0.005 | 0.1 | 6e-05 | 0.1 | 10 | eV | nonphysical complex K realpart |
| 2 | 0.899997 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.2 | 0.005 | 0.1 | 6e-05 | 0.1 | 10 | literal | nonphysical complex K realpart |
| 3 | 0.899997 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.2 | 0.05 | 0.1 | 6e-05 | 0.1 | 0.1 | eV | nonphysical complex K realpart |
| 4 | 0.899997 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.2 | 0.05 | 0.1 | 6e-05 | 0.1 | 0.1 | literal | nonphysical complex K realpart |
| 5 | 0.90001 | A_legacy_A | legacy | DP=-0.2 | 0.01 | 0.005 | 6e-05 | 2 | 2 | eV | not net current; DP=0 nonzero generally |
| 6 | 0.90001 | S_legacy_up_A | legacy | DP=-0.2 | 0.01 | 0.005 | 6e-05 | 2 | 2 | eV | not net current; DP=0 nonzero generally |
| 7 | 0.90001 | A_legacy_A | legacy | DP=-0.2 | 0.01 | 0.005 | 6e-05 | 2 | 2 | literal | not net current; DP=0 nonzero generally |
| 8 | 0.90001 | S_legacy_up_A | legacy | DP=-0.2 | 0.01 | 0.005 | 6e-05 | 2 | 2 | literal | not net current; DP=0 nonzero generally |
| 9 | 0.90001 | A_legacy_A | legacy | DP=-0.2 | 0.02 | 0.005 | 6e-05 | 2 | 0.5 | eV | not net current; DP=0 nonzero generally |
| 10 | 0.90001 | S_legacy_up_A | legacy | DP=-0.2 | 0.02 | 0.005 | 6e-05 | 2 | 0.5 | eV | not net current; DP=0 nonzero generally |
| 11 | 0.90001 | A_legacy_A | legacy | DP=-0.2 | 0.02 | 0.005 | 6e-05 | 2 | 0.5 | literal | not net current; DP=0 nonzero generally |
| 12 | 0.90001 | S_legacy_up_A | legacy | DP=-0.2 | 0.02 | 0.005 | 6e-05 | 2 | 0.5 | literal | not net current; DP=0 nonzero generally |
| 13 | 0.900043 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.5 | 0.005 | 0.1 | 6e-05 | 0.5 | 2 | eV | nonphysical complex K realpart |
| 14 | 0.900043 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.5 | 0.005 | 0.1 | 6e-05 | 0.5 | 2 | literal | nonphysical complex K realpart |
| 15 | 0.900043 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.5 | 0.01 | 0.1 | 6e-05 | 0.5 | 0.5 | eV | nonphysical complex K realpart |
| 16 | 0.900043 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.5 | 0.01 | 0.1 | 6e-05 | 0.5 | 0.5 | literal | nonphysical complex K realpart |
| 17 | 0.899932 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.5 | 0.05 | 0.01 | 6e-05 | 0.01 | 10 | eV | nonphysical complex K realpart |
| 18 | 0.899932 | R | current_decoupled_grid | complex_K_realpart_nonphysical, DP=-0.5 | 0.05 | 0.01 | 6e-05 | 0.01 | 10 | literal | nonphysical complex K realpart |
| 19 | 0.899932 | A_legacy_A | legacy | DP=0.2 | 0.005 | 0.005 | 1e-05 | 2 | 10 | eV | not net current; DP=0 nonzero generally |
| 20 | 0.899932 | S_legacy_up_A | legacy | DP=0.2 | 0.005 | 0.005 | 1e-05 | 2 | 10 | eV | not net current; DP=0 nonzero generally |

## Diagnostics
Best legacy abs value near 0.90: 0.90001 (A_legacy_A, DP=-0.2)
No physical-current candidates.
No windowed candidates.

Interpretation: legacy quantities are proxies, not net currents; exact physical current candidates are only grid-search estimates unless listed as legacy-refined.

## Exact 4x4 check on top legacy candidates
The top refined legacy near-90 cases were checked with Hermitian exact 4x4 Landauer transport in `top_legacy_exact4x4_check.csv`.
For those cases, `P_exact_4x4` is 0 to numerical precision and `A_peak_exact_4x4` is also 0 to numerical precision.

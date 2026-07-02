# Spin-Active Contacts Benchmark Report

This optional benchmark keeps the scalar contact asymmetry phase separate and activates the legacy spin-selective lead parameter `p`.

## How `p` enters

- `val = i (1 + p)`
- `val1 = i (1 - p)`
- `Gamma1 = (val, val)`, `Gamma2 = (val, val1)`, `Gamma3 = (val1, val)`, `Gamma4 = (val1, val1)` at the two ends

## Survey summary

|    p |   median_R_even |   median_abs_I_spin |   max_abs_I_spin |
|-----:|----------------:|--------------------:|-----------------:|
| 0    |     5.64895e-15 |         1.20735e-20 |      1.08854e-16 |
| 0.1  |     3.49863     |         1.11832e-05 |      0.191209    |
| 0.25 |     9.87403     |         2.63372e-05 |      0.455227    |
| 0.5  |    25.1591      |         5.22911e-05 |      0.932181    |

This phase is a benchmark only. Any finite response here is expected because the interface itself becomes spin active, so it should not be confused with the cleaner scalar contact asymmetry mechanism.

# Contact Audit

This short audit verifies the baseline contact implementation before any new numerical campaign is run.

## Verified statements

1. In the legacy stochastic scripts, the physical contact broadening enters through `Gamma1`, `Gamma2`, `Gamma3`, and `Gamma4`.
2. With `p = 0`, the code sets `val = i (1 + p)` and `val1 = i (1 - p)`, so `val = val1 = i`; therefore no effective spin-dependent contact asymmetry is active in the baseline case.
3. The quantities later called `gamma_out_parallel` / `gamma_out_spin_mixing` map onto `gamma_per` / `gamma_per1`, i.e. inter-chain hybridization knobs, not lead asymmetry knobs.
4. The new paper campaign must therefore introduce symmetry breaking explicitly and separately from the baseline lead definition.

## Implementation choice for this workspace

- Baseline and detuning runs use the explicit `Gamma1..Gamma4` contact representation with `p = 0`.
- Scalar contact asymmetry is implemented as a separate left-right lead broadening ratio `alpha = Gamma_R / Gamma_L`, equal for all four channel blocks.
- Spin-active contacts are benchmarked separately through `p`, without mixing them into the scalar asymmetry phase.
- All integrated observables use the corrected Landauer window `mu_L = EF + V/2`, `mu_R = EF - V/2`.

# Summary

## What The Code Computes

The legacy code computes channel-resolved Green-function transport kernels for a two-leg ladder with spin-orbit coupling in an explicit `8N` basis. The coherent script reconstructs a charge-like kernel plus `Gz`, `Gx`, and `Gy`. The disorder and dephasing scripts compute realization-averaged `Gz`.

## What Was Clarified

- The basis ordering is now explicit and centralized.
- The source vector `B` is documented as a simultaneous injection of four left-edge `xi=+1` channels.
- The legacy `Trans1*` and `Trans2*` objects are mapped to unambiguous block names.
- The observable formulas are documented as component-wise channel reconstructions.
- The disorder and dephasing averaging strategy is documented explicitly as amplitude averaging followed by observable assembly.

## What Remains Ambiguous

- Whether the hard-coded phase step `2*pi/(10-1)` is intentional geometry input or a leftover constant.
- Whether the coherent advanced-matrix chain-2 backward Rashba phase uses `(n-1)` by design or by typo.
- Whether the dephasing implementation should be described more strongly than phenomenological random imaginary onsite broadening.

## Numerical Logic Changes

No transport formula was intentionally redefined.

The main code-level change is structural:

- one solve per matrix is now reused to extract all eight output channels, instead of rerunning nearly identical `rho_w*` functions for each component

That change should preserve the numerical content while making the workflow easier to inspect and reproduce.

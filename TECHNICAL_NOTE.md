# Technical Note

## Scope

This note records what is explicit in the legacy code and what is only an informed interpretation. It is intentionally conservative.

## 1. Basis And State Ordering

This part is explicit in the coherent legacy script and consistent with the stochastic variants.

The Hilbert space dimension is `8N`, organized as eight contiguous `N`-site blocks:

1. chain 1, `xi=+1`, spin up
2. chain 1, `xi=+1`, spin down
3. chain 1, `xi=-1`, spin up
4. chain 1, `xi=-1`, spin down
5. chain 2, `xi=+1`, spin up
6. chain 2, `xi=+1`, spin down
7. chain 2, `xi=-1`, spin up
8. chain 2, `xi=-1`, spin down

The cleaned module [basis.py](C:/Users/Nelson/Downloads/orquestador_IA/ciss_ladder_transport/src/ciss_ladder_transport/basis.py) makes this explicit.

## 2. What The Source Vector `B` Does

This is explicit in all three scripts.

The legacy source vector has nonzero entries at:

- `B[0]`
- `B[N]`
- `B[4N]`
- `B[5N]`

all equal to `-1`.

That means the solver is not driven by a single compact lead self-energy trace expression in the code. Instead, it explicitly injects four left-edge channels:

- chain 1, `xi=+1`, spin up
- chain 1, `xi=+1`, spin down
- chain 2, `xi=+1`, spin up
- chain 2, `xi=+1`, spin down

Interpretation:

- Fact: the code solves `A * G = B` with this multi-entry source.
- Inference: the extracted objects should be read as Green-function components in that explicit injected-channel basis, not as a direct literal implementation of a single compact matrix-trace formula.

## 3. What `Trans1u`, `Trans1d`, `Trans1ud`, `Trans1du`, ... Mean

The extracted objects are explicit Green-function components from the solved retarded vector.

From the legacy `den_espectral*` functions:

- `Trans1u` comes from `-GR[N-1]`
- `Trans1d` comes from `-GR[2N-1]`
- `Trans1ud` comes from `-GR[3N-1]`
- `Trans1du` comes from `-GR[4N-1]`
- `Trans2u` comes from `-GR[5N-1]`
- `Trans2d` comes from `-GR[6N-1]`
- `Trans2ud` comes from `-GR[7N-1]`
- `Trans2du` comes from `-GR[8N-1]`

The cleaned names make the block meaning explicit:

- `c1_xi_plus_up`
- `c1_xi_plus_down`
- `c1_xi_minus_up`
- `c1_xi_minus_down`
- `c2_xi_plus_up`
- `c2_xi_plus_down`
- `c2_xi_minus_up`
- `c2_xi_minus_down`

Important ambiguity:

- The legacy suffixes `ud` and `du` are not good scientific names by themselves.
- In code, they correspond to the `xi=-1` branches with spin up and spin down respectively.
- Therefore, reading `ud` or `du` literally as a spin-flip label would be misleading.

## 4. How Transport Observables Are Assembled

This is explicit in the algebra but implicit in the interpretation.

### Charge-like kernel

The coherent script constructs

- a sum of eight component-wise squared moduli

using the `Trans1*` and `Trans2*` amplitudes.

Interpretation:

- Fact: the code builds a charge-like spectral kernel from explicit component amplitudes.
- Inference: this plays the role of a channel-resolved conductance/transmission kernel, but the code itself does not write the compact Caroli trace formula explicitly.

### `Gz`

The code uses positive signs for spin-up blocks and negative signs for spin-down blocks:

- `+ Trans*u`
- `+ Trans*ud`
- `- Trans*d`
- `- Trans*du`

Interpretation:

- Fact: the combination is a spin-`z` projection assembled component by component.
- Inference: it is consistent with a `sigma_z`-type reconstruction in the explicit channel basis.

### `Gx` and `Gy`

Only the coherent script computes these. They are assembled from cross terms such as:

- `Trans_du * conj(Trans_u)`
- `Trans_d * conj(Trans_ud)`

with symmetric real combinations for `Gx` and antisymmetric imaginary combinations for `Gy`.

Interpretation:

- Fact: the code reconstructs transverse spin components from channel cross terms.
- Inference: these are the explicit component-wise analogues of `sigma_x` and `sigma_y` projections.

## 5. Why There Are Two Sign Groups

This is explicit in the code but not fully explained there.

The legacy scripts compute one set of amplitudes using one sign choice for the inter-chain couplings and a second set with sign changes:

- coherent charge block:
  group 2 flips `gamma_per`, but keeps `gamma_per1`
- coherent spin blocks:
  group 2 flips both `gamma_per` and `gamma_per1`
- disorder/dephasing `Gz`:
  group 2 flips `gamma_per1`; `gamma_per` is zero in practice

Interpretation:

- Fact: the observable formulas use amplitudes from two different sign-configured solves.
- Inference: these are not simply chain-1 amplitudes plus chain-2 amplitudes from one common solve. The sign-group construction is part of the historical definition of the published kernels and has been preserved exactly.

## 6. How Disorder Is Implemented

This is explicit in the disorder script.

- For each realization, chain 1 gets a random onsite array `E01[i]`
- chain 2 gets an independent random onsite array `E02[i]`
- each element is uniform in `[-0.5, 0.5]`
- the diagonal term uses these onsite values in the real part
- the small scalar `eta` remains only as a numerical regularizer

The cleaned code keeps this as `build_disorder_samples()`.

What is averaged:

- The script averages the complex amplitude `T`
- Then the final `Gz` kernel is assembled from `|<T>|^2`-type terms

This distinction matters:

- Fact: the code computes `|<T>|^2`-type objects.
- Fact: it does not compute `<|T|^2>` directly.
- Inference: in regimes with strong fluctuations or localization, those two quantities may differ appreciably.

## 7. How Dephasing Is Implemented

This is explicit in the dephasing script, but its methodological interpretation needs caution.

- Each realization draws a site-dependent random array for chain 1 and another independent one for chain 2
- Those arrays enter the diagonal as imaginary onsite terms
- The generator enforces exact zero mean per realization by correcting the last site

So the code does implement random local imaginary broadening.

What should not be overclaimed:

- The code does not solve a self-consistent probe-potential condition
- It does not explicitly enforce zero net current through fictitious probes
- Therefore, the code alone does not justify describing the implementation as a full self-consistent Büttiker-probe calculation

Conservative wording:

- phenomenological local dephasing-like imaginary onsite broadening
- random site-resolved dephasing field with zero mean per realization

## 8. Averaging Strategy

The disorder and dephasing scripts use a double loop:

1. loop over energy
2. loop over realizations
3. solve the Green-function system
4. extract one channel amplitude
5. average the complex amplitude over realizations

The cleaned code preserves that same mathematical averaging rule, but removes the unnecessary repeated solves for different extracted channels. One solved Green vector is now used to read all eight output components for the same matrix.

This is a code-structure change, not a physics change.

## 9. Parameters Present But Not Effectively Used

The legacy signatures contain several variables that are passed around but not used in the matrix construction:

- `theta`
- `U`
- `M1`
- `M2`
- some auxiliary Rashba/Zeeman names such as `l_R12`, `l_EO1`, `l_EO2`, `l_Z`

The cleaned code removes or de-emphasizes these from the computational core and documents them as legacy placeholders.

## 10. Legacy Ambiguities Preserved Deliberately

### Hard-coded phase step

All three scripts use:

- `Dphi = 2*pi/(10-1)`

even when `N` is not `10`.

This may be intentional geometry input or a leftover hard-coded value. The code itself does not decide the issue. The cleaned implementation preserves it and makes it explicit as `phase_site_count=10`.

### Coherent advanced-matrix inconsistency

In the coherent script only, the advanced matrix uses a chain-2 backward Rashba phase based on `(n-1)` where the retarded matrix and the stochastic scripts use `(n-2)`.

This looks like a likely typo or copy-paste inconsistency, but the source of truth is ambiguous. The cleaned coherent runner preserves that exact behavior through the flag:

- `use_legacy_advanced_chain2_backward_phase=True`

and documents it rather than silently changing it.

## 11. Bottom-Line Interpretation

Best-supported description:

- explicit channel-resolved Green-function calculation in an `8N` basis
- component-wise reconstruction of charge-like and spin-projected spectral transport kernels
- disorder treated as realization-dependent real onsite energies
- dephasing treated as realization-dependent imaginary onsite broadening with zero mean per realization

Claims that are not fully supported by code alone:

- exact compact Caroli-trace implementation stated directly in source
- full self-consistent Büttiker-probe enforcement
- rigorous D'Amato-Pastawski-type probe treatment beyond the phenomenological imaginary broadening actually coded

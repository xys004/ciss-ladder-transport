# Mapping Legacy To Clean

## Files

| Legacy file | Cleaned location | Notes |
| --- | --- | --- |
| `trans_cadena_bi_w_so_gammaout0_comentado (1).py` | `legacy/trans_cadena_bi_w_so_gammaout0_comentado (1).py` | Untouched historical source copy. |
| `trans_cadena_bi_w_so_gammaout0_comentado.py` | `legacy/trans_cadena_bi_w_so_gammaout0_comentado.py` | Byte-identical duplicate preserved for traceability. |
| `trans_cadena_bi_w_so_gammaout0_gz_dephasing_n91 (1).py` | `legacy/trans_cadena_bi_w_so_gammaout0_gz_dephasing_n91 (1).py` | Untouched historical source copy. |
| `trans_cadena_bi_w_so_gammaout0_gz_dephasing_n91.py` | `legacy/trans_cadena_bi_w_so_gammaout0_gz_dephasing_n91.py` | Byte-identical duplicate preserved for traceability. |
| `trans_cadena_bi_w_so_gammaout0_gz_disorder_n91 (1).py` | `legacy/trans_cadena_bi_w_so_gammaout0_gz_disorder_n91 (1).py` | Untouched historical source copy. |
| `trans_cadena_bi_w_so_gammaout0_gz_disorder_n91.py` | `legacy/trans_cadena_bi_w_so_gammaout0_gz_disorder_n91.py` | Byte-identical duplicate preserved for traceability. |
| Monolithic coherent workflow | `scripts/run_coherent.py` | Split into explicit coherent cases for charge, `Gz`, `Gx`, and `Gy`. |
| Monolithic dephasing workflow | `scripts/run_dephasing.py` | Keeps the legacy `Gz` stochastic workflow with CLI overrides. |
| Monolithic disorder workflow | `scripts/run_disorder.py` | Keeps the legacy `Gz` stochastic workflow with CLI overrides. |

## Functions And Responsibilities

| Legacy name | Cleaned location | Notes |
| --- | --- | --- |
| `matrix_AR` | `src/ciss_ladder_transport/greens.py::build_green_matrix(..., advanced=False)` | Unified builder for coherent, disorder, and dephasing cases. |
| `matrix_AA` | `src/ciss_ladder_transport/greens.py::build_green_matrix(..., advanced=True)` | Same matrix structure with advanced-sign handling. |
| `den_espectral1u`, `den_espectral1d`, ... | `src/ciss_ladder_transport/greens.py::extract_channel_amplitudes` | All eight output components are extracted from one solved retarded vector. |
| `rho_w1u`, `rho_w1d`, ... | `src/ciss_ladder_transport/greens.py::sweep_channels` and `average_channels` | One sweep now returns all channels at once instead of one function per component. |
| `Random()` in dephasing script | `src/ciss_ladder_transport/randomness.py::zero_mean_dephasing_realization` | Zero-mean random imaginary onsite profile preserved. |
| `Random()` in disorder script | `src/ciss_ladder_transport/randomness.py::anderson_disorder_realization` | Uniform onsite disorder in `[-0.5, 0.5]` preserved. |
| Manual basis comments and `B` construction | `src/ciss_ladder_transport/basis.py` | Centralized basis ordering, source vector, and legacy component mapping. |
| Inline observable formulas | `src/ciss_ladder_transport/observables.py` | Charge-like, `Gz`, `Gx`, and `Gy` kernels made explicit. |

## Variable Renaming

| Legacy variable | Cleaned name | Reason |
| --- | --- | --- |
| `gamma01` | `gamma_chain_1` | Clarifies it is the chain-1 nearest-neighbor hopping. |
| `gamma02` | `gamma_chain_2` | Clarifies it is the chain-2 nearest-neighbor hopping. |
| `gamma_per` | `gamma_per` | Kept, because the original meaning is already fairly specific. |
| `gamma_per1` | `gamma_per1` | Kept, because the original code already uses it consistently. |
| `l_R1` | `rashba_chain_1` | Makes the physical meaning explicit in English. |
| `l_R2` | `rashba_chain_2` | Makes the physical meaning explicit in English. |
| `l_D` | `dresselhaus` | Makes the physical meaning explicit in English. |
| `B` | `source_vector` | Clarifies its solver role. |
| `Trans1u`, `Trans1d`, ... | `c1_xi_plus_up`, `c1_xi_plus_down`, ... via `legacy_components()` | Removes ambiguity in the `u/d/ud/du` suffixes while preserving the original labels in the mapping layer. |
| `list` / `list_N` | CLI arg `--sites` or local `num_sites` | Avoids shadowing Python built-ins. |

## Behavior Changes That Were Deliberately Kept Minimal

- The cleaned code still solves both retarded and advanced systems, even though the extracted amplitudes come only from the retarded vector in the legacy implementation.
- The cleaned code preserves the hard-coded phase step and the coherent advanced-matrix phase inconsistency behind an explicit flag.
- The cleaned code removes repeated identical solves across channel-extraction functions by extracting all eight output channels from the same solved Green vector.

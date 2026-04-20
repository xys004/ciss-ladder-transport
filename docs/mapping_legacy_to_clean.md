# Mapping Legacy To Clean

| Legacy file | Referee-facing entry point | Notes |
| --- | --- | --- |
| `trans_cadena_bi_w_so_gammaout0_comentado.py` | `scripts/run_coherent_T0_Tz.py` and `src/ciss_ladder/observables.py` | Coherent spectral kernels reconstructed from the same channel-resolved logic |
| `trans_cadena_bi_w_so_gammaout0_gz_dephasing_n91.py` | `scripts/run_dephasing_scan.py` and `src/ciss_ladder/dephasing.py` | Preserves realization-averaged dephasing workflow |
| `trans_cadena_bi_w_so_gammaout0_gz_disorder_n91.py` | `scripts/run_disorder_scan.py` and `src/ciss_ladder/disorder.py` | Preserves Anderson-type disorder workflow |
| `current_dat_david_n_dephasing.py` | `scripts/current_from_transmission.py` | Landauer-type integration preserved as a separate post-processing step |
| `current_dat_david_n_disorder.py` | `scripts/current_from_transmission.py` | Same post-processing separation retained |

## Main module mapping

- legacy basis bookkeeping -> `src/ciss_ladder/basis.py`
- legacy effective operator assembly -> `src/ciss_ladder/hamiltonian.py`
- legacy Green-function solves -> `src/ciss_ladder/greens.py`
- legacy kernel assembly -> `src/ciss_ladder/observables.py`

## Traceability rule

The cleaned package is designed to make the legacy workflow easier to read without hiding the original numerical logic.

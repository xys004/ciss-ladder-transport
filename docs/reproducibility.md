# Reproducibility

## Software environment

- Python `>=3.10`
- see `requirements.txt` for package versions

## Random seed policy

- all stochastic scripts accept an optional `--seed`
- if no seed is provided, NumPy's default generator is used without forcing determinism

## Output hygiene

Raw outputs are written under `data/raw/` and may include:

- spectral kernels as CSV files
- metadata sidecars as JSON files

Processed datasets should be written under `data/processed/`.
Figure-ready outputs should be written under `figures/`.

## Figure mapping

- Fig. 2: coherent `T^z(E)` -> `scripts/run_coherent_T0_Tz.py` then `scripts/make_fig2.py`
- Fig. 3: coherent `T^0(E)` -> `scripts/run_coherent_T0_Tz.py` then `scripts/make_fig3.py`
- Fig. 4: integrated `I_z` scan -> post-process raw spectral outputs into `data/processed/fig4_Iz_scan.csv`, then run `scripts/make_fig4.py`

## Runtime expectations

The legacy defaults use large realization counts and are computationally heavy. For quick sanity checks, reduce the number of realizations before reproducing manuscript-scale scans.

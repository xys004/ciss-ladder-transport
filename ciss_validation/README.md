# CISS two-site validation toolkit

This folder contains independent Python checks for the minimal two-site CISS effective model.

## Files

- `ciss_symbolic_audit.py`  
  Uses SymPy to derive the Schur-complement structure of the exact 4x4 Green function.
  It shows explicitly that the off-diagonal spin-mixing term is
  `V*(conj(D_down) + D_up)` and cancels only under the Kramers/TRS condition
  `D_down = -conj(D_up)`.

- `ciss_logic_z3.py`  
  Uses Z3 to prove logical implications:
  1. TRS implies vanishing Schur off-diagonal interference.
  2. Vanishing Schur off-diagonal implies equal spin-flip magnitudes.
  3. Unequal spin-flip magnitudes imply the off-diagonal term is nonzero for `V != 0`.
  4. Hermiticity is encoded as `D2 = D1.conjugate()`.

- `ciss_numeric_benchmark.py`  
  Computes spin-resolved transport with the exact 4x4 Green function and compares it with
  the decoupled-channel analytical limit. Energies are in eV.

- `symbolic_audit_output.txt`, `z3_logic_output.txt`, `numeric_quick_output.txt`  
  Outputs from the first validation pass.

- `results/bias_scan_exact_vs_decoupled.csv` and `results/bias_scan_polarization.png`  
  Example bias scan comparing exact 4x4 transport and the decoupled limit.

## How to run

```bash
python ciss_symbolic_audit.py
python ciss_logic_z3.py
python ciss_numeric_benchmark.py --quick
python ciss_numeric_benchmark.py --scan-bias --outdir results
```

## Dependencies

```bash
pip install numpy scipy sympy z3-solver matplotlib
```

## Main takeaway from the first pass

- The channel-decoupled expression is exact in the TRS/Kramers limit.
- Once `|D_up| != |D_down|`, the Schur off-diagonal term is necessarily nonzero.
- Therefore, the exact 4x4 Green function should be the main numerical benchmark in the TRS-broken effective model.
- The example high-bias benchmark does not show a large 90% polarization for the exact 4x4 model with the default parameters used here; this must be checked against the precise Mathematica parameterization and bias convention used for the published figures.

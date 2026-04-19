# CISS Ladder Transport

Repository for the manuscript-scale codebase used to study Green-function transport and current-induced spin polarization in a two-leg ladder model with spin-orbit coupling.

This repository now has two complementary layers:

- preserved historical scripts and post-processing assets for archival traceability
- cleaned, repository-ready Python modules that make the channel basis, Green-function workflow, and observable reconstruction easier to audit

## Authors

- D. Verrilli
- N. Bolivar

## Repository Layout

- `legacy/`
  Historical source files kept as record. This folder now includes both the normalized legacy filenames from the older repository scaffold and the exact downloaded filenames provided for the latest audit.
- `src/ciss_ladder_transport/`
  Cleaned reusable modules for basis bookkeeping, Hamiltonian / Green-matrix construction, stochastic realizations, observable assembly, and Landauer-style post-processing.
- `scripts/run_coherent.py`
  Clean entry point for the coherent calculations that were previously bundled in one monolithic legacy script.
- `scripts/run_dephasing.py`
  Clean entry point for the realization-averaged dephasing `Gz(E)` workflow.
- `scripts/run_disorder.py`
  Clean entry point for the realization-averaged disorder `Gz(E)` workflow.
- `scripts/current_from_transmission.py`
  Existing post-processing script for current-from-transmission calculations.
- `notebooks/`
  Colab-oriented materials retained from the previous repository version.
- `docs/`
  Existing repository notes plus publication and Colab guidance.
- `TECHNICAL_NOTE.md`
  Detailed audit of what the legacy transport code appears to compute, with facts separated from interpretation.
- `MAPPING_LEGACY_TO_CLEAN.md`
  Map from legacy files/functions/variables to the cleaned layout.
- `SUMMARY.md`
  Short summary of what was clarified and what remains ambiguous.

## Scientific Scope

The preserved scripts cover three main transport workflows:

1. coherent charge-like and spin-resolved spectral kernels in the ladder model
2. spin-`z` transport under phenomenological local dephasing-like imaginary onsite broadening
3. spin-`z` transport under static Anderson-type onsite disorder

The cleaned implementation is best interpreted as a channel-resolved calculation in an explicit `8N` basis rather than as source code written directly in compact matrix-trace notation.

## Basis And Source Convention

The transport core uses an explicit block ordering:

1. chain 1, `xi=+1`, spin up
2. chain 1, `xi=+1`, spin down
3. chain 1, `xi=-1`, spin up
4. chain 1, `xi=-1`, spin down
5. chain 2, `xi=+1`, spin up
6. chain 2, `xi=+1`, spin down
7. chain 2, `xi=-1`, spin up
8. chain 2, `xi=-1`, spin down

The legacy source vector `B` injects the four left-edge `xi=+1` channels simultaneously. Final charge and spin observables are reconstructed explicitly from selected right-edge Green-function components.

## Running The Cleaned Workflows

Examples:

```powershell
python scripts/run_coherent.py
python scripts/run_coherent.py --case gz
python scripts/run_dephasing.py --realizations 100
python scripts/run_disorder.py --realizations 100
python scripts/current_from_transmission.py --config configs/dephasing_current_example.json
```

Notes:

- The stochastic defaults preserve the legacy choices `N=91`, `M=10000`, and `901` energy points, so they are computationally heavy.
- For quick tests, reduce `--realizations` and optionally `--points`.
- The cleaned scripts write outputs under repository-relative folders instead of notebook-era ad hoc paths.

## Key Audit Findings

- The legacy code is highly repetitive but structurally consistent enough to refactor without redefining the main numerical workflow.
- The disorder and dephasing scripts average amplitudes first and only then construct `|<T>|^2`-type kernels. They do not directly compute `<|T|^2>`.
- The dephasing script adds random imaginary onsite terms but does not, from code alone, justify a strong claim of a full self-consistent Buttiker-probe solution.
- All three transport builders use a hard-coded phase increment `2*pi/(10-1)`, which may be intentional or may be a leftover constant. It is preserved and documented, not silently changed.
- The coherent legacy script contains one advanced-matrix phase inconsistency in a chain-2 backward Rashba term. The cleaned code preserves that behavior behind an explicit flag and documents it as a likely coding issue rather than quietly correcting it.

## Recommended Reading

- [TECHNICAL_NOTE.md](C:/Users/Nelson/Downloads/ciss-ladder-transport/TECHNICAL_NOTE.md)
- [MAPPING_LEGACY_TO_CLEAN.md](C:/Users/Nelson/Downloads/ciss-ladder-transport/MAPPING_LEGACY_TO_CLEAN.md)
- [SUMMARY.md](C:/Users/Nelson/Downloads/ciss-ladder-transport/SUMMARY.md)
- [docs/code_audit.md](C:/Users/Nelson/Downloads/ciss-ladder-transport/docs/code_audit.md)

## License

Released under the [MIT License](C:/Users/Nelson/Downloads/ciss-ladder-transport/LICENSE).

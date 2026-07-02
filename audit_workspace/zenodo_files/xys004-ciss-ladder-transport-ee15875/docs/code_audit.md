# Code Audit Notes

These notes summarize issues observed while organizing the original scripts into a repository. They are not accusations of incorrect physics; they are software and reproducibility flags that should be visible before publication.

## Structural findings

- The coherent script mixes four different studies in one file: charge conductance, `Gz`, `Gx`, and `Gy`.
- The dephasing and disorder scripts duplicate most of the Green-function machinery with small changes in diagonal terms and averaging strategy.
- The current-integration scripts duplicate each other and mainly differ in file paths and case labels.
- The legacy scripts rely on top-level execution rather than importable workflow functions.

## Reproducibility findings

- `current_vs_length_dephasing_legacy.py` and `current_vs_length_disorder_legacy.py` require Google Colab and Google Drive.
- File paths are hardcoded to notebook-era directories instead of repository-relative paths.
- Several output names are inconsistent or contain typos, for example `withou` and a disorder plot name reused for dephasing.
- The original scripts do not expose a single command-line interface or configuration layer.

## Numerical and software review findings

- In the transmission builders, `Dphi = 2*pi/(10-1)` is hardcoded. If this is meant to depend on chain geometry rather than fixed site count, it should be documented explicitly.
- The disorder and dephasing scripts compute the average amplitude first and then square it in the final `Gz` expression. That may be intentional, but it is not the same as averaging `|T|^2` over realizations.
- The current scripts use a chemical-potential convention that deserves confirmation before any physics rewrite. This repository keeps a legacy-compatible option for post-processing.
- Many imports are unused, which makes dependency intent harder to read.
- Names such as `list`, `i`, `eta01`, and `l_R12` reduce readability and, in some cases, shadow Python built-ins or appear unused.

## Repository decision taken here

- The original code is preserved under `legacy/`.
- The current-integration stage has a new local script with JSON configuration.
- Documentation now explains the intended data layout and publication workflow.

## Suggested next cleanup pass

- Extract a single reusable Hamiltonian builder for coherent, dephasing, and disorder variants.
- Replace top-level execution blocks with parameterized functions and CLI commands.
- Add a small regression dataset and one smoke test for each workflow.
- Decide whether the public release should preserve the exact historical formulas or include a validated numerical cleanup.

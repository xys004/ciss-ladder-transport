This release reorganizes the repository into a cleaner referee-facing structure while preserving traceability to the historical numerical workflow.

## Added

- referee-facing package under `src/ciss_ladder/` with documented modules for basis ordering, parameters, effective operators, Green-function solves, observable assembly, disorder, dephasing, averaging, I/O, and plotting
- dedicated run scripts for coherent, disorder-averaged, and dephasing-averaged workflows
- figure-generation scripts for the manuscript-facing `T^z(E)`, `T^0(E)`, and `I_z` outputs
- technical documentation covering physics conventions, legacy-to-clean mapping, and reproducibility
- lightweight structural tests for basis indexing, operator dimensions, output writing, and a zero-kernel sanity check

## Changed

- top-level `README.md` now presents the repository as a transparent, qualitative, channel-resolved transport archive rather than as a brand-new implementation
- legacy documentation now explicitly explains how preserved historical scripts map onto the cleaned referee-facing layer
- output layout is now organized around `data/raw/`, `data/processed/`, and `figures/`

## Purpose

This version is intended to support manuscript review by making the codebase easier to audit, reproduce, and interpret physically, while preserving the historical numerical logic used in the associated work.

# Release Notes v0.1.0

Initial public archive for the codebase associated with:

`A Green Function Formalism for Current Induced Spin Polarization`

## Included in this release

- preserved legacy research scripts for coherent, dephasing, and disorder transport calculations,
- a cleaned local post-processing script for current-vs-length integration,
- repository documentation for archival clarity,
- citation metadata for software and manuscript references,
- example JSON configurations for reproducible local post-processing,
- Zenodo metadata for software deposition.

## Notes

- The `legacy/` directory preserves the original scripts as archival reference.
- The post-processing helper in `scripts/current_from_transmission.py` keeps the legacy current-window convention by default to avoid silently changing numerical results.
- The code audit in `docs/code_audit.md` documents software and reproducibility caveats that should remain visible to users of the archive.

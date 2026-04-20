# Notes On Legacy Files

This directory preserves the historical numerical scripts used during the original workflow.

## Preservation policy

- The files in `legacy/` are kept primarily for traceability.
- File-level normalization was allowed where it improved naming consistency.
- The numerical logic was not silently rewritten inside the legacy files.

## What the referee-facing repository does with them

- The `src/ciss_ladder/` package documents the basis, operators, spectral kernels, and averaging rules using cleaner names.
- The `scripts/` directory provides reproducible entry points that map to coherent, disorder, and dephasing workflows.
- The `docs/` directory records what is explicit in the code and what remains an interpretation of the channel-resolved implementation.

## Important honesty note

Some legacy quantities are better understood as channel-resolved Green-function components or transmission-like amplitudes than as fully abstracted compact transport traces. The cleaned repository documents this explicitly rather than hiding it.

## Legacy files in this archive

- coherent transmission scripts
- disorder-averaged spin-`z` transport scripts
- dephasing-averaged spin-`z` transport scripts
- current-vs-length post-processing scripts

When in doubt, the cleaned code should be read as a documented wrapper around the historical numerical logic rather than as a brand-new implementation detached from the legacy files.

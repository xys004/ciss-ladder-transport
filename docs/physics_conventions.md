# Physics Conventions

This repository uses the following terminology consistently.

## Spectral quantities

- `T^0(E)`: charge-transmission kernel
- `T^z(E)`: projected spin-transmission kernel

These are energy-dependent kernels. They are not yet integrated currents.

## Integrated quantities

- `I_z`: integrated spin current obtained after the Landauer-type energy integration

Whenever possible, the repository uses the word `kernel` for spectral outputs and `integrated current` for post-processed observables.

## Coherent / disorder / dephasing

- coherent: no random onsite disorder and no phenomenological dephasing broadening
- disorder: Anderson-type random onsite energies
- dephasing: local complex probe-like self-energy terms introduced phenomenologically

## Honesty policy

The code comments and docs distinguish between:

- what is explicit in the implementation,
- what is a physical interpretation of the channel-resolved workflow,
- what is preserved from the legacy scripts for traceability.

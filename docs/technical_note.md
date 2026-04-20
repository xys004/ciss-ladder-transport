# Technical Note

This repository implements Green-function calculations for a two-leg ladder model in an explicit channel-resolved representation.

## What is explicit in the code

- the ladder basis is flattened into an `8N` vector
- the spectral quantities are assembled from selected channel components
- coherent, disorder, and dephasing workflows are separate
- integrated observables are obtained only after the spectral kernels are computed

## Spectral kernels versus integrated observables

Primary spectral outputs:

- `T^0(E)`: charge-transmission kernel
- `T^z(E)`: projected spin-transmission kernel

Derived outputs:

- `I_z`: integrated spin current obtained through a Landauer-type energy integration
- optional polarization-like ratios assembled afterward

The code therefore computes kernels first and integrated observables second.

## Channel-resolved implementation

The historical scripts do not start from a compact trace expression written directly in code. Instead, they solve an explicit channel-resolved linear problem and reconstruct the relevant transport kernels from selected Green-function components. The cleaned package keeps that logic visible.

## Disorder

The disorder implementation uses Anderson-type random onsite energies drawn independently for each realization. The numerical workflow follows the legacy averaging rule: complex channel amplitudes are averaged over realizations, and the final kernel is assembled afterward.

## Dephasing

The dephasing implementation is phenomenological. It introduces local complex probe-like self-energy terms in the effective operator. This broadens the channel-resolved Green-function description and is best described as a phenomenological dephasing model in the spirit of probe-based approaches.

## Remaining ambiguities

- Some intermediate legacy amplitudes are best interpreted operationally rather than as unique compact observables.
- The historical scripts average amplitudes before kernel assembly in the stochastic workflows.
- The preserved logic should therefore be read as legacy-informed and traceable, not as a freshly derived compact theoretical implementation.

"""Referee-facing helpers for the CISS ladder transport repository.

This package documents the historical numerical workflow in a cleaner layout.
The numerical backend remains traceable to the preserved and earlier-cleaned
implementations under ``legacy/`` and ``src/ciss_ladder_transport/``.
"""

from .averaging import average_channel_components
from .basis import BASIS_BLOCKS, basis_index, describe_basis, make_legacy_source_vector
from .dephasing import build_dephasing_realizations
from .disorder import build_disorder_realizations
from .greens import (
    solve_channel_resolved_green_problem,
    sweep_channel_components,
)
from .hamiltonian import build_effective_operator
from .io_utils import save_metadata_json, save_spectral_kernel_csv
from .observables import (
    charge_transmission_kernel,
    spin_transmission_x_kernel,
    spin_transmission_y_kernel,
    spin_transmission_z_kernel,
)
from .parameters import (
    LadderParameters,
    SweepParameters,
    make_default_coherent_parameters,
    make_default_dephasing_parameters,
    make_default_disorder_parameters,
)

__all__ = [
    "BASIS_BLOCKS",
    "LadderParameters",
    "SweepParameters",
    "average_channel_components",
    "basis_index",
    "build_dephasing_realizations",
    "build_disorder_realizations",
    "build_effective_operator",
    "charge_transmission_kernel",
    "describe_basis",
    "make_default_coherent_parameters",
    "make_default_dephasing_parameters",
    "make_default_disorder_parameters",
    "make_legacy_source_vector",
    "save_metadata_json",
    "save_spectral_kernel_csv",
    "solve_channel_resolved_green_problem",
    "spin_transmission_x_kernel",
    "spin_transmission_y_kernel",
    "spin_transmission_z_kernel",
    "sweep_channel_components",
]

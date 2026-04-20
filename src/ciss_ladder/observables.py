"""Assembly of spectral transport kernels from channel-resolved components.

The primary outputs are energy-dependent kernels:

- ``T^0(E)`` for charge transport
- ``T^z(E)`` for projected spin transport along the propagation axis

Integrated observables such as ``I_z`` are obtained only afterward.
"""

from __future__ import annotations

import numpy as np

from ciss_ladder_transport.observables import charge_kernel, gx_kernel, gy_kernel, gz_kernel


def charge_transmission_kernel(group_1, group_2) -> np.ndarray:
    """Assemble the charge-transmission kernel ``T^0(E)``.

    The input objects are channel-resolved Green-function amplitudes or derived
    transmission-like components as produced by the explicit legacy workflow.
    """

    return charge_kernel(group_1, group_2)


def spin_transmission_z_kernel(group_1, group_2) -> np.ndarray:
    """Assemble the projected spin-transmission kernel ``T^z(E)``."""

    return gz_kernel(group_1, group_2)


def spin_transmission_x_kernel(group_1, group_2) -> np.ndarray:
    """Assemble the transverse spin kernel ``T^x(E)`` when needed."""

    return gx_kernel(group_1, group_2)


def spin_transmission_y_kernel(group_1, group_2) -> np.ndarray:
    """Assemble the transverse spin kernel ``T^y(E)`` when needed."""

    return gy_kernel(group_1, group_2)

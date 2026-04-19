"""Observable reconstruction from the explicit channel amplitudes."""

from __future__ import annotations

from typing import Dict

import numpy as np


ChannelTraces = Dict[str, np.ndarray]


def legacy_components(group_1: ChannelTraces, group_2: ChannelTraces) -> Dict[str, np.ndarray]:
    """Map cleaned channel names onto the historical `Trans1*` / `Trans2*` labels."""

    return {
        "Trans1u": group_1["c1_xi_plus_up"],
        "Trans1d": group_1["c1_xi_plus_down"],
        "Trans1ud": group_1["c1_xi_minus_up"],
        "Trans1du": group_1["c1_xi_minus_down"],
        "Trans2u": group_2["c2_xi_plus_up"],
        "Trans2d": group_2["c2_xi_plus_down"],
        "Trans2ud": group_2["c2_xi_minus_up"],
        "Trans2du": group_2["c2_xi_minus_down"],
    }


def _abs_squared(amplitude: np.ndarray) -> np.ndarray:
    return amplitude * np.conjugate(amplitude)


def charge_kernel(group_1: ChannelTraces, group_2: ChannelTraces) -> np.ndarray:
    """Reproduce the coherent legacy charge kernel."""

    components = legacy_components(group_1, group_2)
    return (
        _abs_squared(components["Trans1u"])
        + _abs_squared(components["Trans1du"])
        + _abs_squared(components["Trans1ud"])
        + _abs_squared(components["Trans1d"])
        + _abs_squared(components["Trans2u"])
        + _abs_squared(components["Trans2du"])
        + _abs_squared(components["Trans2ud"])
        + _abs_squared(components["Trans2d"])
    )


def gz_kernel(group_1: ChannelTraces, group_2: ChannelTraces) -> np.ndarray:
    """Reproduce the legacy spin-z kernel."""

    components = legacy_components(group_1, group_2)
    return (
        _abs_squared(components["Trans1u"])
        - _abs_squared(components["Trans1du"])
        + _abs_squared(components["Trans1ud"])
        - _abs_squared(components["Trans1d"])
        + _abs_squared(components["Trans2u"])
        - _abs_squared(components["Trans2du"])
        + _abs_squared(components["Trans2ud"])
        - _abs_squared(components["Trans2d"])
    )


def gx_kernel(group_1: ChannelTraces, group_2: ChannelTraces) -> np.ndarray:
    """Reproduce the coherent legacy spin-x kernel."""

    components = legacy_components(group_1, group_2)
    return 4.0 * (
        components["Trans1du"] * np.conjugate(components["Trans1u"])
        + components["Trans1u"] * np.conjugate(components["Trans1du"])
        + components["Trans2du"] * np.conjugate(components["Trans2u"])
        + components["Trans2u"] * np.conjugate(components["Trans2du"])
        + components["Trans1d"] * np.conjugate(components["Trans1ud"])
        + components["Trans1ud"] * np.conjugate(components["Trans1d"])
        + components["Trans2d"] * np.conjugate(components["Trans2ud"])
        + components["Trans2ud"] * np.conjugate(components["Trans2d"])
    )


def gy_kernel(group_1: ChannelTraces, group_2: ChannelTraces) -> np.ndarray:
    """Reproduce the coherent legacy spin-y kernel."""

    components = legacy_components(group_1, group_2)
    return 4.0j * (
        components["Trans1du"] * np.conjugate(components["Trans1u"])
        - components["Trans1u"] * np.conjugate(components["Trans1du"])
        + components["Trans2du"] * np.conjugate(components["Trans2u"])
        - components["Trans2u"] * np.conjugate(components["Trans2du"])
        + components["Trans1d"] * np.conjugate(components["Trans1ud"])
        - components["Trans1ud"] * np.conjugate(components["Trans1d"])
        + components["Trans2d"] * np.conjugate(components["Trans2ud"])
        - components["Trans2ud"] * np.conjugate(components["Trans2d"])
    )

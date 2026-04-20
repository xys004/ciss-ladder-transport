"""Green-function solves in the explicit channel-resolved representation."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from ciss_ladder_transport.config import LeadProfiles, SampleArrays
from ciss_ladder_transport.greens import average_channels, solve_green_vectors, sweep_channels

from .parameters import LadderParameters, SweepParameters

ChannelComponents = Dict[str, np.ndarray]


def solve_channel_resolved_green_problem(
    energy: float,
    parameters: LadderParameters,
    leads: LeadProfiles,
    sample: SampleArrays,
    source_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the retarded and advanced channel-resolved linear systems.

    The legacy implementation reconstructs the relevant entrance-to-exit
    Green-function components in an explicit channel basis, rather than forming
    the compact transmission trace directly.
    """

    return solve_green_vectors(
        energy=energy,
        model=parameters.to_backend(),
        leads=leads,
        sample=sample,
        source_vector=source_vector,
    )


def sweep_channel_components(
    parameters: LadderParameters,
    leads: LeadProfiles,
    sample: SampleArrays,
    sweep: SweepParameters,
    source_vector: np.ndarray,
) -> Tuple[np.ndarray, ChannelComponents]:
    """Compute the channel-resolved components for one deterministic sample."""

    return sweep_channels(
        model=parameters.to_backend(),
        leads=leads,
        sample=sample,
        sweep=sweep.to_backend(),
        source_vector=source_vector,
    )


def average_channel_components(
    parameters: LadderParameters,
    leads: LeadProfiles,
    samples: Iterable[SampleArrays],
    sweep: SweepParameters,
    source_vector: np.ndarray,
) -> Tuple[np.ndarray, ChannelComponents]:
    """Average channel-resolved amplitudes over realizations.

    This preserves the legacy rule: the complex amplitudes are averaged first,
    and spectral kernels are assembled afterward from those averaged channel
    components.
    """

    return average_channels(
        model=parameters.to_backend(),
        leads=leads,
        samples=samples,
        sweep=sweep.to_backend(),
        source_vector=source_vector,
    )

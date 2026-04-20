"""Effective-operator construction for the ladder transport problem.

This module marks clearly which terms belong to:

- coherent Hamiltonian couplings,
- lead broadening/self-energy contributions,
- disorder onsite terms,
- phenomenological dephasing-like complex terms.
"""

from __future__ import annotations

from ciss_ladder_transport.config import LeadProfiles, SampleArrays
from ciss_ladder_transport.greens import build_green_matrix

from .parameters import LadderParameters


def build_effective_operator(
    energy: float,
    parameters: LadderParameters,
    leads: LeadProfiles,
    sample: SampleArrays,
    advanced: bool = False,
):
    """Build the channel-resolved effective operator at one energy.

    Mathematically, this is the ``8N x 8N`` linear operator solved by the
    historical scripts at fixed energy. Physically, it contains the coherent
    ladder couplings plus the lead broadening terms and any disorder or
    phenomenological dephasing terms included in the chosen sample.
    """

    return build_green_matrix(
        energy=energy,
        model=parameters.to_backend(),
        leads=leads,
        sample=sample,
        advanced=advanced,
    )

"""Phenomenological dephasing samples for the ladder transport workflow."""

from __future__ import annotations

from ciss_ladder_transport.randomness import build_dephasing_samples, zero_mean_dephasing_realization


def sample_dephasing_profile(num_sites: int, rng):
    """Return one zero-mean local complex-broadening profile."""

    return zero_mean_dephasing_realization(num_sites, rng)


def build_dephasing_realizations(num_sites: int, count: int, eta_d: float, seed: int | None = None):
    """Build phenomenological dephasing realizations.

    The implementation introduces local complex probe-like self-energy terms in
    the effective operator. It should be read as a phenomenological dephasing
    model in the spirit of probe-based broadening schemes, not as a hidden
    claim of a full self-consistent probe network.
    """

    return build_dephasing_samples(num_sites=num_sites, count=count, eta_scale=eta_d, seed=seed)

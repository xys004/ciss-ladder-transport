"""Anderson-type disorder realizations for the ladder transport workflow."""

from __future__ import annotations

from ciss_ladder_transport.randomness import anderson_disorder_realization, build_disorder_samples


def sample_disorder_profile(num_sites: int, rng):
    """Return one Anderson-type onsite profile in the legacy interval."""

    return anderson_disorder_realization(num_sites, rng)


def build_disorder_realizations(num_sites: int, count: int, eta: float, seed: int | None = None):
    """Build independent disorder realizations for both chains.

    Disorder is introduced through random onsite energies. The referee-facing
    code keeps explicit that the historical workflow averages complex channel
    amplitudes over realizations before assembling the final kernel.
    """

    return build_disorder_samples(num_sites=num_sites, count=count, eta=eta, seed=seed)

"""Configuration helpers for the cleaned channel-resolved transport code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SweepSpec:
    """Energy grid used in one spectral sweep."""

    start: float
    stop: float
    num_points: int

    def energies(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.num_points)


@dataclass(frozen=True)
class LadderModel:
    """Model parameters for the two-leg ladder.

    The field names follow the cleaned code. They still map closely onto the
    legacy variables `gamma01`, `gamma02`, `gamma_per`, `gamma_per1`,
    `l_R1`, `l_R2`, and `l_D`.
    """

    num_sites: int
    gamma_chain_1: float
    gamma_chain_2: float
    gamma_per: float
    gamma_per1: float
    rashba_chain_1: float
    rashba_chain_2: float
    dresselhaus: float
    beta: float = np.pi
    phase_site_count: int = 10
    use_legacy_advanced_chain2_backward_phase: bool = False

    def phase_step(self) -> float:
        """Return the legacy Rashba phase increment.

        The original scripts hard-code `2*pi/(10-1)` rather than `2*pi/(N-1)`.
        The cleaned code preserves that behavior by default and documents it as
        a legacy modeling choice / possible typo.
        """

        return 2.0 * np.pi / (self.phase_site_count - 1)


@dataclass(frozen=True)
class LeadProfiles:
    """Diagonal lead-broadening profiles for the eight explicit channels."""

    chain1_block_gammas: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    chain2_block_gammas: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    def validate(self, num_sites: int) -> None:
        for gamma_profile in (*self.chain1_block_gammas, *self.chain2_block_gammas):
            if len(gamma_profile) != num_sites:
                raise ValueError("All lead profiles must have length num_sites.")


@dataclass(frozen=True)
class SampleArrays:
    """One realization of onsite terms and imaginary broadening terms."""

    onsite_chain_1: np.ndarray
    onsite_chain_2: np.ndarray
    imag_chain_1: np.ndarray
    imag_chain_2: np.ndarray

    def validate(self, num_sites: int) -> None:
        arrays = (
            self.onsite_chain_1,
            self.onsite_chain_2,
            self.imag_chain_1,
            self.imag_chain_2,
        )
        if any(len(array) != num_sites for array in arrays):
            raise ValueError("All sample arrays must have length num_sites.")


def constant_real_array(num_sites: int, value: float = 0.0) -> np.ndarray:
    """Create a real-valued onsite array."""

    return np.full(num_sites, value, dtype=float)


def constant_imaginary_shift(num_sites: int, eta: float) -> np.ndarray:
    """Create the real prefactor of the `i*eta` diagonal term."""

    return np.full(num_sites, eta, dtype=float)


def make_uniform_sample(num_sites: int, onsite_1: float = 0.0, onsite_2: float = 0.0, eta: float = 0.0) -> SampleArrays:
    """Create one spatially uniform sample."""

    return SampleArrays(
        onsite_chain_1=constant_real_array(num_sites, onsite_1),
        onsite_chain_2=constant_real_array(num_sites, onsite_2),
        imag_chain_1=constant_imaginary_shift(num_sites, eta),
        imag_chain_2=constant_imaginary_shift(num_sites, eta),
    )


def _edge_profile(num_sites: int, left: complex, right: complex) -> np.ndarray:
    """Create one lead-broadening array with support only at the chain ends."""

    profile = np.zeros(num_sites, dtype=complex)
    profile[0] = left
    profile[-1] = right
    return profile


def make_coherent_leads(num_sites: int, p: float = 0.0) -> LeadProfiles:
    """Reproduce the two-profile lead choice from the coherent legacy script."""

    val = 1.0 * (1.0 + p) * 1j
    val1 = 1.0 * 1j
    gamma_chain_1 = _edge_profile(num_sites, val, val)
    gamma_chain_2 = _edge_profile(num_sites, val1, val1)
    return LeadProfiles(
        chain1_block_gammas=(gamma_chain_1,) * 4,
        chain2_block_gammas=(gamma_chain_2,) * 4,
    )


def make_spin_resolved_leads(num_sites: int, p: float = 0.0) -> LeadProfiles:
    """Reproduce the four `Gamma1..Gamma4` arrays from the stochastic scripts."""

    val = 1.0 * (1.0 + p) * 1j
    val1 = 1.0 * (1.0 - p) * 1j
    gamma1 = _edge_profile(num_sites, val, val)
    gamma2 = _edge_profile(num_sites, val, val1)
    gamma3 = _edge_profile(num_sites, val1, val)
    gamma4 = _edge_profile(num_sites, val1, val1)
    return LeadProfiles(
        chain1_block_gammas=(gamma1, gamma2, gamma3, gamma4),
        chain2_block_gammas=(gamma1, gamma2, gamma3, gamma4),
    )

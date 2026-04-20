"""Model and sweep parameters for the referee-facing ladder package.

This module keeps the historical numerical choices visible while giving them
names that are easier to interpret in the manuscript and repository docs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ciss_ladder_transport.config import LadderModel, SweepSpec


@dataclass(frozen=True)
class LadderParameters:
    """Named parameter set for the two-leg ladder model.

    The object is a thin referee-facing wrapper around the backend
    ``LadderModel`` dataclass. The numerical logic is unchanged; the goal here
    is interpretability and explicit terminology.
    """

    num_sites: int
    gamma_in_chain_1: float
    gamma_in_chain_2: float
    gamma_out_parallel: float
    gamma_out_spin_mixing: float
    lambda_soc_chain_1: float
    lambda_soc_chain_2: float
    dresselhaus: float = 0.0
    beta: float = 3.141592653589793
    phase_site_count: int = 10
    use_legacy_advanced_chain2_backward_phase: bool = False

    def to_backend(self) -> LadderModel:
        return LadderModel(
            num_sites=self.num_sites,
            gamma_chain_1=self.gamma_in_chain_1,
            gamma_chain_2=self.gamma_in_chain_2,
            gamma_per=self.gamma_out_parallel,
            gamma_per1=self.gamma_out_spin_mixing,
            rashba_chain_1=self.lambda_soc_chain_1,
            rashba_chain_2=self.lambda_soc_chain_2,
            dresselhaus=self.dresselhaus,
            beta=self.beta,
            phase_site_count=self.phase_site_count,
            use_legacy_advanced_chain2_backward_phase=self.use_legacy_advanced_chain2_backward_phase,
        )


@dataclass(frozen=True)
class SweepParameters:
    """Energy-window parameters for one spectral calculation."""

    energy_min: float
    energy_max: float
    num_points: int

    def to_backend(self) -> SweepSpec:
        return SweepSpec(self.energy_min, self.energy_max, self.num_points)


def make_default_coherent_parameters(num_sites: int = 10) -> LadderParameters:
    """Return the coherent parameter set used for the referee-facing examples."""

    return LadderParameters(
        num_sites=num_sites,
        gamma_in_chain_1=1.0,
        gamma_in_chain_2=1.0,
        gamma_out_parallel=0.0,
        gamma_out_spin_mixing=1.0,
        lambda_soc_chain_1=0.1,
        lambda_soc_chain_2=0.1,
        use_legacy_advanced_chain2_backward_phase=True,
    )


def make_default_dephasing_parameters(num_sites: int = 91) -> LadderParameters:
    """Return the dephasing parameter set used in the historical workflow."""

    return LadderParameters(
        num_sites=num_sites,
        gamma_in_chain_1=1.0,
        gamma_in_chain_2=1.0,
        gamma_out_parallel=0.0,
        gamma_out_spin_mixing=1.0,
        lambda_soc_chain_1=0.1,
        lambda_soc_chain_2=0.1,
    )


def make_default_disorder_parameters(num_sites: int = 91) -> LadderParameters:
    """Return the disorder parameter set used in the historical workflow."""

    return make_default_dephasing_parameters(num_sites=num_sites)

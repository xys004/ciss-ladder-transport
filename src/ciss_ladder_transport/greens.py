"""Green-function builders and spectral sweeps for the cleaned ladder code."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .basis import CHANNEL_OUTPUT_ORDER, channel_output_indices, site_index
from .config import LadderModel, LeadProfiles, SampleArrays, SweepSpec


ChannelTraces = Dict[str, np.ndarray]


def build_green_matrix(
    energy: float,
    model: LadderModel,
    leads: LeadProfiles,
    sample: SampleArrays,
    advanced: bool = False,
) -> sparse.csr_matrix:
    """Build the legacy 8N x 8N linear system for one energy.

    This function follows the historical code channel by channel. The main
    cleanup is structural: the coherent, disorder, and dephasing cases all use
    the same builder with different onsite and imaginary arrays.
    """

    num_sites = model.num_sites
    leads.validate(num_sites)
    sample.validate(num_sites)

    phase_step = model.phase_step()
    imag_sign = -1.0 if advanced else 1.0
    matrix = np.zeros((8 * num_sites, 8 * num_sites), dtype=complex)

    for n in range(1, num_sites + 1):
        site = n - 1
        diagonal_site = (n - 1) % num_sites

        chain1_gammas = leads.chain1_block_gammas
        chain2_gammas = leads.chain2_block_gammas

        for block_index, gamma_profile in enumerate(chain1_gammas):
            row = site_index(block_index, site, num_sites)
            matrix[row, row] = -(
                energy
                - sample.onsite_chain_1[site]
                + gamma_profile[diagonal_site]
                + imag_sign * sample.imag_chain_1[site] * 1j
            )

        for block_index, gamma_profile in enumerate(chain2_gammas):
            row = site_index(block_index + 4, site, num_sites)
            matrix[row, row] = -(
                energy
                - sample.onsite_chain_2[site]
                + gamma_profile[diagonal_site]
                + imag_sign * sample.imag_chain_2[site] * 1j
            )

        # Perpendicular inter-chain couplings.
        matrix[site_index(0, site, num_sites), site_index(4, site, num_sites)] = model.gamma_per
        matrix[site_index(1, site, num_sites), site_index(5, site, num_sites)] = model.gamma_per
        matrix[site_index(2, site, num_sites), site_index(6, site, num_sites)] = model.gamma_per
        matrix[site_index(3, site, num_sites), site_index(7, site, num_sites)] = model.gamma_per
        matrix[site_index(4, site, num_sites), site_index(0, site, num_sites)] = model.gamma_per
        matrix[site_index(5, site, num_sites), site_index(1, site, num_sites)] = model.gamma_per
        matrix[site_index(6, site, num_sites), site_index(2, site, num_sites)] = model.gamma_per
        matrix[site_index(7, site, num_sites), site_index(3, site, num_sites)] = model.gamma_per

        matrix[site_index(0, site, num_sites), site_index(6, site, num_sites)] = model.gamma_per1
        matrix[site_index(1, site, num_sites), site_index(7, site, num_sites)] = model.gamma_per1
        matrix[site_index(2, site, num_sites), site_index(4, site, num_sites)] = model.gamma_per1
        matrix[site_index(3, site, num_sites), site_index(5, site, num_sites)] = model.gamma_per1
        matrix[site_index(4, site, num_sites), site_index(2, site, num_sites)] = model.gamma_per1
        matrix[site_index(5, site, num_sites), site_index(3, site, num_sites)] = model.gamma_per1
        matrix[site_index(6, site, num_sites), site_index(0, site, num_sites)] = model.gamma_per1
        matrix[site_index(7, site, num_sites), site_index(1, site, num_sites)] = model.gamma_per1

        # Hopping toward n+1.
        forward_site = n % num_sites
        forward_phase = (n - 1) * phase_step
        forward_hopping_chain_1 = model.gamma_chain_1 if n < num_sites else 0.0
        forward_hopping_chain_2 = model.gamma_chain_2 if n < num_sites else 0.0
        forward_rashba_chain_1 = model.rashba_chain_1 if n < num_sites else 0.0
        forward_rashba_chain_2 = model.rashba_chain_2 if n < num_sites else 0.0

        for block_index in range(4):
            matrix[site_index(block_index, site, num_sites), site_index(block_index, forward_site, num_sites)] += forward_hopping_chain_1
            matrix[site_index(block_index + 4, site, num_sites), site_index(block_index + 4, forward_site, num_sites)] += forward_hopping_chain_2

        matrix[site_index(0, site, num_sites), site_index(2, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_1 * np.exp(-1j * forward_phase)
            + model.dresselhaus * np.exp(1j * forward_phase)
        )
        matrix[site_index(1, site, num_sites), site_index(3, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_1 * np.exp(1j * forward_phase)
            - model.dresselhaus * np.exp(-1j * forward_phase)
        )
        matrix[site_index(2, site, num_sites), site_index(0, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_1 * np.exp(1j * forward_phase)
            - model.dresselhaus * np.exp(-1j * forward_phase)
        )
        matrix[site_index(3, site, num_sites), site_index(1, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_1 * np.exp(-1j * forward_phase)
            + model.dresselhaus * np.exp(1j * forward_phase)
        )

        matrix[site_index(4, site, num_sites), site_index(6, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_2 * np.exp(-1j * forward_phase) * np.exp(-1j * model.beta)
            + model.dresselhaus * np.exp(1j * forward_phase)
        )
        matrix[site_index(5, site, num_sites), site_index(7, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_2 * np.exp(1j * forward_phase) * np.exp(1j * model.beta)
            - model.dresselhaus * np.exp(-1j * forward_phase)
        )
        matrix[site_index(6, site, num_sites), site_index(4, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_2 * np.exp(1j * forward_phase) * np.exp(1j * model.beta)
            - model.dresselhaus * np.exp(-1j * forward_phase)
        )
        matrix[site_index(7, site, num_sites), site_index(5, forward_site, num_sites)] = (
            -1j * forward_rashba_chain_2 * np.exp(-1j * forward_phase) * np.exp(-1j * model.beta)
            + model.dresselhaus * np.exp(1j * forward_phase)
        )

        # Hopping toward n-1.
        backward_site = (n - 2) % num_sites
        backward_phase_chain_1 = (n - 2) * phase_step
        if advanced and model.use_legacy_advanced_chain2_backward_phase:
            backward_phase_chain_2 = (n - 1) * phase_step
        else:
            backward_phase_chain_2 = (n - 2) * phase_step

        backward_hopping_chain_1 = model.gamma_chain_1 if n > 1 else 0.0
        backward_hopping_chain_2 = model.gamma_chain_2 if n > 1 else 0.0
        backward_rashba_chain_1 = model.rashba_chain_1 if n > 1 else 0.0
        backward_rashba_chain_2 = model.rashba_chain_2 if n > 1 else 0.0

        for block_index in range(4):
            matrix[site_index(block_index, site, num_sites), site_index(block_index, backward_site, num_sites)] += backward_hopping_chain_1
            matrix[site_index(block_index + 4, site, num_sites), site_index(block_index + 4, backward_site, num_sites)] += backward_hopping_chain_2

        matrix[site_index(0, site, num_sites), site_index(2, backward_site, num_sites)] += (
            1j * backward_rashba_chain_1 * np.exp(-1j * backward_phase_chain_1)
            - model.dresselhaus * np.exp(1j * backward_phase_chain_1)
        )
        matrix[site_index(1, site, num_sites), site_index(3, backward_site, num_sites)] += (
            1j * backward_rashba_chain_1 * np.exp(1j * backward_phase_chain_1)
            + model.dresselhaus * np.exp(-1j * backward_phase_chain_1)
        )
        matrix[site_index(2, site, num_sites), site_index(0, backward_site, num_sites)] += (
            1j * backward_rashba_chain_1 * np.exp(1j * backward_phase_chain_1)
            + model.dresselhaus * np.exp(-1j * backward_phase_chain_1)
        )
        matrix[site_index(3, site, num_sites), site_index(1, backward_site, num_sites)] += (
            1j * backward_rashba_chain_1 * np.exp(-1j * backward_phase_chain_1)
            - model.dresselhaus * np.exp(1j * backward_phase_chain_1)
        )

        matrix[site_index(4, site, num_sites), site_index(6, backward_site, num_sites)] += (
            1j * backward_rashba_chain_2 * np.exp(-1j * backward_phase_chain_2) * np.exp(-1j * model.beta)
            - model.dresselhaus * np.exp(1j * backward_phase_chain_2)
        )
        matrix[site_index(5, site, num_sites), site_index(7, backward_site, num_sites)] += (
            1j * backward_rashba_chain_2 * np.exp(1j * backward_phase_chain_2) * np.exp(1j * model.beta)
            + model.dresselhaus * np.exp(-1j * backward_phase_chain_2)
        )
        matrix[site_index(6, site, num_sites), site_index(4, backward_site, num_sites)] += (
            1j * backward_rashba_chain_2 * np.exp(1j * backward_phase_chain_2) * np.exp(1j * model.beta)
            + model.dresselhaus * np.exp(-1j * backward_phase_chain_2)
        )
        matrix[site_index(7, site, num_sites), site_index(5, backward_site, num_sites)] += (
            1j * backward_rashba_chain_2 * np.exp(-1j * backward_phase_chain_2) * np.exp(-1j * model.beta)
            - model.dresselhaus * np.exp(1j * backward_phase_chain_2)
        )

    return sparse.csr_matrix(matrix)


def solve_green_vectors(
    energy: float,
    model: LadderModel,
    leads: LeadProfiles,
    sample: SampleArrays,
    source_vector: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the retarded and advanced linear systems for one energy."""

    matrix_retarded = build_green_matrix(energy, model, leads, sample, advanced=False)
    retarded_vector = np.asarray(spsolve(matrix_retarded, source_vector)).ravel()

    matrix_advanced = build_green_matrix(energy, model, leads, sample, advanced=True)
    advanced_vector = np.asarray(spsolve(matrix_advanced, source_vector)).ravel()

    return retarded_vector, advanced_vector


def extract_channel_amplitudes(retarded_vector: np.ndarray, num_sites: int) -> Dict[str, complex]:
    """Extract the eight legacy channel amplitudes from one solved Green vector."""

    output_indices = channel_output_indices(num_sites)
    return {
        channel_name: -retarded_vector[index]
        for channel_name, index in output_indices.items()
    }


def _empty_channel_traces(num_points: int) -> ChannelTraces:
    return {
        channel_name: np.zeros(num_points, dtype=complex)
        for channel_name in CHANNEL_OUTPUT_ORDER
    }


def sweep_channels(
    model: LadderModel,
    leads: LeadProfiles,
    sample: SampleArrays,
    sweep: SweepSpec,
    source_vector: np.ndarray,
) -> Tuple[np.ndarray, ChannelTraces]:
    """Compute all eight channel amplitudes for one deterministic sample."""

    energies = sweep.energies()
    traces = _empty_channel_traces(len(energies))

    for energy_index, energy in enumerate(energies):
        retarded_vector, _advanced_vector = solve_green_vectors(
            energy=energy,
            model=model,
            leads=leads,
            sample=sample,
            source_vector=source_vector,
        )
        amplitudes = extract_channel_amplitudes(retarded_vector, model.num_sites)
        for channel_name, amplitude in amplitudes.items():
            traces[channel_name][energy_index] = amplitude

    return energies, traces


def average_channels(
    model: LadderModel,
    leads: LeadProfiles,
    samples: Iterable[SampleArrays],
    sweep: SweepSpec,
    source_vector: np.ndarray,
) -> Tuple[np.ndarray, ChannelTraces]:
    """Average the extracted amplitudes over many realizations.

    This preserves the legacy averaging rule:
    first average the complex amplitude `T`, then assemble observables from the
    averaged amplitudes. The cleaned code makes that behavior explicit.
    """

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("At least one sample is required.")

    energies = sweep.energies()
    traces = _empty_channel_traces(len(energies))

    for energy_index, energy in enumerate(energies):
        channel_sums = {channel_name: 0.0j for channel_name in CHANNEL_OUTPUT_ORDER}

        for sample in sample_list:
            retarded_vector, _advanced_vector = solve_green_vectors(
                energy=energy,
                model=model,
                leads=leads,
                sample=sample,
                source_vector=source_vector,
            )
            amplitudes = extract_channel_amplitudes(retarded_vector, model.num_sites)
            for channel_name, amplitude in amplitudes.items():
                channel_sums[channel_name] += amplitude

        normalization = float(len(sample_list))
        for channel_name in CHANNEL_OUTPUT_ORDER:
            traces[channel_name][energy_index] = channel_sums[channel_name] / normalization

    return energies, traces

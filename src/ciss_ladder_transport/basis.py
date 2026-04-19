"""Basis and channel bookkeeping for the legacy two-leg ladder transport code."""

from __future__ import annotations

from typing import Dict

import numpy as np

BLOCK_LABELS = (
    "chain_1_xi_plus_spin_up",
    "chain_1_xi_plus_spin_down",
    "chain_1_xi_minus_spin_up",
    "chain_1_xi_minus_spin_down",
    "chain_2_xi_plus_spin_up",
    "chain_2_xi_plus_spin_down",
    "chain_2_xi_minus_spin_up",
    "chain_2_xi_minus_spin_down",
)

CHANNEL_OUTPUT_ORDER = (
    "c1_xi_plus_up",
    "c1_xi_plus_down",
    "c1_xi_minus_up",
    "c1_xi_minus_down",
    "c2_xi_plus_up",
    "c2_xi_plus_down",
    "c2_xi_minus_up",
    "c2_xi_minus_down",
)

LEGACY_COMPONENT_RENAMING = {
    "Trans1u": "group_1['c1_xi_plus_up']",
    "Trans1d": "group_1['c1_xi_plus_down']",
    "Trans1ud": "group_1['c1_xi_minus_up']",
    "Trans1du": "group_1['c1_xi_minus_down']",
    "Trans2u": "group_2['c2_xi_plus_up']",
    "Trans2d": "group_2['c2_xi_plus_down']",
    "Trans2ud": "group_2['c2_xi_minus_up']",
    "Trans2du": "group_2['c2_xi_minus_down']",
}


def block_offset(block_index: int, num_sites: int) -> int:
    """Return the first flat index of one N-sized block inside the 8N basis."""

    return block_index * num_sites


def site_index(block_index: int, site: int, num_sites: int) -> int:
    """Return the flat basis index for one block/site pair."""

    return block_offset(block_index, num_sites) + site


def channel_output_indices(num_sites: int) -> Dict[str, int]:
    """Return the output-site indices used by the legacy `den_espectral*` helpers."""

    return {
        "c1_xi_plus_up": num_sites - 1,
        "c1_xi_plus_down": 2 * num_sites - 1,
        "c1_xi_minus_up": 3 * num_sites - 1,
        "c1_xi_minus_down": 4 * num_sites - 1,
        "c2_xi_plus_up": 5 * num_sites - 1,
        "c2_xi_plus_down": 6 * num_sites - 1,
        "c2_xi_minus_up": 7 * num_sites - 1,
        "c2_xi_minus_down": 8 * num_sites - 1,
    }


def make_source_vector(num_sites: int) -> np.ndarray:
    """Build the legacy source vector B.

    The historical scripts inject four left-edge channels simultaneously:
    chain 1 / xi=+1 / spin up, chain 1 / xi=+1 / spin down,
    chain 2 / xi=+1 / spin up, and chain 2 / xi=+1 / spin down.
    """

    source = np.zeros(8 * num_sites, dtype=complex)
    source[0] = -1.0
    source[num_sites] = -1.0
    source[4 * num_sites] = -1.0
    source[5 * num_sites] = -1.0
    return source

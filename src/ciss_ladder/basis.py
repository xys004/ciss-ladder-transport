"""Basis ordering used throughout the explicit channel-resolved calculation.

The ladder is represented in an ``8N`` basis. Each block corresponds to a
chain / propagation-sector / spin combination, resolved site by site.
"""

from __future__ import annotations

from typing import Dict

from ciss_ladder_transport.basis import BLOCK_LABELS, make_source_vector, site_index

BASIS_BLOCKS = BLOCK_LABELS


def describe_basis() -> str:
    """Return a human-readable description of the flattened basis ordering."""

    lines = [
        "Explicit 8N channel basis used by the ladder transport calculation:",
        "  block 0: chain 1, sector xi=+1, spin up",
        "  block 1: chain 1, sector xi=+1, spin down",
        "  block 2: chain 1, sector xi=-1, spin up",
        "  block 3: chain 1, sector xi=-1, spin down",
        "  block 4: chain 2, sector xi=+1, spin up",
        "  block 5: chain 2, sector xi=+1, spin down",
        "  block 6: chain 2, sector xi=-1, spin up",
        "  block 7: chain 2, sector xi=-1, spin down",
        "Each block is resolved site by site from 0 to N-1 in the flattened vector.",
    ]
    return "\n".join(lines)


def _block_index(chain: int, sector: int, spin: str) -> int:
    sector_map: Dict[int, int] = {+1: 0, -1: 2}
    spin_map = {"up": 0, "down": 1}
    if chain not in (1, 2):
        raise ValueError("chain must be 1 or 2")
    if sector not in sector_map:
        raise ValueError("sector must be +1 or -1")
    if spin not in spin_map:
        raise ValueError("spin must be 'up' or 'down'")
    return (chain - 1) * 4 + sector_map[sector] + spin_map[spin]


def basis_index(chain: int, sector: int, spin: str, site: int, num_sites: int) -> int:
    """Return the flattened basis index for one channel-resolved site.

    This helper makes explicit which chain / sector / spin combination is being
    addressed when reconstructing selected Green-function components.
    """

    return site_index(_block_index(chain, sector, spin), site, num_sites)


def make_legacy_source_vector(num_sites: int):
    """Return the historical source vector used by the legacy scripts."""

    return make_source_vector(num_sites)

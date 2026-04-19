"""Random realization builders for the disorder and dephasing scripts."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import SampleArrays, constant_imaginary_shift, constant_real_array


def zero_mean_dephasing_realization(num_sites: int, rng: np.random.Generator) -> np.ndarray:
    """Reproduce the legacy dephasing draw with exact zero mean per realization."""

    realization = rng.random(num_sites) - 0.5
    realization[-1] += -np.sum(realization)
    return realization


def anderson_disorder_realization(num_sites: int, rng: np.random.Generator) -> np.ndarray:
    """Reproduce the legacy onsite disorder draw in [-0.5, 0.5]."""

    return rng.random(num_sites) - 0.5


def build_dephasing_samples(
    num_sites: int,
    count: int,
    eta_scale: float = 1.0,
    seed: Optional[int] = None,
) -> List[SampleArrays]:
    """Create dephasing samples with zero onsite disorder and random imaginary terms."""

    rng = np.random.default_rng(seed)
    zeros = constant_real_array(num_sites, 0.0)
    samples: List[SampleArrays] = []

    for _ in range(count):
        samples.append(
            SampleArrays(
                onsite_chain_1=zeros.copy(),
                onsite_chain_2=zeros.copy(),
                imag_chain_1=eta_scale * zero_mean_dephasing_realization(num_sites, rng),
                imag_chain_2=eta_scale * zero_mean_dephasing_realization(num_sites, rng),
            )
        )

    return samples


def build_disorder_samples(
    num_sites: int,
    count: int,
    eta: float,
    seed: Optional[int] = None,
) -> List[SampleArrays]:
    """Create Anderson-disorder samples with a small uniform numerical broadening."""

    rng = np.random.default_rng(seed)
    imag = constant_imaginary_shift(num_sites, eta)
    samples: List[SampleArrays] = []

    for _ in range(count):
        samples.append(
            SampleArrays(
                onsite_chain_1=anderson_disorder_realization(num_sites, rng),
                onsite_chain_2=anderson_disorder_realization(num_sites, rng),
                imag_chain_1=imag.copy(),
                imag_chain_2=imag.copy(),
            )
        )

    return samples

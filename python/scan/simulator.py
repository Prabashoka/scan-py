"""Simulation helpers for SCAN examples and benchmarks."""

from __future__ import annotations

import gc
import time as pytime
from typing import List

import numpy as np

from .detector import scan_cpd


def simulate_time_series(
    n: int = 10**6,
    n_cps: int = 250,
    min_seg_len: int = 1000,
    change_type: str = "meanvar",
    seed: int = 123,
):
    """Simulate a univariate time series with multiple change points.

    Parameters
    ----------
    n
        Length of the time series.
    n_cps
        Number of change points.
    min_seg_len
        Minimum segment length.
    change_type
        One of ``"mean"``, ``"var"``, or ``"meanvar"``.
    seed
        Random seed.

    Returns
    -------
    x, cps, means, sigmas
        Simulated observations, true change-points, segment means, and segment
        standard deviations.
    """

    change_type = str(change_type).lower()
    if change_type not in {"mean", "var", "meanvar"}:
        raise ValueError("change_type must be one of {'mean', 'var', 'meanvar'}")

    rng = np.random.default_rng(seed)
    n_segments = int(n_cps) + 1

    if n_segments * int(min_seg_len) > int(n):
        raise ValueError("min_seg_len is too large for the requested number of change points")

    remaining = int(n) - n_segments * int(min_seg_len)
    extra_lengths = rng.multinomial(remaining, np.ones(n_segments) / n_segments)
    segment_lengths = int(min_seg_len) + extra_lengths
    cps = np.cumsum(segment_lengths)[:-1]

    means = np.zeros(n_segments)
    sigmas = np.ones(n_segments)

    if change_type in {"mean", "meanvar"}:
        jumps = rng.choice([-1, 1], size=n_segments) * rng.uniform(0.8, 2.0, size=n_segments)
        means = np.cumsum(jumps)
        means = means - np.mean(means)

    if change_type in {"var", "meanvar"}:
        sigmas = rng.uniform(0.6, 2.0, size=n_segments)

    x = np.empty(int(n), dtype=np.float64)
    start = 0
    for j, seg_len in enumerate(segment_lengths):
        end = start + int(seg_len)
        x[start:end] = rng.normal(loc=means[j], scale=sigmas[j], size=int(seg_len))
        start = end

    return x, cps, means, sigmas


def safe_min_seg_len(n: int, k: int, spacing_hint: int, safety_fraction: float = 0.8) -> int:
    """Clamp a requested spacing to a feasible minimum segment length."""
    max_feasible = int(n) // (int(k) + 1)
    safe_value = int(float(safety_fraction) * max_feasible)
    min_seg_len = min(int(spacing_hint), safe_value)
    return max(5, min_seg_len)


def choose_window_sizes(n: int, n_windows: int = 7, seed: int = 500) -> List[int]:
    """Choose a reproducible set of safe scan window sizes."""
    rng = np.random.default_rng(seed)
    upper = int(np.sqrt(int(n)))

    if upper >= 100:
        lower = 100
    else:
        lower = max(10, int(0.5 * upper))

    if lower >= upper:
        lower = max(2, upper // 2)

    candidates = np.arange(lower, upper + 1)
    if len(candidates) == 0:
        candidates = np.arange(2, max(3, upper + 1))

    if len(candidates) >= int(n_windows):
        window_sizes = np.sort(rng.choice(candidates, size=int(n_windows), replace=False))
    else:
        window_sizes = np.sort(candidates)

    return window_sizes.astype(int).tolist()



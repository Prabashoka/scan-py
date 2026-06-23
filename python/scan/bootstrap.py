"""Bootstrap utilities for research experiments."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .statistics import wasserstein_statistic


def _taper_window(length: int, taper: str = "tukey") -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    value = taper.lower()
    if value in {"none", "flat", "rectangular"}:
        return np.ones(length)
    if value != "tukey":
        raise ValueError("taper must be one of {'tukey', 'none'}")
    return np.hanning(length + 2)[1:-1]


def tapered_block_bootstrap(
    x: Iterable[float],
    sample_length: int,
    block_length: Optional[int] = None,
    n_boot: int = 1,
    taper: str = "tukey",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate tapered block bootstrap samples."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("x must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains NaN or infinite values")

    sample_length = int(sample_length)
    n_boot = int(n_boot)
    if sample_length <= 0 or n_boot <= 0:
        raise ValueError("sample_length and n_boot must be positive")

    block_length = int(block_length or max(3, round(arr.size ** (1 / 3))))
    block_length = min(block_length, arr.size)
    starts = np.arange(0, arr.size - block_length + 1)
    weights = _taper_window(block_length, taper=taper)
    weights = weights * (block_length**0.5 / np.sqrt(np.sum(weights**2)))

    rng = np.random.default_rng(random_state)
    out = np.empty((n_boot, sample_length), dtype=np.float64)
    n_blocks = int(np.ceil(sample_length / block_length))

    for i in range(n_boot):
        pieces = []
        for _ in range(n_blocks):
            start = int(rng.choice(starts))
            pieces.append(arr[start : start + block_length] * weights)
        out[i] = np.concatenate(pieces)[:sample_length]
    return out


def adaptive_threshold(
    left: Iterable[float],
    right: Iterable[float],
    alpha: float = 0.05,
    n_boot: int = 400,
    block_length: Optional[int] = None,
    taper: str = "tukey",
    random_state: Optional[int] = None,
) -> float:
    """Compute a local bootstrap threshold for a two-window comparison."""
    left_arr = np.asarray(left, dtype=np.float64).ravel()
    right_arr = np.asarray(right, dtype=np.float64).ravel()
    if left_arr.size == 0 or right_arr.size == 0:
        raise ValueError("left and right must be non-empty")

    pooled = np.r_[left_arr - left_arr.mean(), right_arr - right_arr.mean()]
    samples = tapered_block_bootstrap(
        pooled,
        sample_length=left_arr.size + right_arr.size,
        block_length=block_length,
        n_boot=n_boot,
        taper=taper,
        random_state=random_state,
    )
    stats = [wasserstein_statistic(s[: left_arr.size], s[left_arr.size :]) for s in samples]
    return float(np.quantile(stats, 1 - float(alpha)))

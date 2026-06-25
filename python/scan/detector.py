import os
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

from . import _scan_rust
from .result import ScanResult, WindowResult, scan_result_from_raw

_ALLOWED_CHANGE_TYPES = {"mean", "var", "distribution"}


def _as_float_list(x: Iterable[float]) -> List[float]:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 3:
        raise ValueError("x must contain at least 3 observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains NaN or infinite values")
    return arr.tolist()


def _as_window_list(window_sizes: Iterable[int]) -> List[int]:
    windows = sorted({int(w) for w in window_sizes})
    if not windows:
        raise ValueError("window_sizes must not be empty")
    if any(w <= 0 for w in windows):
        raise ValueError("window_sizes must contain positive integers")
    return windows


def _default_window_sizes(n: int, min_window: int, max_window: Optional[int]) -> List[int]:
    min_window = int(min_window)
    if min_window <= 0:
        raise ValueError("min_window must be positive")

    upper = int(max_window) if max_window is not None else int(np.sqrt(n))
    upper = max(min_window, upper)
    upper = min(upper, max(1, n // 2))

    if min_window > upper:
        raise ValueError("min_window is too large for the length of x")

    n_grid = min(5, upper - min_window + 1)
    return _as_window_list(np.linspace(min_window, upper, n_grid, dtype=int))


def _validate_change_type(value: str) -> str:
    normalized = str(value).lower()
    if normalized not in _ALLOWED_CHANGE_TYPES:
        raise ValueError("change_type must be one of {'mean', 'var', 'distribution'}")
    return normalized


def _change_type_from_ipm(ipm: str, change_type: Optional[str]) -> str:
    if change_type is not None:
        return _validate_change_type(change_type)

    value = str(ipm).lower()
    if value in _ALLOWED_CHANGE_TYPES:
        return value
    if value in {"wasserstein", "ipm"}:
        return "distribution"
    raise ValueError("ipm must be one of {'wasserstein', 'mean', 'var', 'distribution'}")


def _normalize_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return alpha


def _normalize_taper(taper: str) -> float:
    value = str(taper).lower()
    if value in {"tukey", "cosine"}:
        return 0.5
    if value in {"none", "flat", "rectangular"}:
        return 0.0
    raise ValueError("taper must be one of {'tukey', 'none'}")


def _resolve_n_jobs(n_jobs: Optional[int]) -> tuple[Optional[int], int, int]:
    cpu_count = os.cpu_count() or 1

    if n_jobs is None:
        return None, max(1, cpu_count - 1), cpu_count

    requested = int(n_jobs)
    if requested == -1:
        return requested, cpu_count, cpu_count
    if requested < 1:
        raise ValueError("n_jobs must be None, -1, or a positive integer")

    return requested, min(requested, cpu_count), cpu_count


def scan_cpd(
    x: Iterable[float],
    window_sizes: Optional[Sequence[int]] = None,
    alpha: float = 0.05,
    n_boot: int = 400,
    vote_threshold: float = 0.5,
    min_window: int = 15,
    max_window: Optional[int] = None,
    block_length: Optional[int] = None,
    block_length_rule: str = "n^(1/3)",
    taper: str = "tukey",
    ipm: str = "wasserstein",
    tolerance: Optional[int] = None,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    return_all: bool = True,
    *,
    change_type: Optional[str] = None,
    eps: float = 1e-12,
    batch_size: int = 32,
) -> ScanResult:
    """Run SCAN / Ensemble SCAN and return a structured result.

    Change points use Python split indexing: a returned value ``t`` denotes the
    split between ``x[:t]`` and ``x[t:]``.
    """

    series = _as_float_list(x)
    n = len(series)
    windows = _as_window_list(window_sizes) if window_sizes is not None else _default_window_sizes(n, min_window, max_window)

    if any(2 * w > n for w in windows):
        raise ValueError("each window size must satisfy 2 * window_size <= len(x)")

    alpha = _normalize_alpha(alpha)
    n_boot = int(n_boot)
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")

    vote_threshold = float(vote_threshold)
    if not 0 <= vote_threshold <= 1:
        raise ValueError("vote_threshold must be between 0 and 1")

    if block_length is not None and int(block_length) <= 0:
        raise ValueError("block_length must be positive when provided")

    if block_length is None and block_length_rule != "n^(1/3)":
        raise ValueError("only block_length_rule='n^(1/3)' is currently supported")

    tol = int(tolerance) if tolerance is not None else min(windows)
    seed = 0 if random_state is None else int(random_state)
    rust_change_type = _change_type_from_ipm(ipm, change_type)
    taper_ratio = _normalize_taper(taper)
    requested_n_jobs, resolved_n_jobs, cpu_count = _resolve_n_jobs(n_jobs)

    parameters = {
        "window_sizes": windows,
        "alpha": alpha,
        "n_boot": n_boot,
        "vote_threshold": vote_threshold,
        "min_window": min_window,
        "max_window": max_window,
        "block_length": block_length,
        "block_length_rule": block_length_rule,
        "taper": taper,
        "ipm": ipm,
        "change_type": rust_change_type,
        "tolerance": tol,
        "random_state": random_state,
        "n_jobs": requested_n_jobs,
        "return_all": return_all,
        "eps": eps,
        "batch_size": batch_size,
    }

    start = time.perf_counter()
    raw = _scan_rust.scan_detector(
        series,
        windows,
        n_boot,
        alpha,
        seed,
        tol,
        resolved_n_jobs,
        "thread",
        rust_change_type,
        float(eps),
        None if block_length is None else int(block_length),
        taper_ratio,
        True,
        int(batch_size),
    )
    elapsed = time.perf_counter() - start

    metadata = {
        "n_obs": n,
        "elapsed_seconds": elapsed,
        "index_base": "python_split_0_based",
        "rust_backend": "scan._scan_rust",
        "cpu_count": cpu_count,
        "resolved_n_jobs": resolved_n_jobs,
    }

    result = scan_result_from_raw(
        raw,
        vote_threshold=vote_threshold,
        parameters=parameters,
        metadata=metadata,
    )

    if return_all:
        return result

    return ScanResult(
        change_points=result.change_points,
        scores=result.scores,
        votes=result.votes,
        window_results={},
        thresholds={},
        parameters=result.parameters,
        metadata=result.metadata,
        segments=result.segments,
        raw={},
    )


def scan_single_window(
    x: Iterable[float],
    window_size: int,
    *,
    alpha: float = 0.05,
    n_boot: int = 400,
    block_length: Optional[int] = None,
    taper: str = "tukey",
    ipm: str = "wasserstein",
    random_state: Optional[int] = None,
    change_type: Optional[str] = None,
    eps: float = 1e-12,
    batch_size: int = 32,
) -> WindowResult:
    """Run SCAN for one window size and return per-location diagnostics."""

    series = _as_float_list(x)
    window_size = int(window_size)
    if 2 * window_size > len(series):
        raise ValueError("window_size must satisfy 2 * window_size <= len(x)")

    raw = _scan_rust.scan_single_window(
        series,
        window_size,
        int(n_boot),
        _normalize_alpha(alpha),
        0 if random_state is None else int(random_state),
        _change_type_from_ipm(ipm, change_type),
        float(eps),
        None if block_length is None else int(block_length),
        _normalize_taper(taper),
        True,
        int(batch_size),
    )
    return WindowResult.from_raw(window_size, raw)
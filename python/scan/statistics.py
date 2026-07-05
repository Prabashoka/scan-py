"""Statistical discrepancy and localization helpers for SCAN.

These functions expose the Rust backend's one-dimensional Wasserstein, IPM,
SWAL, and CUSUM routines with Python validation and return-type normalization.
"""

from typing import Iterable, Tuple

import numpy as np

from . import _scan_rust

_ALLOWED_CHANGE_TYPES = {"mean", "var", "distribution"}


def _as_float_list(x: Iterable[float]) -> list[float]:
    """Convert a finite one-dimensional sample into a Python float list."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("sample must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("sample contains NaN or infinite values")
    return arr.tolist()


def _validate_change_type(change_type: str) -> str:
    """Validate the backend-supported localization change type."""
    value = str(change_type).lower()
    if value not in _ALLOWED_CHANGE_TYPES:
        raise ValueError("change_type must be one of {'mean', 'var', 'distribution'}")
    return value


def wasserstein_statistic(left: Iterable[float], right: Iterable[float]) -> float:
    """Compute the empirical one-dimensional 1-Wasserstein distance.

    Parameters
    ----------
    left, right:
        Numeric samples to compare.

    Returns
    -------
    float
        Empirical 1-Wasserstein distance between the two samples.
    """
    return float(_scan_rust.wasserstein_statistic(_as_float_list(left), _as_float_list(right)))


def ipm_statistic(left: Iterable[float], right: Iterable[float], ipm: str = "wasserstein") -> float:
    """Compute the local IPM discrepancy between two windows.

    Parameters
    ----------
    left, right:
        Numeric samples to compare.
    ipm:
        Discrepancy name. Currently only ``"wasserstein"`` and ``"ipm"`` are
        accepted aliases for the backend Wasserstein statistic.

    Returns
    -------
    float
        Local discrepancy score.
    """
    if str(ipm).lower() not in {"wasserstein", "ipm"}:
        raise ValueError("only ipm='wasserstein' is currently supported")
    return float(_scan_rust.ipm_statistic(_as_float_list(left), _as_float_list(right)))


def swal_statistic(x: Iterable[float], change_type: str = "distribution") -> int:
    """Return the SWAL/CUSUM localizer inside a flagged local region.

    Parameters
    ----------
    x:
        Candidate local region containing at least three observations.
    change_type:
        Type of change to localize: ``"mean"``, ``"var"``, or
        ``"distribution"``.

    Returns
    -------
    int
        Split index inside ``x`` selected by the backend localizer.
    """
    arr = _as_float_list(x)
    if len(arr) < 3:
        raise ValueError("x must contain at least 3 observations")
    return int(_scan_rust.swal_statistic(arr, _validate_change_type(change_type)))


def refine_cusum(x: Iterable[float]) -> int:
    """Return the CUSUM localizer inside one candidate block.

    Parameters
    ----------
    x:
        Candidate local region containing at least three observations.

    Returns
    -------
    int
        Split index inside ``x`` with the strongest CUSUM evidence.
    """
    arr = _as_float_list(x)
    if len(arr) < 3:
        raise ValueError("x must contain at least 3 observations")
    return int(_scan_rust.refine_cusum(arr))


def refine_wasserstein(x: Iterable[float]) -> Tuple[int, list[float]]:
    """Return the Wasserstein localizer and split statistics.

    Parameters
    ----------
    x:
        Candidate local region containing at least three observations.

    Returns
    -------
    tuple[int, list[float]]
        Local split index and the Wasserstein statistic evaluated at each
        candidate split.
    """
    arr = _as_float_list(x)
    if len(arr) < 3:
        raise ValueError("x must contain at least 3 observations")
    cp, stats = _scan_rust.refine_wasserstein(arr)
    return int(cp), [float(v) for v in stats]

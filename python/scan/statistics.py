from typing import Iterable, Tuple

import numpy as np

from . import _scan_rust

_ALLOWED_CHANGE_TYPES = {"mean", "var", "distribution"}


def _as_float_list(x: Iterable[float]) -> list[float]:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("sample must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("sample contains NaN or infinite values")
    return arr.tolist()


def _validate_change_type(change_type: str) -> str:
    value = str(change_type).lower()
    if value not in _ALLOWED_CHANGE_TYPES:
        raise ValueError("change_type must be one of {'mean', 'var', 'distribution'}")
    return value


def wasserstein_statistic(left: Iterable[float], right: Iterable[float]) -> float:
    """Compute the empirical one-dimensional 1-Wasserstein distance."""
    return float(_scan_rust.wasserstein_statistic(_as_float_list(left), _as_float_list(right)))


def ipm_statistic(left: Iterable[float], right: Iterable[float], ipm: str = "wasserstein") -> float:
    """Compute the local IPM discrepancy between two windows."""
    if str(ipm).lower() not in {"wasserstein", "ipm"}:
        raise ValueError("only ipm='wasserstein' is currently supported")
    return float(_scan_rust.ipm_statistic(_as_float_list(left), _as_float_list(right)))


def swal_statistic(x: Iterable[float], change_type: str = "distribution") -> int:
    """Return the SWAL/CUSUM localizer inside a flagged local region."""
    arr = _as_float_list(x)
    if len(arr) < 3:
        raise ValueError("x must contain at least 3 observations")
    return int(_scan_rust.swal_statistic(arr, _validate_change_type(change_type)))


def refine_cusum(x: Iterable[float]) -> int:
    """Return the CUSUM localizer inside one candidate block."""
    arr = _as_float_list(x)
    if len(arr) < 3:
        raise ValueError("x must contain at least 3 observations")
    return int(_scan_rust.refine_cusum(arr))


def refine_wasserstein(x: Iterable[float]) -> Tuple[int, list[float]]:
    """Return the Wasserstein localizer and split statistics."""
    arr = _as_float_list(x)
    if len(arr) < 3:
        raise ValueError("x must contain at least 3 observations")
    cp, stats = _scan_rust.refine_wasserstein(arr)
    return int(cp), [float(v) for v in stats]
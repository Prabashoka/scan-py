"""
SCAN: Sequential Change-point Analysis via Nonparametric window screening.

This package keeps the public Python API close to the previous MACS code, but
renames the method and entry points to SCAN and runs the expensive core in Rust.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np

from . import _scanpy

__version__ = "0.1.0"


def _as_float_list(series: Iterable[float]) -> List[float]:
    return np.asarray(series, dtype=np.float64).ravel().tolist()


def _as_window_list(window_sizes: Iterable[int]) -> List[int]:
    return [int(w) for w in window_sizes]


def scan_detector(
    series: Iterable[float],
    window_sizes: Optional[List[int]] = None,
    n_perm: int = 300,
    alpha_q: float = 1.0,
    seed: int = 123,
    tol: int = 2,
    workers: Optional[int] = None,
    backend: str = "thread",
    change_type: str = "mean",
    eps: float = 1e-12,
    b: Optional[int] = None,
    taper_ratio: float = 0.5,
    center: bool = True,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """
    Run the full SCAN detection pipeline and return the raw multi-window output.

    Returns a dictionary with the same high-level structure as the old MACS
    detector: cp_dict, timings, total_time, segments, and out.
    """
    return _scan_rust.scan_detector(
        _as_float_list(series),
        None if window_sizes is None else _as_window_list(window_sizes),
        int(n_perm),
        float(alpha_q),
        int(seed),
        int(tol),
        None if workers is None else int(workers),
        str(backend),
        str(change_type),
        float(eps),
        None if b is None else int(b),
        float(taper_ratio),
        bool(center),
        int(batch_size),
    )


def _scan_cpd_base(
    series: Iterable[float],
    window_sizes: List[int],
    n_perm: int = 300,
    alpha_q: float = 1.0,
    seed: int = 123,
    tol: Optional[int] = None,
    workers: Optional[int] = 8,
    backend: str = "thread",
    threshold: float = 0.5,
    change_type: str = "mean",
    eps: float = 1e-12,
    b: Optional[int] = None,
    taper_ratio: float = 0.5,
    center: bool = True,
    batch_size: int = 512,
) -> Tuple[List[int], float]:
    window_sizes = _as_window_list(window_sizes)
    tol_value = None if tol is None else int(tol)

    if change_type == "mean":
        return _scan_rust.scan_cpd_mean(
            _as_float_list(series),
            window_sizes,
            int(n_perm),
            float(alpha_q),
            int(seed),
            tol_value,
            None if workers is None else int(workers),
            str(backend),
            float(threshold),
            float(eps),
            None if b is None else int(b),
            float(taper_ratio),
            bool(center),
            int(batch_size),
        )
    if change_type == "var":
        return _scan_rust.scan_cpd_var(
            _as_float_list(series),
            window_sizes,
            int(n_perm),
            float(alpha_q),
            int(seed),
            tol_value,
            None if workers is None else int(workers),
            str(backend),
            float(threshold),
            float(eps),
            None if b is None else int(b),
            float(taper_ratio),
            bool(center),
            int(batch_size),
        )
    if change_type == "meanvar":
        return _scan_rust.scan_cpd_meanvar(
            _as_float_list(series),
            window_sizes,
            int(n_perm),
            float(alpha_q),
            int(seed),
            tol_value,
            None if workers is None else int(workers),
            str(backend),
            float(threshold),
            float(eps),
            None if b is None else int(b),
            float(taper_ratio),
            bool(center),
            int(batch_size),
        )
    raise ValueError("change_type must be one of {'mean', 'var', 'meanvar'}")


def scan_cpd_mean(
    series: Iterable[float],
    window_sizes: List[int],
    n_perm: int = 300,
    alpha_q: float = 1.0,
    seed: int = 123,
    tol: Optional[int] = None,
    workers: Optional[int] = 8,
    backend: str = "thread",
    threshold: float = 0.5,
    eps: float = 1e-12,
    b: Optional[int] = None,
    taper_ratio: float = 0.5,
    center: bool = True,
    batch_size: int = 512,
) -> Tuple[List[int], float]:
    """SCAN with CUSUM localization for mean shifts. Returns 1-based CP indices."""
    return _scan_cpd_base(
        series,
        window_sizes,
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        threshold,
        "mean",
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )


def scan_cpd_var(
    series: Iterable[float],
    window_sizes: List[int],
    n_perm: int = 400,
    alpha_q: float = 10.0,
    seed: int = 123,
    tol: Optional[int] = None,
    workers: Optional[int] = 8,
    backend: str = "thread",
    threshold: float = 5.0,
    eps: float = 1e-12,
    b: Optional[int] = None,
    taper_ratio: float = 0.5,
    center: bool = True,
    batch_size: int = 512,
) -> Tuple[List[int], float]:
    """
    SCAN with Wasserstein localization for variance-sensitive changes.

    Note: the default threshold=5.0 preserves your current Python default. Since
    leader scores are clipped to [0, 1], threshold=0.5 is usually more natural.
    """
    return _scan_cpd_base(
        series,
        window_sizes,
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        threshold,
        "var",
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )


def scan_cpd_meanvar(
    series: Iterable[float],
    window_sizes: List[int],
    n_perm: int = 300,
    alpha_q: float = 1.0,
    seed: int = 123,
    tol: Optional[int] = None,
    workers: Optional[int] = 8,
    backend: str = "thread",
    threshold: float = 0.5,
    eps: float = 1e-12,
    b: Optional[int] = None,
    taper_ratio: float = 0.5,
    center: bool = True,
    batch_size: int = 512,
) -> Tuple[List[int], float]:
    """SCAN with Wasserstein localization for joint mean/variance changes."""
    return _scan_cpd_base(
        series,
        window_sizes,
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        threshold,
        "meanvar",
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )


def refine_cusum(series: Iterable[float]) -> int:
    """Return the CUSUM localizer inside one candidate block."""
    return int(_scan_rust.refine_cusum(_as_float_list(series)))


def refine_wasserstein(series: Iterable[float]):
    """Return the scaled Wasserstein localizer and all split statistics."""
    return _scan_rust.refine_wasserstein(_as_float_list(series))


def plot_detected_changepoints(
    series: Iterable[float],
    cpts: List[int],
    title: str = "Detected Change Points",
    figsize=(10, 4),
):
    """Plot a time series with detected change points marked by dashed lines."""
    import matplotlib.pyplot as plt

    y = np.asarray(series, dtype=float).ravel()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=figsize)
    plt.plot(y, color="#1f77b4", lw=2, label="Series")

    for i, cp in enumerate(cpts):
        plt.axvline(
            x=cp,
            color="#d62728",
            linestyle="--",
            lw=1.8,
            alpha=0.8,
            label="Change Point" if i == 0 else None,
        )

    plt.title(title, fontsize=13, fontweight="semibold")
    plt.xlabel("Time Index", fontsize=11)
    plt.ylabel("Value", fontsize=11)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.show()


__all__ = [
    "scan_detector",
    "scan_cpd_mean",
    "scan_cpd_var",
    "scan_cpd_meanvar",
    "refine_cusum",
    "refine_wasserstein",
    "plot_detected_changepoints",
]

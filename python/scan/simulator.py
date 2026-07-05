"""Simulation and benchmarking utilities for SCAN examples.

The simulator creates univariate time series with controlled mean, variance,
or distributional changes. The benchmark helpers use those series to exercise
the high-level detector with reproducible defaults.
"""

import gc
import math
import time as pytime
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np

from .detector import scan_cpd

_ALLOWED_CHANGE_TYPES = {"mean", "var", "variance", "distribution", "both"}


def _normalize_change_type(change_type: str) -> str:
    """Normalize user change-type aliases to simulator-internal names."""
    value = str(change_type).lower()
    if value not in _ALLOWED_CHANGE_TYPES:
        raise ValueError("change_type must be one of {'mean', 'var', 'variance', 'distribution'}")
    if value == "var":
        return "variance"
    if value == "distribution":
        return "both"
    return value


@dataclass
class UnivariateSeriesSimulator:
    """Simulator for shifted univariate time series.

    Parameters
    ----------
    len_series:
        Number of observations to simulate.
    initial_value:
        Initial value used by autoregressive generators.
    seed:
        Seed used by NumPy random number generators.
    """

    len_series: int
    initial_value: float = 0.0
    seed: int = 123

    def __post_init__(self) -> None:
        """Validate and normalize dataclass fields after initialization."""
        self.len_series = int(self.len_series)
        if self.len_series < 3:
            raise ValueError("len_series must be at least 3")
        self.initial_value = float(self.initial_value)
        self.seed = int(self.seed)

    def simulate_ar_series(
        self,
        rho: float,
        error_type: str,
        error_mean: float = 0.0,
        error_variance: float = 1.0,
        error_location: float = 0.0,
        error_scale: float = 0.03,
    ) -> np.ndarray:
        """Simulate an AR(1) series with normal or Cauchy innovations.

        Parameters
        ----------
        rho:
            Autoregressive coefficient.
        error_type:
            Innovation family: ``"normal"`` or ``"cauchy"``.
        error_mean, error_variance:
            Mean and variance used for normal innovations.
        error_location, error_scale:
            Location and scale used for Cauchy innovations.

        Returns
        -------
        numpy.ndarray
            Simulated series of length ``len_series``.
        """
        rng = np.random.default_rng(self.seed)
        x = np.empty(self.len_series, dtype=float)
        x_prev = self.initial_value
        for t in range(self.len_series):
            if error_type == "normal":
                eps = rng.normal(error_mean, np.sqrt(error_variance))
            elif error_type == "cauchy":
                eps = error_location + error_scale * rng.standard_cauchy()
            else:
                raise ValueError("error_type must be one of {'normal', 'cauchy'}")
            x_prev = float(rho) * x_prev + eps
            x[t] = x_prev
        return x

    def simulate_ar_unif(self, error_variance: float = 1.0) -> np.ndarray:
        """Simulate an AR process with random time-varying coefficients.

        Parameters
        ----------
        error_variance:
            Variance of Gaussian innovations.

        Returns
        -------
        numpy.ndarray
            Simulated series with ``rho_t`` drawn uniformly from ``[0, 1]``.
        """
        rng = np.random.default_rng(self.seed)
        rho = rng.uniform(0.0, 1.0, size=self.len_series)
        eps = rng.normal(0.0, np.sqrt(error_variance), size=self.len_series)
        x = np.empty(self.len_series, dtype=float)
        x[0] = self.initial_value
        for t in range(1, self.len_series):
            x[t] = rho[t] * x[t - 1] + eps[t]
        return x

    def simulate_arma(
        self,
        phi: Union[Sequence[float], float],
        theta: Union[Sequence[float], float],
        error_mean: float = 0.0,
        error_variance: float = 1.0,
    ) -> np.ndarray:
        """Simulate an ARMA process using statsmodels when it is installed.

        Parameters
        ----------
        phi:
            Autoregressive coefficients.
        theta:
            Moving-average coefficients.
        error_mean, error_variance:
            Mean and variance of Gaussian innovations.

        Returns
        -------
        numpy.ndarray
            Simulated ARMA series.
        """
        try:
            from statsmodels.tsa.arima_process import ArmaProcess
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ImportError("simulate_arma requires statsmodels. Install it with `pip install statsmodels`.") from exc

        rng = np.random.default_rng(self.seed)
        phi_arr = np.atleast_1d(phi).astype(float)
        theta_arr = np.atleast_1d(theta).astype(float)
        ar = np.r_[1.0, -phi_arr]
        ma = np.r_[1.0, theta_arr]
        process = ArmaProcess(ar, ma)
        return process.generate_sample(
            nsample=self.len_series,
            distrvs=lambda size: rng.normal(loc=error_mean, scale=np.sqrt(error_variance), size=size),
        )

    @staticmethod
    def fracint_weights(d: float, m: int) -> np.ndarray:
        """Return fractional integration weights.

        Parameters
        ----------
        d:
            Fractional differencing/integration parameter.
        m:
            Number of weights to generate.

        Returns
        -------
        numpy.ndarray
            Fractional integration weights of length ``m``.
        """
        m = int(m)
        if m <= 0:
            raise ValueError("m must be positive")
        w = np.empty(m, dtype=float)
        w[0] = 1.0
        for k in range(1, m):
            w[k] = w[k - 1] * (k + float(d) - 1.0) / k
        return w

    def arfima_sim(
        self,
        d: float = 0.3,
        ar: Optional[Sequence[float]] = None,
        ma: Optional[Sequence[float]] = None,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """Simulate an ARFIMA(p,d,q) series.

        Parameters
        ----------
        d:
            Fractional integration parameter.
        ar:
            Optional autoregressive coefficients.
        ma:
            Optional moving-average coefficients.
        sigma:
            Innovation standard deviation.

        Returns
        -------
        numpy.ndarray
            Simulated ARFIMA series of length ``len_series``.
        """
        rng = np.random.default_rng(self.seed)
        ar_arr = np.atleast_1d(ar).astype(float) if ar is not None else np.array([])
        ma_arr = np.atleast_1d(ma).astype(float) if ma is not None else np.array([])
        p, q = len(ar_arr), len(ma_arr)

        max_frac_lag = 5000
        k_lag = max(1, min(max_frac_lag, self.len_series + max(p, q) + 5000))
        pad = k_lag
        total = self.len_series + pad

        eps = rng.normal(scale=sigma, size=total + q)
        innovations = np.zeros(total + q, dtype=float)
        for t in range(q, total + q):
            innovations[t] = eps[t]
            if q > 0:
                innovations[t] += np.dot(ma_arr, eps[t - np.arange(1, q + 1)])
        innovations = innovations[q:]

        weights = self.fracint_weights(d, k_lag + 1)
        fractional = np.convolve(innovations, weights, mode="full")[:total]

        x = np.zeros(total, dtype=float)
        for t in range(total):
            value = fractional[t]
            if p and t > 0:
                k = min(p, t)
                lag_idx = np.arange(1, k + 1)
                value += np.dot(ar_arr[:k], x[t - lag_idx])
            x[t] = value

        return x[-self.len_series:]

    @staticmethod
    def determine_change_points(n: int, ratio: int = 100) -> int:
        """Determine a default number of change points from a length ratio.

        Parameters
        ----------
        n:
            Series length.
        ratio:
            Approximate observations per change point.

        Returns
        -------
        int
            Floor of ``n / ratio``.
        """
        return int(math.floor(int(n) // int(ratio)))

    def select_change_point_locations(
        self,
        min_points: int,
        ratio: int = 100,
        min_first_cp: int = 30,
        n_cps: Optional[int] = None,
    ) -> np.ndarray:
        """Select sorted split locations with a minimum spacing constraint.

        Parameters
        ----------
        min_points:
            Minimum distance between consecutive selected split locations.
        ratio:
            Used to infer the number of change points when ``n_cps`` is not
            provided.
        min_first_cp:
            Earliest allowed split location.
        n_cps:
            Optional explicit number of change points to select.

        Returns
        -------
        numpy.ndarray
            Sorted integer split locations.
        """
        n = self.len_series
        count = self.determine_change_points(n, ratio=ratio) if n_cps is None else int(n_cps)
        if count <= 0:
            return np.array([], dtype=int)

        min_points = int(min_points)
        min_first_cp = int(min_first_cp)
        if min_points < 1:
            raise ValueError("min_points must be >= 1")
        if not 1 <= min_first_cp <= n - 1:
            raise ValueError(f"min_first_cp must be in [1, {n - 1}]")

        slack = (min_points - 1) * (count - 1)
        max_base = (n - 1) - slack - (min_first_cp - 1)
        if max_base < count:
            raise ValueError(
                "Parameters are too tight for the requested number of change points, "
                "minimum spacing, and first change-point constraint."
            )

        rng = np.random.default_rng(self.seed)
        base = np.sort(rng.choice(np.arange(1, max_base + 1), size=count, replace=False))
        points = (min_first_cp - 1) + base + np.arange(count) * (min_points - 1)
        return points.astype(int)

    def apply_random_shifts(
        self,
        series: Sequence[float],
        change_point_locations: Sequence[int],
        min_shift: float = 3.0,
        max_shift: float = 10.0,
        shifts: Optional[Union[Sequence[float], float]] = None,
        seed: Optional[int] = None,
        change_type: str = "mean",
        variance_multipliers: Optional[Union[Sequence[float], float]] = None,
        lognormal_mean: float = 0.0,
        lognormal_sigma: Optional[float] = None,
        variance_center: Union[str, float] = "pre_change",
        variance_reference: str = "first_segment",
        eps: float = 1e-12,
    ) -> dict[str, np.ndarray]:
        """Apply mean and/or variance shifts at supplied split locations.

        Parameters
        ----------
        series:
            Base one-dimensional series to transform.
        change_point_locations:
            Split locations where segment parameters change.
        min_shift, max_shift:
            Range for randomly generated mean-shift magnitudes.
        shifts:
            Optional scalar or per-change mean shifts.
        seed:
            Optional seed overriding the simulator seed for shift generation.
        change_type:
            Shift type: ``"mean"``, ``"variance"``, or ``"distribution"``.
        variance_multipliers:
            Optional scalar or per-change variance multipliers.
        lognormal_mean, lognormal_sigma:
            Parameters used for random variance multipliers.
        variance_center:
            Center used when rescaling segment variance.
        variance_reference:
            Reference variance source for multipliers.
        eps:
            Small positive floor for near-zero variances.

        Returns
        -------
        dict[str, numpy.ndarray]
            Original series, shifted series, change points, mean shifts, and
            variance multipliers.
        """
        x = np.asarray(series, dtype=float)
        if x.ndim != 1:
            raise ValueError("series must be one-dimensional")
        n = x.shape[0]
        if n != self.len_series:
            raise ValueError("series length must equal len_series")

        change_kind = _normalize_change_type(change_type)
        cps = np.unique(np.sort(np.asarray(change_point_locations, dtype=int).ravel()))
        if cps.size == 0:
            return {
                "original_series": x.copy(),
                "change_points": cps,
                "shifted_series": x.copy(),
                "mean_shifts": np.array([]),
                "variance_multipliers": np.array([]),
            }
        if np.any(cps <= 0) or np.any(cps >= n):
            raise ValueError("change_point_locations must be split locations in 1..len(series)-1")

        rng = np.random.default_rng(self.seed if seed is None else seed)

        mean_shifts = np.array([])
        if change_kind in {"mean", "both"}:
            if shifts is None:
                magnitudes = rng.uniform(min_shift, max_shift, size=cps.size)
                signs = rng.choice([-1.0, 1.0], size=cps.size)
                mean_shifts = magnitudes * signs
            else:
                mean_shifts = np.asarray(shifts, dtype=float).ravel()
                if mean_shifts.size == 1:
                    mean_shifts = np.full(cps.size, float(mean_shifts[0]))
                if mean_shifts.size != cps.size:
                    raise ValueError(f"shifts must have length {cps.size} or be scalar")

        if lognormal_sigma is None:
            lognormal_sigma = np.log(10.0) / 2.0

        variance_mults = np.array([])
        if change_kind in {"variance", "both"}:
            if variance_multipliers is None:
                variance_mults = rng.lognormal(
                    mean=lognormal_mean,
                    sigma=lognormal_sigma,
                    size=cps.size,
                )
            else:
                variance_mults = np.asarray(variance_multipliers, dtype=float).ravel()
                if variance_mults.size == 1:
                    variance_mults = np.full(cps.size, float(variance_mults[0]))
                if variance_mults.size != cps.size:
                    raise ValueError(f"variance_multipliers must have length {cps.size} or be scalar")
                if np.any(variance_mults <= 0):
                    raise ValueError("all variance multipliers must be positive")

        bounds = np.r_[0, cps, n]
        n_segments = len(bounds) - 1
        cumulative_offsets = np.zeros(n_segments, dtype=float)
        if change_kind in {"mean", "both"}:
            cumulative_offsets[1:] = np.cumsum(mean_shifts)

        y = np.empty_like(x)
        first_seg = x[bounds[0] : bounds[1]] if bounds[1] > bounds[0] else x
        first_var = float(np.var(first_seg, ddof=1)) if first_seg.size > 1 else float(np.var(x, ddof=1))
        global_var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0

        for seg_idx in range(n_segments):
            start, end = int(bounds[seg_idx]), int(bounds[seg_idx + 1])
            seg = x[start:end].copy()
            if seg.size == 0:
                continue

            if change_kind in {"mean", "both"}:
                seg = seg + cumulative_offsets[seg_idx]

            if seg_idx == 0 or change_kind == "mean":
                y[start:end] = seg
                continue

            if variance_reference == "first_segment":
                ref_var = first_var
            elif variance_reference == "previous_segment":
                prev_start, prev_end = int(bounds[seg_idx - 1]), int(bounds[seg_idx])
                prev_seg = y[prev_start:prev_end]
                ref_var = float(np.var(prev_seg, ddof=1)) if prev_seg.size > 1 else first_var
            elif variance_reference == "global":
                ref_var = global_var
            else:
                raise ValueError("variance_reference must be one of {'first_segment', 'previous_segment', 'global'}")

            variance_factor = variance_mults[seg_idx - 1]
            target_std = np.sqrt(max(float(variance_factor) * ref_var, eps))

            # Preserve the selected center while stretching or regenerating the
            # segment to match the requested variance multiplier.
            if variance_center == "pre_change":
                center_value = float(np.mean(seg))
            elif variance_center == "global":
                center_value = float(np.mean(x))
                if change_kind in {"mean", "both"}:
                    center_value += cumulative_offsets[seg_idx]
            elif isinstance(variance_center, (int, float)):
                center_value = float(variance_center)
                if change_kind in {"mean", "both"}:
                    center_value += cumulative_offsets[seg_idx]
            else:
                raise ValueError("variance_center must be 'pre_change', 'global', or a numeric value")

            seg_std = float(np.std(seg, ddof=1)) if seg.size > 1 else 0.0
            if seg_std < eps:
                y[start:end] = rng.normal(loc=center_value, scale=target_std, size=seg.size)
            else:
                y[start:end] = center_value + (seg - np.mean(seg)) * (target_std / seg_std)

        return {
            "original_series": x.copy(),
            "change_points": cps,
            "shifted_series": y,
            "mean_shifts": mean_shifts,
            "variance_multipliers": variance_mults,
        }


def simulate_time_series(
    n: int = 10**6,
    n_cps: int = 250,
    min_seg_len: int = 235,
    change_type: str = "distribution",
    seed: int = 123,
):
    """Simulate a shifted AR(1) series using ``UnivariateSeriesSimulator``.

    Parameters
    ----------
    n:
        Series length.
    n_cps:
        Number of change points to insert.
    min_seg_len:
        Minimum spacing between change points.
    change_type:
        Type of change to apply.
    seed:
        Random seed used by the simulator.

    Returns
    -------
    tuple
        Shifted series, true change points, segment means, and segment standard
        deviations.
    """
    simulator = UnivariateSeriesSimulator(len_series=n, seed=seed)
    base = simulator.simulate_ar_series(rho=0, error_type="normal")
    cps = simulator.select_change_point_locations(min_points=min_seg_len, n_cps=n_cps)
    shifted = simulator.apply_random_shifts(
        base,
        cps,
        change_type=_normalize_change_type(change_type),
    )

    mean_shifts = shifted["mean_shifts"]
    variance_multipliers = shifted["variance_multipliers"]
    means = np.zeros(len(cps) + 1, dtype=float)
    sigmas = np.ones(len(cps) + 1, dtype=float)
    if mean_shifts.size:
        means[1:] = np.cumsum(mean_shifts)
    if variance_multipliers.size:
        sigmas[1:] = np.sqrt(variance_multipliers)

    return shifted["shifted_series"], shifted["change_points"], means, sigmas


def safe_min_seg_len(n: int, k: int, spacing_hint: int, safety_fraction: float = 0.8) -> int:
    """Clamp a requested spacing to a feasible minimum segment length.

    The returned value is at least 5 and no larger than a safety fraction of
    the average segment length.
    """
    max_feasible = int(n) // (int(k) + 1)
    safe_value = int(float(safety_fraction) * max_feasible)
    min_seg_len = min(int(spacing_hint), safe_value)
    return max(5, min_seg_len)


def choose_window_sizes(series_length: int, n_windows: int = 7, seed: int = 500) -> list[int]:
    """Choose a reproducible set of safe scan window sizes.

    Parameters
    ----------
    series_length:
        Length of the series that will be scanned.
    n_windows:
        Maximum number of window sizes to return.
    seed:
        Random seed used when sampling from candidate sizes.

    Returns
    -------
    list[int]
        Sorted scan window sizes.
    """
    rng = np.random.default_rng(seed)
    upper = int(np.sqrt(int(series_length)))

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


def run_one_benchmark(
    n: int,
    k: int,
    spacing_hint: int,
    change_type: str = "distribution",
    n_windows: int = 7,
    n_boot: int = 400,
    alpha: float = 1,
    vote_threshold: float = 0.7,
    n_jobs: Optional[int] = None,
    batch_size: int = 32,
    seed: int = 500,
) -> dict[str, Any]:
    """Simulate one benchmark series and run ``scan_cpd`` on it.

    Parameters
    ----------
    n:
        Series length.
    k:
        Number of true change points.
    spacing_hint:
        Requested minimum segment length before feasibility clamping.
    change_type:
        Type of simulated change.
    n_windows:
        Number of scan window sizes to sample.
    n_boot:
        Bootstrap replicates per local scan.
    alpha:
        Bootstrap significance level.
    vote_threshold:
        Ensemble vote score required to retain a point.
    n_jobs:
        Optional number of detector worker threads.
    batch_size:
        Backend batch size.
    seed:
        Random seed shared by simulation and detection defaults.

    Returns
    -------
    dict[str, Any]
        Benchmark metadata, true and detected change points, timing, and the
        full :class:`scan.result.ScanResult`.
    """
    min_seg_len = safe_min_seg_len(n=n, k=k, spacing_hint=spacing_hint, safety_fraction=0.8)

    x, true_cps, means, sigmas = simulate_time_series(
        n=n,
        n_cps=k,
        min_seg_len=min_seg_len,
        change_type=change_type,
        seed=seed,
    )

    x = np.asarray(x, dtype=np.float64)
    x_std = (x - np.mean(x)) / np.std(x)
    window_sizes = choose_window_sizes(series_length=n, n_windows=n_windows, seed=seed)

    start_wall = pytime.perf_counter()
    result = scan_cpd(
        x_std,
        window_sizes=window_sizes,
        n_boot=n_boot,
        alpha=alpha,
        vote_threshold=vote_threshold,
        random_state=seed,
        n_jobs=n_jobs,
        change_type=change_type,
        batch_size=batch_size,
    )
    wall_time = pytime.perf_counter() - start_wall

    out: dict[str, Any] = {
        "T": int(n),
        "K_true": int(k),
        "spacing_hint": int(spacing_hint),
        "min_seg_len_used": int(min_seg_len),
        "window_sizes": window_sizes,
        "n_boot": int(n_boot),
        "vote_threshold": float(vote_threshold),
        "n_jobs": n_jobs,
        "batch_size": int(batch_size),
        "n_detected": len(result.change_points),
        "elapsed_seconds": result.metadata["elapsed_seconds"],
        "wall_time": wall_time,
        "true_cps": true_cps,
        "detected_cps": result.change_points,
        "result": result,
        "means": means,
        "sigmas": sigmas,
    }

    del x, x_std
    gc.collect()
    return out

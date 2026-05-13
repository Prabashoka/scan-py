import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scan import scan_cpd_mean

def simulate_time_series(
    n=10**6,
    n_cps=250,
    min_seg_len=1000,
    change_type="meanvar",
    seed=123
):
    """
    Simulate a univariate time series with multiple change points.

    Parameters
    ----------
    n : int
        Length of the time series.
    n_cps : int
        Number of change points.
    min_seg_len : int
        Minimum segment length.
    change_type : {"mean", "var", "meanvar"}
        Type of distributional change.
    seed : int
        Random seed.

    Returns
    -------
    x : np.ndarray
        Simulated time series of length n.
    cps : np.ndarray
        True change point locations.
    means : np.ndarray
        Segment means.
    sigmas : np.ndarray
        Segment standard deviations.
    """

    rng = np.random.default_rng(seed)

    n_segments = n_cps + 1

    if n_segments * min_seg_len > n:
        raise ValueError("min_seg_len is too large for the requested number of change points.")

    # --------------------------------------------------
    # 1. Random segment lengths with minimum spacing
    # --------------------------------------------------
    remaining = n - n_segments * min_seg_len
    extra_lengths = rng.multinomial(remaining, np.ones(n_segments) / n_segments)
    segment_lengths = min_seg_len + extra_lengths

    cps = np.cumsum(segment_lengths)[:-1]

    # --------------------------------------------------
    # 2. Generate segment-level means and variances
    # --------------------------------------------------
    means = np.zeros(n_segments)
    sigmas = np.ones(n_segments)

    if change_type in ["mean", "meanvar"]:
        jumps = rng.choice([-1, 1], size=n_segments) * rng.uniform(0.8, 2.0, size=n_segments)
        means = np.cumsum(jumps)
        means = means - np.mean(means)

    if change_type in ["var", "meanvar"]:
        sigmas = rng.uniform(0.6, 2.0, size=n_segments)

    # --------------------------------------------------
    # 3. Generate observations segment by segment
    # --------------------------------------------------
    x = np.empty(n, dtype=np.float64)

    start = 0
    for j, seg_len in enumerate(segment_lengths):
        end = start + seg_len
        x[start:end] = rng.normal(loc=means[j], scale=sigmas[j], size=seg_len)
        start = end

    return x, cps, means, sigmas

import gc
import time as pytime
import numpy as np
import pandas as pd

# ======================================================
# Benchmark configuration
# ======================================================

benchmark_config = [
    # T,       K,    spacing_hint
    (500,      4,    125),
    (1000,     25,   67),
    (5000,     42,   116),
    (10000,    53,   188),
    (20000,    67,   298),
    (50000,    92,   537),
    (100000,   116,  862),    # inferred from 100000 / 862 ≈ 116
    (1000000,  250,  4000),
    (10000000, 538,  4000)
]


# ======================================================
# Helper 1: safe minimum segment length
# ======================================================

def safe_min_seg_len(n, k, spacing_hint, safety_fraction=0.8):
    """
    For K change points, there are K + 1 segments.

    If min_seg_len is too large, simulation becomes impossible.
    This function safely clamps the requested spacing.
    """
    max_feasible = n // (k + 1)

    # Keep some flexibility for random placement of change points.
    safe_value = int(safety_fraction * max_feasible)

    min_seg_len = min(spacing_hint, safe_value)

    # Avoid extremely tiny segments for small T.
    min_seg_len = max(5, min_seg_len)

    return min_seg_len


# ======================================================
# Helper 2: choose window sizes safely for all T
# ======================================================

def choose_window_sizes(n, n_windows=7, seed=500):
    """
    Chooses window sizes for each series length.

    Your previous code used:
        upper = int(n ** (1/2))
        np.arange(100, upper + 1)

    But this fails for small n such as 500 or 1000 because sqrt(n) < 100.
    So we use a dynamic lower bound.
    """
    rng = np.random.default_rng(seed)

    upper = int(np.sqrt(n))

    # Dynamic lower bound.
    # For large n, keep your original lower bound 100.
    # For small n, use smaller valid windows.
    if upper >= 100:
        lower = 100
    else:
        lower = max(10, int(0.5 * upper))

    if lower >= upper:
        lower = max(2, upper // 2)

    candidates = np.arange(lower, upper + 1)

    if len(candidates) == 0:
        candidates = np.arange(2, max(3, upper + 1))

    if len(candidates) >= n_windows:
        window_sizes = np.sort(
            rng.choice(candidates, size=n_windows, replace=False)
        )
    else:
        # If there are not enough unique values, use all available candidates.
        window_sizes = np.sort(candidates)

    return window_sizes.astype(int).tolist()


# ======================================================
# Helper 3: run one simulation and detection
# ======================================================

def run_one_benchmark(
    n,
    k,
    spacing_hint,
    change_type="meanvar",
    n_windows=7,
    n_perm=400,
    alpha_q=1,
    threshold=0.7,
    workers=48,
    backend="thread",
    batch_size=32,
    seed=500,
):
    min_seg_len = safe_min_seg_len(
        n=n,
        k=k,
        spacing_hint=spacing_hint,
        safety_fraction=0.8,
    )

    print("=" * 80)
    print(f"Simulating T={n}, K={k}, spacing_hint={spacing_hint}, min_seg_len={min_seg_len}")

    x, true_cps, means, sigmas = simulate_time_series(
        n=n,
        n_cps=k,
        min_seg_len=min_seg_len,
        change_type=change_type,
        seed=seed,
    )

    x = np.asarray(x, dtype=np.float64)
    x_std = (x - np.mean(x)) / np.std(x)

    window_sizes = choose_window_sizes(
        n=n,
        n_windows=n_windows,
        seed=seed,
    )

    print("Window sizes:", window_sizes)

    start_wall = pytime.perf_counter()

    # Since the simulated series is meanvar, use the meanvar detector.
    cpts, rust_time = scan_cpd_mean(
        series=x_std,
        window_sizes=window_sizes,
        n_perm=n_perm,
        alpha_q=alpha_q,
        threshold=threshold,
        workers=workers,
        backend=backend,
        batch_size=batch_size,
        seed=seed,
    )

    wall_time = pytime.perf_counter() - start_wall

    out = {
        "T": n,
        "K_true": k,
        "spacing_hint": spacing_hint,
        "min_seg_len_used": min_seg_len,
        "window_sizes": window_sizes,
        "n_perm": n_perm,
        "threshold": threshold,
        "workers": workers,
        "batch_size": batch_size,
        "n_detected": len(cpts),
        "rust_time": rust_time,
        "wall_time": wall_time,
        "true_cps": true_cps,
        "detected_cps": cpts,
    }

    print(f"True K: {k}")
    print(f"Detected K: {len(cpts)}")
    print(f"Rust time: {rust_time:.4f} seconds")
    print(f"Wall time: {wall_time:.4f} seconds")

    del x, x_std
    gc.collect()

    return out


# ======================================================
# Run full benchmark
# ======================================================

results = []
computation_times = []

for n, k, spacing_hint in benchmark_config:
    result = run_one_benchmark(
        n=n,
        k=k,
        spacing_hint=spacing_hint,
        change_type="mean",
        n_windows=7,
        n_perm=400,
        alpha_q=1,
        threshold=0.7,
        workers=48,
        backend="thread",
        batch_size=10,
        seed=500,
    )

    results.append(result)
    computation_times.append(result["rust_time"])


# ======================================================
# Summary table
# ======================================================

summary_df = pd.DataFrame([
    {
        "T": r["T"],
        "K_true": r["K_true"],
        "spacing_hint": r["spacing_hint"],
        "min_seg_len_used": r["min_seg_len_used"],
        "window_sizes": r["window_sizes"],
        "n_detected": r["n_detected"],
        "rust_time": r["rust_time"],
        "wall_time": r["wall_time"],
    }
    for r in results
])

print("\nComputation times list:")
print(computation_times)

summary_df
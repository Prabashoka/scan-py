# API Reference

## Detector Functions

### `scan_cpd(...)`

Runs the full SCAN / Ensemble SCAN detector and returns a `ScanResult`.

```python
scan_cpd(
    x,
    window_sizes=None,
    alpha=0.05,
    n_boot=400,
    vote_threshold=0.5,
    min_window=15,
    max_window=None,
    block_length=None,
    block_length_rule="n^(1/3)",
    taper="tukey",
    ipm="wasserstein",
    tolerance=None,
    random_state=None,
    n_jobs=None,
    return_all=True,
    *,
    change_type=None,
    eps=1e-12,
    batch_size=32,
)
```

Parameters:

- `x`: one-dimensional numeric sequence. NaN and infinite values are rejected.
- `window_sizes`: scan window sizes. Each `w` compares adjacent windows of length `w`.
- `alpha`: bootstrap tail probability. Larger values are less conservative.
- `n_boot`: number of tapered block bootstrap replications per local comparison.
- `vote_threshold`: minimum normalized ensemble vote score retained in `result.change_points`.
- `min_window`, `max_window`: automatic window grid controls when `window_sizes=None`.
- `block_length`: optional bootstrap block length.
- `block_length_rule`: currently only `"n^(1/3)"` is supported when `block_length=None`.
- `taper`: bootstrap taper, either `"tukey"` or `"none"`.
- `ipm`: discrepancy family. Supported values are `"wasserstein"`, `"mean"`, `"var"`, and `"distribution"`.
- `tolerance`: distance used to merge nearby detections across windows. Defaults to the smallest window size.
- `random_state`: seed for reproducible bootstrap sampling.
- `n_jobs`: number of Rust/Rayon worker threads. `None` automatically uses `cpu_count - 1`, `-1` uses all available cores, and a positive integer requests that many workers capped at the CPU count. The resolved value is stored in `result.metadata["resolved_n_jobs"]`.
- `return_all`: if `False`, omits detailed diagnostics from the returned object.
- `change_type`: explicit override for `ipm`; use `"mean"`, `"var"`, or `"distribution"`.
- `eps`: numerical stability constant for local standardization.
- `batch_size`: internal bootstrap batch size.

Output:

- `ScanResult` with final change-points, scores, votes, per-window diagnostics, thresholds, parameters, metadata, and raw backend output.

### `scan_single_window(...)`

Runs SCAN for one window size and returns a `WindowResult`.

```python
scan_single_window(
    x,
    window_size,
    alpha=0.05,
    n_boot=400,
    block_length=None,
    taper="tukey",
    ipm="wasserstein",
    random_state=None,
    change_type=None,
    eps=1e-12,
    batch_size=32,
)
```

Use this for debugging one window size, inspecting scan statistics, and studying local thresholds.

## Local Statistics and Localization

### `wasserstein_statistic(left, right)`

Computes empirical one-dimensional 1-Wasserstein distance between two samples.

### `ipm_statistic(left, right, ipm="wasserstein")`

Computes a local IPM discrepancy. Currently `"wasserstein"` is supported.

### `swal_statistic(x, change_type="distribution")`

Localizes a change point inside one flagged local region. Returns an integer split index relative to the supplied block.

### `refine_cusum(x)`

CUSUM-style localizer for mean shifts. Returns an integer split index relative to `x`.

### `refine_wasserstein(x)`

Wasserstein/SWAL localizer for distributional changes. Returns `(split_index, split_scores)`.

## Bootstrap Utilities

### `tapered_block_bootstrap(x, sample_length, block_length=None, n_boot=1, taper="tukey", random_state=None)`

Generates tapered block bootstrap samples. Returns a NumPy array of shape `(n_boot, sample_length)`.

### `adaptive_threshold(left, right, alpha=0.05, n_boot=400, block_length=None, taper="tukey", random_state=None)`

Computes a local bootstrap threshold for one two-window comparison. Returns a float threshold.

## Ensemble Utilities

### `merge_change_points(change_points, tolerance=10)`

Clusters nearby candidate change-points. Returns a list of clusters.

### `ensemble_vote(window_results, vote_threshold=0.5, tolerance=10)`

Applies ensemble voting to a `{window_size: change_points}` mapping. Returns `(selected_change_points, scores, votes)`.

## Evaluation Metrics

### `match_change_points(true_cps, estimated_cps, tolerance=10)`

Greedily matches true and estimated change-points within a tolerance. Returns `(true_cp, estimated_cp)` pairs.

### `precision_recall_cpd(true_cps, estimated_cps, tolerance=10)`

Returns tolerant `(precision, recall)`.

### `f1_score_cpd(true_cps, estimated_cps, tolerance=10)`

Returns tolerant F1 score.

### `covering_metric(true_cps, estimated_cps, n)`

Computes a weighted segment-covering score in `[0, 1]`.

## Simulation Helpers

### `UnivariateSeriesSimulator(len_series, initial_value=0.0, seed=123)`

Class-based simulator for univariate series. It can generate AR(1), AR with uniform time-varying coefficients, ARMA, and ARFIMA baseline series, select spaced change-point locations, and apply random mean and/or variance shifts.

```python
from scan import UnivariateSeriesSimulator

sim = UnivariateSeriesSimulator(len_series=1000, initial_value=0.0, seed=123)
base = sim.simulate_ar_series(rho=0.4, error_type="normal")
cps = sim.select_change_point_locations(min_points=150, n_cps=4)
out = sim.apply_random_shifts(base, cps, change_type="both")

x = out["shifted_series"]
true_cps = out["change_points"]
```

`simulate_arma` requires `statsmodels`; install it with `pip install "scan-py[simulation]"`. Other simulator methods only require NumPy.

### `simulate_time_series(n=10**6, n_cps=250, min_seg_len=235, change_type="distribution", seed=123)`

Convenience wrapper using `UnivariateSeriesSimulator`: it simulates an AR(1) baseline with `rho=0.4`, selects spaced change-point locations, applies random shifts, and returns `(x, cps, means, sigmas)`.

### `safe_min_seg_len(n, k, spacing_hint, safety_fraction=0.8)`

Computes a feasible minimum segment length for simulation designs.

### `choose_window_sizes(series_length, n_windows=7, seed=500)`

Chooses reproducible scan window sizes.

### `run_one_benchmark(...)`

Simulates one benchmark series and runs `scan_cpd`. Returns a summary dictionary with the full `ScanResult` under `summary["result"]`.

## Plotting Functions

All plotting functions return plotnine `ggplot` objects.

- `plot_change_points(x, result)`: time series with detected change-points.
- `plot_swal_curve(x, start, end)`: localization curve inside one flagged local region.
- `plot_vote_scree(result)`: retained change-points versus voting threshold.
- `plot_window_votes(result, max_x_labels=12, x_label_angle=45)`: vote-count bar plot.
- `plot_thresholds(result, window_size=None)`: observed scan statistics and adaptive thresholds.
# SCAN Py

SCAN Py provides tools for detecting change points in univariate time series using sequential nonparametric window screening. It is aimed at research workflows where users need to simulate time series, detect changes across multiple window sizes, localize change-point positions, and visualize the resulting segmentation. The Python interface is backed by a Rust/PyO3 core for efficient computation.

## Installation

SCAN Py can be installed from PyPI using `pip`:

```bash
pip install scan-py
```

For local development, clone the repository and build the package in an isolated virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip maturin
maturin develop --release
```

## Change-point detection

The main function provided by SCAN Py is `scan_cpd`, which provides a unified interface for running the SCAN change-point detection framework. It detects change points in a one-dimensional time series by scanning the data with multiple local window sizes. The type of change to detect is controlled by the `change_type` argument.

Supported change types include:

| `change_type` | Explanation |
|---|---|
| `"mean"` | Detects changes mainly in the location or average level of the series. |
| `"variance"` | Detects changes mainly in the variability or scale of the series. |
| `"distribution"` | Detects broader distributional changes, not restricted to only mean or variance shifts (includes both mean and variance together). |

## Example: Detecting Multiple Mean Changes

The following example simulates a long univariate time series with multiple mean changes and applies `scan_cpd` using several window sizes. This illustrates a typical research workflow: simulate data, standardize the series, run SCAN, and compare the number of detected change points with the true number.

```python
import numpy as np
from scan import scan_cpd, simulate_time_series

T = 20_000
K = 67
min_seg_len = 235
seed = 500

window_sizes = [104, 114, 115, 120, 123, 126, 133]

x, true_cps, means, sigmas = simulate_time_series(
    n=T,
    n_cps=K,
    min_seg_len=min_seg_len,
    change_type="mean",
    seed=seed,
)

x_std = (x - np.mean(x)) / np.std(x)

result = scan_cpd(
    x_std,
    window_sizes=window_sizes,
    n_boot=400,
    alpha=1,
    vote_threshold=0.5,
    random_state=seed,
    n_jobs=8,
    change_type="mean",
    batch_size=32,
)
```
## Output

`scan_cpd` returns a `ScanResult` object containing the final detected `change points`, `scores`, `votes`, `thresholds`, `diagnostics`, `parameters`, and `metadata`.

The main attribute is:

```python
result.change_points
```

The `scan_cpd` function returns a `ScanResult` object. This object collects the main outputs of the SCAN procedure in one place, including the final detected change points, detection scores, voting information, per-window diagnostics, threshold values, input parameters, metadata, and the raw backend output.

| Attribute | Description | Why it is useful |
|---|---|---|
| `change_points` | Final estimated change-point locations returned by SCAN. | This is the main output users usually need for downstream analysis or plotting. |
| `scores` | Detection scores associated with candidate or final change points. | Helps assess the relative strength of detected changes. |
| `votes` | Number or proportion of window sizes that support each detected change point. | Useful for understanding how stable a detection is across multiple window sizes. |
| `per_window_diagnostics` | Diagnostic information from each individual window size. | Helps inspect which window sizes contributed to each detection. |
| `thresholds` | Bootstrap or calibration thresholds used during detection. | Important for reproducibility and for understanding the rejection rule. |
| `parameters` | The parameter values used in the call to `scan_cpd`, such as `window_sizes`, `alpha`, `n_boot`, and `vote_threshold`. | Makes the result self-contained and easier to reproduce. |
| `metadata` | Additional run information, such as package version, runtime information, random seed, or backend details. | Useful for experiments, reporting, debugging, and reproducibility. |
| `raw_backend_output` | Raw output returned by the Rust/PyO3 backend before post-processing. | Mainly useful for advanced users, debugging, or development. |

```python
print("Detected K:", len(result.change_points))
print("Detected change points:", result.change_points)
````

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
    n_jobs=1,
    return_all=True,
    *,
    change_type=None,
    eps=1e-12,
    batch_size=32,
)
```

Parameters:

- `x`: one-dimensional numeric sequence. `NaN` and infinite values are rejected.
- `window_sizes`: scan window sizes. Each `w` compares adjacent windows of length `w`. If `None`, a small grid is generated from `min_window` to `max_window` or `sqrt(n)`.
- `alpha`: bootstrap tail probability. Larger values are less conservative.
- `n_boot`: number of tapered block bootstrap replications per local comparison.
- `vote_threshold`: minimum normalized ensemble vote score retained in `result.change_points`.
- `min_window`, `max_window`: automatic window grid controls when `window_sizes=None`.
- `block_length`: optional bootstrap block length. If `None`, the backend uses its default rule.
- `block_length_rule`: currently only `"n^(1/3)"` is accepted when `block_length=None`.
- `taper`: bootstrap taper, either `"tukey"` or `"none"`.
- `ipm`: discrepancy family. Supported aliases include `"wasserstein"`, `"mean"`, `"var"`, and `"meanvar"`.
- `tolerance`: distance used to merge nearby detections across windows. Defaults to the smallest window size.
- `random_state`: seed for reproducible bootstrap sampling.
- `n_jobs`: number of Rust/Rayon worker threads. Use `None` for default Rayon behavior.
- `return_all`: if `False`, returns only final result fields and omits detailed diagnostics.
- `change_type`: explicit override for `ipm`; use `"mean"`, `"var"`, or `"meanvar"`.
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

Useful for debugging one window size, inspecting scan statistics, and studying local thresholds.

Output fields:

- `window_size`: scanned window size.
- `change_points`: candidate change-points from this window size.
- `starts`: scan block start indices.
- `statistics`: observed IPM statistics at each scan start.
- `lower_thresholds`, `upper_thresholds`: adaptive bootstrap thresholds.
- `localized_regions`: flagged local regions used for refinement.

```python
window_result = scan_single_window(x_std, window_size=104, n_boot=100, random_state=500)
print(window_result.change_points[:10])
print(window_result.upper_thresholds[:5])
```
## Result Objects

### `ScanResult`

Returned by `scan_cpd`.

Fields:

- `change_points`: sorted final detected change-points after ensemble voting.
- `scores`: `{change_point: normalized_vote_score}`. Scores are usually in `[0, 1]`.
- `votes`: `{change_point: vote_count}` before normalization.
- `window_results`: `{window_size: WindowResult}` diagnostics for every window size.
- `thresholds`: `{window_size: {starts, lower, upper, statistics}}` for plotting/debugging thresholds.
- `parameters`: detector parameters used for the run.
- `metadata`: run metadata such as `n_obs`, elapsed seconds, index convention, and Rust backend name.
- `segments`: merged candidate segments used during voting.
- `raw`: raw dictionary returned by the Rust backend.
- `cp_dict`: property returning `{window_size: candidate_change_points}`.

```python
print(result.change_points)
print(result.scores)
print(result.votes)
print(result.parameters)
print(result.metadata)
print(result.cp_dict)
```

### `WindowResult`

Returned by `scan_single_window` and stored inside `result.window_results`.

Fields:

- `window_size`: integer window size.
- `change_points`: candidates detected by that window size.
- `starts`: scan starts.
- `statistics`: observed scan statistics.
- `lower_thresholds`: lower bootstrap thresholds.
- `upper_thresholds`: upper bootstrap thresholds.
- `localized_regions`: `(start, end)` local regions that were refined.

```python
wr = result.window_results[window_sizes[0]]
print(wr.window_size)
print(wr.change_points[:10])
print(wr.starts[:10])
print(wr.statistics[:10])
print(wr.upper_thresholds[:10])
```

## Local Statistics and Localization

### `wasserstein_statistic(left, right)`

Computes the empirical one-dimensional 1-Wasserstein distance between two samples.

Parameters:

- `left`, `right`: non-empty numeric iterables.

Output:

- `float` Wasserstein distance.

```python
w = wasserstein_statistic(x_std[:100], x_std[100:200])
print(w)
```

### `ipm_statistic(left, right, ipm="wasserstein")`

Computes a local IPM discrepancy. Currently `"wasserstein"` is supported.

Parameters:

- `left`, `right`: non-empty numeric iterables.
- `ipm`: discrepancy name. Use `"wasserstein"`.

Output:

- `float` discrepancy value.

```python
stat = ipm_statistic(x_std[:100], x_std[100:200])
print(stat)
```

### `localize_cp(x, change_type="meanvar")`

Localizes a change point inside one flagged local region.

Parameters:

- `x`: local block containing a suspected change.
- `change_type`: `"mean"`, `"var"`, or `"meanvar"`.

Output:

- integer split index relative to the supplied block.

```python
local_cp = localize_cp(x_std[150:260], change_type="mean")
print(local_cp)
```

### `refine_cusum(x)`

CUSUM-style localizer for mean shifts.

Output:

- integer split index relative to `x`.

```python
split = refine_cusum(x_std[150:260])
print(split)
```

### `refine_wasserstein(x)`

Wasserstein/SWAL localizer for distributional changes.

Output:

- `(split_index, split_scores)` where `split_scores` contains the localization curve.

```python
split, curve = refine_wasserstein(x_std[150:260])
print(split)
print(curve[:5])
```

## Bootstrap Utilities

### `tapered_block_bootstrap(x, sample_length, block_length=None, n_boot=1, taper="tukey", random_state=None)`

Generates tapered block bootstrap samples.

Parameters:

- `x`: reference numeric sequence.
- `sample_length`: length of each bootstrap sample.
- `block_length`: block length; default uses a cube-root style rule.
- `n_boot`: number of samples.
- `taper`: `"tukey"` or `"none"`.
- `random_state`: reproducibility seed.

Output:

- NumPy array of shape `(n_boot, sample_length)`.

```python
samples = tapered_block_bootstrap(x_std[:500], sample_length=100, n_boot=5, random_state=123)
print(samples.shape)
```

### `adaptive_threshold(left, right, alpha=0.05, n_boot=400, block_length=None, taper="tukey", random_state=None)`

Computes a local bootstrap threshold for one two-window comparison.

Output:

- `float` threshold.

```python
threshold = adaptive_threshold(x_std[:100], x_std[100:200], n_boot=100, random_state=123)
print(threshold)
```

## Ensemble Utilities

### `merge_change_points(change_points, tolerance=10)`

Clusters nearby candidate change-points.

Output:

- list of clusters, e.g. `[[99, 101], [250]]`.

```python
clusters = merge_change_points([99, 101, 250], tolerance=5)
print(clusters)
```

### `ensemble_vote(window_results, vote_threshold=0.5, tolerance=10)`

Applies ensemble voting to per-window candidate lists.

Parameters:

- `window_results`: mapping like `{40: [200, 400], 60: [202, 399]}`.
- `vote_threshold`: normalized vote cutoff.
- `tolerance`: merge distance for nearby candidates.

Output:

- `(selected_change_points, scores, votes)`.

```python
selected, scores, votes = ensemble_vote({40: [200, 400], 60: [202, 399]}, tolerance=5)
print(selected)
print(scores)
print(votes)
```

## Evaluation Metrics

### `match_change_points(true_cps, estimated_cps, tolerance=10)`

Greedily matches true and estimated change-points within a tolerance.

Output:

- list of `(true_cp, estimated_cp)` pairs.

### `precision_recall_cpd(true_cps, estimated_cps, tolerance=10)`

Returns tolerant precision and recall.

Output:

- `(precision, recall)`.

### `f1_score_cpd(true_cps, estimated_cps, tolerance=10)`

Returns tolerant F1 score.

Output:

- `float` F1 score.

### `covering_metric(true_cps, estimated_cps, n)`

Computes a weighted segment-covering score in `[0, 1]`.

Parameters:

- `true_cps`: true change-points.
- `estimated_cps`: detected change-points.
- `n`: sequence length.

Output:

- `float` covering score.

```python
matches = match_change_points(true_cps, result.change_points, tolerance=25)
precision, recall = precision_recall_cpd(true_cps, result.change_points, tolerance=25)
f1 = f1_score_cpd(true_cps, result.change_points, tolerance=25)
covering = covering_metric(true_cps, result.change_points, n=len(x_std))

print(matches[:5])
print(precision, recall, f1, covering)
```
## Simulation Helpers

### `simulate_time_series(n=10**6, n_cps=250, min_seg_len=1000, change_type="meanvar", seed=123)`

Simulates a univariate series with multiple change-points.

Parameters:

- `n`: sequence length.
- `n_cps`: number of change-points.
- `min_seg_len`: minimum segment length.
- `change_type`: `"mean"`, `"var"`, or `"meanvar"`.
- `seed`: random seed.

Output:

- `(x, cps, means, sigmas)`.

```python
x, cps, means, sigmas = simulate_time_series(
    n=2000,
    n_cps=5,
    min_seg_len=200,
    change_type="mean",
    seed=123,
)
```

### `safe_min_seg_len(n, k, spacing_hint, safety_fraction=0.8)`

Computes a feasible minimum segment length for simulation designs.

Output:

- integer minimum segment length.

```python
min_len = safe_min_seg_len(n=20_000, k=67, spacing_hint=298)
print(min_len)  # 235
```

### `choose_window_sizes(n, n_windows=7, seed=500)`

Chooses reproducible scan window sizes.

Output:

- sorted list of integer window sizes.

```python
windows = choose_window_sizes(n=20_000, n_windows=7, seed=500)
print(windows)  # [104, 114, 115, 120, 123, 126, 133]
```

### `run_one_benchmark(...)`

Simulates one benchmark series and runs `scan_cpd`.

```python
summary = run_one_benchmark(
    n=20_000,
    k=67,
    spacing_hint=298,
    change_type="mean",
    n_boot=400,
    alpha=1,
    vote_threshold=0.7,
    n_jobs=8,
    seed=500,
)

print(summary["n_detected"])
print(summary["detected_cps"][:10])
```

Output:

- dictionary containing simulation configuration, detected CPs, true CPs, timings, means, sigmas, and the full `ScanResult` under `summary["result"]`.

### `BENCHMARK_CONFIG`

Default benchmark configurations as `(T, K, spacing_hint)` tuples.

```python
from scan import BENCHMARK_CONFIG
print(BENCHMARK_CONFIG)
```

## Plotting Functions

Plotting uses plotnine and returns `ggplot` objects. In notebooks, put the plot object as the last expression in a cell. In scripts, call `.save(...)`.

### `plot_change_points(x, result)`

Plots the time series with detected change-points as vertical dashed lines.

```python
plot_change_points(x_std, result)
```

### `plot_swal_curve(x, start, end)`

Plots the SWAL/Wasserstein localization curve inside one flagged local region.

```python
start, end = result.window_results[window_sizes[0]].localized_regions[0]
plot_swal_curve(x_std, start, end)
```

### `plot_vote_scree(result)`

Plots number of retained change-points versus voting threshold.

```python
plot_vote_scree(result)
```

### `plot_window_votes(result, max_x_labels=12, x_label_angle=45)`

Plots ensemble vote counts as a bar chart.

Parameters:

- `max_x_labels`: maximum number of x-axis tick labels.
- `x_label_angle`: rotation angle for x-axis labels.

```python
plot_window_votes(result, max_x_labels=15, x_label_angle=45)
```

### `plot_thresholds(result, window_size=None)`

Plots observed scan statistics and adaptive bootstrap thresholds.

Parameters:

- `window_size`: if provided, show one window size; otherwise facet over all window sizes.

```python
plot_thresholds(result)
plot_thresholds(result, window_size=window_sizes[0])
```

### Saving plots

```python
from pathlib import Path

out_dir = Path("scan_plots")
out_dir.mkdir(exist_ok=True)

plot_change_points(x_std, result).save(out_dir / "change_points.png", width=11, height=4.8, dpi=150)
plot_vote_scree(result).save(out_dir / "vote_scree.png", width=8, height=4.8, dpi=150)
plot_window_votes(result).save(out_dir / "window_votes.png", width=10, height=4.8, dpi=150)
plot_thresholds(result).save(out_dir / "thresholds.png", width=12, height=7, dpi=150)
```

## Compatibility Wrappers

The previous wrappers remain available and return `(change_points, elapsed_seconds)`:

```python
from scan import scan_cpd_mean, scan_cpd_var, scan_cpd_meanvar

cps, elapsed = scan_cpd_mean(y, window_sizes=[20, 30, 40])
```

New projects should prefer `scan_cpd()` because it returns diagnostics, thresholds, votes, and metadata.

## Full Example Notebook

See `example-usage.ipynb` for executable examples covering:

- simulation helpers
- `scan_cpd`
- `ScanResult` and `WindowResult`
- lower-level statistics and localization
- bootstrap utilities
- ensemble voting
- evaluation metrics
- all plotnine plotting functions
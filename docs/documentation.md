# Change-point detection

## `scan_cpd(...)`

`scan_cpd` runs the full SCAN / Ensemble SCAN change-point detection procedure. It takes a one-dimensional numeric time series, scans it using one or more local window sizes, applies the selected discrepancy measure, and returns a `ScanResult` object containing the detected change points and related diagnostic information.

```python
scan_cpd(x, window_sizes=None, alpha=0.05, n_boot=400, vote_threshold=0.5,
         min_window=15, max_window=None, block_length=None, block_length_rule="n^(1/3)",
         taper="tukey", ipm="wasserstein", tolerance=None, random_state=None,
         n_jobs=None, return_all=True, *, change_type=None, eps=1e-12, batch_size=32)


```
| Parameter           | Type / Default                    | Description                                                                                                                                                                                                                                                          |
| ------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x`                 | One-dimensional numeric sequence  | Input time series. This can be a list, tuple, or NumPy array. The values must be finite; `NaN` and infinite values are rejected.                                                                                                                                     |
| `window_sizes`      | List of integers, default `None`  | Window sizes used for scanning the time series. Each window size `w` compares adjacent local windows of length `w`. If `None`, SCAN automatically generates a small grid of window sizes using `min_window`, `max_window`, and the length of the series.             |
| `alpha`             | Float, default `0.05`             | Bootstrap tail probability used for local thresholding. Smaller values are more conservative, while larger values allow more candidate detections.                                                                                                                   |
| `n_boot`            | Integer, default `400`            | Number of tapered block bootstrap replications used for each local comparison. Larger values give more stable thresholds but increase computation time.                                                                                                              |
| `vote_threshold`    | Float, default `0.5`              | Minimum normalized ensemble vote required for a candidate change point to be retained in `result.change_points`. Higher values require stronger agreement across window sizes.                                                                                       |
| `min_window`        | Integer, default `15`             | Smallest window size used when `window_sizes=None`. Ignored if `window_sizes` is provided explicitly.                                                                                                                                                                |
| `max_window`        | Integer or `None`, default `None` | Largest window size used when `window_sizes=None`. If `None`, an automatic upper value is chosen based on the length of the input series.                                                                                                                            |
| `block_length`      | Integer or `None`, default `None` | Block length used in the tapered block bootstrap. If `None`, the backend uses the rule specified by `block_length_rule`.                                                                                                                                             |
| `block_length_rule` | String, default `"n^(1/3)"`       | Rule used to choose the bootstrap block length when `block_length=None`. Currently, `"n^(1/3)"` is supported.                                                                                                                                                        |
| `taper`             | String, default `"tukey"`         | Tapering method used in the block bootstrap. Use `"tukey"` for Tukey tapering or `"none"` for no tapering.                                                                                                                                                           |
| `ipm`               | String, default `"wasserstein"`   | Discrepancy measure used to compare adjacent windows. Supported values are `"wasserstein"`, `"mean"`, `"var"`, and `"distribution"`.                                                                                                                                 |
| `tolerance`         | Integer or `None`, default `None` | Distance used to merge nearby detections across different window sizes. If `None`, the smallest window size is used as the default merging tolerance.                                                                                                                |
| `random_state`      | Integer or `None`, default `None` | Random seed used for bootstrap sampling. Set this value to make results reproducible.                                                                                                                                                                                |
| `n_jobs`            | Integer, `-1`, or `None`, default `None` | Number of Rust/Rayon worker threads used for parallel computation. `None` automatically uses one fewer than the available CPU cores, `-1` uses all cores, and a positive integer requests that many workers, capped at the system CPU count.                                                                                                                                     |
| `return_all`        | Boolean, default `True`           | If `True`, returns detailed diagnostics in addition to the final detected change points. If `False`, omits some detailed intermediate outputs.                                                                                                                       |
| `change_type`       | String or `None`, default `None`  | Explicitly specifies the type of change to detect. Supported options are exactly `"mean"`, `"var"`, and `"distribution"`. When provided, this can be used as a clearer alternative to setting `ipm` directly. |
| `eps`               | Float, default `1e-12`            | Small numerical constant used for stability, especially in local standardization or scale-related calculations.                                                                                                                                                      |
| `batch_size`        | Integer, default `32`             | Internal batch size used when processing bootstrap computations. Larger values may improve throughput but can increase memory use.                                                                                                                                   |

`scan_cpd` returns a `ScanResult` object. Most users need only `result.change_points`, but the object also stores scores, votes, thresholds, per-window diagnostics, parameters, and metadata. See [Results](docs/scanresult.md) documentation for the full output description.

### `scan_single_window(...)`

Runs SCAN for one window size and returns a `WindowResult`.

```python
scan_single_window(x, window_size, alpha=0.05, n_boot=400, block_length=None,
                   taper="tukey", ipm="wasserstein", random_state=None,
                   change_type=None, eps=1e-12, batch_size=32)
```

Useful for debugging one window size, inspecting scan statistics, and studying local thresholds and how each detector in the ensemble model works.

| Parameter      | Type / Default                    | Description                                                                                                                                                                      |
| -------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `x`            | One-dimensional numeric sequence  | Input time series. This can be a list, tuple, or NumPy array. The values must be finite; `NaN` and infinite values are rejected.                                                 |
| `window_size`  | Integer                           | Window size used for the scan. Adjacent local windows of length `window_size` are compared across the series.                                                                    |
| `alpha`        | Float, default `0.05`             | Bootstrap tail probability used for local thresholding. Smaller values are more conservative, while larger values allow more candidate detections.                               |
| `n_boot`       | Integer, default `400`            | Number of tapered block bootstrap replications used for each local comparison. Larger values give more stable thresholds but increase computation time.                          |
| `block_length` | Integer or `None`, default `None` | Block length used in the tapered block bootstrap. If `None`, the backend uses its default block-length rule.                                                                     |
| `taper`        | String, default `"tukey"`         | Tapering method used in the block bootstrap. Use `"tukey"` for Tukey tapering or `"none"` for no tapering.                                                                       |
| `ipm`          | String, default `"wasserstein"`   | Discrepancy measure used to compare adjacent windows. Supported values are `"wasserstein"`, `"mean"`, `"var"`, and `"distribution"`.                                             |
| `random_state` | Integer or `None`, default `None` | Random seed used for bootstrap sampling. Set this value to make results reproducible.                                                                                            |
| `change_type`  | String or `None`, default `None`  | Explicitly specifies the type of change to detect. Supported options are exactly `"mean"`, `"var"`, and `"distribution"`. |
| `eps`          | Float, default `1e-12`            | Small numerical constant used for numerical stability, especially in local standardization or scale-related calculations.                                                        |
| `batch_size`   | Integer, default `32`             | Internal batch size used when processing bootstrap computations. Larger values may improve throughput but can increase memory use.                                               |

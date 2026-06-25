# Change-point detection

## `scan_cpd(...)`

`scan_cpd` runs the full SCAN / Ensemble SCAN change-point detection procedure. It takes a one-dimensional numeric time series, scans it using one or more local window sizes, applies the selected discrepancy measure, and returns a `ScanResult` object containing the detected change points and related diagnostic information.

```python
scan_cpd(x, window_sizes=None, alpha=0.05, n_boot=400, vote_threshold=0.5,
         min_window=15, max_window=None, block_length=None, block_length_rule="n^(1/3)",
         taper="tukey", ipm="wasserstein", tolerance=None, random_state=None,
         n_jobs=1, return_all=True, *, change_type=None, eps=1e-12, batch_size=32)


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
| `ipm`               | String, default `"wasserstein"`   | Discrepancy measure used to compare adjacent windows. Supported aliases include `"wasserstein"`, `"mean"`, `"var"`, and `"meanvar"`.                                                                                                                                 |
| `tolerance`         | Integer or `None`, default `None` | Distance used to merge nearby detections across different window sizes. If `None`, the smallest window size is used as the default merging tolerance.                                                                                                                |
| `random_state`      | Integer or `None`, default `None` | Random seed used for bootstrap sampling. Set this value to make results reproducible.                                                                                                                                                                                |
| `n_jobs`            | Integer or `None`, default `1`    | Number of Rust/Rayon worker threads used for parallel computation. Use `None` to allow Rayon to use its default thread behavior.                                                                                                                                     |
| `return_all`        | Boolean, default `True`           | If `True`, returns detailed diagnostics in addition to the final detected change points. If `False`, omits some detailed intermediate outputs.                                                                                                                       |
| `change_type`       | String or `None`, default `None`  | Explicitly specifies the type of change to detect. Supported options include `"mean"`, `"var"`, `"meanvar"`, and broader distributional settings through the discrepancy choice. When provided, this can be used as a clearer alternative to setting `ipm` directly. |
| `eps`               | Float, default `1e-12`            | Small numerical constant used for stability, especially in local standardization or scale-related calculations.                                                                                                                                                      |
| `batch_size`        | Integer, default `32`             | Internal batch size used when processing bootstrap computations. Larger values may improve throughput but can increase memory use.                                                                                                                                   |

`scan_cpd` returns a `ScanResult` object. Most users need only `result.change_points`, but the object also stores scores, votes, thresholds, per-window diagnostics, parameters, and metadata.


### `ScanResult`

The `scan_cpd` function returns a `ScanResult` object. This object collects the main outputs of the SCAN procedure in one place, including the final detected change points, detection scores, voting information, per-window diagnostics, threshold values, input parameters, metadata, and the raw backend output.

| Attribute | Description | Usage |
|---|---|---|
| `change_points` | Final estimated change-point locations returned by SCAN. | This is the main output users usually need for downstream analysis or plotting. |
| `scores` | Detection scores associated with candidate or final change points. | Helps assess the relative strength of detected changes. |
| `votes` | Number or proportion of window sizes that support each detected change point. | Useful for understanding how stable a detection is across multiple window sizes. |
| `per_window_diagnostics` | Diagnostic information from each individual window size. | Helps inspect which window sizes contributed to each detection. |
| `thresholds` | Bootstrap or calibration thresholds used during detection. | Important for reproducibility and for understanding the rejection rule. |
| `parameters` | The parameter values used in the call to `scan_cpd`, such as `window_sizes`, `alpha`, `n_boot`, and `vote_threshold`. | Makes the result self-contained and easier to reproduce. |
| `metadata` | Additional run information, such as package version, runtime information, random seed, or backend details. | Useful for experiments, reporting, debugging, and reproducibility. |
| `raw_backend_output` | Raw output returned by the Rust/PyO3 backend before post-processing. | Mainly useful for advanced users, debugging, or development. |

The object returned by `scan_cpd` stores the main outputs of the detection procedure as attributes. Each attribute can be accessed using the dot operator (`.`), which makes it easy to inspect the detected change points, detection scores, voting information, run parameters, metadata, and per-window results.

For example:
```python
print(result.change_points)
print(result.scores)
print(result.votes)
print(result.parameters)
print(result.metadata)
print(result.cp_dict)
```

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
| `ipm`          | String, default `"wasserstein"`   | Discrepancy measure used to compare adjacent windows. Supported aliases include `"wasserstein"`, `"mean"`, `"var"`, and `"meanvar"`.                                             |
| `random_state` | Integer or `None`, default `None` | Random seed used for bootstrap sampling. Set this value to make results reproducible.                                                                                            |
| `change_type`  | String or `None`, default `None`  | Explicitly specifies the type of change to detect. Supported options include `"mean"`, `"var"`, `"meanvar"`, and broader distributional settings through the discrepancy choice. |
| `eps`          | Float, default `1e-12`            | Small numerical constant used for numerical stability, especially in local standardization or scale-related calculations.                                                        |
| `batch_size`   | Integer, default `32`             | Internal batch size used when processing bootstrap computations. Larger values may improve throughput but can increase memory use.                                               |

### `WindowResult`

The `WindowResult` object contains the diagnostics from running SCAN at one specific window size. It is returned directly by `scan_single_window`, and it also appears inside `ScanResult.window_results`, where each key is a window size and each value is a `WindowResult`.

| Attribute | Description | Why it is useful |
|---|---|---|
| `window_size` | The integer window size used for this scan. | Identifies which local scale produced the diagnostics and candidate change points. |
| `change_points` | Candidate change-point locations detected using this single window size. | Helps inspect how one window size behaves before ensemble voting merges results across windows. |
| `starts` | Starting indices of the scanned local regions. | Shows where the detector placed each adjacent-window comparison along the time series. |
| `statistics` | Observed scan statistics computed at each scan start. | Useful for seeing where the local discrepancy between adjacent windows is large. |
| `tapered_block_bootstrap_threshold` | Tapered block bootstrap thresholds for each scanned region. | Provides the rejection threshold; detections occur when the observed statistic exceeds this threshold. |
| `localized_regions` | Local `(start, end)` regions that were flagged and then refined into candidate change points. | Helps trace each detected candidate back to the region where localization was performed. |

Per-window results are stored in `result.window_results`, which is indexed by window size. This allows users to inspect the candidate change points and diagnostics produced by each individual window size before the final voting or aggregation step. In the example below, `40` refers to the window size used during the scan.

```python
wr = result.window_results[40]
print("Window size:", wr.window_size)
print("Candidate CPs:", wr.change_points)
```

### `swal_statistic(...)`


`swal_statistic(x)`

Returns the localized change-point position inside a flagged local region using the SWAL statistic used in the localization step.

This function is useful when a candidate region has already been identified and you want to refine the estimated change-point location within that region. This function can be used as a custom 

| Parameter | Type / Default | Description |
|---|---|---|
| `x` | One-dimensional numeric sequence | Local segment of the time series containing a suspected change point. The sequence must contain at least 3 finite observations. |
| `change_type` | String, default `"distribution"` | Type of change to localize. Supported values are `"mean"`, `"var"`, and `"distribution"`. |
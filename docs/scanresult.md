# Results

## `ScanResult`

The `scan_cpd` function returns a `ScanResult` object. This object collects the main outputs of the SCAN procedure in one place, including the final detected change points, detection scores, voting information, per-window diagnostics, threshold values, input parameters, metadata, and the raw backend output.

| Attribute | Description | Usage |
|---|---|---|
| `change_points` | Final estimated change-point locations returned by SCAN. | This is the main output users usually need for downstream analysis or plotting. |
| `scores` | Detection scores associated with candidate or final change points. | Helps assess the relative strength of detected changes. |
| `votes` | Number or proportion of window sizes that support each detected change point. | Useful for understanding how stable a detection is across multiple window sizes. |
| `per_window_diagnostics` | Diagnostic information from each individual window size. | Helps inspect which window sizes contributed to each detection. |
| `thresholds` | Bootstrap or calibration thresholds used during detection. | Important for reproducibility and for understanding the rejection rule. |
| `parameters` | The parameter values used in the call to `scan_cpd`, such as `window_sizes`, `alpha`, `n_boot`, and `vote_threshold`. | Makes the result self-contained and easier to reproduce. |
| `metadata` | Additional run information, such as runtime, backend details, `cpu_count`, and `resolved_n_jobs`. | Useful for experiments, reporting, debugging, and reproducibility, especially when `n_jobs=None` chooses workers automatically. |
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

Worker selection is also recorded in the result metadata:

```python
print(result.parameters["n_jobs"])        # requested value, often None for automatic
print(result.metadata["cpu_count"])       # CPU count seen by Python
print(result.metadata["resolved_n_jobs"]) # worker count passed to the Rust backend
```

## `WindowResult`

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

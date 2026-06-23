# Theory Notes

SCAN Py detects change-points by comparing adjacent local windows over a time series.

For a window size `w` and scan start `s`, the detector compares:

```text
left  = x[s : s + w]
right = x[s + w : s + 2w]
```

A local discrepancy statistic is computed between the two windows. The current Rust backend uses Wasserstein-style discrepancies for the scan statistic and then applies a bootstrap threshold to decide whether the local region is flagged.

## Adaptive Thresholding

Thresholds are computed locally using tapered block bootstrap samples. This helps account for local scale and dependence structure. The main controls are:

- `n_boot`: number of bootstrap replications.
- `block_length`: optional bootstrap block length.
- `taper`: tapering scheme, currently `"tukey"` or `"none"`.
- `alpha`: tail probability for the threshold.

## Localization

Once a local region is flagged, SCAN Py refines the change-point location within that region:

- `change_type="mean"`: CUSUM-style localization.
- `change_type="var"` or `"meanvar"`: Wasserstein/SWAL-style localization.

The localized change-point is returned using Python split indexing.

## Ensemble Voting

Multiple window sizes can produce nearby candidate change-points. SCAN Py merges nearby candidates using `tolerance`, then selects final change-points using `vote_threshold`.

This ensemble design is useful because different window sizes can be sensitive to different spacing and signal-strength regimes.

## Index Convention

A detected change-point `t` means:

```python
left_segment = x[:t]
right_segment = x[t:]
```

This is 0-based Python split indexing, not 1-based indexing.
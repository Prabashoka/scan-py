# SCAN CPD

SCAN CPD is a Python package for **S**equential **C**hange-point **A**nalysis via
**N**onparametric window screening. It provides a small Python API backed by a
Rust/PyO3 extension for fast multi-window change-point detection.

The package is useful when you have a one-dimensional sequence and want to find
locations where the mean, variance, or full distribution appears to change.

## Installation

```bash
pip install scan-cpd
```

For local development from this repository:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip maturin
maturin develop --release
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Quick Start

```python
import numpy as np
from scan import scan_cpd_mean

rng = np.random.default_rng(123)

# Simulated data with true changes at indices 150 and 300.
y = np.r_[
    rng.normal(0.0, 1.0, 150),
    rng.normal(1.8, 1.0, 150),
    rng.normal(-0.7, 1.0, 150),
]

cpts, elapsed = scan_cpd_mean(
    y,
    window_sizes=[20, 30, 40],
    n_perm=50,
    threshold=0.5,
    workers=2,
    seed=123,
)

print(cpts)
print(f"Elapsed seconds: {elapsed:.4f}")
```

Example output:

```text
[145, 301]
Elapsed seconds: 0.0007
```

Detected change points are returned as **1-based indices**. In this example the
true changes are near 150 and 300, so small offsets are expected.

## Simulated Data Examples

### Mean Changes

Use `scan_cpd_mean` when the primary signal is a shift in average level.

```python
import numpy as np
from scan import scan_cpd_mean

rng = np.random.default_rng(42)
y = np.r_[
    rng.normal(2.0, 0.8, 200),
    rng.normal(5.0, 0.8, 200),
    rng.normal(1.0, 0.8, 200),
]

cpts, elapsed = scan_cpd_mean(
    y,
    window_sizes=[25, 40, 60],
    n_perm=100,
    alpha_q=1.0,
    threshold=0.5,
    workers=4,
    seed=42,
)

print("Detected mean changes:", cpts)
```

### Variance Changes

Use `scan_cpd_var` when the mean stays similar but the spread changes.

```python
import numpy as np
from scan import scan_cpd_var

rng = np.random.default_rng(7)
y = np.r_[
    rng.normal(0.0, 0.5, 150),
    rng.normal(0.0, 1.8, 150),
    rng.normal(0.0, 0.8, 150),
]

cpts, elapsed = scan_cpd_var(
    y,
    window_sizes=[20, 30, 40],
    n_perm=100,
    threshold=0.5,
    workers=4,
    seed=7,
)

print("Detected variance changes:", cpts)
```

`scan_cpd_var` keeps `threshold=5.0` as its historical default. For most
practical use, set `threshold=0.5` because leader scores are clipped to `[0, 1]`.

### Joint Mean and Variance Changes

Use `scan_cpd_meanvar` when both the center and spread may change.

```python
import numpy as np
from scan import scan_cpd_meanvar

rng = np.random.default_rng(99)
y = np.r_[
    rng.normal(0.0, 0.7, 180),
    rng.normal(2.0, 1.5, 180),
    rng.normal(-1.0, 0.9, 180),
]

cpts, elapsed = scan_cpd_meanvar(
    y,
    window_sizes=[25, 35, 50],
    n_perm=100,
    threshold=0.5,
    workers=4,
    seed=99,
)

print("Detected mean/variance changes:", cpts)
```

## Plotting Results

The package includes a small plotting helper for exploratory work.

```python
import numpy as np
from scan import scan_cpd_mean, plot_detected_changepoints

rng = np.random.default_rng(123)
y = np.r_[
    rng.normal(0.0, 1.0, 250),
    rng.normal(1.5, 1.0, 250),
    rng.normal(-0.8, 1.0, 250),
]

cpts, elapsed = scan_cpd_mean(
    y,
    window_sizes=[25, 35, 50],
    n_perm=100,
    threshold=0.5,
    workers=4,
    seed=123,
)

plot_detected_changepoints(y, cpts, title="SCAN detected change points")
```

## Raw Detector Output

`scan_detector` returns the full multi-window result rather than only the final
filtered change points.

```python
import numpy as np
from scan import scan_detector

rng = np.random.default_rng(123)
y = np.r_[
    rng.normal(0.0, 1.0, 150),
    rng.normal(1.8, 1.0, 150),
    rng.normal(-0.7, 1.0, 150),
]

result = scan_detector(
    y,
    window_sizes=[20, 30, 40],
    n_perm=50,
    change_type="mean",
    workers=2,
    seed=123,
)

print(result.keys())
print(result["cp_dict"])
print(result["out"]["leaders_scores"])
```

The result dictionary contains:

| Key | Description |
| --- | --- |
| `cp_dict` | Candidate change points detected for each window size. |
| `timings` | Runtime in seconds for each window size. |
| `total_time` | Total detector runtime in seconds. |
| `segments` | Nearby candidates grouped into tolerance-based segments. |
| `out` | Aggregated leader votes, scores, probabilities, and CDF. |

## Local Refinement Helpers

If you already have a candidate block and only want to localize the split inside
that block, use the refinement helpers directly.

```python
import numpy as np
from scan import refine_cusum, refine_wasserstein

rng = np.random.default_rng(2024)
block = np.r_[rng.normal(0, 1, 40), rng.normal(2, 1, 40)]

mean_split = refine_cusum(block)
wasserstein_split, split_scores = refine_wasserstein(block)

print(mean_split)
print(wasserstein_split)
```

These helpers return split locations relative to the block passed into the
function.

## API Reference

### `scan_cpd_mean(series, window_sizes, ...)`

Detect mean shifts with CUSUM localization.

Returns:

```python
(change_points: list[int], elapsed_seconds: float)
```

### `scan_cpd_var(series, window_sizes, ...)`

Detect variance-sensitive changes with Wasserstein localization.

Returns:

```python
(change_points: list[int], elapsed_seconds: float)
```

### `scan_cpd_meanvar(series, window_sizes, ...)`

Detect joint mean and variance changes with Wasserstein localization.

Returns:

```python
(change_points: list[int], elapsed_seconds: float)
```

### `scan_detector(series, window_sizes=None, ...)`

Run the complete detector and return the raw multi-window output dictionary.

### `refine_cusum(series)`

Return the CUSUM split location inside a candidate block.

### `refine_wasserstein(series)`

Return the Wasserstein split location and all split statistics inside a
candidate block.

### Common Parameters

| Parameter | Description |
| --- | --- |
| `series` | One-dimensional iterable of finite numeric values. |
| `window_sizes` | Positive integer window sizes to scan. Use several sizes when the change duration is unknown. |
| `n_perm` | Number of bootstrap/permutation replications used to estimate thresholds. Larger values are more stable but slower. |
| `alpha_q` | Quantile control. Values `<= 1.0` are interpreted as proportions and values `> 1.0` as percentages. |
| `seed` | Random seed for deterministic bootstrap behavior. |
| `tol` | Distance used to group nearby candidate change points. If omitted in high-level functions, the minimum window size is used. |
| `workers` | Number of Rayon worker threads. Use `None` for the default Rayon behavior. |
| `backend` | Accepts `"thread"` or `"process"` for API compatibility. The Rust implementation uses Rayon threads internally. |
| `threshold` | Minimum leader score for returning a final change point from high-level functions. |
| `eps` | Small positive value used for numerical stability. |
| `b` | Optional block length for bootstrap sampling. |
| `taper_ratio` | Fraction of the block used for tapering. |
| `center` | Whether to center local bootstrap blocks before threshold estimation. |
| `batch_size` | Number of bootstrap replications processed per internal batch. |

## Practical Tips

- Start with three to five window sizes around the shortest segment length you
  expect to detect.
- Increase `n_perm` for final analysis after prototyping with smaller values.
- Use `scan_detector` when you want to inspect candidate votes before choosing a
  threshold.
- Clean or impute missing values first. The Rust core rejects `NaN` and infinite
  values.
- Returned high-level change points are 1-based indices. Convert to Python
  0-based positions with `[cp - 1 for cp in cpts]` if needed.

## Build and Publish

Build a wheel locally:

```bash
maturin build --release
```

The wheel will be written under `target/wheels/`.

Before publishing to PyPI, check that the package metadata uses this README:

```toml
[project]
readme = "README.md"
```

Publish with Maturin after configuring your PyPI credentials:

```bash
maturin publish --release
```

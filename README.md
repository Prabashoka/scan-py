# scan-py: Sequentially Detecting Change-points via Adaptive Nonparametric Inference

`scan-py` provides tools for detecting change points general distributional shifts in long univariate time series using Integral Probability Metrics (IPMs). It is aimed at research workflows where users need to simulate time series, detect changes across multiple window sizes, localize change-point positions, evaluate accuracy, and visualize diagnostics. The Python interface is backed by a Rust/PyO3 computation core.

## Installation

SCAN Py can be installed from PyPI using pip:

```bash
pip install scan-py
```
For local development, clone the repository and build the package in an isolated virtual environment:

```python
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

### Example usage

The following example simulates a univariate time series with multiple mean changes and applies `scan_cpd` using several window sizes.


```python
import numpy as np
from scan import scan_cpd, simulate_time_series

T = 20_000 # Series length
K = 67 # Number of change points
min_seg_len = 235 # Minimum distance between two change points
seed = 500 # For reproducibility of the results

x, true_cps, means, sigmas = simulate_time_series(
    n=T,
    n_cps=K,
    min_seg_len=min_seg_len,
    change_type="mean",
    seed=seed,
)
```
select window sizes required for the ensemble model. This can be done using the `choose_window_sizes` function. With 

```python
from scan import choose_window_sizes

window_sizes = choose_window_sizes(
    n=20_000,
    n_windows=7,
    seed=500,
)

```
Output:

```python
print(window_sizes)

[104, 114, 115, 120, 123, 126, 133]
```
Standerdize the series and then detect change-points using the `scan_cpd` function:

```python

x_std = (x - np.mean(x)) / np.std(x)

result = scan_cpd(
    x_std,
    window_sizes=window_sizes,
    n_boot=400,
    alpha=1,
    vote_threshold=0.5,
    random_state=seed,
```
The function returns a results object, change points can be accessed with the 

Output:
```python
print(result.change_points)
array([  287,   588,   889,  1185,  1474,  1763,  2057,  2360,  2652,
        2949,  3230,  3539,  3844,  4132,  4431,  4731,  5012,  5314,
        5607,  5895,  6180,  6450,  6663,  7050,  7342,  7643,  7926,
        8211,  8505,  8804,  9085,  9377,  9662,  9976, 10269, 10579,
       10861, 11151, 11456, 11740, 12018, 12315, 12622, 12904, 13198,
       13503, 13801, 14101, 14390, 14684, 14976, 15272, 15572, 15886,
       16175, 16463, 16746, 17040, 17344, 17628, 17915, 18217, 18520,
       18817, 19112, 19424, 19712])
```


```python
plot_change_points(x, result)
```


## Documentation

More detailed documentation is available in the `docs/` folder. The README provides a short overview and a minimal example, while the documentation files give more complete guidance on installation, usage, outputs, and development.

- [Home](README.mdmd): overview of the package and where to start.
- [Documentation](docs/documentation.md): a short end-to-end example showing how to simulate data, run `scan_cpd`, and inspect the detected change points.
- [Results](docs/scanresult.md): explanation of the `ScanResult` object returned by `scan_cpd`, including change points, scores, votes, thresholds, diagnostics, parameters, and metadata.
- [Examples](docs/examples.md): additional examples for different types of change-point detection workflows.
- [Development](docs/development.md): notes for contributors, including local builds, tests, formatting checks, and package release checks.

A runnable tutorial notebook is also available at [example-usage.ipynb](example-usage.ipynb).

## Citation

Include the citation to the paper here
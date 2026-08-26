# scan-cpd: Sequentially Detecting Change-points via Adaptive Nonparametric Inference

<p align="center">
  <a href="https://pypi.org/project/scan-cpd/"><img src="https://img.shields.io/pypi/v/scan-cpd?label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/scan-cpd/"><img src="https://img.shields.io/pypi/pyversions/scan-cpd?label=Python" alt="Python versions"></a>
  <a href="https://pepy.tech/project/scan-cpd"><img src="https://static.pepy.tech/badge/scan-cpd" alt="Downloads"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg" alt="GPL-3.0 License"></a>
</p>

<p align="center">
  <a href="https://github.com/Prabashoka/scan-py/actions/workflows/tests.yml"><img src="https://github.com/Prabashoka/scan-py/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/Prabashoka/scan-py/actions/workflows/tests.yml"><img src="https://img.shields.io/badge/Linux-tested-success?logo=linux" alt="Linux"></a>
  <a href="https://github.com/Prabashoka/scan-py/actions/workflows/tests.yml"><img src="https://img.shields.io/badge/Windows-tested-success?logo=windows" alt="Windows"></a>
  <a href="https://github.com/Prabashoka/scan-py/actions/workflows/tests.yml"><img src="https://img.shields.io/badge/macOS-tested-success?logo=apple" alt="macOS"></a>
</p>

`scan-cpd` provides tools for detecting change points general distributional shifts in long univariate time series using Integral Probability Metrics (IPMs). It is aimed at research workflows where users need to simulate time series, detect changes across multiple window sizes, localize change-point positions, evaluate accuracy, and visualize diagnostics. The Python interface is backed by a Rust/PyO3 computation core.

## Installation

SCAN Py can be installed from PyPI using pip:

```bash
pip install scan-cpd
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
| `"var"` | Detects changes mainly in the variability or scale of the series. |
| `"distribution"` | Detects broader distributional changes, not restricted to only mean or variance shifts (includes both mean and variance together). |

### Example usage

The following example simulates a univariate time series with multiple mean changes and applies `scan_cpd` using several window sizes.


```python
T = 200_000 # Length of the time series
K = 100 # Number of change-points
min_seg_len = 1000 # Minimum distance between two change-points
seed = 2000 # Seed for reproducibility

x, true_cps, _, _ = simulate_time_series(
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
    series_length=200_000,
    n_windows=7,
    seed=500,
)

```
Output:

```python
print(window_sizes)

[134, 225, 241, 294, 323, 325, 394]
```
#### Detecting change-points
Standerdize the series and then detect change-points using the `scan_cpd` function:

```python

x_std = (x - np.mean(x)) / np.std(x)

result = scan_cpd(
    x_std,
    window_sizes=window_sizes,
    n_boot=400,
    alpha=5, # significance level
    vote_threshold=0.5,
    random_state=1000, # for reproducibility of the tapred block bootstrap
    n_jobs=-1,  # use all available CPU threads
)
```
The function returns a results object, change points can be accessed with the 

Output:
```python
print(result.change_points)
```

#### Visualizing

**Visualizing the detected change-points**
The detected change points can also be visualized using `plot_change_points`. The function returns a `plotnine` plot object showing the time series as a dark blue line and the detected change points as vertical dashed orange lines.

```python
plot_change_points(x, result)
```
![Detected change points](https://github.com/Prabashoka/scan-py/blob/main/plots/change_points.png)
To save the plot as an image:
```python
plot = plot_change_points(x, result)
plot.save("plots/change_points.png", width=12, height=4.5, dpi=300)
```

**Determine vote threshold using the scree plot**

The voting scree plot helps inspect how many candidate change points are retained as the ensemble voting threshold changes. It is useful for choosing a sensible `vote_threshold` before finalizing the detected change points.

```python
from scan import plot_vote_scree
plot_vote_scree(result)
```
The x-axis shows the voting threshold, denoted by ν, and the y-axis shows the number of retained change points at each threshold. A lower threshold keeps more candidate change points, while a higher threshold keeps only candidates supported by more window sizes.
In practice, choose a value near the point where the curve begins to flatten. This avoids keeping many weak detections while preserving stable change points that are supported across multiple window sizes.

![Majority voting scree plot](https://github.com/Prabashoka/scan-py/blob/main/plots/vote_scree.png)

### SWAL Statistic

The `swal_statistic` function is a single change-point detection tool for univariate time series. It is useful after a suspicious local region has been identified and you want to estimate the most likely split point inside that region. Additionally, this can be used as a custom cost function with other change-point detction methods such as bianr segmentation, PELT.

It can be used for changes in:

See the following usage examples for changes in mean and general distributional shifts.

**Change in mean**

```python
import numpy as np
from scan import swal_statistic

rng = np.random.default_rng(123)

# Simulate a time seires with a single change in mean
x_region = np.r_[
    rng.normal(0.0, 1.0, 80),
    rng.normal(2.0, 1.0, 80),
]

local_cp = swal_statistic(x_region)

print(local_cp)
```

```python
time_series_plot = plot_time_series(
    x_region,
    y_label="Value",
    title="Local time series",
)

swal_curve_plot = plot_swal_curve(x_region)
```

<p align="center">
  <img src="https://github.com/Prabashoka/scan-py/blob/main/plots/local_time_series.png" width="48%">
  <img src="https://github.com/Prabashoka/scan-py/blob/main/plots/swal_curve.png" width="48%">
</p>

**Change in distribution**

```python
import numpy as np
from scan import swal_statistic, plot_time_series, plot_swal_curve

rng = np.random.default_rng(123)

# Distributional change:
# first segment is standard normal,
# second segment is centered exponential.
x_region = np.r_[
    rng.normal(0.0, 1.0, 100),
    rng.exponential(scale=1.0, size=100) - 1.0,
]

local_cp = swal_statistic(
    x_region,
)

print(local_cp)
```
<p align="center">
  <img src="https://github.com/Prabashoka/scan-py/blob/main/plots/distribution_change_time_series.png" width="48%">
  <img src="https://github.com/Prabashoka/scan-py/blob/main/plots/distribution_change_swal_curve.png" width="48%">
</p>


## Documentation

More detailed documentation is available here: [scan-cpd documentation](https://prabashoka.github.io/scan-py-documentation/)

## Citation

Include the citation to the paper here

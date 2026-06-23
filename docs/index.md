# SCAN Py Documentation

SCAN Py is a research-oriented change-point detection package for univariate time series. It combines a Python API for simulation, diagnostics, plotting, and evaluation with a Rust/PyO3 core for computation.

Install the distribution as `scan-py` and import it as `scan`.

```python
from scan import scan_cpd
```

Change points use Python split indexing: a detected value `t` denotes the split between `x[:t]` and `x[t:]`.

## Contents

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [API Reference](api.md)
- [ScanResult and WindowResult](scanresult.md)
- [Examples](examples.md)
- [Development](development.md)
- [Theory Notes](theory.md)

## Main Workflow

```python
import numpy as np
from scan import scan_cpd, simulate_time_series

x, true_cps, means, sigmas = simulate_time_series(
    n=2000,
    n_cps=5,
    min_seg_len=200,
    change_type="mean",
    seed=123,
)
x = (x - np.mean(x)) / np.std(x)

result = scan_cpd(x, window_sizes=[40, 60, 80], random_state=123)
print(result.change_points)
```
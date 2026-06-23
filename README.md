# SCAN Py

SCAN Py is a Python package for **S**equential **C**hange-point **A**nalysis via **N**onparametric window screening. It exposes a research-friendly Python API backed by a Rust/PyO3 computation core.

The package is designed around one normal entry point:

```python
from scan import scan_cpd

result = scan_cpd(x)
print(result.change_points)
```

Change points use Python split indexing: a returned value `t` denotes the split between `x[:t]` and `x[t:]`.

## Installation

```bash
pip install scan-py
```

For local development:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip maturin
maturin develop --release
```

## Quick Start

```python
import numpy as np
from scan import scan_cpd, plot_change_points

rng = np.random.default_rng(123)
y = np.r_[
    rng.normal(0.0, 1.0, 150),
    rng.normal(1.8, 1.0, 150),
    rng.normal(-0.7, 1.0, 150),
]

result = scan_cpd(
    y,
    window_sizes=[20, 30, 40],
    alpha=0.05,
    n_boot=100,
    vote_threshold=0.5,
    random_state=123,
    n_jobs=4,
)

print(result.change_points)
print(result.scores)
print(result.window_results[30])

plot = plot_change_points(y, result)
print(plot)
```

## Reproducible Mean-Change Simulation

```python
import numpy as np

from scan import (
    covering_metric,
    f1_score_cpd,
    plot_change_points,
    plot_swal_curve,
    plot_thresholds,
    plot_vote_scree,
    plot_window_votes,
    precision_recall_cpd,
    scan_cpd,
    simulate_time_series,
)

# Simulating T=20000, K=67, spacing_hint=298, min_seg_len=235
T = 20_000
K = 67
spacing_hint = 298
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

print(f"Simulating T={T}, K={K}, spacing_hint={spacing_hint}, min_seg_len={min_seg_len}")
print("Window sizes:", window_sizes)

result = scan_cpd(
    x_std,
    window_sizes=window_sizes,
    n_boot=400,
    alpha=1,
    vote_threshold=0.7,
    random_state=seed,
    n_jobs=8,
    change_type="mean",
    batch_size=32,
)

print("True K:", len(true_cps))
print("Detected K:", len(result.change_points))

# Expected summary for this seed/configuration:
# Simulating T=20000, K=67, spacing_hint=298, min_seg_len=235
# Window sizes: [104, 114, 115, 120, 123, 126, 133]
# True K: 67
# Detected K: 67
```

## Inspecting the Result Object

```python
print("change_points:", result.change_points)
print("scores:", result.scores)
print("votes:", result.votes)
print("window_results keys:", sorted(result.window_results.keys()))
print("thresholds keys:", sorted(result.thresholds.keys()))
print("parameters:", result.parameters)
print("metadata:", result.metadata)
print("segments:", result.segments)
print("cp_dict:", result.cp_dict)

first_window = window_sizes[0]
window_result = result.window_results[first_window]

print("window_size:", window_result.window_size)
print("window change_points:", window_result.change_points)
print("scan starts:", window_result.starts[:10])
print("observed statistics:", window_result.statistics[:10])
print("lower thresholds:", window_result.lower_thresholds[:10])
print("upper thresholds:", window_result.upper_thresholds[:10])
print("localized regions:", window_result.localized_regions[:10])

precision, recall = precision_recall_cpd(true_cps, result.change_points, tolerance=25)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1_score_cpd(true_cps, result.change_points, tolerance=25))
print("covering:", covering_metric(true_cps, result.change_points, n=len(x_std)))
```

## Plotting

```python
from pathlib import Path

out_dir = Path("scan_plots")
out_dir.mkdir(exist_ok=True)

plot_change_points(x_std, result).save(
    out_dir / "change_points.png",
    width=11,
    height=4.8,
    dpi=150,
    verbose=False,
)

plot_vote_scree(result).save(
    out_dir / "vote_scree.png",
    width=8,
    height=4.8,
    dpi=150,
    verbose=False,
)

plot_window_votes(result).save(
    out_dir / "window_votes.png",
    width=10,
    height=4.8,
    dpi=150,
    verbose=False,
)

plot_thresholds(result).save(
    out_dir / "thresholds.png",
    width=12,
    height=7,
    dpi=150,
    verbose=False,
)

for window_result in result.window_results.values():
    if window_result.localized_regions:
        start, end = window_result.localized_regions[0]
        plot_swal_curve(x_std, start, end).save(
            out_dir / "swal_curve.png",
            width=8,
            height=4.8,
            dpi=150,
            verbose=False,
        )
        break
```

## Main API

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
)
```

`scan_cpd()` returns a `ScanResult`:

```python
result.change_points
result.scores
result.votes
result.window_results
result.thresholds
result.parameters
result.metadata
result.segments
result.cp_dict
```

## Lower-Level Research API

```python
from scan import (
    scan_single_window,
    localize_cp,
    wasserstein_statistic,
    ipm_statistic,
    tapered_block_bootstrap,
    adaptive_threshold,
    merge_change_points,
    ensemble_vote,
    simulate_time_series,
    safe_min_seg_len,
    choose_window_sizes,
    run_one_benchmark,
)
```

These functions are useful for threshold diagnostics, localization experiments, simulation studies, and reproducing paper figures.

## Evaluation

```python
from scan import f1_score_cpd, covering_metric, match_change_points, precision_recall_cpd

f1 = f1_score_cpd(true_cps, result.change_points, tolerance=10)
covering = covering_metric(true_cps, result.change_points, n=len(y))
```

## Compatibility Wrappers

The previous wrappers remain available and return `(change_points, elapsed_seconds)`:

```python
from scan import scan_cpd_mean, scan_cpd_var, scan_cpd_meanvar

cps, elapsed = scan_cpd_mean(y, window_sizes=[20, 30, 40])
```

New projects should prefer `scan_cpd()` because it returns diagnostics, thresholds, votes, and metadata.
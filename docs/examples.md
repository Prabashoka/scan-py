# Examples

## Detect Mean Changes

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

result = scan_cpd(
    x,
    window_sizes=[40, 60, 80],
    n_boot=100,
    alpha=0.05,
    vote_threshold=0.5,
    random_state=123,
    change_type="mean",
)

print(result.change_points)
```

## Reproducible Larger Simulation

```python
import numpy as np
from scan import scan_cpd, simulate_time_series

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
x = (x - np.mean(x)) / np.std(x)

result = scan_cpd(
    x,
    window_sizes=window_sizes,
    n_boot=400,
    alpha=1,
    vote_threshold=0.7,
    random_state=seed,
    n_jobs=8,
    change_type="mean",
    batch_size=32,
)

print(f"Simulating T={T}, K={K}, spacing_hint={spacing_hint}, min_seg_len={min_seg_len}")
print("Window sizes:", window_sizes)
print("True K:", len(true_cps))
print("Detected K:", len(result.change_points))
```

Expected summary for this seed/configuration:

```text
Simulating T=20000, K=67, spacing_hint=298, min_seg_len=235
Window sizes: [104, 114, 115, 120, 123, 126, 133]
True K: 67
Detected K: 67
```

## Evaluate Detection Accuracy

```python
from scan import covering_metric, f1_score_cpd, precision_recall_cpd

precision, recall = precision_recall_cpd(true_cps, result.change_points, tolerance=25)
f1 = f1_score_cpd(true_cps, result.change_points, tolerance=25)
covering = covering_metric(true_cps, result.change_points, n=len(x))

print(precision, recall, f1, covering)
```

## Create Plots

```python
from pathlib import Path
from scan import plot_change_points, plot_thresholds, plot_vote_scree, plot_window_votes

out_dir = Path("scan_plots")
out_dir.mkdir(exist_ok=True)

plot_change_points(x, result).save(out_dir / "change_points.png", width=11, height=4.8, dpi=150)
plot_vote_scree(result).save(out_dir / "vote_scree.png", width=8, height=4.8, dpi=150)
plot_window_votes(result).save(out_dir / "window_votes.png", width=10, height=4.8, dpi=150)
plot_thresholds(result).save(out_dir / "thresholds.png", width=12, height=7, dpi=150)
```

## Notebook

See `example-usage.ipynb` in the repository root for runnable examples covering the full API.
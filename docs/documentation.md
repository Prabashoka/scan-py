# Quickstart

```python
import numpy as np
from scan import scan_cpd, plot_change_points, simulate_time_series

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
    alpha=0.05,
    n_boot=100,
    vote_threshold=0.5,
    random_state=123,
    change_type="mean",
)

print("Detected change-points:", result.change_points)
print("Scores:", result.scores)
print("Votes:", result.votes)

plot_change_points(x, result)
```

`scan_cpd` returns a `ScanResult` object. Most users need only `result.change_points`, but the object also stores scores, votes, thresholds, per-window diagnostics, parameters, and metadata.
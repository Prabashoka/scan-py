"""Generate diagnostic SCAN plots for a small synthetic example."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scan import (
    plot_change_points,
    plot_swal_curve,
    plot_thresholds,
    plot_vote_scree,
    plot_window_votes,
    scan_cpd,
)


OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(123)
x = np.r_[
    rng.normal(0.0, 1.0, 200),
    rng.normal(3.0, 1.4, 200),
    rng.normal(-2.0, 0.7, 200),
]

result = scan_cpd(
    x,
    window_sizes=[40, 60, 80],
    n_boot=100,
    alpha=0.05,
    vote_threshold=0.5,
    random_state=123,
)

print("Detected change-points:", result.change_points)
print("Scores:", result.scores)

plots = {
    "change_points.png": plot_change_points(x, result),
    "vote_scree.png": plot_vote_scree(result),
    "window_votes.png": plot_window_votes(result),
    "thresholds.png": plot_thresholds(result),
}

# The first 400 observations contain the single change point at index 200.
plots["swal_curve.png"] = plot_swal_curve(x[:400])

for filename, plot in plots.items():
    path = OUT_DIR / filename
    plot.save(path, width=9, height=4.8, dpi=150, verbose=False)
    print("Saved", path)

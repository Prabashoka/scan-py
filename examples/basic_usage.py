import numpy as np

from scan import plot_change_points, scan_cpd

rng = np.random.default_rng(123)
y = np.r_[
    rng.normal(0.0, 1.0, 250),
    rng.normal(1.5, 1.0, 250),
    rng.normal(-0.8, 1.0, 250),
]

result = scan_cpd(
    y,
    window_sizes=[25, 35, 50],
    alpha=0.05,
    n_boot=100,
    vote_threshold=0.5,
    n_jobs=None,
    random_state=123,
)

print("Detected CPs:", result.change_points)
print("Scores:", result.scores)
print("Elapsed seconds:", result.metadata["elapsed_seconds"])

plot = plot_change_points(y, result)
print(plot)

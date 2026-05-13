import numpy as np

from scan import scan_cpd_mean, plot_detected_changepoints

rng = np.random.default_rng(123)
y = np.r_[
    rng.normal(0.0, 1.0, 250),
    rng.normal(1.5, 1.0, 250),
    rng.normal(-0.8, 1.0, 250),
]

window_sizes = [25, 35, 50]
cpts, elapsed = scan_cpd_mean(
    y,
    window_sizes=window_sizes,
    n_perm=100,
    alpha_q=1.0,
    threshold=0.5,
    workers=4,
    seed=123,
)

print("Detected CPs:", cpts)
print("Elapsed seconds:", elapsed)
plot_detected_changepoints(y, cpts, title="SCAN detected change points")

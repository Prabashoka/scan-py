import numpy as np

def _extract_cps(result):
    """Allow detector to return either a list or a result object."""
    if hasattr(result, "change_points"):
        return list(result.change_points)
    return list(result)


def _matched_all(true_cps, pred_cps, tolerance=20):
    """Check whether every true change-point has a detected point nearby."""
    return all(any(abs(p - t) <= tolerance for p in pred_cps) for t in true_cps)


def simulate_mean_change(n=1000, cps=(200, 400), seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    x[cps[0] : cps[1]] += 3.0
    x[cps[1] :] -= 2.5
    return x, list(cps)


def simulate_variance_change(n=1800, cps=(600, 1200), seed=2):
    """Simulate long low-, high-, then low-variance regimes."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[: cps[0]] = rng.normal(0, 1.0, cps[0])
    x[cps[0] : cps[1]] = rng.normal(0, 2.0, cps[1] - cps[0])
    x[cps[1] :] = rng.normal(0, 0.1, n - cps[1])
    return x, list(cps)


def simulate_mean_variance_change(n=1000, cps=(200, 400), seed=3):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[: cps[0]] = rng.normal(0, 1.0, cps[0])
    x[cps[0] : cps[1]] = rng.normal(3.0, 2.0, cps[1] - cps[0])
    x[cps[1] :] = rng.normal(-2.0, 0.7, n - cps[1])
    return x, list(cps)



def simulate_distribution_change(n=1000, cps=(300, 800), seed=100):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    cp1, cp2 = cps
    x[:cp1] = rng.normal(0, 1, cp1)
    x[cp1:cp2] = rng.exponential(1.0, cp2 - cp1)
    x[cp2:] = rng.normal(3, 1, n - cp2)
    return x, list(cps)


def simulate_ar1_mean_change(n=1000, cps=(300, 550), rho=0.6, seed=5):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, n)
    x = np.zeros(n)

    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]

    x[cps[0] : cps[1]] += 3.0
    x[cps[1] :] -= 2.0
    return x, list(cps)


def simulate_no_change(n=600, seed=6):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n), []


def test_detects_mean_changes(scan_cpd):
    x, true_cps = simulate_mean_change()

    result = scan_cpd(
        x,
        window_sizes=[20, 25, 30],
        n_boot=400,
        alpha=0.05,
        vote_threshold=0.5,
        random_state=123,
    )

    pred_cps = _extract_cps(result)
    assert _matched_all(true_cps, pred_cps, tolerance=25)


def test_detects_variance_changes(scan_cpd):
    x, true_cps = simulate_variance_change()

    result = scan_cpd(
        x,
        window_sizes=[30, 40, 50, 60, 75],
        n_boot=500,
        alpha=0.05,
        vote_threshold=0.5,
        random_state=123,
        change_type="var",
    )

    pred_cps = _extract_cps(result)
    assert _matched_all(true_cps, pred_cps, tolerance=25)


def test_detects_mean_variance_changes(scan_cpd):
    x, true_cps = simulate_mean_variance_change()

    result = scan_cpd(
        x,
        window_sizes=[20, 25, 30],
        n_boot=100,
        alpha=0.05,
        vote_threshold=0.5,
        random_state=123,
    )

    pred_cps = _extract_cps(result)
    assert _matched_all(true_cps, pred_cps, tolerance=25)


def test_detects_distribution_change(scan_cpd):
    x, true_cps = simulate_distribution_change()

    result = scan_cpd(
        x,
        window_sizes=[30, 40, 50],
        n_boot=400,
        alpha=0.05,
        vote_threshold=0.5,
        random_state=123,
    )

    pred_cps = _extract_cps(result)
    assert _matched_all(true_cps, pred_cps, tolerance=10)


def test_detects_dependent_ar1_changes(scan_cpd):
    x, true_cps = simulate_ar1_mean_change()

    result = scan_cpd(
        x,
        window_sizes=[20, 25, 30, 40, 50],
        n_boot=100,
        alpha=0.05,
        vote_threshold=0.8,
        random_state=123,
    )

    pred_cps = _extract_cps(result)
    assert _matched_all(true_cps, pred_cps, tolerance=10)


def test_no_change_has_few_false_positives(scan_cpd):
    x, _ = simulate_no_change()

    result = scan_cpd(
        x,
        window_sizes=[40, 60, 80],
        n_boot=400,
        alpha=0.01,
        vote_threshold=0.5,
        random_state=123,
    )

    pred_cps = _extract_cps(result)
    assert len(pred_cps) <= 1


def test_returns_sorted_unique_change_points(scan_cpd):
    x, _ = simulate_mean_change()

    result = scan_cpd(
        x,
        window_sizes=[40, 60, 80],
        n_boot=400,
        alpha=0.05,
        vote_threshold=0.5,
        random_state=123,
    )

    pred_cps = _extract_cps(result)

    assert pred_cps == sorted(pred_cps)
    assert len(pred_cps) == len(set(pred_cps))

"""Evaluation metrics for estimated change-point sets."""

from typing import Iterable, List, Tuple


def _clean_cps(cps: Iterable[int]) -> List[int]:
    """Normalize change-point collections to sorted unique integers."""
    return sorted({int(cp) for cp in cps})


def match_change_points(
    true_cps: Iterable[int],
    estimated_cps: Iterable[int],
    tolerance: int = 10,
) -> List[Tuple[int, int]]:
    """Match true and estimated change points within a tolerance.

    The matching is greedy by increasing absolute distance, so each true and
    estimated change point can appear in at most one pair.

    Parameters
    ----------
    true_cps:
        Ground-truth split locations.
    estimated_cps:
        Estimated split locations.
    tolerance:
        Maximum absolute distance allowed for a match.

    Returns
    -------
    list[tuple[int, int]]
        Matched ``(true_cp, estimated_cp)`` pairs.
    """
    true = _clean_cps(true_cps)
    estimated = _clean_cps(estimated_cps)
    tolerance = int(tolerance)

    candidates = []
    for t in true:
        for e in estimated:
            distance = abs(t - e)
            if distance <= tolerance:
                candidates.append((distance, t, e))

    matches: List[Tuple[int, int]] = []
    used_true = set()
    used_est = set()
    for _, t, e in sorted(candidates):
        if t not in used_true and e not in used_est:
            matches.append((t, e))
            used_true.add(t)
            used_est.add(e)
    return matches


def precision_recall_cpd(
    true_cps: Iterable[int],
    estimated_cps: Iterable[int],
    tolerance: int = 10,
) -> Tuple[float, float]:
    """Return tolerant precision and recall.

    Empty sets are handled in the usual change-point evaluation convention:
    precision is 1 when there are no estimates and no true points, and recall
    is 1 when there are no true points and no estimates.
    """
    true = _clean_cps(true_cps)
    estimated = _clean_cps(estimated_cps)
    matches = match_change_points(true, estimated, tolerance=tolerance)

    precision = len(matches) / len(estimated) if estimated else float(not true)
    recall = len(matches) / len(true) if true else float(not estimated)
    return precision, recall


def f1_score_cpd(
    true_cps: Iterable[int],
    estimated_cps: Iterable[int],
    tolerance: int = 10,
) -> float:
    """Return tolerant F1 score for change-point estimates.

    Parameters
    ----------
    true_cps, estimated_cps:
        Ground-truth and estimated split locations.
    tolerance:
        Maximum absolute distance allowed when matching points.

    Returns
    -------
    float
        Harmonic mean of tolerant precision and recall.
    """
    precision, recall = precision_recall_cpd(true_cps, estimated_cps, tolerance=tolerance)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _segments(cps: List[int], n: int) -> List[Tuple[int, int]]:
    """Convert change points into half-open segments over ``[0, n)``."""
    valid = [cp for cp in cps if 0 < cp < n]
    bounds = [0, *valid, n]
    return list(zip(bounds[:-1], bounds[1:]))


def covering_metric(true_cps: Iterable[int], estimated_cps: Iterable[int], n: int) -> float:
    """Compute a weighted segment-covering score in ``[0, 1]``.

    Each true segment is compared with all estimated segments using
    intersection-over-union, then weighted by the true segment length.

    Parameters
    ----------
    true_cps:
        Ground-truth split locations.
    estimated_cps:
        Estimated split locations.
    n:
        Total series length.

    Returns
    -------
    float
        Weighted covering score, where 1 is perfect segment agreement.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    true_segments = _segments(_clean_cps(true_cps), n)
    estimated_segments = _segments(_clean_cps(estimated_cps), n)

    total = 0.0
    for true_start, true_end in true_segments:
        true_len = true_end - true_start
        best = 0.0
        for est_start, est_end in estimated_segments:
            overlap = max(0, min(true_end, est_end) - max(true_start, est_start))
            union = max(true_end, est_end) - min(true_start, est_start)
            if union > 0:
                best = max(best, overlap / union)
        total += true_len * best
    return total / n

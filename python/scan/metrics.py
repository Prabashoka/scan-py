from typing import Iterable, List, Tuple


def _clean_cps(cps: Iterable[int]) -> List[int]:
    return sorted({int(cp) for cp in cps})


def match_change_points(
    true_cps: Iterable[int],
    estimated_cps: Iterable[int],
    tolerance: int = 10,
) -> List[Tuple[int, int]]:
    """Match true and estimated change-points within a tolerance."""
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
    """Return tolerant precision and recall."""
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
    """Return tolerant F1 score for change-point estimates."""
    precision, recall = precision_recall_cpd(true_cps, estimated_cps, tolerance=tolerance)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _segments(cps: List[int], n: int) -> List[Tuple[int, int]]:
    valid = [cp for cp in cps if 0 < cp < n]
    bounds = [0, *valid, n]
    return list(zip(bounds[:-1], bounds[1:]))


def covering_metric(true_cps: Iterable[int], estimated_cps: Iterable[int], n: int) -> float:
    """Compute a weighted segment-covering score in [0, 1]."""
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

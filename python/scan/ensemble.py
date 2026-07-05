"""Ensemble voting helpers for combining window-specific candidates."""

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def merge_change_points(change_points: Iterable[int], tolerance: int = 10) -> List[List[int]]:
    """Cluster nearby candidate change points.

    Parameters
    ----------
    change_points:
        Candidate split locations to group.
    tolerance:
        Maximum gap between consecutive candidates in the same cluster.

    Returns
    -------
    list[list[int]]
        Sorted clusters of nearby candidate locations.
    """
    cps = sorted({int(cp) for cp in change_points})
    if not cps:
        return []

    tolerance = int(tolerance)
    clusters: List[List[int]] = [[cps[0]]]
    for cp in cps[1:]:
        if cp - clusters[-1][-1] <= tolerance:
            clusters[-1].append(cp)
        else:
            clusters.append([cp])
    return clusters


def ensemble_vote(
    window_results: Mapping[int, Sequence[int]],
    vote_threshold: float = 0.5,
    tolerance: int = 10,
) -> Tuple[List[int], Dict[int, float], Dict[int, int]]:
    """Apply ensemble voting across window-specific candidate lists.

    Parameters
    ----------
    window_results:
        Mapping from window size to candidate split locations.
    vote_threshold:
        Minimum fraction of windows required to retain a cluster leader.
    tolerance:
        Maximum gap used to cluster nearby candidates before voting.

    Returns
    -------
    tuple[list[int], dict[int, float], dict[int, int]]
        Selected change points, normalized vote scores, and raw vote counts.
    """
    if not window_results:
        return [], {}, {}

    counts: Counter[int] = Counter()
    for cps in window_results.values():
        counts.update(set(map(int, cps)))

    leaders: Dict[int, int] = {}
    for cluster in merge_change_points(counts.keys(), tolerance=tolerance):
        leader = max(cluster, key=lambda cp: (counts[cp], -cp))
        leaders[leader] = sum(counts[cp] for cp in cluster)

    n_windows = len(window_results)
    scores = {cp: min(1.0, vote / n_windows) for cp, vote in leaders.items()}
    selected = sorted(cp for cp, score in scores.items() if score >= vote_threshold)
    return selected, scores, leaders

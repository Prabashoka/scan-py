"""Dataclasses and conversion helpers for SCAN detector output.

The Rust backend returns nested dictionaries. This module converts those raw
records into stable Python dataclasses that are easier to inspect, document,
and pass to plotting or evaluation helpers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class WindowResult:
    """Detailed diagnostics for one scan window size.

    Attributes
    ----------
    window_size:
        Half-window size used by the local scan.
    change_points:
        Candidate split locations found for this window size.
    starts:
        Start indices of local windows evaluated by the backend.
    statistics:
        Local discrepancy statistics aligned with ``starts``.
    tapered_block_bootstrap_threshold:
        Bootstrap thresholds aligned with ``starts``.
    localized_regions:
        Inclusive/exclusive local regions used to refine candidate change
        points.
    """

    window_size: int
    change_points: List[int]
    starts: List[int]
    statistics: List[float]
    tapered_block_bootstrap_threshold: List[float]
    localized_regions: List[Tuple[int, int]]

    @classmethod
    def from_raw(cls, window_size: int, raw: Mapping[str, Any]) -> "WindowResult":
        """Build a :class:`WindowResult` from one raw backend dictionary."""
        return cls(
            window_size=int(window_size),
            change_points=[int(v) for v in raw.get("change_points", [])],
            starts=[int(v) for v in raw.get("starts", [])],
            statistics=[float(v) for v in raw.get("statistics", [])],
            tapered_block_bootstrap_threshold=[
                float(v) for v in raw.get("tapered_block_bootstrap_threshold", [])
            ],
            localized_regions=[tuple(map(int, pair)) for pair in raw.get("localized_regions", [])],
        )


@dataclass(frozen=True)
class ScanResult:
    """Research-friendly SCAN output returned by :func:`scan.scan_cpd`.

    Attributes
    ----------
    change_points:
        Final selected split locations after ensemble voting.
    scores:
        Vote score by candidate change point.
    votes:
        Raw vote count by candidate change point.
    window_results:
        Per-window diagnostics keyed by window size.
    thresholds:
        Convenience threshold/statistic arrays keyed by window size.
    parameters:
        Normalized detector parameters used for the run.
    metadata:
        Runtime metadata such as elapsed seconds and worker counts.
    segments:
        Segment metadata returned by the backend, when available.
    raw:
        Raw backend dictionary. Hidden from ``repr`` to keep interactive output
        readable.
    """

    change_points: List[int]
    scores: Dict[int, float]
    votes: Dict[int, int]
    window_results: Dict[int, WindowResult]
    thresholds: Dict[int, Dict[str, List[float]]]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    segments: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def cp_dict(self) -> Dict[int, List[int]]:
        """Candidate change-points by window size."""
        return {w: result.change_points for w, result in self.window_results.items()}


def scan_result_from_raw(
    raw: Mapping[str, Any],
    *,
    vote_threshold: float,
    parameters: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> ScanResult:
    """Normalize the Rust dictionary into a stable Python dataclass.

    Parameters
    ----------
    raw:
        Dictionary returned by ``scan._scan_rust.scan_detector``.
    vote_threshold:
        Minimum score required to retain a final change point.
    parameters:
        Detector parameters to attach to the result.
    metadata:
        Runtime metadata to attach to the result.

    Returns
    -------
    ScanResult
        Parsed result with numeric keys and values converted to Python types.
    """

    out = raw.get("out", {})
    scores = {int(k): float(v) for k, v in out.get("leaders_scores", {}).items()}
    votes = {int(k): int(v) for k, v in out.get("leaders_segment_votes", {}).items()}

    change_points = sorted(cp for cp, score in scores.items() if score >= vote_threshold)

    window_results = {
        int(w): WindowResult.from_raw(int(w), info)
        for w, info in raw.get("window_results", {}).items()
    }

    thresholds = {
        w: {
            "starts": result.starts,
            "statistics": result.statistics,
            "tapered_block_bootstrap_threshold": result.tapered_block_bootstrap_threshold,
        }
        for w, result in window_results.items()
    }

    return ScanResult(
        change_points=change_points,
        scores=scores,
        votes=votes,
        window_results=window_results,
        thresholds=thresholds,
        parameters=dict(parameters),
        metadata=dict(metadata),
        segments=dict(raw.get("segments", {})),
        raw=dict(raw),
    )

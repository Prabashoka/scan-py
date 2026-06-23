"""Structured result objects for SCAN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class WindowResult:
    """Detailed diagnostics for one scan window size."""

    window_size: int
    change_points: List[int]
    starts: List[int]
    statistics: List[float]
    lower_thresholds: List[float]
    upper_thresholds: List[float]
    localized_regions: List[Tuple[int, int]]

    @classmethod
    def from_raw(cls, window_size: int, raw: Mapping[str, Any]) -> "WindowResult":
        return cls(
            window_size=int(window_size),
            change_points=[int(v) for v in raw.get("change_points", [])],
            starts=[int(v) for v in raw.get("starts", [])],
            statistics=[float(v) for v in raw.get("statistics", [])],
            lower_thresholds=[float(v) for v in raw.get("lower_thresholds", [])],
            upper_thresholds=[float(v) for v in raw.get("upper_thresholds", [])],
            localized_regions=[tuple(map(int, pair)) for pair in raw.get("localized_regions", [])],
        )


@dataclass(frozen=True)
class ScanResult:
    """Research-friendly SCAN output returned by :func:`scan.scan_cpd`."""

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
    """Normalize the Rust dictionary into a stable Python dataclass."""

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
            "lower": result.lower_thresholds,
            "upper": result.upper_thresholds,
            "statistics": result.statistics,
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

"""Plotnine visualizations for SCAN results."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    theme_minimal,
)

from .result import ScanResult
from .statistics import refine_wasserstein


def plot_change_points(x: Iterable[float], result: ScanResult):
    """Plot a time series with detected change-points."""
    y = np.asarray(x, dtype=np.float64).ravel()
    df = pd.DataFrame({"t": np.arange(y.size), "value": y})
    cps = pd.DataFrame({"cp": result.change_points})

    p = (
        ggplot(df, aes("t", "value"))
        + geom_line(color="#1f4e79", size=0.7)
        + labs(x="Time index", y="Value", title="Detected change-points")
        + theme_minimal()
    )
    if not cps.empty:
        p = p + geom_vline(cps, aes(xintercept="cp"), linetype="dashed", color="#b23a48", alpha=0.85)
    return p


def plot_swal_curve(x: Iterable[float], start: int, end: int):
    """Plot the SWAL/Wasserstein localization curve inside one region."""
    y = np.asarray(x, dtype=np.float64).ravel()
    start = int(start)
    end = int(end)
    if start < 0 or end > y.size or end - start < 3:
        raise ValueError("start/end must define a valid region with at least 3 observations")

    cp, stats = refine_wasserstein(y[start:end])
    df = pd.DataFrame({"split": start + np.arange(len(stats)), "score": stats})
    df = df[np.isfinite(df["score"])]

    return (
        ggplot(df, aes("split", "score"))
        + geom_line(color="#2f6f4e", size=0.8)
        + geom_vline(xintercept=start + cp, linetype="dashed", color="#b23a48")
        + labs(x="Candidate split", y="Scaled Wasserstein statistic", title="SWAL localization curve")
        + theme_minimal()
    )


def plot_vote_scree(result: ScanResult):
    """Plot number of retained change-points versus voting threshold."""
    thresholds = np.linspace(0, 1, 101)
    scores = np.array(list(result.scores.values()), dtype=float)
    counts = [int(np.sum(scores >= t)) for t in thresholds]
    df = pd.DataFrame({"vote_threshold": thresholds, "n_change_points": counts})

    return (
        ggplot(df, aes("vote_threshold", "n_change_points"))
        + geom_line(color="#1f4e79", size=0.8)
        + geom_point(color="#1f4e79", size=0.8)
        + labs(x="Voting threshold", y="Retained change-points", title="Vote scree")
        + theme_minimal()
    )


def plot_window_votes(result: ScanResult):
    """Plot ensemble vote counts for candidate change-points as bars."""
    rows = [
        {
            "change_point": int(cp),
            "votes": int(vote),
            "score": float(result.scores.get(cp, 0.0)),
            "selected": cp in set(result.change_points),
        }
        for cp, vote in result.votes.items()
    ]
    df = pd.DataFrame(rows, columns=["change_point", "votes", "score", "selected"])

    n_windows = max(1, len(result.window_results))
    vote_threshold = float(result.parameters.get("vote_threshold", 0.5))
    threshold_votes = vote_threshold * n_windows

    p = (
        ggplot(df, aes("factor(change_point)", "votes", fill="selected"))
        + geom_col(width=0.72, alpha=0.9)
        + geom_hline(yintercept=threshold_votes, linetype="dashed", color="#b23a48")
        + labs(
            x="Candidate change-point",
            y="Window votes",
            fill="Selected",
            title="Window vote counts",
        )
        + theme_minimal()
    )
    return p


def plot_thresholds(result: ScanResult, window_size: Optional[int] = None):
    """Plot observed scan statistics and adaptive thresholds."""
    rows = []
    for w, threshold_info in result.thresholds.items():
        if window_size is not None and int(window_size) != int(w):
            continue
        starts = threshold_info["starts"]
        stats = threshold_info["statistics"]
        upper = threshold_info["upper"]
        for start, stat, bound in zip(starts, stats, upper):
            rows.append({"window_size": str(w), "start": start, "value": stat, "series": "Observed"})
            rows.append({"window_size": str(w), "start": start, "value": bound, "series": "Upper threshold"})

    df = pd.DataFrame(rows, columns=["window_size", "start", "value", "series"])
    p = (
        ggplot(df, aes("start", "value", color="series"))
        + geom_line(size=0.75)
        + labs(x="Scan start", y="Statistic", color="Series", title="Adaptive scan thresholds")
        + theme_minimal()
    )
    if window_size is None:
        p = p + facet_wrap("~window_size", scales="free_y")
    return p

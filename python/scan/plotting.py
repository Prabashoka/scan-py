from typing import Iterable, Optional

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_col,
    geom_hline,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_x_continuous,
    theme,
    theme_minimal,
)

from .result import ScanResult
from .statistics import refine_wasserstein


SCAN_BLUE = "#00008B"
SCAN_ORANGE = "#f5710a"
SCAN_GRID = "#D9D9D9"


def _comma_labels(values):
    return [f"{value:,.0f}" for value in values]


def _plot_labs(x_label: str, y_label: str, title: Optional[str] = None, **kwargs):
    """Create plot labels, adding a title only when one is provided."""
    labels = {
        "x": x_label,
        "y": y_label,
        **kwargs,
    }

    if title:
        labels["title"] = title

    return labs(**labels)


def _scan_plot_theme(legend_position="none"):
    """Shared publication style for SCAN plots."""
    return theme_minimal(base_family="serif", base_size=11) + theme(
        figure_size=(6.8, 4.2),
        panel_background=element_rect(fill="white", color=None),
        plot_background=element_rect(fill="white", color=None),
        panel_grid_major=element_line(color=SCAN_GRID, alpha=0.25, size=0.7),
        panel_grid_minor=element_blank(),
        axis_line=element_line(color="black", size=0.9),
        axis_text_x=element_text(size=10.5),
        axis_text_y=element_text(size=10.5),
        axis_title_x=element_text(size=12),
        axis_title_y=element_text(size=12),
        plot_title=element_text(size=12),
        legend_position=legend_position,
        legend_title=element_blank(),
        legend_background=element_blank(),
        legend_box_background=element_blank(),
    )

def plot_time_series(
    x: Iterable[float],
    change_points: Optional[Iterable[int]] = None,
    true_change_points: Optional[Iterable[int]] = None,
    index: Optional[Iterable[float]] = None,
    x_label: str = "Time",
    y_label: str = "Value",
    title: str = "Time series",
):
    """Plot a univariate time series with optional change-point markers."""
    y = np.asarray(x, dtype=np.float64).ravel()

    if index is None and hasattr(x, "index"):
        t = np.asarray(x.index)
    elif index is None:
        t = np.arange(y.size)
    else:
        t = np.asarray(list(index))

    if len(t) != y.size:
        raise ValueError("index must have the same length as x")

    def _map_cps_to_axis(cps):
        if cps is None:
            return []

        mapped = []
        for cp in cps:
            cp_int = int(cp)
            if 0 <= cp_int < len(t):
                mapped.append(t[cp_int])
            else:
                mapped.append(cp)
        return mapped

    df = pd.DataFrame(
        {
            "t": t,
            "value": y,
        }
    )

    detected_df = pd.DataFrame(
        {
            "cp": _map_cps_to_axis(change_points),
            "kind": "Detected change points",
        }
    )

    true_cp_df = pd.DataFrame(columns=["cp", "kind"])
    if true_change_points is not None:
        true_cp_df = pd.DataFrame(
            {
                "cp": _map_cps_to_axis(true_change_points),
                "kind": "True change points",
            }
        )

    p = (
        ggplot(df, aes("t", "value"))
        + geom_line(color="#00008B", size=0.9, show_legend=False)
        + labs(
            x=x_label,
            y=y_label,
            color=None,
            linetype=None,
            title=title,
        )
        + _scan_plot_theme(legend_position="bottom")
    )

    if not detected_df.empty:
        p = p + geom_vline(
            detected_df,
            aes(xintercept="cp", color="kind", linetype="kind"),
            size=0.7,
            alpha=0.9,
        )

    if not true_cp_df.empty:
        p = p + geom_vline(
            true_cp_df,
            aes(xintercept="cp", color="kind", linetype="kind"),
            size=0.7,
            alpha=0.85,
        )

    p = (
        p
        + scale_color_manual(
            values={
                "Detected change points": "#f5710a",
                "True change points": "black",
            }
        )
        + scale_linetype_manual(
            values={
                "Detected change points": "dashed",
                "True change points": "dotted",
            }
        )
    )

    if np.issubdtype(np.asarray(t).dtype, np.number):
        p = p + scale_x_continuous(labels=_comma_labels)

    return p


def plot_change_points(
    x: Iterable[float],
    result: ScanResult,
    true_change_points: Optional[Iterable[int]] = None,
    index: Optional[Iterable[float]] = None,
    x_label: str = "Time",
    y_label: str = "Series",
    title: Optional[str] = None,
):
    """Plot a time series with detected and optional true change points.

    Detected change points are taken from ``result.change_points``.
    True change points should be provided separately using
    ``true_change_points``.
    """
    y = np.asarray(x, dtype=np.float64).ravel()

    if index is None and hasattr(x, "index"):
        t = np.asarray(x.index)
    elif index is None:
        t = np.arange(y.size)
    else:
        t = np.asarray(list(index))

    if len(t) != y.size:
        raise ValueError("index must have the same length as x")

    def _map_cps_to_axis(cps):
        if cps is None:
            return []

        mapped = []
        for cp in cps:
            cp_int = int(cp)
            if 0 <= cp_int < len(t):
                mapped.append(t[cp_int])
            else:
                mapped.append(cp)
        return mapped

    detected_cps = _map_cps_to_axis(result.change_points)
    true_cps = _map_cps_to_axis(true_change_points)

    df = pd.DataFrame(
        {
            "t": t,
            "value": y,
        }
    )

    detected_df = pd.DataFrame(
        {
            "cp": detected_cps,
            "kind": "Detected change points",
        }
    )

    true_cp_df = pd.DataFrame(columns=["cp", "kind"])
    if true_change_points is not None:
        true_cp_df = pd.DataFrame(
            {
                "cp": true_cps,
                "kind": "True change points",
            }
        )

    legend_kinds = []
    if not detected_df.empty:
        legend_kinds.append("Detected change points")
    if not true_cp_df.empty:
        legend_kinds.append("True change points")

    color_values = {
        "Detected change points": SCAN_ORANGE,
        "True change points": "black",
    }

    linetype_values = {
        "Detected change points": "dashed",
        "True change points": "dotted",
    }

    legend_df = pd.DataFrame(columns=["t", "value", "kind"])
    if legend_kinds:
        x0 = t[0]
        x1 = t[1] if len(t) > 1 else t[0]
        y0 = y[0]

        legend_df = pd.DataFrame(
            [{"t": x0, "value": y0, "kind": kind} for kind in legend_kinds]
            + [{"t": x1, "value": y0, "kind": kind} for kind in legend_kinds]
        )

    p = (
        ggplot(df, aes("t", "value"))
        + geom_line(color=SCAN_BLUE, size=1.0, show_legend=False)
    )

    if not detected_df.empty:
        p = p + geom_vline(
            detected_df,
            aes(xintercept="cp", color="kind", linetype="kind"),
            size=0.75,
            alpha=0.9,
            show_legend=False,
        )

    if not true_cp_df.empty:
        p = p + geom_vline(
            true_cp_df,
            aes(xintercept="cp", color="kind", linetype="kind"),
            size=0.75,
            alpha=0.85,
            show_legend=False,
        )

    if not legend_df.empty:
        p = p + geom_line(
            legend_df,
            aes("t", "value", color="kind", linetype="kind", group="kind"),
            size=1.0,
            alpha=0.8,
            show_legend=True,
        )

    p = (
        p
        + scale_color_manual(
            values=color_values,
            breaks=legend_kinds,
        )
        + scale_linetype_manual(
            values=linetype_values,
            breaks=legend_kinds,
        )
        + _plot_labs(
            x_label=x_label,
            y_label=y_label,
            title=title,
            color=None,
            linetype=None,
        )
        + _scan_plot_theme(legend_position="bottom")
        + theme(figure_size=(10.5, 4.2))
    )

    if np.issubdtype(np.asarray(t).dtype, np.number):
        p = p + scale_x_continuous(labels=_comma_labels)

    return p

def plot_swal_curve(
    x: Iterable[float],
    start: int,
    end: int,
    x_label: str = "Time series",
    y_label: str = "Scaled Wasserstein statistic",
    title: Optional[str] = None,
):
    """Plot the SWAL/Wasserstein localization curve inside one region."""
    y = np.asarray(x, dtype=np.float64).ravel()
    start = int(start)
    end = int(end)

    if start < 0 or end > y.size or end - start < 3:
        raise ValueError("start/end must define a valid region with at least 3 observations")

    cp, stats = refine_wasserstein(y[start:end])

    df = pd.DataFrame(
        {
            "split": start + np.arange(len(stats)),
            "score": stats,
        }
    )
    df = df[np.isfinite(df["score"])]

    return (
        ggplot(df, aes("split", "score"))
        + geom_line(color=SCAN_BLUE, size=1.0)
        + geom_vline(
            xintercept=start + cp,
            linetype="dashed",
            color=SCAN_ORANGE,
            size=1.2,
        )
        + scale_x_continuous(labels=_comma_labels)
        + _plot_labs(x_label=x_label, y_label=y_label, title=title)
        + _scan_plot_theme(legend_position="none")
    )


def plot_vote_scree(
    result: ScanResult,
    x_label: str = r"Voting threshold $(\nu)$",
    y_label: str = "Number of retained change points",
    title: Optional[str] = None,
):
    """Plot number of retained change points versus voting threshold."""
    thresholds = np.linspace(0, 1, 101)
    scores = np.array(list(result.scores.values()), dtype=float)
    counts = [int(np.sum(scores >= t)) for t in thresholds]

    df = pd.DataFrame(
        {
            "vote_threshold": thresholds,
            "n_change_points": counts,
        }
    )

    selected_threshold = float(result.parameters.get("vote_threshold", 0.5))

    return (
        ggplot(df, aes("vote_threshold", "n_change_points"))
        + geom_line(color=SCAN_BLUE, size=1.0)
        + geom_point(color=SCAN_BLUE, size=2.4)
        + geom_vline(
            xintercept=selected_threshold,
            linetype="dashed",
            color=SCAN_ORANGE,
            size=1.2,
        )
        + scale_x_continuous(
            breaks=np.arange(0.0, 1.01, 0.10),
            limits=(-0.02, 1.02),
        )
        + _plot_labs(x_label=x_label, y_label=y_label, title=title)
        + _scan_plot_theme(legend_position="none")
    )


def plot_window_votes(
    result: ScanResult,
    max_x_labels: int = 12,
    x_label_angle: int = 45,
    x_label: str = "Candidate change point",
    y_label: str = "Window votes",
    title: Optional[str] = None,
):
    """Plot ensemble vote counts for candidate change points."""
    n_windows = max(1, len(result.window_results))
    vote_threshold = float(result.parameters.get("vote_threshold", 0.5))
    threshold_votes = vote_threshold * n_windows

    rows = [
        {
            "change_point": int(cp),
            "votes": int(vote),
            "score": float(result.scores.get(cp, 0.0)),
            "threshold_group": (
                "At or above threshold"
                if float(vote) >= threshold_votes
                else "Below threshold"
            ),
        }
        for cp, vote in result.votes.items()
    ]

    df = pd.DataFrame(
        rows,
        columns=["change_point", "votes", "score", "threshold_group"],
    )
    df = df.sort_values("change_point")

    if df.empty:
        bar_width = 0.72
        x_breaks = []
    else:
        cps = df["change_point"].to_numpy(dtype=float)
        gaps = np.diff(np.unique(cps))
        positive_gaps = gaps[gaps > 0]

        if len(positive_gaps):
            bar_width = float(max(1.0, 0.65 * np.median(positive_gaps)))
        else:
            bar_width = 1.0

        n_labels = min(max(2, int(max_x_labels)), len(cps))
        x_breaks = np.linspace(float(cps.min()), float(cps.max()), n_labels)
        x_breaks = np.unique(np.round(x_breaks).astype(int)).tolist()

    return (
        ggplot(df, aes("change_point", "votes", fill="threshold_group"))
        + geom_col(width=bar_width, alpha=0.9)
        + geom_hline(
            yintercept=threshold_votes,
            linetype="dashed",
            color="black",
            size=0.9,
        )
        + scale_fill_manual(
            values={
                "Below threshold": SCAN_BLUE,
                "At or above threshold": SCAN_ORANGE,
            }
        )
        + scale_x_continuous(breaks=x_breaks)
        + _plot_labs(
            x_label=x_label,
            y_label=y_label,
            title=title,
            fill=None,
        )
        + _scan_plot_theme(legend_position="bottom")
        + theme(axis_text_x=element_text(rotation=x_label_angle, ha="right"))
    )


def plot_thresholds():
    pass
"""SCAN: research-ready change-point detection with a Rust core."""

from __future__ import annotations

from .bootstrap import adaptive_threshold, tapered_block_bootstrap
from .detector import scan_cpd, scan_single_window
from .ensemble import ensemble_vote, merge_change_points
from .metrics import covering_metric, f1_score_cpd, match_change_points, precision_recall_cpd
from .plotting import (
    plot_change_points,
    plot_swal_curve,
    plot_thresholds,
    plot_time_series,
    plot_vote_scree,
    plot_window_votes,
)
from .result import ScanResult, WindowResult
from .simulator import (
    UnivariateSeriesSimulator,
    choose_window_sizes,
    run_one_benchmark,
    safe_min_seg_len,
    simulate_time_series,
)
from .statistics import (
    ipm_statistic,
    refine_cusum,
    refine_wasserstein,
    swal_statistic,
    wasserstein_statistic,
)

__version__ = "0.1.0"

__all__ = [
    "ScanResult",
    "UnivariateSeriesSimulator",
    "WindowResult",
    "adaptive_threshold",
    "choose_window_sizes",
    "covering_metric",
    "ensemble_vote",
    "f1_score_cpd",
    "ipm_statistic",
    "match_change_points",
    "merge_change_points",
    "plot_change_points",
    "plot_swal_curve",
    "plot_thresholds",
    "plot_time_series",
    "plot_vote_scree",
    "plot_window_votes",
    "precision_recall_cpd",
    "refine_cusum",
    "refine_wasserstein",
    "run_one_benchmark",
    "safe_min_seg_len",
    "scan_cpd",
    "scan_single_window",
    "simulate_time_series",
    "swal_statistic",
    "tapered_block_bootstrap",
    "wasserstein_statistic",
]
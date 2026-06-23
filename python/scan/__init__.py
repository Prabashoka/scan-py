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
    plot_vote_scree,
    plot_window_votes,
)
from .result import ScanResult, WindowResult
from .simulator import (
    choose_window_sizes,
    run_one_benchmark,
    safe_min_seg_len,
    simulate_time_series,
)
from .statistics import (
    ipm_statistic,
    localize_cp,
    refine_cusum,
    refine_wasserstein,
    wasserstein_statistic,
)

__version__ = "0.1.0"

__all__ = [
    "ScanResult",
    "WindowResult",
    "adaptive_threshold",
    "choose_window_sizes",
    "covering_metric",
    "ensemble_vote",
    "f1_score_cpd",
    "ipm_statistic",
    "localize_cp",
    "match_change_points",
    "merge_change_points",
    "plot_change_points",
    "plot_swal_curve",
    "plot_thresholds",
    "plot_vote_scree",
    "plot_window_votes",
    "precision_recall_cpd",
    "refine_cusum",
    "refine_wasserstein",
    "run_one_benchmark",
    "safe_min_seg_len",
    "scan_cpd",
    "scan_cpd_mean",
    "scan_cpd_meanvar",
    "scan_cpd_var",
    "scan_single_window",
    "simulate_time_series",
    "tapered_block_bootstrap",
    "wasserstein_statistic",
]
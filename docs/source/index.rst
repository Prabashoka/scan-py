scan-py documentation
=====================

API reference for the main SCAN change-point detection workflow, grouped by
how the package is commonly used.

SCAN
====

.. autofunction:: scan.detector.scan_cpd

.. autofunction:: scan.detector.scan_single_window

Plotting
=========

.. autofunction:: scan.plotting.plot_time_series

.. autofunction:: scan.plotting.plot_change_points

.. autofunction:: scan.plotting.plot_swal_curve

.. autofunction:: scan.plotting.plot_vote_scree

.. autofunction:: scan.plotting.plot_window_votes

.. autofunction:: scan.plotting.plot_thresholds

Evaluation Metrics
==================

.. autofunction:: scan.metrics.covering_metric

.. autofunction:: scan.metrics.precision_recall_cpd

.. autofunction:: scan.metrics.f1_score_cpd

.. autofunction:: scan.metrics.match_change_points

Univariate Simulator
====================

.. autoclass:: scan.simulator.UnivariateSeriesSimulator
   :members: apply_random_shifts, arfima_sim, determine_change_points, fracint_weights, select_change_point_locations, simulate_ar_series, simulate_ar_unif, simulate_arma

.. autofunction:: scan.simulator.simulate_time_series


Utils
======

.. autofunction:: scan.simulator.choose_window_sizes

.. autofunction:: scan.ensemble.ensemble_vote

.. autofunction:: scan.ensemble.merge_change_points

.. autofunction:: scan.bootstrap.adaptive_threshold

.. autofunction:: scan.bootstrap.tapered_block_bootstrap

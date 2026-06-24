use crate::aggregate::{cdf_from_segment_votes, compute_change_points_with_votes};
use crate::bootstrap::compute_tapered_block_bootstrap_threshold;
use crate::refine::refine_for_change_type;
use crate::stats::PrefixStats;
use crate::types::{ChangeType, ScanResult, WindowScanResult};
use crate::validation::{validate_series, validate_window_sizes};
use crate::wasserstein::wasserstein_1d;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Scan one chosen window size over the series.
///
/// The function compares adjacent windows, computes a local tapered block
/// bootstrap threshold for each comparison, then refines each flagged region to
/// a single candidate change-point. The detailed vectors are intentionally kept
/// for the Python research API.
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_for_window(
    series: &[f64],
    prefix: &PrefixStats,
    w: usize,
    n_boot: usize,
    alpha_q_percent: f64,
    seed: u64,
    change_type: ChangeType,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<(usize, WindowScanResult)> {
    let n = series.len();
    let delta_w = w;

    let corrected_q = alpha_q_percent;

    let mut change_points = Vec::new();
    let mut starts = Vec::new();
    let mut statistics = Vec::new();
    let mut tapered_block_bootstrap_threshold_values = Vec::new();
    let mut localized_regions = Vec::new();

    let mut start = 0usize;
    while start + w + delta_w <= n {
        let block_end = start + w + delta_w;
        let block = &series[start..block_end];

        let tapered_block_bootstrap_threshold = compute_tapered_block_bootstrap_threshold(
            series,
            prefix,
            start,
            w,
            delta_w,
            n_boot,
            seed,
            corrected_q,
            b,
            taper_ratio,
            center,
            eps,
            batch_size,
        )?;

        let observed = wasserstein_1d(&block[..w], &block[w..w + delta_w]);

        starts.push(start);
        statistics.push(observed);
        tapered_block_bootstrap_threshold_values.push(tapered_block_bootstrap_threshold);

        if observed > tapered_block_bootstrap_threshold {
            let k_loc = refine_for_change_type(block, change_type)?;
            let cp = (start + k_loc).clamp(start + 1, block_end - 1);
            change_points.push(cp);
            localized_regions.push((start, block_end));

            // Skip past the flagged region to avoid returning repeated local
            // detections for the same structural change.
            start += w + delta_w;
        } else {
            start += delta_w;
        }
    }

    Ok((
        w,
        WindowScanResult {
            change_points,
            starts,
            statistics,
            tapered_block_bootstrap_threshold: tapered_block_bootstrap_threshold_values,
            localized_regions,
        },
    ))
}

/// Main Rust engine called by all Python-facing wrappers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_scan_detector(
    series: Vec<f64>,
    window_sizes: Option<Vec<usize>>,
    n_boot: usize,
    alpha_q: f64,
    seed: u64,
    tol: usize,
    workers: Option<usize>,
    backend: &str,
    change_type: &str,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<ScanResult> {
    validate_series(&series)?;

    let window_sizes = window_sizes.unwrap_or_else(|| (10usize..=20usize).collect());
    validate_window_sizes(&window_sizes)?;

    let backend_lower = backend.to_ascii_lowercase();
    if backend_lower != "thread" && backend_lower != "process" {
        return Err(PyValueError::new_err(
            "backend must be 'thread' or 'process'. Rust uses Rayon threads internally for both options.",
        ));
    }

    let ct = ChangeType::parse(change_type)?;

    // Accept either 0.01-style or 1.0-style percentage inputs.
    let alpha_percent = if alpha_q <= 1.0 {
        100.0 * alpha_q
    } else {
        alpha_q
    };
    let alpha_percent_corrected = alpha_percent / window_sizes.len().max(1) as f64;
    let batch_size = batch_size.max(1);

    let prefix = PrefixStats::from_series(&series);

    let compute = || -> Vec<PyResult<(usize, WindowScanResult)>> {
        window_sizes
            .par_iter()
            .map(|&w| {
                detect_for_window(
                    &series,
                    &prefix,
                    w,
                    n_boot,
                    alpha_percent_corrected,
                    seed,
                    ct,
                    eps,
                    b,
                    taper_ratio,
                    center,
                    batch_size,
                )
            })
            .collect()
    };

    let results = if let Some(n_threads) = workers.filter(|&n| n > 0) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| PyValueError::new_err(format!("failed to build Rayon thread pool: {e}")))?
            .install(compute)
    } else {
        compute()
    };

    let mut cp_dict = BTreeMap::new();
    let mut window_results = BTreeMap::new();

    for item in results {
        let (w, result) = item?;
        cp_dict.insert(w, result.change_points.clone());
        window_results.insert(w, result);
    }

    let segments = compute_change_points_with_votes(&cp_dict, tol);
    let out = cdf_from_segment_votes(&segments, cp_dict.len())?;

    Ok(ScanResult {
        cp_dict,
        window_results,
        segments,
        out,
    })
}

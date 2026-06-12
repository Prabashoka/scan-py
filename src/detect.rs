use crate::aggregate::{cdf_from_segment_votes, compute_change_points_with_votes};
use crate::bootstrap::compute_bounds_tbb;
use crate::refine::refine_for_change_type;
use crate::stats::PrefixStats;
use crate::types::{ChangeType, ScanResult};
use crate::validation::{validate_series, validate_window_sizes};
use crate::wasserstein::wasserstein_1d;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Scan one chosen window size over the series.
///
/// The function compares adjacent windows, uses a TBB threshold, then refines
/// the detected region to a single candidate change-point.
///
/// It returns only:
/// - the window size `w`
/// - the predicted change-points for that window size
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_for_window(
    series: &[f64],
    prefix: &PrefixStats,
    w: usize,
    n_perm: usize,
    alpha_q_percent: f64,
    seed: u64,
    change_type: ChangeType,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<(usize, Vec<usize>)> {
    let n = series.len();
    let delta_w = w;

    // Bonferroni-like correction over the number of window comparisons.
    let approx_n_tests = if n >= 2 * w {
        usize::max(1, (n - 2 * w) / w + 1)
    } else {
        1
    };
    let corrected_q = alpha_q_percent / approx_n_tests as f64;

    let mut cps = Vec::new();

    let bounds_at = |start_idx: usize| -> PyResult<(f64, f64)> {
        if start_idx + w + delta_w > n {
            Ok((f64::NEG_INFINITY, f64::INFINITY))
        } else {
            compute_bounds_tbb(
                series,
                prefix,
                start_idx,
                w,
                delta_w,
                n_perm,
                seed,
                corrected_q,
                b,
                taper_ratio,
                center,
                eps,
                batch_size,
            )
        }
    };

    let mut start = 0usize;
    let (mut lower, mut upper) = bounds_at(start)?;

    while start + w + delta_w <= n {
        let block = &series[start..start + w + delta_w];

        // Compare the adjacent left and right windows.
        //
        // We use the single general Wasserstein function here. It works for
        // both equal and unequal sample sizes, so we do not need separate
        // equal-length helper functions.
        let x = wasserstein_1d(&block[..w], &block[w..w + delta_w]);

        if x > upper || x < lower {
            let k_loc = refine_for_change_type(block, change_type)?;
            let mut cp = start + k_loc;
            cp = cp.clamp(start + 1, start + w + delta_w - 1);
            cps.push(cp);

            // Skip past the detected region to avoid repeatedly flagging the same change.
            start += w + delta_w;
            let bounds = bounds_at(start)?;
            lower = bounds.0;
            upper = bounds.1;
        } else {
            start += delta_w;
        }
    }

    Ok((w, cps))
}

/// Main Rust engine called by all Python-facing wrappers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_scan_detector(
    series: Vec<f64>,
    window_sizes: Option<Vec<usize>>,
    n_perm: usize,
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

    // Build prefix stats once and share them across all window sizes.
    let prefix = PrefixStats::from_series(&series);

    let compute = || -> Vec<PyResult<(usize, Vec<usize>)>> {
        window_sizes
            .par_iter()
            .map(|&w| {
                detect_for_window(
                    &series,
                    &prefix,
                    w,
                    n_perm,
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

    for item in results {
        let (w, cps) = item?;
        cp_dict.insert(w, cps);
    }

    let segments = compute_change_points_with_votes(&cp_dict, tol);
    let out = cdf_from_segment_votes(&segments, cp_dict.len())?;

    Ok(ScanResult {
        cp_dict,
        segments,
        out,
    })
}

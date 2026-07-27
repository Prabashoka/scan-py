use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use scan_core::{
    detect_for_window, refine_cp_cusum, refine_cp_wasserstein, refine_for_change_type,
    run_scan_detector, validate_series, wasserstein_1d, ChangeType, PrefixStats, ScanResult,
    WindowScanResult,
};

fn core_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn window_result_to_py<'py>(
    py: Python<'py>,
    result: WindowScanResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("change_points", result.change_points)?;
    out.set_item("starts", result.starts)?;
    out.set_item("statistics", result.statistics)?;
    out.set_item(
        "tapered_block_bootstrap_threshold",
        result.tapered_block_bootstrap_threshold,
    )?;
    Ok(out)
}

/// Convert the internal Rust result into a Python dictionary.
fn scan_result_to_py<'py>(py: Python<'py>, result: ScanResult) -> PyResult<Bound<'py, PyDict>> {
    let root = PyDict::new(py);

    let cp_dict = PyDict::new(py);
    for (w, cps) in result.cp_dict {
        cp_dict.set_item(w, cps)?;
    }
    root.set_item("cp_dict", cp_dict)?;

    let window_results = PyDict::new(py);
    for (w, info) in result.window_results {
        window_results.set_item(w, window_result_to_py(py, info)?)?;
    }
    root.set_item("window_results", window_results)?;

    let out = PyDict::new(py);

    let leaders_segment_votes = PyDict::new(py);
    for (cp, vote) in result.out.leaders_segment_votes {
        leaders_segment_votes.set_item(cp, vote)?;
    }
    out.set_item("leaders_segment_votes", leaders_segment_votes)?;

    let leaders_scores = PyDict::new(py);
    for (cp, score) in result.out.leaders_scores {
        leaders_scores.set_item(cp, score)?;
    }
    out.set_item("leaders_scores", leaders_scores)?;

    let leaders_probs = PyDict::new(py);
    for (cp, prob) in result.out.leaders_probs {
        leaders_probs.set_item(cp, prob)?;
    }
    out.set_item("leaders_probs", leaders_probs)?;
    out.set_item("cdf", result.out.cdf)?;

    root.set_item("out", out)?;

    Ok(root)
}

#[pyfunction]
#[pyo3(signature = (series, window_sizes=None, n_perm=300, alpha_q=1.0, seed=123, tol=2, workers=None, backend="thread", change_type="mean", b=None, taper_ratio=0.5, center=true, batch_size=32))]
#[allow(clippy::too_many_arguments)]
fn scan_detector<'py>(
    py: Python<'py>,
    series: Vec<f64>,
    window_sizes: Option<Vec<usize>>,
    n_perm: usize,
    alpha_q: f64,
    seed: u64,
    tol: usize,
    workers: Option<usize>,
    backend: &str,
    change_type: &str,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let result = run_scan_detector(
        series,
        window_sizes,
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        change_type,
        1e-12,
        b,
        taper_ratio,
        center,
        batch_size,
    )
    .map_err(core_error)?;

    scan_result_to_py(py, result)
}

#[pyfunction]
#[pyo3(signature = (series, window_size, n_perm=300, alpha_q=1.0, seed=123, change_type="mean", b=None, taper_ratio=0.5, center=true, batch_size=32))]
#[allow(clippy::too_many_arguments)]
fn scan_single_window<'py>(
    py: Python<'py>,
    series: Vec<f64>,
    window_size: usize,
    n_perm: usize,
    alpha_q: f64,
    seed: u64,
    change_type: &str,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    validate_series(&series).map_err(core_error)?;
    if window_size == 0 {
        return Err(PyValueError::new_err("window_size must be positive"));
    }

    let alpha_percent = if alpha_q <= 1.0 {
        100.0 * alpha_q
    } else {
        alpha_q
    };
    let prefix = PrefixStats::from_series(&series);
    let (_, result) = detect_for_window(
        &series,
        &prefix,
        window_size,
        n_perm,
        alpha_percent,
        seed,
        ChangeType::parse(change_type).map_err(core_error)?,
        1e-12,
        b,
        taper_ratio,
        center,
        batch_size.max(1),
    )
    .map_err(core_error)?;

    window_result_to_py(py, result)
}

#[pyfunction]
fn refine_cusum(series: Vec<f64>) -> PyResult<usize> {
    validate_series(&series).map_err(core_error)?;
    refine_cp_cusum(&series).map_err(core_error)
}

#[pyfunction]
fn refine_wasserstein(series: Vec<f64>) -> PyResult<(usize, Vec<f64>)> {
    validate_series(&series).map_err(core_error)?;
    refine_cp_wasserstein(&series).map_err(core_error)
}

#[pyfunction]
#[pyo3(signature = (series, change_type="distribution"))]
fn swal_statistic(series: Vec<f64>, change_type: &str) -> PyResult<usize> {
    validate_series(&series).map_err(core_error)?;
    refine_for_change_type(&series, ChangeType::parse(change_type).map_err(core_error)?)
        .map_err(core_error)
}

#[pyfunction]
fn wasserstein_statistic(left: Vec<f64>, right: Vec<f64>) -> PyResult<f64> {
    if left.is_empty() || right.is_empty() {
        return Err(PyValueError::new_err(
            "left and right samples must be non-empty",
        ));
    }
    if left.iter().chain(right.iter()).any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err(
            "samples contain NaN or infinite values",
        ));
    }
    Ok(wasserstein_1d(&left, &right))
}

#[pyfunction]
fn ipm_statistic(left: Vec<f64>, right: Vec<f64>) -> PyResult<f64> {
    wasserstein_statistic(left, right)
}

/// Register all Python-callable functions with the module declared in `lib.rs`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_detector, m)?)?;
    m.add_function(wrap_pyfunction!(scan_single_window, m)?)?;
    m.add_function(wrap_pyfunction!(refine_cusum, m)?)?;
    m.add_function(wrap_pyfunction!(refine_wasserstein, m)?)?;
    m.add_function(wrap_pyfunction!(swal_statistic, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein_statistic, m)?)?;
    m.add_function(wrap_pyfunction!(ipm_statistic, m)?)?;
    Ok(())
}

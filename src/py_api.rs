use crate::detect::run_scan_detector;
use crate::refine::{refine_cp_cusum, refine_cp_wasserstein};
use crate::types::ScanResult;
use crate::validation::validate_series;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Convert the internal Rust result into a Python dictionary.
///
/// Timing information is intentionally omitted. This detailed function keeps
/// the per-window and voting information, while the `scan_cpd_*` wrappers below
/// return only the final predicted change-point list.
fn scan_result_to_py<'py>(py: Python<'py>, result: ScanResult) -> PyResult<Bound<'py, PyDict>> {
    let root = PyDict::new(py);

    let cp_dict = PyDict::new(py);
    for (w, cps) in result.cp_dict {
        cp_dict.set_item(w, cps)?;
    }
    root.set_item("cp_dict", cp_dict)?;

    let segments = PyDict::new(py);
    for (name, info) in result.segments {
        let info_dict = PyDict::new(py);
        info_dict.set_item("change_points", info.change_points)?;

        let votes = PyDict::new(py);
        for (cp, vote) in info.votes {
            votes.set_item(cp, vote)?;
        }
        info_dict.set_item("votes", votes)?;
        info_dict.set_item("segment_vote", info.segment_vote)?;
        segments.set_item(name, info_dict)?;
    }
    root.set_item("segments", segments)?;

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
#[pyo3(signature = (series, window_sizes=None, n_perm=300, alpha_q=1.0, seed=123, tol=2, workers=None, backend="thread", change_type="mean", eps=1e-12, b=None, taper_ratio=0.5, center=true, batch_size=32))]
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
    eps: f64,
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
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )?;

    scan_result_to_py(py, result)
}

#[allow(clippy::too_many_arguments)]
fn scan_cpd_base(
    series: Vec<f64>,
    window_sizes: Vec<usize>,
    n_perm: usize,
    alpha_q: f64,
    seed: u64,
    tol: Option<usize>,
    workers: Option<usize>,
    backend: &str,
    threshold: f64,
    change_type: &str,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<Vec<usize>> {
    let tol = tol.unwrap_or_else(|| *window_sizes.iter().min().unwrap_or(&2));

    let result = run_scan_detector(
        series,
        Some(window_sizes),
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        change_type,
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )?;

    let mut cpts: Vec<usize> = result
        .out
        .leaders_scores
        .iter()
        .filter_map(|(&cp, &score)| {
            if score >= threshold {
                Some(cp + 1)
            } else {
                None
            }
        })
        .collect();

    cpts.sort_unstable();
    Ok(cpts)
}

#[pyfunction]
#[pyo3(signature = (series, window_sizes, n_perm=300, alpha_q=1.0, seed=123, tol=None, workers=None, backend="thread", threshold=0.5, eps=1e-12, b=None, taper_ratio=0.5, center=true, batch_size=32))]
#[allow(clippy::too_many_arguments)]
fn scan_cpd_mean(
    series: Vec<f64>,
    window_sizes: Vec<usize>,
    n_perm: usize,
    alpha_q: f64,
    seed: u64,
    tol: Option<usize>,
    workers: Option<usize>,
    backend: &str,
    threshold: f64,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<Vec<usize>> {
    scan_cpd_base(
        series,
        window_sizes,
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        threshold,
        "mean",
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )
}

#[pyfunction]
#[pyo3(signature = (series, window_sizes, n_perm=400, alpha_q=10.0, seed=123, tol=None, workers=None, backend="thread", threshold=5.0, eps=1e-12, b=None, taper_ratio=0.5, center=true, batch_size=32))]
#[allow(clippy::too_many_arguments)]
fn scan_cpd_var(
    series: Vec<f64>,
    window_sizes: Vec<usize>,
    n_perm: usize,
    alpha_q: f64,
    seed: u64,
    tol: Option<usize>,
    workers: Option<usize>,
    backend: &str,
    threshold: f64,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<Vec<usize>> {
    scan_cpd_base(
        series,
        window_sizes,
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        threshold,
        "var",
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )
}

#[pyfunction]
#[pyo3(signature = (series, window_sizes, n_perm=300, alpha_q=1.0, seed=123, tol=None, workers=None, backend="thread", threshold=0.5, eps=1e-12, b=None, taper_ratio=0.5, center=true, batch_size=32))]
#[allow(clippy::too_many_arguments)]
fn scan_cpd_meanvar(
    series: Vec<f64>,
    window_sizes: Vec<usize>,
    n_perm: usize,
    alpha_q: f64,
    seed: u64,
    tol: Option<usize>,
    workers: Option<usize>,
    backend: &str,
    threshold: f64,
    eps: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    batch_size: usize,
) -> PyResult<Vec<usize>> {
    scan_cpd_base(
        series,
        window_sizes,
        n_perm,
        alpha_q,
        seed,
        tol,
        workers,
        backend,
        threshold,
        "meanvar",
        eps,
        b,
        taper_ratio,
        center,
        batch_size,
    )
}

#[pyfunction]
fn refine_cusum(series: Vec<f64>) -> PyResult<usize> {
    validate_series(&series)?;
    refine_cp_cusum(&series)
}

#[pyfunction]
fn refine_wasserstein(series: Vec<f64>) -> PyResult<(usize, Vec<f64>)> {
    validate_series(&series)?;
    refine_cp_wasserstein(&series)
}

/// Register all Python-callable functions with the module declared in `lib.rs`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_detector, m)?)?;
    m.add_function(wrap_pyfunction!(scan_cpd_mean, m)?)?;
    m.add_function(wrap_pyfunction!(scan_cpd_var, m)?)?;
    m.add_function(wrap_pyfunction!(scan_cpd_meanvar, m)?)?;
    m.add_function(wrap_pyfunction!(refine_cusum, m)?)?;
    m.add_function(wrap_pyfunction!(refine_wasserstein, m)?)?;
    Ok(())
}

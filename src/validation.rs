use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Validate the input time series before any scanning/refinement.
pub(crate) fn validate_series(series: &[f64]) -> PyResult<()> {
    if series.len() < 3 {
        return Err(PyValueError::new_err(
            "series must contain at least 3 values",
        ));
    }
    if series.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err(
            "series contains NaN or infinite values; clean/impute before calling scan",
        ));
    }
    Ok(())
}

pub(crate) fn validate_window_sizes(window_sizes: &[usize]) -> PyResult<()> {
    if window_sizes.is_empty() {
        return Err(PyValueError::new_err("window_sizes must not be empty"));
    }
    if let Some(w) = window_sizes.iter().find(|&&w| w == 0) {
        return Err(PyValueError::new_err(format!(
            "window_sizes must be positive, got {w}"
        )));
    }
    Ok(())
}

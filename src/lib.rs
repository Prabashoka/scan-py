//! Rust backend for the Python `scan` package.
//!
//! The implementation is split into small modules:
//! - `stats`: prefix sums and simple summaries
//! - `wasserstein`: 1D Wasserstein distance
//! - `bootstrap`: tapered block bootstrap thresholding
//! - `refine`: local change-point refinement
//! - `detect`: main window-based scan engine
//! - `aggregate`: ensemble/voting aggregation
//! - `py_api`: Python-facing PyO3 functions

use pyo3::prelude::*;

mod aggregate;
mod bootstrap;
mod detect;
mod py_api;
mod refine;
mod stats;
mod types;
mod validation;
mod wasserstein;

/// Native Rust extension module for the Python `scan` package.
#[pymodule]
fn _scan_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_api::register(m)
}

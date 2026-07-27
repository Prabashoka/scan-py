//! Rust backend for the Python `scan` package.
//!
//! The language-neutral implementation lives in the `scan-core` crate.

use pyo3::prelude::*;

mod py_api;

/// Native Rust extension module for the Python `scan` package.
#[pymodule]
fn _scan_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_api::register(m)
}

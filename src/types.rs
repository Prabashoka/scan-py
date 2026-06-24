use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// Type of distributional change targeted by the refinement step.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ChangeType {
    Mean,
    Var,
    MeanVar,
}

impl ChangeType {
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "var" | "variance" => Ok(Self::Var),
            "meanvar" | "distribution" | "wasserstein" => Ok(Self::MeanVar),
            other => Err(PyValueError::new_err(format!(
                "change_type must be one of {{'mean', 'var', 'meanvar'}}, got {other:?}"
            ))),
        }
    }
}

/// One merged segment of nearby candidate change-points and their votes.
#[derive(Clone, Debug)]
pub(crate) struct SegmentInfo {
    pub(crate) change_points: Vec<usize>,
    pub(crate) votes: BTreeMap<usize, usize>,
    pub(crate) segment_vote: usize,
}

/// Detailed output for a single scan window.
#[derive(Clone, Debug)]
pub(crate) struct WindowScanResult {
    pub(crate) change_points: Vec<usize>,
    pub(crate) starts: Vec<usize>,
    pub(crate) statistics: Vec<f64>,
    pub(crate) tapered_block_bootstrap_threshold: Vec<f64>,
    pub(crate) localized_regions: Vec<(usize, usize)>,
}

/// Aggregated voting output used by the Python API.
#[derive(Clone, Debug)]
pub(crate) struct AggregatedOut {
    pub(crate) leaders_segment_votes: BTreeMap<usize, usize>,
    pub(crate) leaders_scores: BTreeMap<usize, f64>,
    pub(crate) leaders_probs: BTreeMap<usize, f64>,
    pub(crate) cdf: Vec<(usize, f64)>,
}

/// Full internal scan result before conversion to Python dictionaries.
#[derive(Clone, Debug)]
pub(crate) struct ScanResult {
    pub(crate) cp_dict: BTreeMap<usize, Vec<usize>>,
    pub(crate) window_results: BTreeMap<usize, WindowScanResult>,
    pub(crate) segments: BTreeMap<String, SegmentInfo>,
    pub(crate) out: AggregatedOut,
}

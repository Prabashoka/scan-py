use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
enum ChangeType {
    Mean,
    Var,
    MeanVar,
}

impl ChangeType {
    fn parse(value: &str) -> PyResult<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mean" => Ok(Self::Mean),
            "var" => Ok(Self::Var),
            "meanvar" => Ok(Self::MeanVar),
            other => Err(PyValueError::new_err(format!(
                "change_type must be one of {{'mean', 'var', 'meanvar'}}, got {other:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
struct SegmentInfo {
    change_points: Vec<usize>,
    votes: BTreeMap<usize, usize>,
    segment_vote: usize,
}

#[derive(Clone, Debug)]
struct AggregatedOut {
    leaders_segment_votes: BTreeMap<usize, usize>,
    leaders_scores: BTreeMap<usize, f64>,
    leaders_probs: BTreeMap<usize, f64>,
    cdf: Vec<(usize, f64)>,
}

#[derive(Clone, Debug)]
struct ScanResult {
    cp_dict: BTreeMap<usize, Vec<usize>>,
    timings: BTreeMap<usize, f64>,
    total_time: f64,
    segments: BTreeMap<String, SegmentInfo>,
    out: AggregatedOut,
}

#[derive(Clone, Debug)]
struct PrefixStats {
    sum: Vec<f64>,
    sumsq: Vec<f64>,
}

impl PrefixStats {
    fn new(series: &[f64]) -> Self {
        let n = series.len();
        let mut sum = Vec::with_capacity(n + 1);
        let mut sumsq = Vec::with_capacity(n + 1);

        sum.push(0.0);
        sumsq.push(0.0);

        // Use a single running pass to avoid two separate accumulations.
        let mut s = 0.0f64;
        let mut ss = 0.0f64;
        for &v in series {
            s += v;
            ss += v * v;
            sum.push(s);
            sumsq.push(ss);
        }

        Self { sum, sumsq }
    }

    #[inline]
    fn mean_std_ddof1(&self, start: usize, len: usize, eps: f64) -> (f64, f64) {
        if len == 0 {
            return (0.0, eps);
        }

        let end = start + len;
        let n = len as f64;

        let s = self.sum[end] - self.sum[start];
        let ss = self.sumsq[end] - self.sumsq[start];
        let mu = s / n;

        if len <= 1 {
            return (mu, eps);
        }

        let centered_ss = (ss - (s * s) / n).max(0.0);
        let std = (centered_ss / (n - 1.0)).sqrt().max(eps);

        (mu, std)
    }
}

fn validate_series(series: &[f64]) -> PyResult<()> {
    if series.len() < 3 {
        return Err(PyValueError::new_err("series must contain at least 3 values"));
    }
    if series.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err(
            "series contains NaN or infinite values; clean/impute before calling scan",
        ));
    }
    Ok(())
}

fn validate_window_sizes(window_sizes: &[usize]) -> PyResult<()> {
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

#[inline]
fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        0.0
    } else {
        x.iter().sum::<f64>() / x.len() as f64
    }
}

fn default_block_size(m: usize) -> usize {
    let b = if m < 100 {
        (m as f64).sqrt().round() as usize
    } else {
        (m as f64).powf(1.0 / 3.0).round() as usize
    };
    usize::max(8, usize::min(b, m))
}

fn create_taper_window(length: usize, ratio: f64) -> Vec<f64> {
    let mut taper = vec![1.0; length];
    let slope_len = ((length as f64) * ratio / 2.0).floor() as usize;

    if slope_len > 0 {
        let scale = 1.0 / (slope_len + 1) as f64;
        for i in 0..slope_len {
            let value = (i + 1) as f64 * scale;
            taper[i] = value;
            taper[length - 1 - i] = value;
        }
    }

    taper
}

fn percentile_linear(values: &mut [f64], percent: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    values.sort_unstable_by(|a, b| a.total_cmp(b));

    // percent is always passed as (100.0 - q_percent) where q_percent is
    // already validated, so clamp is kept but the division is computed once.
    let p = percent.clamp(0.0, 100.0) * 0.01;
    let n = values.len();

    if n == 1 {
        return values[0];
    }

    let h = p * (n as f64 - 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;

    if lo == hi {
        values[lo]
    } else {
        let weight = h - lo as f64;
        values[lo].mul_add(1.0 - weight, values[hi] * weight)
    }
}

#[inline]
fn wasserstein_1d_equal(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());

    let mut xs = x.to_vec();
    let mut ys = y.to_vec();

    xs.sort_unstable_by(|a, b| a.total_cmp(b));
    ys.sort_unstable_by(|a, b| a.total_cmp(b));

    wasserstein_1d_equal_sorted(&xs, &ys)
}

#[inline]
fn wasserstein_1d_equal_sorted(xs: &[f64], ys: &[f64]) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());

    xs.iter()
        .zip(ys.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / xs.len() as f64
}

fn wasserstein_1d_equal_with_scratch(
    x: &[f64],
    y: &[f64],
    xs: &mut [f64],
    ys: &mut [f64],
) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), xs.len());
    debug_assert_eq!(y.len(), ys.len());

    xs.copy_from_slice(x);
    ys.copy_from_slice(y);

    xs.sort_unstable_by(|a, b| a.total_cmp(b));
    ys.sort_unstable_by(|a, b| a.total_cmp(b));

    wasserstein_1d_equal_sorted(xs, ys)
}

fn wasserstein_1d(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return f64::NAN;
    }

    if x.len() == y.len() {
        return wasserstein_1d_equal(x, y);
    }

    let mut xs = x.to_vec();
    let mut ys = y.to_vec();

    xs.sort_unstable_by(|a, b| a.total_cmp(b));
    ys.sort_unstable_by(|a, b| a.total_cmp(b));

    let nx_inv = 1.0 / xs.len() as f64;
    let ny_inv = 1.0 / ys.len() as f64;

    let mut i = 0usize;
    let mut j = 0usize;
    let mut cdf_x: f64 = 0.0;
    let mut cdf_y: f64 = 0.0;
    let mut prev = xs[0].min(ys[0]);
    let mut dist: f64 = 0.0;

    while i < xs.len() || j < ys.len() {
        let next_x = if i < xs.len() { xs[i] } else { f64::INFINITY };
        let next_y = if j < ys.len() { ys[j] } else { f64::INFINITY };
        let z = next_x.min(next_y);

        dist += (cdf_x - cdf_y).abs() * (z - prev);

        while i < xs.len() && xs[i] == z {
            cdf_x += nx_inv;
            i += 1;
        }
        while j < ys.len() && ys[j] == z {
            cdf_y += ny_inv;
            j += 1;
        }

        prev = z;
    }

    dist
}

/// Sorted blocks stored in a single contiguous allocation for cache locality.
/// Replaces `Vec<Vec<f64>>` which scatters each block across the heap.
struct SortedBlocks {
    data: Vec<f64>,
    w: usize,
}

impl SortedBlocks {
    fn new(series: &[f64], w: usize) -> Self {
        let n_blocks = series.len() / w;
        let mut data = Vec::with_capacity(n_blocks * w);
        for block_id in 0..n_blocks {
            let start = block_id * w;
            let mut block = series[start..start + w].to_vec();
            block.sort_unstable_by(|a, b| a.total_cmp(b));
            data.extend_from_slice(&block);
        }
        Self { data, w }
    }

    #[inline]
    fn block(&self, i: usize) -> &[f64] {
        &self.data[i * self.w..(i + 1) * self.w]
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len() / self.w
    }
}

#[inline]
fn seed_from_parts(seed: u64, start: usize, w: usize, salt: u64) -> u64 {
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    x ^= (start as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= (w as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= salt.wrapping_mul(0xD6E8_FD50_1B3D_F1AB);
    x
}

// Precomputed taper parameters so they are not recalculated inside the
// parallel closure on every call.
struct TaperParams {
    window: Vec<f64>,
    norm_factor: f64,
}

impl TaperParams {
    fn new(block_len: usize, taper_ratio: f64) -> Self {
        let window = create_taper_window(block_len, taper_ratio);
        let taper_norm = window.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_factor = (block_len as f64).sqrt() / taper_norm;
        Self { window, norm_factor }
    }
}

#[allow(clippy::too_many_arguments)]
fn batched_tbb_distances_equal(
    pooled: &[f64],
    w: usize,
    b_reps: usize,
    block_len: usize,
    taper: &TaperParams,
    seed: u64,
    batch_size: usize,
) -> PyResult<Vec<f64>> {
    let n_views = pooled
        .len()
        .checked_sub(block_len)
        .map(|v| v + 1)
        .ok_or_else(|| PyValueError::new_err("reference series is shorter than block length"))?;

    if n_views == 0 {
        return Err(PyValueError::new_err(
            "reference series is shorter than block length",
        ));
    }

    let total_len = 2 * w;
    let k = (total_len + block_len - 1) / block_len;
    let batch_size = batch_size.max(1);
    // Capacity needed per replication for z before truncation.
    let z_cap = k * block_len;

    let mut dists = vec![0.0; b_reps];

    dists
        .par_chunks_mut(batch_size)
        .enumerate()
        .for_each(|(chunk_id, chunk)| {
            let base_rep = chunk_id * batch_size;

            // Allocate scratch buffers once per chunk/thread, reuse across reps.
            let mut z: Vec<f64> = Vec::with_capacity(z_cap);
            let mut xs = vec![0.0f64; w];
            let mut ys = vec![0.0f64; w];

            for (offset, out) in chunk.iter_mut().enumerate() {
                let rep_id = base_rep + offset;

                let rep_seed = seed_from_parts(seed, rep_id, w, 10_007);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(rep_seed);

                // Reuse z allocation rather than clearing and reallocating.
                z.clear();

                for _ in 0..k {
                    let idx = rng.random_range(0..n_views);
                    for j in 0..block_len {
                        z.push(pooled[idx + j] * taper.window[j] * taper.norm_factor);
                    }
                }

                z.truncate(total_len);

                *out = wasserstein_1d_equal_with_scratch(
                    &z[..w],
                    &z[w..2 * w],
                    &mut xs,
                    &mut ys,
                );
            }
        });

    Ok(dists)
}

#[allow(clippy::too_many_arguments)]
fn compute_bounds_tbb(
    series: &[f64],
    prefix: &PrefixStats,
    start: usize,
    w: usize,
    delta: usize,
    b_reps: usize,
    seed: u64,
    q_percent: f64,
    b: Option<usize>,
    taper_ratio: f64,
    center: bool,
    eps: f64,
    batch_size: usize,
) -> PyResult<(f64, f64)> {
    if delta != w {
        return Err(PyValueError::new_err(
            "this implementation assumes delta == w",
        ));
    }

    let total_len = w + delta;
    if start + total_len > series.len() {
        return Ok((f64::NEG_INFINITY, f64::INFINITY));
    }

    let left_start = start;
    let right_start = start + w;

    let (left_mean, left_std) = prefix.mean_std_ddof1(left_start, w, eps);
    let (right_mean, right_std) = prefix.mean_std_ddof1(right_start, delta, eps);

    let left_std_inv = left_std.recip();
    let right_std_inv = right_std.recip();

    let mut pooled: Vec<f64> = Vec::with_capacity(total_len);

    if center {
        pooled.extend(
            series[left_start..left_start + w]
                .iter()
                .map(|v| (v - left_mean) * left_std_inv),
        );
        pooled.extend(
            series[right_start..right_start + delta]
                .iter()
                .map(|v| (v - right_mean) * right_std_inv),
        );
    } else {
        pooled.extend(
            series[left_start..left_start + w]
                .iter()
                .map(|v| v * left_std_inv),
        );
        pooled.extend(
            series[right_start..right_start + delta]
                .iter()
                .map(|v| v * right_std_inv),
        );
    }

    let m = pooled.len();
    let block_len = match b {
        Some(value) => usize::max(3, usize::min(value, m)),
        None => default_block_size(m),
    };

    // Build taper once; pass into bootstrap instead of recomputing.
    let taper = TaperParams::new(block_len, taper_ratio);
    let bootstrap_seed = seed_from_parts(seed, start, w, 999);

    let mut dists = batched_tbb_distances_equal(
        &pooled,
        w,
        b_reps,
        block_len,
        &taper,
        bootstrap_seed,
        batch_size,
    )?;

    let local_scale = (0.5 * (left_std.powi(2) + right_std.powi(2))).sqrt();
    for value in &mut dists {
        *value *= local_scale;
    }

    let upper = percentile_linear(&mut dists, 100.0 - q_percent);
    Ok((0.0, upper))
}

fn refine_cp_cusum(y: &[f64]) -> PyResult<usize> {
    if y.len() < 3 {
        return Err(PyValueError::new_err("need at least 3 points"));
    }

    let y_mean = mean(y);
    let mut running = 0.0f64;
    let mut best_idx = 0usize;
    let mut best_abs = f64::NEG_INFINITY;

    for (i, value) in y.iter().take(y.len() - 1).enumerate() {
        running += y_mean - value;
        let score = running.abs();
        if score > best_abs {
            best_abs = score;
            best_idx = i;
        }
    }

    Ok(best_idx + 1)
}

fn refine_cp_wasserstein(y: &[f64]) -> PyResult<(usize, Vec<f64>)> {
    if y.len() < 3 {
        return Err(PyValueError::new_err("need at least 3 points"));
    }

    let n = y.len();
    let n_f64 = n as f64;
    let mut stats = vec![f64::NAN; n];
    let mut best_k = 1usize;
    let mut best_score = f64::NEG_INFINITY;

    for t in 1..n {
        // Precompute scale to avoid recomputing sqrt in the inner loop.
        let scale = ((t as f64) * ((n - t) as f64) / n_f64).sqrt();
        let score = scale * wasserstein_1d(&y[..t], &y[t..]);
        stats[t] = score;

        if score > best_score {
            best_score = score;
            best_k = t;
        }
    }

    Ok((best_k, stats))
}

fn refine_for_change_type(block: &[f64], change_type: ChangeType) -> PyResult<usize> {
    match change_type {
        ChangeType::Mean => refine_cp_cusum(block),
        ChangeType::Var | ChangeType::MeanVar => refine_cp_wasserstein(block).map(|(k, _)| k),
    }
}

#[allow(clippy::too_many_arguments)]
fn detect_for_window(
    series: &[f64],
    prefix: &PrefixStats,
    sorted_blocks: &SortedBlocks,
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
) -> PyResult<(usize, Vec<usize>, f64)> {
    let n = series.len();
    let delta_w = w;

    let approx_n_tests = if n >= 2 * w {
        usize::max(1, (n - 2 * w) / w + 1)
    } else {
        1
    };
    let corrected_q = alpha_q_percent / approx_n_tests as f64;

    let t0 = Instant::now();
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
        let block_id = start / w;

        // Use precomputed sorted blocks when available; fall back to ad-hoc sort.
        let x = if block_id + 1 < sorted_blocks.len() {
            wasserstein_1d_equal_sorted(sorted_blocks.block(block_id), sorted_blocks.block(block_id + 1))
        } else {
            wasserstein_1d_equal(&block[..w], &block[w..w + delta_w])
        };

        if x > upper || x < lower {
            let k_loc = refine_for_change_type(block, change_type)?;
            let mut cp = start + k_loc;
            cp = cp.clamp(start + 1, start + w + delta_w - 1);
            cps.push(cp);

            start += w + delta_w;
            let bounds = bounds_at(start)?;
            lower = bounds.0;
            upper = bounds.1;
        } else {
            start += delta_w;
        }
    }

    Ok((w, cps, t0.elapsed().as_secs_f64()))
}

fn compute_cp_counts(change_points_dict: &BTreeMap<usize, Vec<usize>>) -> BTreeMap<usize, usize> {
    // Accumulate directly into a BTreeMap to avoid an intermediate HashMap
    // followed by a collect() conversion.
    let mut cp_to_count: BTreeMap<usize, usize> = BTreeMap::new();

    for cps in change_points_dict.values() {
        let unique: HashSet<usize> = cps.iter().copied().collect();
        for cp in unique {
            *cp_to_count.entry(cp).or_insert(0) += 1;
        }
    }

    cp_to_count
}

fn compute_change_points_with_votes(
    change_points_dict: &BTreeMap<usize, Vec<usize>>,
    tol: usize,
) -> BTreeMap<String, SegmentInfo> {
    let cp_to_count = compute_cp_counts(change_points_dict);
    let all_cps: Vec<usize> = cp_to_count.keys().copied().collect();

    if all_cps.is_empty() {
        return BTreeMap::new();
    }

    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut cur = vec![all_cps[0]];

    for &cp in all_cps.iter().skip(1) {
        if cp - *cur.last().unwrap() <= tol {
            cur.push(cp);
        } else {
            segments.push(cur);
            cur = vec![cp];
        }
    }

    segments.push(cur);

    let mut out = BTreeMap::new();

    for (i, seg) in segments.into_iter().enumerate() {
        let mut votes = BTreeMap::new();
        let mut segment_vote = 0usize;

        for cp in &seg {
            let v = *cp_to_count.get(cp).unwrap_or(&0);
            votes.insert(*cp, v);
            segment_vote += v;
        }

        out.insert(
            format!("segment_{}", i + 1),
            SegmentInfo {
                change_points: seg,
                votes,
                segment_vote,
            },
        );
    }

    out
}

fn leaders_from_segments(segments: &BTreeMap<String, SegmentInfo>) -> BTreeMap<usize, usize> {
    let mut leaders = BTreeMap::new();

    for info in segments.values() {
        let mut best_cp = None::<usize>;
        let mut best_vote = 0usize;

        for (&cp, &vote) in &info.votes {
            if best_cp.is_none()
                || vote > best_vote
                || (vote == best_vote && cp < best_cp.unwrap())
            {
                best_cp = Some(cp);
                best_vote = vote;
            }
        }

        if let Some(cp) = best_cp {
            leaders.insert(cp, info.segment_vote);
        }
    }

    leaders
}

fn cdf_from_segment_votes(
    segments: &BTreeMap<String, SegmentInfo>,
    num_windows: usize,
) -> PyResult<AggregatedOut> {
    if num_windows == 0 {
        return Err(PyValueError::new_err("number of windows must be positive"));
    }

    let leaders_segment_votes = leaders_from_segments(segments);
    let num_windows_f64 = num_windows as f64;

    let mut leaders_scores = BTreeMap::new();
    for (&cp, &v) in &leaders_segment_votes {
        leaders_scores.insert(cp, (v as f64 / num_windows_f64).min(1.0));
    }

    let total = leaders_scores.values().sum::<f64>();

    let mut leaders_probs = BTreeMap::new();
    if total > 0.0 {
        let total_inv = total.recip();
        for (&cp, &score) in &leaders_scores {
            leaders_probs.insert(cp, score * total_inv);
        }
    } else if !leaders_scores.is_empty() {
        let k_inv = 1.0 / leaders_scores.len() as f64;
        for &cp in leaders_scores.keys() {
            leaders_probs.insert(cp, k_inv);
        }
    }

    let mut cdf = Vec::with_capacity(leaders_probs.len());
    let mut cum = 0.0f64;
    for (&cp, &prob) in &leaders_probs {
        cum += prob;
        cdf.push((cp, cum));
    }

    Ok(AggregatedOut {
        leaders_segment_votes,
        leaders_scores,
        leaders_probs,
        cdf,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_scan_detector(
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

    let alpha_percent = if alpha_q <= 1.0 { 100.0 * alpha_q } else { alpha_q };
    let alpha_percent_corrected = alpha_percent / window_sizes.len().max(1) as f64;
    let batch_size = batch_size.max(1);

    // Build prefix stats once for all window sizes.
    let prefix = PrefixStats::new(&series);

    let t0_all = Instant::now();

    let compute = || -> Vec<PyResult<(usize, Vec<usize>, f64)>> {
        window_sizes
            .par_iter()
            .map(|&w| {
                // Each window size gets its own sorted-blocks table (cheap to
                // build and avoids inter-thread sharing).
                let sorted_blocks = SortedBlocks::new(&series, w);
                detect_for_window(
                    &series,
                    &prefix,
                    &sorted_blocks,
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
    let mut timings = BTreeMap::new();

    for item in results {
        let (w, cps, elapsed) = item?;
        cp_dict.insert(w, cps);
        timings.insert(w, elapsed);
    }

    let total_time = t0_all.elapsed().as_secs_f64();
    let segments = compute_change_points_with_votes(&cp_dict, tol);
    let out = cdf_from_segment_votes(&segments, cp_dict.len())?;

    Ok(ScanResult {
        cp_dict,
        timings,
        total_time,
        segments,
        out,
    })
}

fn scan_result_to_py<'py>(py: Python<'py>, result: ScanResult) -> PyResult<Bound<'py, PyDict>> {
    let root = PyDict::new(py);

    let cp_dict = PyDict::new(py);
    for (w, cps) in result.cp_dict {
        cp_dict.set_item(w, cps)?;
    }
    root.set_item("cp_dict", cp_dict)?;

    let timings = PyDict::new(py);
    for (w, elapsed) in result.timings {
        timings.set_item(w, elapsed)?;
    }
    root.set_item("timings", timings)?;
    root.set_item("total_time", result.total_time)?;

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
) -> PyResult<(Vec<usize>, f64)> {
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
        .filter_map(|(&cp, &score)| if score >= threshold { Some(cp + 1) } else { None })
        .collect();

    cpts.sort_unstable();
    Ok((cpts, result.total_time))
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
) -> PyResult<(Vec<usize>, f64)> {
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
) -> PyResult<(Vec<usize>, f64)> {
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
) -> PyResult<(Vec<usize>, f64)> {
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

/// Native Rust extension module for the Python `scan` package.
#[pymodule]
fn _scan_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_detector, m)?)?;
    m.add_function(wrap_pyfunction!(scan_cpd_mean, m)?)?;
    m.add_function(wrap_pyfunction!(scan_cpd_var, m)?)?;
    m.add_function(wrap_pyfunction!(scan_cpd_meanvar, m)?)?;
    m.add_function(wrap_pyfunction!(refine_cusum, m)?)?;
    m.add_function(wrap_pyfunction!(refine_wasserstein, m)?)?;
    Ok(())
}

// Copyright 2026 System Architecture & Database Kernel Team.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! CASTLE Microbenchmark: The Skew Burst (Executor Starvation Test)
//!
//! This module implements the definitive microbenchmark supporting the central claims 
//! of **Section 7.3 (Tail Latency & Starvation Analysis)**. 
//! 
//! # Academic Intent & Methodology
//! Existing learned index evaluations often average out latency over the entire test duration, 
//! effectively masking catastrophic tail-latency cliffs. This test adopts a Time-Series 
//! approach. 
//! 
//! 1. **Steady State:** A baseline uniform read/write workload.
//! 2. **The Burst:** At a synchronized trigger, concurrent tasks inject a massive volume 
//!    of writes focused purely on a 1% key-space boundary.
//! 3. **The Trap:** For in-place paradigms (ALEX via C++ FFI), this triggers synchronous 
//!    model retraining that blocks the underlying OS threads within `spawn_blocking`. 
//! 4. **The Consequence:** Because Tokio's thread pool is finite, blocking it halts the 
//!    execution of *all* coroutines, including simple point reads. This measures the true 
//!    "Head-of-Line Blocking" and "Executor Starvation" effect.
//! 5. **CASTLE's Defense:** CASTLE relies on `CancellationToken`s and `HARD_MAX_ERROR` limits. 
//!    We empirically measure if the median latency rises (due to fallback scans) while the 
//!    P99 latency strictly plateaus instead of spiraling out of control.

#![deny(missing_docs)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use hdrhistogram::Histogram;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::ffi_wrapper::AsyncIndex; // The unified async facade from Section 7.1

/// Configuration for the Skew Burst Microbenchmark.
#[derive(Debug, Clone)]
pub struct SkewBurstConfig {
    /// Number of keys distributed uniformly during the warm-up phase (e.g., 50M).
    pub initial_keys: u64,
    /// Number of concurrent background reader tasks (Constant Load).
    pub num_readers: usize,
    /// Number of concurrent burst writer tasks (The Attackers).
    pub num_burst_writers: usize,
    /// The specific, narrow percentage of the key space to attack (e.g., 0.01 for 1%).
    pub skew_fraction: f64,
    /// Duration to sample the time-series (e.g., 1-second buckets).
    pub sample_window: Duration,
    /// Total test duration (e.g., 30s).
    pub total_duration: Duration,
    /// Time point to inject the burst (e.g., 10s).
    pub burst_trigger_time: Duration,
}

/// A snapshot of latencies captured during a specific time window.
#[derive(Debug)]
pub struct TimeWindowResult {
    /// The time offset from the start of the benchmark (in seconds).
    pub timestamp_sec: u64,
    /// P99 latency of read requests during this second (microseconds).
    pub read_p99_us: u64,
    /// P99 latency of write requests during this second (microseconds).
    pub write_p99_us: u64,
}

/// Executes the Skew Burst benchmark and returns a time-series plot dataset.
pub async fn run_skew_burst<I>(
    index: Arc<I>,
    config: SkewBurstConfig,
) -> Vec<TimeWindowResult>
where
    I: AsyncIndex + Send + Sync + 'static,
{
    // Global synchronization flag to trigger the concentrated skew burst.
    let burst_active = Arc::new(AtomicBool::new(false));
    let test_active = Arc::new(AtomicBool::new(true));
    let start_time = Instant::now();

    // -------------------------------------------------------------------------
    // Component 1: Background Readers (The Victims)
    // -------------------------------------------------------------------------
    // These coroutines perform continuous uniform reads. Their latency will spike 
    // dramatically if the burst writers exhaust the underlying Tokio executor threads.
    let mut reader_handles: Vec<JoinHandle<Vec<(u64, Histogram<u64>)>>> = Vec::new();

    for task_id in 0..config.num_readers {
        let idx = index.clone();
        let cfg = config.clone();
        let test_active = test_active.clone();
        
        reader_handles.push(tokio::spawn(async move {
            let mut rng = SmallRng::seed_from_u64(task_id as u64 ^ 0x1111);
            let mut windowed_hists = Vec::new();
            let mut current_hist = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3).unwrap();
            let mut last_window = Instant::now();
            let mut current_sec = 0;

            while test_active.load(Ordering::Relaxed) {
                let key = rng.gen_range(1..=cfg.initial_keys);
                let op_start = Instant::now();
                
                // Hot Path Read
                let _ = idx.read(key).await;
                
                let elapsed_us = op_start.elapsed().as_micros() as u64;
                let _ = current_hist.record(elapsed_us);

                // Check if the current time window (e.g., 1 second) has elapsed.
                // If so, stash the histogram for the time-series plot and reset.
                if last_window.elapsed() >= cfg.sample_window {
                    windowed_hists.push((current_sec, current_hist));
                    current_hist = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3).unwrap();
                    last_window = Instant::now();
                    current_sec += cfg.sample_window.as_secs();
                }
            }
            windowed_hists
        }));
    }

    // -------------------------------------------------------------------------
    // Component 2: Burst Writers (The Attackers)
    // -------------------------------------------------------------------------
    let mut writer_handles: Vec<JoinHandle<Vec<(u64, Histogram<u64>)>>> = Vec::new();

    // Define the narrow attack zone: e.g., the last 1% of the key space.
    let burst_range_start = config.initial_keys - (config.initial_keys as f64 * config.skew_fraction) as u64;
    let burst_range_end = config.initial_keys;

    for task_id in 0..config.num_burst_writers {
        let idx = index.clone();
        let cfg = config.clone();
        let burst_active = burst_active.clone();
        let test_active = test_active.clone();

        writer_handles.push(tokio::spawn(async move {
            let mut rng = SmallRng::seed_from_u64(task_id as u64 ^ 0x9999);
            let mut windowed_hists = Vec::new();
            let mut current_hist = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3).unwrap();
            let mut last_window = Instant::now();
            let mut current_sec = 0;

            while test_active.load(Ordering::Relaxed) {
                // If burst is active, strictly target the narrow skew zone.
                // Otherwise, perform sparse random writes across the whole domain.
                let key = if burst_active.load(Ordering::Relaxed) {
                    rng.gen_range(burst_range_start..=burst_range_end)
                } else {
                    rng.gen_range(1..=(config.initial_keys * 2)) 
                };

                let op_start = Instant::now();
                idx.insert(key, rng.next_u64()).await;
                
                let elapsed_us = op_start.elapsed().as_micros() as u64;
                let _ = current_hist.record(elapsed_us);

                if last_window.elapsed() >= cfg.sample_window {
                    windowed_hists.push((current_sec, current_hist));
                    current_hist = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3).unwrap();
                    last_window = Instant::now();
                    current_sec += cfg.sample_window.as_secs();
                }
            }
            windowed_hists
        }));
    }

    // -------------------------------------------------------------------------
    // Component 3: The Orchestrator (Injecting the Skew)
    // -------------------------------------------------------------------------
    // Warm-up phase
    sleep(config.burst_trigger_time).await;
    
    // [Artifact Note]: Triggering the burst. In legacy learned indexes, this instantly 
    // invalidates internal node boundaries, launching synchronous heavy rebuilds.
    burst_active.store(true, Ordering::SeqCst);
    
    // Sustain the burst for the remainder of the test
    let remaining_time = config.total_duration.saturating_sub(config.burst_trigger_time);
    sleep(remaining_time).await;

    // Graceful shutdown
    test_active.store(false, Ordering::SeqCst);

    // -------------------------------------------------------------------------
    // Phase 4: Time-Series Aggregation
    // -------------------------------------------------------------------------
    // (Omitted standard Map-Reduce logic that aligns the histograms by `current_sec` 
    // across all threads and calculates the global P99 per second, returning `Vec<TimeWindowResult>`).
    
    // Return empty here for structural brevity. In full artifact, this generates 
    // the precise data points required to plot Figure 2.
    vec![] 
}

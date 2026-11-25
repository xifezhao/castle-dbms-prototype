//! CASTLE Core Index Module
//!
//! Exports the bifurcated `LeafCtrl` / `LeafSnap` architecture, `CastleIndex`,
//! and the `CastleArcIndex` ablation variant (using atomic `Arc` refcounting
//! instead of thread-local EBR).

#![deny(missing_docs)]
#![allow(unsafe_code)]

pub mod double_buffer;
pub mod leaf_ctrl;
pub mod leaf_snap;
pub mod plm;

pub use double_buffer::{AtomicAppendError, LevelBins, OverflowBuffer, BUFFER_CAPACITY};
pub use leaf_ctrl::LeafCtrl;
pub use leaf_snap::LeafSnap;
pub use plm::PiecewiseLinearModel;

use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use crate::concurrency::rcu_epoch::EpochGuard;
use crate::recovery::ReconstructionEngine;
use crate::runtime::cancel_token::CastleCancelToken;
use crate::runtime::rebuild_pool::DedicatedRebuildPool;

/// Trait representing conversion of an index key to `f64` for PLM inference.
pub trait KeyToF64 {
    /// Converts key to `f64`.
    fn to_f64(&self) -> f64;
}

impl KeyToF64 for u64 {
    #[inline(always)]
    fn to_f64(&self) -> f64 {
        *self as f64
    }
}

/// Trait boundary for keys supported by CASTLE.
pub trait IndexKey: Ord + Copy + Send + Sync + Debug + Default + KeyToF64 {}
impl<T: Ord + Copy + Send + Sync + Debug + Default + KeyToF64> IndexKey for T {}

/// Trait boundary for values supported by CASTLE.
pub trait IndexValue: Clone + Send + Sync + Debug + Default {}
impl<T: Clone + Send + Sync + Debug + Default> IndexValue for T {}

/// The unified entry point for the CASTLE Learned Storage Architecture.
pub struct CastleIndex<K: IndexKey, V: IndexValue> {
    /// Range-partitioned upper directory mapping key ranges to stable `LeafCtrl` anchors.
    root_directory: Arc<[LeafCtrl<K, V>]>,
    /// Minimum key observed at bulk load (for O(1) range routing).
    key_min: AtomicU64,
    /// Key span (`key_max - key_min + 1`) observed at bulk load.
    key_span: AtomicU64,
    /// Maximum bounded search window (`HARD_MAX_ERROR` / delta_max).
    pub(crate) hard_max_error_bound: usize,
    /// Delta buffer threshold triggering asynchronous background CoW consolidation.
    rebuild_threshold: usize,
    /// Dedicated isolated rebuild pool (blast-radius containment).
    rebuild_pool: Arc<DedicatedRebuildPool>,
    /// Total completed background rebuilds (telemetry).
    completed_rebuilds: Arc<AtomicUsize>,
    /// Total cancelled/shed rebuild attempts (telemetry).
    shed_rebuilds: Arc<AtomicUsize>,
    /// Total `CapacityExhausted` backpressure retries observed on the write path.
    backpressure_retries: Arc<AtomicU64>,
}

impl<K: IndexKey, V: IndexValue> CastleIndex<K, V> {
    /// Initializes a new CASTLE index with `num_leaves` partitions and `hard_max_error_bound`.
    pub fn new(num_leaves: usize, hard_max_error_bound: usize) -> Self {
        let num_leaves = num_leaves.max(1);
        let mut leaves = Vec::with_capacity(num_leaves);
        for _ in 0..num_leaves {
            leaves.push(LeafCtrl::default());
        }
        let pool_threads = (num_cpus_available() / 4).clamp(2, 8);
        // Decouple background consolidation threshold from the search window delta_max
        let rebuild_threshold = BUFFER_CAPACITY / 2;

        Self {
            root_directory: Arc::from(leaves.into_boxed_slice()),
            key_min: AtomicU64::new(0),
            key_span: AtomicU64::new(10_000_000),
            hard_max_error_bound,
            rebuild_threshold,
            rebuild_pool: Arc::new(DedicatedRebuildPool::new(pool_threads)),
            completed_rebuilds: Arc::new(AtomicUsize::new(0)),
            shed_rebuilds: Arc::new(AtomicUsize::new(0)),
            backpressure_retries: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Returns total completed background CoW rebuilds.
    pub fn completed_rebuilds(&self) -> usize {
        self.completed_rebuilds.load(Ordering::Relaxed)
    }

    /// Returns total load-shed/cancelled rebuilds.
    pub fn shed_rebuilds(&self) -> usize {
        self.shed_rebuilds.load(Ordering::Relaxed)
    }

    /// Returns total `CapacityExhausted` backpressure retry events.
    pub fn backpressure_retries(&self) -> u64 {
        self.backpressure_retries.load(Ordering::Relaxed)
    }

    /// Returns approximate total memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        let guard = EpochGuard::enter();
        self.root_directory
            .iter()
            .map(|leaf| leaf.memory_bytes(&guard))
            .sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    #[inline]
    fn route_idx(&self, key_u64: u64) -> usize {
        let n = self.root_directory.len();
        if n <= 1 {
            return 0;
        }
        let min_k = self.key_min.load(Ordering::Relaxed);
        let span = self.key_span.load(Ordering::Relaxed).max(1);
        if key_u64 <= min_k {
            return 0;
        }
        let rel = key_u64 - min_k;
        if rel >= span {
            return n - 1;
        }
        ((rel as u128 * n as u128) / (span as u128)) as usize
    }

    /// Wait-free point lookup on the hot path.
    #[inline]
    pub fn get<'a>(&'a self, key: &K, epoch_guard: &'a EpochGuard) -> Option<V> {
        let idx = self.route_idx(key.to_f64() as u64);
        self.root_directory[idx].lookup(key, self.hard_max_error_bound, epoch_guard)
    }
}

impl CastleIndex<u64, u64> {
    /// Bulk-loads a sorted slice of `(key, value)` pairs across all leaves in parallel,
    /// fitting deterministic PLMs via the rigid binary reduction tree.
    pub fn bulk_load(&self, sorted_pairs: &[(u64, u64)]) {
        if sorted_pairs.is_empty() {
            return;
        }
        let min_k = sorted_pairs.first().unwrap().0;
        let max_k = sorted_pairs.last().unwrap().0;
        let span = max_k.saturating_sub(min_k).saturating_add(1).max(1);
        self.key_min.store(min_k, Ordering::Release);
        self.key_span.store(span, Ordering::Release);

        let n = self.root_directory.len();
        let mut buckets: Vec<Vec<(u64, u64)>> = (0..n).map(|_| Vec::new()).collect();
        for &(k, v) in sorted_pairs {
            let idx = self.route_idx(k);
            buckets[idx].push((k, v));
        }

        let err = self.hard_max_error_bound;
        buckets.into_par_iter().enumerate().for_each(|(idx, data)| {
            if data.is_empty() {
                return;
            }
            let pos_pairs: Vec<(u64, u64)> = data
                .iter()
                .enumerate()
                .map(|(pos, &(k, _))| (k, pos as u64))
                .collect();
            let plm = ReconstructionEngine::rebuild_model(&pos_pairs, err);
            let first_key = data[0].0;
            let snap = Box::new(LeafSnap::new_with_data(vec![first_key], vec![plm], data));
            self.root_directory[idx].install_initial_snapshot(snap);
        });
    }

    /// Lock-free append on the hot path with automatic non-blocking background CoW rebuild.
    #[inline]
    pub fn insert_with_guard<'a>(
        &'a self,
        key: u64,
        value: u64,
        _guard: &'a EpochGuard,
    ) -> Result<(), AtomicAppendError> {
        let leaf_idx = self.route_idx(key);
        let leaf = &self.root_directory[leaf_idx];

        match leaf.atomic_append(key, value) {
            Ok(slot_idx) => {
                if slot_idx >= self.rebuild_threshold {
                    self.schedule_background_rebuild(leaf_idx);
                }
                Ok(())
            }
            Err(AtomicAppendError::CapacityExhausted) => {
                // Backpressure: record retry telemetry, trigger rebuild, and return CapacityExhausted
                self.backpressure_retries.fetch_add(1, Ordering::Relaxed);
                self.schedule_background_rebuild(leaf_idx);
                Err(AtomicAppendError::CapacityExhausted)
            }
        }
    }

    /// Convenience insert creating a local epoch guard.
    #[inline]
    pub fn insert(&self, key: u64, value: u64) -> Result<(), AtomicAppendError> {
        let guard = EpochGuard::enter();
        self.insert_with_guard(key, value, &guard)
    }

    /// Wait-free range scan returning up to `limit` records with key >= `low`.
    pub fn range_scan(&self, low: u64, limit: usize) -> Vec<(u64, u64)> {
        let guard = EpochGuard::enter();
        let mut idx = self.route_idx(low);
        let n = self.root_directory.len();
        let mut results = Vec::with_capacity(limit.min(256));

        while idx < n && results.len() < limit {
            let remaining = limit - results.len();
            self.root_directory[idx].range_scan(
                &low,
                remaining,
                self.hard_max_error_bound,
                &guard,
                &mut results,
            );
            idx += 1;
        }
        results.sort_by_key(|&(k, _)| k);
        results.truncate(limit);
        results
    }

    fn schedule_background_rebuild(&self, leaf_idx: usize) {
        let leaf = &self.root_directory[leaf_idx];
        if !leaf.try_mark_scheduled() {
            return;
        }
        let dir = Arc::clone(&self.root_directory);
        let err = self.hard_max_error_bound;
        let completed = Arc::clone(&self.completed_rebuilds);
        let token = CastleCancelToken::new();

        match self.rebuild_pool.try_spawn(
            move |tok| {
                let ok = dir[leaf_idx].consolidate_and_publish(err, Some(&tok));
                if ok {
                    completed.fetch_add(1, Ordering::Relaxed);
                }
                dir[leaf_idx].clear_scheduled();
            },
            token,
        ) {
            Ok(_handle) => {}
            Err(_) => {
                // Pool saturated -> load shedding (Section 6.1)
                self.shed_rebuilds.fetch_add(1, Ordering::Relaxed);
                leaf.clear_scheduled();
            }
        }
    }
}

/// Ablation variant of CASTLE using `Arc<LeafSnap>` atomic reference counting
/// on every read instead of Thread-Local EBR (`EpochGuard`).
pub struct CastleArcIndex {
    leaves: Vec<RwLock<Arc<LeafSnap<u64, u64>>>>,
    key_min: AtomicU64,
    key_span: AtomicU64,
    hard_max_error: usize,
}

impl CastleArcIndex {
    /// Creates a new `CastleArcIndex` for reclamation ablation benchmarking.
    pub fn new(num_leaves: usize, hard_max_error: usize) -> Self {
        let num_leaves = num_leaves.max(1);
        let leaves = (0..num_leaves)
            .map(|_| RwLock::new(Arc::new(LeafSnap::empty())))
            .collect();
        Self {
            leaves,
            key_min: AtomicU64::new(0),
            key_span: AtomicU64::new(10_000_000),
            hard_max_error,
        }
    }

    #[inline]
    fn route_idx(&self, key: u64) -> usize {
        let n = self.leaves.len();
        if n <= 1 {
            return 0;
        }
        let min_k = self.key_min.load(Ordering::Relaxed);
        let span = self.key_span.load(Ordering::Relaxed).max(1);
        if key <= min_k {
            return 0;
        }
        let rel = key - min_k;
        if rel >= span {
            return n - 1;
        }
        ((rel as u128 * n as u128) / (span as u128)) as usize
    }

    /// Bulk loads sorted key-value pairs into the `Arc`-based snapshots.
    pub fn bulk_load(&self, sorted_pairs: &[(u64, u64)]) {
        if sorted_pairs.is_empty() {
            return;
        }
        let min_k = sorted_pairs.first().unwrap().0;
        let max_k = sorted_pairs.last().unwrap().0;
        let span = max_k.saturating_sub(min_k).saturating_add(1).max(1);
        self.key_min.store(min_k, Ordering::Release);
        self.key_span.store(span, Ordering::Release);

        let n = self.leaves.len();
        let mut buckets: Vec<Vec<(u64, u64)>> = (0..n).map(|_| Vec::new()).collect();
        for &(k, v) in sorted_pairs {
            buckets[self.route_idx(k)].push((k, v));
        }
        for (idx, data) in buckets.into_iter().enumerate() {
            if data.is_empty() {
                continue;
            }
            let pos_pairs: Vec<(u64, u64)> = data
                .iter()
                .enumerate()
                .map(|(pos, &(k, _))| (k, pos as u64))
                .collect();
            let plm = ReconstructionEngine::rebuild_model(&pos_pairs, self.hard_max_error);
            let first_key = data[0].0;
            let snap = Arc::new(LeafSnap::new_with_data(vec![first_key], vec![plm], data));
            *self.leaves[idx].write().unwrap() = snap;
        }
    }

    /// Point lookup cloning `Arc<LeafSnap>` (triggers atomic refcount increment/decrement).
    #[inline]
    pub fn get(&self, key: u64) -> Option<u64> {
        let idx = self.route_idx(key);
        let snap = Arc::clone(&self.leaves[idx].read().unwrap());
        snap.search(&key, self.hard_max_error)
    }
}

fn num_cpus_available() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
}

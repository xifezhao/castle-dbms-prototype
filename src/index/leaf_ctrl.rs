//! CASTLE Core: The Mutable Synchronization Anchor (`LeafCtrl`)
//!
//! Implements the globally stable leaf control anchor with cache-line-isolated
//! 64-bit tagged-pointer `state_word`, phase-indexed `inflight_writers[2]`
//! (`SeqCst` TOCTOU defense), and lock-free double-buffered delta arrays.

#![deny(missing_docs)]

use std::sync::atomic::Ordering;

#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicUsize};

use crate::concurrency::rcu_epoch::EpochGuard;
use crate::concurrency::write_intent::WriteIntentGuard;
use crate::index::double_buffer::{AtomicAppendError, OverflowBuffer};
use crate::index::leaf_snap::LeafSnap;
use crate::recovery::ReconstructionEngine;
use crate::runtime::cancel_token::CastleCancelToken;
use crate::{IndexKey, IndexValue};

// Pointer tagging bit layout (64-byte aligned LeafSnap guarantees lowest 6 bits == 0)
const TAG_BUF_MASK: usize = 0b01;
const TAG_REBUILD_MASK: usize = 0b10;
const TAG_MASK: usize = TAG_BUF_MASK | TAG_REBUILD_MASK;
const PTR_MASK: usize = !TAG_MASK;

#[inline(always)]
fn pack_state<K: IndexKey, V: IndexValue>(
    ptr: *mut LeafSnap<K, V>,
    buffer_idx: usize,
    is_rebuilding: bool,
) -> usize {
    debug_assert!(buffer_idx <= 1, "Buffer index overflow");
    let ptr_val = ptr as usize;
    debug_assert!(ptr_val & TAG_MASK == 0, "Unaligned LeafSnap pointer");
    let rebuild_bit = if is_rebuilding { TAG_REBUILD_MASK } else { 0 };
    ptr_val | buffer_idx | rebuild_bit
}

#[inline(always)]
fn unpack_state<K: IndexKey, V: IndexValue>(packed_val: usize) -> (*mut LeafSnap<K, V>, usize, bool) {
    let ptr = (packed_val & PTR_MASK) as *mut LeafSnap<K, V>;
    let buffer_idx = packed_val & TAG_BUF_MASK;
    let is_rebuilding = (packed_val & TAG_REBUILD_MASK) != 0;
    (ptr, buffer_idx, is_rebuilding)
}

/// Cache-line padded wrapper to isolate hot atomics from write-heavy counters.
#[repr(align(64))]
struct CachePadded<T>(T);

/// The stable control block anchoring concurrent operations for a leaf key range.
#[repr(align(64))]
pub struct LeafCtrl<K: IndexKey, V: IndexValue> {
    /// CL 0: Packed 64-bit state word (62-bit `LeafSnap` ptr | `TAG_REBUILD` | `TAG_BUF`).
    state_word: CachePadded<AtomicUsize>,
    /// CL 1: Phase-indexed in-flight writer counters for `SeqCst` TOCTOU quiescence.
    inflight_writers: CachePadded<[AtomicUsize; 2]>,
    /// Background scheduling deduplication flag.
    async_scheduled_flag: AtomicBool,
    /// Double-buffered lock-free delta arrays.
    buffers: [OverflowBuffer<K, V>; 2],
}

impl<K: IndexKey, V: IndexValue> Drop for LeafCtrl<K, V> {
    fn drop(&mut self) {
        let packed = self.state_word.0.load(Ordering::Relaxed);
        let (ptr, _, _) = unpack_state::<K, V>(packed);
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

impl<K: IndexKey, V: IndexValue> Default for LeafCtrl<K, V> {
    fn default() -> Self {
        let init_snap = Box::into_raw(Box::new(LeafSnap::empty()));
        Self {
            state_word: CachePadded(AtomicUsize::new(pack_state::<K, V>(init_snap, 0, false))),
            inflight_writers: CachePadded([AtomicUsize::new(0), AtomicUsize::new(0)]),
            async_scheduled_flag: AtomicBool::new(false),
            buffers: [OverflowBuffer::new(), OverflowBuffer::new()],
        }
    }
}

impl<K: IndexKey, V: IndexValue> LeafCtrl<K, V> {
    /// Returns approximate memory footprint of this leaf (control + active snapshot + buffers).
    pub fn memory_bytes(&self, guard: &EpochGuard) -> usize {
        let snap = self.load_snapshot(guard);
        std::mem::size_of::<Self>()
            + snap.memory_bytes()
            + self.buffers[0].memory_bytes()
            + self.buffers[1].memory_bytes()
    }

    // =========================================================================
    // READ PATH (Wait-Free Inference + Bounded Search + Delta Reconciliation)
    // =========================================================================

    /// Safely loads the immutable `LeafSnap` via `Acquire`, pinned by `EpochGuard`.
    #[inline]
    pub fn load_snapshot<'a>(&'a self, _guard: &'a EpochGuard) -> &'a LeafSnap<K, V> {
        let packed_val = self.state_word.0.load(Ordering::Acquire);
        let (ptr, _, _) = unpack_state::<K, V>(packed_val);
        unsafe { &*ptr }
    }

    /// Wait-free point lookup combining delta buffer check and PLM-guided base search.
    #[inline]
    pub fn lookup(&self, key: &K, hard_max_error: usize, _guard: &EpochGuard) -> Option<V> {
        let packed_val = self.state_word.0.load(Ordering::Acquire);
        let (snap_ptr, active_idx, is_rebuilding) = unpack_state::<K, V>(packed_val);

        // 1. Check most recent delta buffer(s) first (read-your-writes consistency)
        if is_rebuilding {
            if let Some(val) = self.buffers[1 - active_idx].scan(key) {
                return Some(val);
            }
        }
        if let Some(val) = self.buffers[active_idx].scan(key) {
            return Some(val);
        }

        // 2. Search immutable sorted base run using PLM prediction bounded by delta_max
        let snap = unsafe { &*snap_ptr };
        snap.search(key, hard_max_error)
    }

    /// Backward-compatible bounded search helper.
    #[inline]
    pub fn bounded_search(&self, key: &K, _predicted_pos: usize, hard_bound: usize) -> Option<V> {
        let guard = EpochGuard::enter();
        self.lookup(key, hard_bound, &guard)
    }

    /// Wait-free range scan merging contiguous `LeafSnap::sorted_data` and active deltas.
    #[inline]
    pub fn range_scan(
        &self,
        low: &K,
        limit: usize,
        hard_max_error: usize,
        _guard: &EpochGuard,
        out: &mut Vec<(K, V)>,
    ) {
        let packed_val = self.state_word.0.load(Ordering::Acquire);
        let (snap_ptr, active_idx, is_rebuilding) = unpack_state::<K, V>(packed_val);
        let snap = unsafe { &*snap_ptr };

        snap.range_scan(low, limit, hard_max_error, out);
        self.buffers[active_idx].scan_ge(low, out);
        if is_rebuilding {
            self.buffers[1 - active_idx].scan_ge(low, out);
        }
    }

    // =========================================================================
    // WRITE PATH (Zero-Blocking Append with Phase-Indexed SeqCst TOCTOU Guard)
    // =========================================================================

    /// Lock-free append returning the allocated delta slot index.
    #[inline]
    pub fn atomic_append(&self, key: K, value: V) -> Result<usize, AtomicAppendError> {
        let mut target_buf_idx;
        let _intent_guard = loop {
            let initial_packed = self.state_word.0.load(Ordering::Acquire);
            let (_, init_active, init_rebuilding) = unpack_state::<K, V>(initial_packed);
            target_buf_idx = if init_rebuilding {
                1 - init_active
            } else {
                init_active
            };

            let guard = WriteIntentGuard::new(&self.inflight_writers.0[target_buf_idx]);
            let verified_packed = self.state_word.0.load(Ordering::SeqCst);
            let (_, ver_active, ver_rebuilding) = unpack_state::<K, V>(verified_packed);
            let verified_buf_idx = if ver_rebuilding {
                1 - ver_active
            } else {
                ver_active
            };

            if verified_buf_idx == target_buf_idx {
                break guard;
            }
            drop(guard);
        };

        self.buffers[target_buf_idx].append(key, value)
    }

    // =========================================================================
    // BACKGROUND CO-W PUBLICATION & DETERMINISTIC CONSOLIDATION
    // =========================================================================

    /// Attempts to claim the async scheduling slot to prevent duplicate rebuild tasks.
    #[inline]
    pub fn try_mark_scheduled(&self) -> bool {
        self.async_scheduled_flag
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Releases the async scheduling slot.
    #[inline]
    pub fn clear_scheduled(&self) {
        self.async_scheduled_flag.store(false, Ordering::Release);
    }

    /// Directly installs an initial bulk-loaded snapshot (called before concurrent traffic).
    pub fn install_initial_snapshot(&self, snap: Box<LeafSnap<K, V>>) {
        let new_ptr = Box::into_raw(snap);
        let old_packed = self
            .state_word
            .0
            .swap(pack_state(new_ptr, 0, false), Ordering::SeqCst);
        let (old_ptr, _, _) = unpack_state::<K, V>(old_packed);
        if !old_ptr.is_null() {
            unsafe {
                drop(Box::from_raw(old_ptr));
            }
        }
    }

    /// Freezes the primary active buffer by setting `TAG_REBUILD` in `state_word`
    /// and awaiting quiescence solely on `inflight_writers[active_idx]`.
    pub fn try_freeze_primary_buffer(&self) -> Option<usize> {
        let mut current_packed = self.state_word.0.load(Ordering::Acquire);
        let frozen_idx = loop {
            let (ptr, active_idx, is_rebuilding) = unpack_state::<K, V>(current_packed);
            if is_rebuilding {
                return None;
            }
            let rebuilding_packed = pack_state(ptr, active_idx, true);
            match self.state_word.0.compare_exchange_weak(
                current_packed,
                rebuilding_packed,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => break active_idx,
                Err(updated) => current_packed = updated,
            }
        };

        crate::concurrency::write_intent::await_quiescence(&self.inflight_writers.0[frozen_idx]);
        Some(frozen_idx)
    }

    /// Cancels an in-progress rebuild by clearing `TAG_REBUILD` without swapping snapshots.
    pub fn abort_rebuild(&self) {
        let mut current_packed = self.state_word.0.load(Ordering::Acquire);
        loop {
            let (ptr, active_idx, is_rebuilding) = unpack_state::<K, V>(current_packed);
            if !is_rebuilding {
                return;
            }
            let restored = pack_state(ptr, active_idx, false);
            match self.state_word.0.compare_exchange_weak(
                current_packed,
                restored,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(updated) => current_packed = updated,
            }
        }
    }

    /// Atomically publishes a newly constructed `LeafSnap` and flips the active delta buffer.
    pub fn publish_new_snapshot(&self, new_snap: Box<LeafSnap<K, V>>) {
        let new_ptr = Box::into_raw(new_snap);
        let mut current_packed = self.state_word.0.load(Ordering::Acquire);
        loop {
            let (old_ptr, old_idx, _) = unpack_state::<K, V>(current_packed);
            let new_idx = 1 - old_idx;
            let new_packed = pack_state(new_ptr, new_idx, false);

            match self.state_word.0.compare_exchange_weak(
                current_packed,
                new_packed,
                Ordering::SeqCst,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if !old_ptr.is_null() {
                        crate::concurrency::rcu_epoch::retire_snapshot(old_ptr);
                    }
                    self.buffers[old_idx].clear();
                    break;
                }
                Err(updated) => current_packed = updated,
            }
        }
    }
}

impl LeafCtrl<u64, u64> {
    /// Executes the complete Pure CoW consolidation path on a background thread:
    /// freeze primary buffer -> merge-sort -> deterministic PLM fit -> single-word CAS swap.
    pub fn consolidate_and_publish(
        &self,
        hard_max_error: usize,
        cancel_token: Option<&CastleCancelToken>,
    ) -> bool {
        let frozen_idx = match self.try_freeze_primary_buffer() {
            Some(idx) => idx,
            None => return false,
        };

        let guard = EpochGuard::enter();
        let old_snap = self.load_snapshot(&guard);
        let base_data = old_snap.sorted_data();
        let mut delta = self.buffers[frozen_idx].snapshot_entries();

        if delta.is_empty() {
            self.abort_rebuild();
            return false;
        }

        // Cooperative cancellation checkpoint 1
        if let Some(tok) = cancel_token {
            if tok.is_cancelled() {
                self.abort_rebuild();
                return false;
            }
        }

        // Sort delta stably so newer writes to the same key come later
        delta.sort_by_key(|&(k, _)| k);

        // Two-way merge of base_data and delta (latest update wins on duplicate key)
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(base_data.len() + delta.len());
        let (mut i, mut j) = (0usize, 0usize);
        let mut processed = 0usize;

        while i < base_data.len() && j < delta.len() {
            processed += 1;
            if processed & 1023 == 0 {
                if let Some(tok) = cancel_token {
                    if tok.is_cancelled() {
                        self.abort_rebuild();
                        return false;
                    }
                }
            }
            let (bk, bv) = base_data[i];
            let (dk, dv) = delta[j];
            if bk < dk {
                if merged.last().map(|&(lk, _)| lk) == Some(bk) {
                    merged.last_mut().unwrap().1 = bv;
                } else {
                    merged.push((bk, bv));
                }
                i += 1;
            } else if dk < bk {
                if merged.last().map(|&(lk, _)| lk) == Some(dk) {
                    merged.last_mut().unwrap().1 = dv;
                } else {
                    merged.push((dk, dv));
                }
                j += 1;
            } else {
                // Same key: delta supersedes base
                if merged.last().map(|&(lk, _)| lk) == Some(dk) {
                    merged.last_mut().unwrap().1 = dv;
                } else {
                    merged.push((dk, dv));
                }
                i += 1;
                j += 1;
            }
        }
        while i < base_data.len() {
            let (bk, bv) = base_data[i];
            if merged.last().map(|&(lk, _)| lk) == Some(bk) {
                merged.last_mut().unwrap().1 = bv;
            } else {
                merged.push((bk, bv));
            }
            i += 1;
        }
        while j < delta.len() {
            let (dk, dv) = delta[j];
            if merged.last().map(|&(lk, _)| lk) == Some(dk) {
                merged.last_mut().unwrap().1 = dv;
            } else {
                merged.push((dk, dv));
            }
            j += 1;
        }

        drop(guard);

        // Cooperative cancellation checkpoint 2 before PLM regression
        if let Some(tok) = cancel_token {
            if tok.is_cancelled() {
                self.abort_rebuild();
                return false;
            }
        }

        // Fit deterministic Piecewise Linear Model mapping key -> array position
        let pos_pairs: Vec<(u64, u64)> = merged
            .iter()
            .enumerate()
            .map(|(pos, &(k, _))| (k, pos as u64))
            .collect();
        let plm = ReconstructionEngine::rebuild_model(&pos_pairs, hard_max_error);
        let first_key = merged.first().map(|&(k, _)| k).unwrap_or(0);
        let new_snap = Box::new(LeafSnap::new_with_data(vec![first_key], vec![plm], merged));

        // Single-word atomic publication & buffer flip
        self.publish_new_snapshot(new_snap);
        true
    }
}

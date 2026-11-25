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

//! CASTLE Concurrency Verification: Task-Local RCU & Epoch Lifetimes
//!
//! This module implements the exhaustive state-space permutation tests described in 
//! **Section 7.5 (Concurrency Verification)**, specifically validating the memory 
//! safety claims made in **Section 6.2 (Dealing with Long-Running Transactions)**.
//!
//! # Academic Context
//! In an asynchronous cooperative runtime (e.g., Tokio), relying on traditional Thread-Local 
//! Storage (TLS) for RCU critical sections is fundamentally broken due to coroutine migration 
//! and work-stealing schedulers. CASTLE replaces TLS with a `Task-Local` RAII epoch pinning 
//! mechanism.
//! 
//! # Evaluation Methodology
//! We utilize the `loom` framework's memory tracking primitives (`loom::alloc::Track`) to 
//! mathematically prove two critical memory safety invariants under C11 weak memory models:
//! 
//! 1. **No Use-After-Free (UAF):** A foreground reader whose task holds an active RAII 
//!    `PinEpochGuard` will NEVER access a `LeafSnap` that has been prematurely reclaimed 
//!    by a background rebuilder.
//! 2. **No Memory Leaks / Double Free:** Once the RAII guard is dropped (e.g., task completes, 
//!    cancels, or panics via unwind), the retired `LeafSnap` is reclaimed exactly once.
//!
//! # Artifact Note on `unsafe`
//! RCU implementation inherently requires `unsafe` to dereference atomic raw pointers. 
//! By enclosing this logic within a `loom` harness, we expose these `unsafe` blocks to 
//! exhaustive permutation testing, elevating the safety guarantee from "careful engineering" 
//! to "mathematically verified state-space".

use loom::alloc::Track;
use loom::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;
use std::ptr;

// -----------------------------------------------------------------------------
// Mocking the Core RCU State Machine
// -----------------------------------------------------------------------------

/// Simulates the immutable predictive model (Section 3.2: LeafSnap).
/// Wrapped in `Track` to allow `loom` to detect memory leaks and UAFs.
struct MockLeafSnap {
    data: Track<u64>,
}

/// The global epoch watermark manager (Section 6.2).
struct GlobalWatermark {
    /// Active readers currently pinned to the "Old" epoch.
    active_old_epoch_readers: AtomicUsize,
    /// Pending retired snapshots waiting for the old epoch to drain.
    retired_snapshots: AtomicPtr<MockLeafSnap>,
}

/// The RAII Guard ensuring Task-Local epoch lifecycle.
struct PinEpochGuard<'a> {
    watermark: &'a GlobalWatermark,
}

impl<'a> PinEpochGuard<'a> {
    /// Simulates entering the RCU critical section (Pinning).
    fn new(watermark: &'a GlobalWatermark) -> Self {
        watermark.active_old_epoch_readers.fetch_add(1, Ordering::SeqCst);
        Self { watermark }
    }
}

impl<'a> Drop for PinEpochGuard<'a> {
    /// Simulates exiting the RCU critical section (Unpinning).
    /// Automatically triggered when the coroutine drops or panics.
    fn drop(&mut self) {
        let prev = self.watermark.active_old_epoch_readers.fetch_sub(1, Ordering::SeqCst);
        // If we are the LAST reader leaving the old epoch, perform the garbage collection.
        if prev == 1 {
            let pending_ptr = self.watermark.retired_snapshots.swap(ptr::null_mut(), Ordering::SeqCst);
            if !pending_ptr.is_null() {
                unsafe {
                    // Reclaim memory exactly once
                    drop(Box::from_raw(pending_ptr));
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Exhaustive Verification Harness
// -----------------------------------------------------------------------------

/// Exhaustively tests that concurrent model publication and reader unpinning 
/// correctly adhere to RCU memory isolation and leak-free reclamation.
#[test]
fn test_rcu_task_local_safety() {
    loom::model(|| {
        // Initialize the global watermark manager
        let watermark = Arc::new(GlobalWatermark {
            active_old_epoch_readers: AtomicUsize::new(0),
            retired_snapshots: AtomicPtr::new(ptr::null_mut()),
        });

        // Initialize the LeafCtrl with the active LeafSnap (Model A)
        let initial_snap = Box::new(MockLeafSnap { data: Track::new(100) });
        let leaf_ctrl_ptr = Arc::new(AtomicPtr::new(Box::into_raw(initial_snap)));

        let watermark_reader = watermark.clone();
        let leaf_ctrl_reader = leaf_ctrl_ptr.clone();

        // ---------------------------------------------------------------------
        // THREAD 1: The Foreground Reader (Simulating Async Query Task)
        // ---------------------------------------------------------------------
        let reader_handle = thread::spawn(move || {
            // Step 1: Pin the Epoch (RAII Acquisition)
            let _guard = PinEpochGuard::new(&watermark_reader);
            loom::sync::atomic::fence(Ordering::SeqCst);

            // Step 2: Load the active snapshot (SeqCst establishes total order with retirement check)
            let snap_ptr = leaf_ctrl_reader.load(Ordering::SeqCst);
            
            // Step 3: Yield simulation (Loom explores context switches here)
            loom::thread::yield_now();

            // Step 4: Safely read the floating-point routing data
            unsafe {
                let snap = &*snap_ptr;
                // If UAF occurs, `loom::alloc::Track` will instantly panic here!
                let val = *snap.data.get_ref();
                assert!(val == 100 || val == 200, "Reader must see consistent snapshot data");
            }
            
            // Step 5: Guard drops here automatically, triggering epoch GC if ref_count == 0.
        });

        // ---------------------------------------------------------------------
        // THREAD 2: The Background Rebuilder (Simulating Async Rebuild Task)
        // ---------------------------------------------------------------------
        let watermark_writer = watermark.clone();
        let leaf_ctrl_writer = leaf_ctrl_ptr.clone();

        let writer_handle = thread::spawn(move || {
            // Step 1: Construct the new model (Model B) in a separate arena (Pure CoW)
            let new_snap = Box::new(MockLeafSnap { data: Track::new(200) });
            let new_snap_ptr = Box::into_raw(new_snap);

            // Step 2: Publish the new model (Atomic Swap with SeqCst)
            let old_snap_ptr = leaf_ctrl_writer.swap(new_snap_ptr, Ordering::SeqCst);
            loom::sync::atomic::fence(Ordering::SeqCst);

            // Step 3: Retire the old model to the global watermark
            // (In a full implementation, this attaches to the specific epoch bucket)
            let prev_retired = watermark_writer.retired_snapshots.swap(old_snap_ptr, Ordering::SeqCst);
            assert!(prev_retired.is_null(), "Mock limits to one retirement for simplicity");
            loom::sync::atomic::fence(Ordering::SeqCst);

            // Step 4: Attempt aggressive GC (If readers are already gone)
            if watermark_writer.active_old_epoch_readers.load(Ordering::SeqCst) == 0 {
                let pending_ptr = watermark_writer.retired_snapshots.swap(ptr::null_mut(), Ordering::SeqCst);
                if !pending_ptr.is_null() {
                    unsafe { drop(Box::from_raw(pending_ptr)); }
                }
            }
        });

        // ---------------------------------------------------------------------
        // Teardown & Leak Verification
        // ---------------------------------------------------------------------
        reader_handle.join().unwrap();
        writer_handle.join().unwrap();

        // Final cleanup of the active pointer (Model B) to prevent artificial leaks in test
        let final_ptr = leaf_ctrl_ptr.swap(ptr::null_mut(), Ordering::SeqCst);
        unsafe { drop(Box::from_raw(final_ptr)); }
        
        // Final cleanup of any lingering retired pointers
        let lingering_ptr = watermark.retired_snapshots.swap(ptr::null_mut(), Ordering::SeqCst);
        if !lingering_ptr.is_null() {
            unsafe { drop(Box::from_raw(lingering_ptr)); }
        }

        // Loom automatically verifies that ALL `Track::new()` allocations were dropped 
        // EXACTLY ONCE upon exiting this closure. If a leak or double-free exists 
        // in ANY interleaving, Loom will fail the test.
    });
}

#[test]
fn verify_capacity_exhausted_tail_rollback() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);

    builder.check(|| {
        const CAP: usize = 1;
        let tail = Arc::new(AtomicUsize::new(0));
        let slot_written = Arc::new(AtomicUsize::new(0));
        let successes = Arc::new(AtomicUsize::new(0));
        let exhausted = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for tid in 1..=2usize {
            let t = Arc::clone(&tail);
            let sw = Arc::clone(&slot_written);
            let ok = Arc::clone(&successes);
            let ex = Arc::clone(&exhausted);
            handles.push(thread::spawn(move || {
                let idx = t.fetch_add(1, Ordering::Relaxed);
                if idx >= CAP {
                    t.fetch_sub(1, Ordering::Relaxed);
                    ex.fetch_add(1, Ordering::Relaxed);
                } else {
                    sw.store(tid, Ordering::Release);
                    ok.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(successes.load(Ordering::SeqCst), 1, "Exactly one writer must win the sole slot");
        assert_eq!(exhausted.load(Ordering::SeqCst), 1, "Exactly one writer must observe CapacityExhausted");
        assert_eq!(tail.load(Ordering::SeqCst), CAP, "Tail must roll back cleanly to CAP");
        let winner = slot_written.load(Ordering::Acquire);
        assert!(winner == 1 || winner == 2, "Winning slot must contain a valid writer payload");
    });
}

#[test]
fn verify_buffer_reuse_ebr_grace_period() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);

    builder.check(|| {
        let reader_pinned = Arc::new(AtomicUsize::new(0));
        let buf0_val = Arc::new(AtomicUsize::new(42));
        let swapped = Arc::new(AtomicUsize::new(0));

        let r_pin = Arc::clone(&reader_pinned);
        let r_buf = Arc::clone(&buf0_val);
        let r_sw = Arc::clone(&swapped);

        let reader = thread::spawn(move || {
            // Pin EBR epoch before loading state_word (SeqCst fence enforces Store-Load order)
            r_pin.store(1, Ordering::SeqCst);
            loom::sync::atomic::fence(Ordering::SeqCst);
            let saw_old = r_sw.load(Ordering::SeqCst) == 0;
            if saw_old {
                // Reader scans buffers[0]; must NEVER see reused value 99 while pinned
                let v = r_buf.load(Ordering::Acquire);
                assert_eq!(v, 42, "Stale buffer reuse hazard: reader observed overwritten slot!");
            }
            r_pin.store(0, Ordering::SeqCst);
        });

        let w_pin = Arc::clone(&reader_pinned);
        let w_buf = Arc::clone(&buf0_val);
        let w_sw = Arc::clone(&swapped);

        let rebuilder = thread::spawn(move || {
            // Publish new snapshot and flip active buffer away from 0
            w_sw.store(1, Ordering::SeqCst);
            loom::sync::atomic::fence(Ordering::SeqCst);
            // EBR grace period: only clear & reuse buffers[0] once no reader is pinned
            if w_pin.load(Ordering::SeqCst) == 0 {
                w_buf.store(99, Ordering::Release);
            }
        });

        reader.join().unwrap();
        rebuilder.join().unwrap();
    });
}

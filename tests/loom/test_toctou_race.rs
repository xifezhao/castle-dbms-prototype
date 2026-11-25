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

//! CASTLE Concurrency Verification: TOCTOU StoreLoad Litmus Test
//!
//! This module implements the exhaustive state-space permutation tests described in 
//! **Section 7.5 (Concurrency Verification)** of the VLDB 2027 paper.
//!
//! # Academic Context
//! In **Section 4.3 (Memory Ordering Guarantees)**, we argued that coordinating the 
//! `InflightWriters` count and the `RebuildingFlag` forms a classic Store-Load pattern.
//! Relying on standard `Acquire/Release` semantics is vulnerable to the Store Buffer 
//! anomaly found in weak memory models (and x86-TSO).
//! 
//! # Evaluation Methodology
//! We utilize `loom`, which does not rely on OS thread scheduling luck. Instead, it 
//! deterministically explores *every possible interleaving* and simulates delayed 
//! visibility of stores (Store Buffers) permitted by the C11 memory model.
//!
//! - `test_seqcst_prevents_toctou`: Proves that `Ordering::SeqCst` establishes a 
//!   Total Global Order, successfully preventing the Write-Write conflict.
//! - `test_acq_rel_fails_toctou`: A negative control test (`#[should_panic]`). It proves 
//!   that without `SeqCst`, the hardware memory model WILL allow both threads to observe 
//!   the old state (0) and crash the database invariant.

#![deny(unsafe_code)]

use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

/// Simulates the exact synchronization boundary between a foreground Writer 
/// and a background Rebuilder, using strict Sequential Consistency.
#[test]
fn test_seqcst_prevents_toctou() {
    loom::model(|| {
        let inflight_writers = Arc::new(AtomicUsize::new(0));
        let rebuilding_flag = Arc::new(AtomicUsize::new(0));

        let inflight_clone = inflight_writers.clone();
        let rebuilding_clone = rebuilding_flag.clone();

        // ---------------------------------------------------------------------
        // THREAD 1: The Writer (Simulating WriteIntentGuard initialization)
        // ---------------------------------------------------------------------
        let writer_handle = thread::spawn(move || {
            // STEP 1: Store (Declare intent to write)
            inflight_clone.fetch_add(1, Ordering::SeqCst);
            loom::sync::atomic::fence(Ordering::SeqCst);
            
            // STEP 2: Load (Check if rebuild is actively claiming the node)
            let is_rebuilding = rebuilding_clone.load(Ordering::SeqCst);
            
            is_rebuilding
        });

        // ---------------------------------------------------------------------
        // THREAD 2: The Rebuilder (Simulating TryAcquireRAII success)
        // ---------------------------------------------------------------------
        let rebuilder_handle = thread::spawn(move || {
            // STEP 1: Store (Claim the rebuild lock via CAS or Swap)
            // Simplified to swap here as we only care about the visibility boundary
            rebuilding_flag.swap(1, Ordering::SeqCst);
            loom::sync::atomic::fence(Ordering::SeqCst);
            
            // STEP 2: Load (Check if any writers are currently in flight)
            let active_writers = inflight_writers.load(Ordering::SeqCst);
            
            active_writers
        });

        // ---------------------------------------------------------------------
        // Assertion & Verification Phase
        // ---------------------------------------------------------------------
        let writer_saw_rebuilding = writer_handle.join().unwrap() == 1;
        let rebuilder_saw_writer = rebuilder_handle.join().unwrap() >= 1;

        // THE INVARIANT:
        // Because both used `SeqCst`, there is a total global order. It is mathematically 
        // IMPOSSIBLE for both the writer to think there is no rebuilder AND the rebuilder 
        // to think there is no writer. At least one MUST see the other's operation.
        // If both read `false/0`, they would both proceed, causing a catastrophic data race.
        let both_missed = !writer_saw_rebuilding && !rebuilder_saw_writer;
        
        assert!(
            !both_missed,
            "FATAL: StoreLoad reordering occurred despite SeqCst! The C11 memory model is broken."
        );
    });
}

/// NEGATIVE CONTROL TEST: Proves that `Acquire/Release` is insufficient.
/// 
/// This test actively demonstrates the Store Buffer anomaly. Loom will explore 
/// the relaxed visibility paths and eventually find an interleaving where the 
/// stores are buffered, causing BOTH threads to load `0` simultaneously.
/// The `#[should_panic]` attribute verifies that the vulnerability is real and 
/// successfully caught by our assertion.
#[test]
#[should_panic(expected = "TOCTOU Race Detected")]
fn test_acq_rel_fails_toctou() {
    loom::model(|| {
        let inflight_writers = Arc::new(AtomicUsize::new(0));
        let rebuilding_flag = Arc::new(AtomicUsize::new(0));

        let inflight_clone = inflight_writers.clone();
        let rebuilding_clone = rebuilding_flag.clone();

        let writer_handle = thread::spawn(move || {
            // Using Release/Acquire instead of SeqCst
            inflight_clone.fetch_add(1, Ordering::Release);
            rebuilding_clone.load(Ordering::Acquire)
        });

        let rebuilder_handle = thread::spawn(move || {
            // Using Release/Acquire instead of SeqCst
            rebuilding_flag.swap(1, Ordering::Release);
            inflight_writers.load(Ordering::Acquire)
        });

        let writer_saw_rebuilding = writer_handle.join().unwrap() == 1;
        let rebuilder_saw_writer = rebuilder_handle.join().unwrap() >= 1;

        let both_missed = !writer_saw_rebuilding && !rebuilder_saw_writer;
        
        // This will panic under Loom's state exploration, proving Section 4.3's claim!
        if both_missed {
            panic!("TOCTOU Race Detected: Store Buffer anomaly allowed both threads to bypass locks.");
        }
    });
}

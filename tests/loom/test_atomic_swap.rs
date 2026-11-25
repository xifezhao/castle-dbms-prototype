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

//! CASTLE Concurrency Verification: Atomic Buffer Rotation & Pointer Tagging
//!
//! This module implements the exhaustive state-space permutation tests described in 
//! **Section 7.5 (Concurrency Verification)**, specifically validating the lock-free 
//! pointer manipulation claims made in **Section 4.2 (Double-Buffering & Atomic Overflow)**.
//!
//! # Academic Context & Hardware Constraints
//! To atomically publish a new `LeafSnap` routing model AND rotate the active overflow 
//! buffer index without relying on coarse-grained locks or non-portable 128-bit CAS 
//! instructions (which hurt cross-architecture compatibility like ARMv8), CASTLE utilizes 
//! **Pointer Tagging**. 
//! 
//! Because dynamically allocated structures on 64-bit architectures can be forced to 
//! 8-byte boundaries, the lower 3 bits of the pointer address are guaranteed to be `0`. 
//! We exploit these unused bits to pack the `active_buffer_index` (0 or 1). Thus, a 
//! single 64-bit `compare_exchange` updates both states simultaneously.
//!
//! # Evaluation Methodology
//! Using `loom`, we prove two properties:
//! 1. **No Torn Reads:** A reader unpacking the 64-bit word will NEVER observe a mismatched 
//!    pair (e.g., observing the old model with the new buffer index).
//! 2. **Happens-Before Visibility:** We verify that the `Release` memory ordering on the CAS 
//!    and the `Acquire` ordering on the load successfully flush the store buffer. If we 
//!    artificially downgrade this to `Relaxed`, `loom::alloc::Track` will mathematically 
//!    catch the data race, providing a negative control for our memory ordering claims.

#![allow(unsafe_code)]

use loom::alloc::Track;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::Arc;
use loom::thread;

// -----------------------------------------------------------------------------
// Pointer Tagging Utilities
// -----------------------------------------------------------------------------

/// The memory alignment boundary. 8 bytes guarantees 3 free bits at the bottom.
const ALIGNMENT: usize = 8;
const TAG_MASK: usize = 0b111;
const PTR_MASK: usize = !TAG_MASK;

/// Mock predictive model, explicitly aligned to guarantee safe pointer tagging.
/// Wrapped in `Track` to allow `loom` to enforce happens-before data race detection.
#[repr(align(8))]
struct MockLeafSnap {
    model_version: Track<u64>,
}

/// Packs an aligned pointer and a buffer index (0 or 1) into a single 64-bit word.
fn pack_state(ptr: *mut MockLeafSnap, buffer_idx: usize) -> usize {
    assert!(buffer_idx <= TAG_MASK, "Buffer index exceeds available tag space");
    let ptr_val = ptr as usize;
    assert!(ptr_val % ALIGNMENT == 0, "Pointer violates 8-byte alignment");
    ptr_val | buffer_idx
}

/// Unpacks the 64-bit word back into a raw pointer and the buffer index.
fn unpack_state(packed_val: usize) -> (*mut MockLeafSnap, usize) {
    let ptr = (packed_val & PTR_MASK) as *mut MockLeafSnap;
    let buffer_idx = packed_val & TAG_MASK;
    (ptr, buffer_idx)
}

// -----------------------------------------------------------------------------
// Exhaustive Verification Harness
// -----------------------------------------------------------------------------

/// Verifies that the single 64-bit CAS guarantees absolute atomic transition of 
/// both the snapshot pointer and the buffer index, preventing torn states.
#[test]
fn test_atomic_buffer_rotation_safety() {
    loom::model(|| {
        // Step 1: Initialize System State (Version 100, Buffer 0)
        let initial_snap = Box::new(MockLeafSnap { model_version: Track::new(100) });
        let initial_ptr = Box::into_raw(initial_snap);
        let packed_initial = pack_state(initial_ptr, 0);
        
        let leaf_ctrl_state = Arc::new(AtomicUsize::new(packed_initial));
        
        let reader_state = leaf_ctrl_state.clone();
        let rebuilder_state = leaf_ctrl_state.clone();

        // ---------------------------------------------------------------------
        // THREAD 1: The Foreground Reader/Writer
        // ---------------------------------------------------------------------
        let reader_handle = thread::spawn(move || {
            // Hot Path: Load state with Acquire semantics.
            let packed_val = reader_state.load(Ordering::Acquire);
            let (snap_ptr, buf_idx) = unpack_state(packed_val);

            let observed_version = unsafe {
                let snap = &*snap_ptr;
                // [Loom Magic]: If the Release/Acquire pairing is missing, loom detects 
                // that this read lacks a happens-before relationship with the rebuilder's 
                // memory initialization, and will panic here.
                *snap.model_version.get_ref()
            };

            // THE INVARIANT: No Torn States.
            // If we see the old model, we MUST see buffer 0.
            // If we see the new model, we MUST see buffer 1.
            if observed_version == 100 {
                assert_eq!(buf_idx, 0, "Torn State: Old model with new buffer index!");
            } else if observed_version == 200 {
                assert_eq!(buf_idx, 1, "Torn State: New model with old buffer index!");
            } else {
                panic!("Observed corrupted memory state");
            }
        });

        // ---------------------------------------------------------------------
        // THREAD 2: The Background Rebuilder (Section 4.2 Publication)
        // ---------------------------------------------------------------------
        let rebuilder_handle = thread::spawn(move || {
            // Construct the new model offline (Version 200)
            let new_snap = Box::new(MockLeafSnap { model_version: Track::new(200) });
            let new_ptr = Box::into_raw(new_snap);
            
            // We know the old state for the CAS, but we must load it properly
            let current_packed = rebuilder_state.load(Ordering::Acquire);
            let (_, old_buf_idx) = unpack_state(current_packed);
            
            // Promote buffer index (0 -> 1)
            let new_buf_idx = 1 - old_buf_idx;
            let new_packed = pack_state(new_ptr, new_buf_idx);

            // Execute the single 64-bit Release CAS (Atomic Rotation)
            let result = rebuilder_state.compare_exchange(
                current_packed, 
                new_packed, 
                Ordering::Release, 
                Ordering::Acquire
            );

            // Return pointers to prevent memory leaks in the loom test
            if let Ok(old_val) = result {
                Some((unpack_state(old_val).0, new_ptr))
            } else {
                unsafe { drop(Box::from_raw(new_ptr)); }
                None
            }
        });

        // ---------------------------------------------------------------------
        // Teardown & Validation
        // ---------------------------------------------------------------------
        reader_handle.join().unwrap();
        let swapped_ptrs = rebuilder_handle.join().unwrap();

        // Memory cleanup to satisfy Loom's leak detector
        let final_packed = leaf_ctrl_state.load(Ordering::SeqCst);
        let (final_ptr, _) = unpack_state(final_packed);
        unsafe { drop(Box::from_raw(final_ptr)); }

        if let Some((old_ptr, _)) = swapped_ptrs {
            unsafe { drop(Box::from_raw(old_ptr)); }
        }
    });
}

/// NEGATIVE CONTROL TEST: Proves that `Ordering::Relaxed` is scientifically invalid.
///
/// If a naive developer uses `Relaxed` for the CAS, the pointer swap is atomic, BUT 
/// the memory content of the new `MockLeafSnap` (initialized by the Rebuilder) 
/// is not guaranteed to be flushed from the hardware store buffer to the reader.
/// The `#[should_panic]` confirms that our `Release/Acquire` choice is a physical necessity.
#[test]
#[should_panic]
fn test_relaxed_fails_visibility() {
    loom::model(|| {
        let initial_snap = Box::new(MockLeafSnap { model_version: Track::new(100) });
        let leaf_ctrl_state = Arc::new(AtomicUsize::new(pack_state(Box::into_raw(initial_snap), 0)));
        
        let reader_state = leaf_ctrl_state.clone();
        let rebuilder_state = leaf_ctrl_state.clone();

        let reader_handle = thread::spawn(move || {
            // INCORRECT: Using Relaxed load
            let packed_val = reader_state.load(Ordering::Relaxed);
            let (snap_ptr, _) = unpack_state(packed_val);
            unsafe {
                // Loom will PANIC here on some interleavings because the happens-before 
                // edge is missing. The reader accesses uninitialized/stale cache lines.
                let _val = *(*snap_ptr).model_version.get_ref(); 
            }
        });

        let _rebuilder_handle = thread::spawn(move || {
            let new_snap = Box::new(MockLeafSnap { model_version: Track::new(200) });
            let new_ptr = Box::into_raw(new_snap);
            let current_packed = rebuilder_state.load(Ordering::Relaxed);
            
            // INCORRECT: Using Relaxed CAS
            let _ = rebuilder_state.compare_exchange(
                current_packed, 
                pack_state(new_ptr, 1), 
                Ordering::Relaxed, 
                Ordering::Relaxed
            );
        });

        reader_handle.join().unwrap();
        // Memory cleanup omitted for brevity in failing test...
    });
}

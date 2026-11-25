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

//! CASTLE Concurrency: Global Watermark & LRT Circuit Breaker
//!
//! This module implements the defense-in-depth mechanisms described in **Section 6.2 
//! (Dealing with Long-Running Transactions)** of the VLDB 2027 paper.
//!
//! # Academic Context & The RCU Stagnation Hazard
//! In any pure Copy-on-Write (CoW) or MVCC architecture, the garbage collection (GC) of 
//! obsolete snapshots is blocked by the oldest active reader. A Long-Running Transaction (LRT) 
//! (e.g., a slow analytical scan or a stalled network client) pinning an old RCU epoch 
//! will stall the Global Safe Watermark. During write-heavy bursts, this prevents the 
//! reclamation of millions of transient models, inevitably triggering an Out-Of-Memory (OOM) 
//! cascade across the DBMS.
//!
//! # The CASTLE Circuit Breaker Protocol
//! Instead of immediately aborting the offending LRT (which degrades user experience), 
//! CASTLE executes a graceful degradation protocol:
//! 1. **Telemetry:** The `WatermarkManager` continuously tracks the absolute wall-clock 
//!    age of each active epoch.
//! 2. **Threshold Trip:** If an epoch exceeds `MAX_LRT_DURATION_MS`, it is flagged as stagnant.
//! 3. **Localized Deep-Copy (Snapshot Isolation):** The offending task detects this flag, 
//!    performs a deep-copy (`Clone`) of the specific `LeafSnap` it requires, and transfers 
//!    ownership to its own local task heap.
//! 4. **Forceful Unpin:** The task drops its global `EpochGuard`, allowing the global 
//!    watermark to advance and unblocking system-wide GC.
//!
//! This is a deliberate architectural trade-off: trading bounded, localized memory overhead 
//! (the deep copy) for global system stability (preventing OOM).

#![deny(missing_docs)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::concurrency::rcu_epoch::EpochGuard;

/// The number of epoch buckets, aligning with rcu_epoch.rs
const NUM_EPOCHS: usize = 3;

/// The maximum duration (in milliseconds) a coroutine is allowed to pin a global epoch 
/// before the circuit breaker trips. (e.g., 5 seconds).
const MAX_LRT_DURATION_MS: u64 = 5000;

/// The absolute hard limit (in milliseconds). If a task holds the isolated snapshot 
/// beyond this time, the DBMS transaction manager will deterministically abort it.
const ABSOLUTE_SESSION_TIMEOUT_MS: u64 = 60_000;

/// Global tracking structure for Epoch age and circuit breaker flags.
pub struct WatermarkManager {
    /// Tracks the absolute UNIX timestamp (in ms) when each epoch was activated.
    epoch_start_times: [AtomicU64; NUM_EPOCHS],
}

impl WatermarkManager {
    /// Initializes the Watermark Manager.
    pub const fn new() -> Self {
        Self {
            epoch_start_times: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
        }
    }

    /// Records the exact wall-clock time when a new epoch boundary is established.
    /// Called atomically by the global epoch advancement routine.
    #[inline]
    pub fn mark_epoch_start(&self, epoch_idx: usize) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64;
        
        self.epoch_start_times[epoch_idx].store(now_ms, Ordering::Release);
    }

    /// Checks if a specific epoch has exceeded the safety threshold.
    /// 
    /// Evaluated lock-free by foreground readers to determine if they need to 
    /// trigger self-isolation (deep copy).
    #[inline]
    pub fn is_epoch_stagnant(&self, epoch_idx: usize) -> bool {
        let start_time = self.epoch_start_times[epoch_idx].load(Ordering::Acquire);
        if start_time == 0 {
            return false;
        }

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // If the epoch has been active for longer than the threshold, trip the breaker.
        now_ms.saturating_sub(start_time) > MAX_LRT_DURATION_MS
    }
}

// Global instance (aligned with `GLOBAL_RCU` in rcu_epoch.rs)
static GLOBAL_WATERMARK: WatermarkManager = WatermarkManager::new();

/// Type alias aligning with the public re-export in concurrency/mod.rs
pub type GlobalWatermark = WatermarkManager;

/// Re-export alias for the circuit-breaker protocol
pub type LRTBreaker = WatermarkManager;

// -----------------------------------------------------------------------------
// Snapshot Isolation Wrapper (Deep-Copy Hand-off)
// -----------------------------------------------------------------------------

/// An isolated, heap-owned wrapper for a predictive model.
///
/// When the circuit breaker trips, CASTLE converts a shared RCU reference (`&LeafSnap`) 
/// into this `IsolatedSnapshot`. By owning the `T` (via `.clone()`), the task explicitly 
/// drops the `EpochGuard`, releasing its grip on the global RCU watermark.
pub struct IsolatedSnapshot<T: Clone> {
    /// The deep-copied data structure (e.g., the specific LeafSnap needed by the LRT).
    /// Bound strictly to the task's local memory footprint.
    pub owned_data: T,
    
    /// Timestamp when isolation occurred. Used by the upper transaction manager 
    /// to enforce `ABSOLUTE_SESSION_TIMEOUT_MS`.
    isolation_timestamp_ms: u64,
}

impl<T: Clone> IsolatedSnapshot<T> {
    /// Executes the isolation protocol.
    ///
    /// Takes the shared reference and the `EpochGuard`. By consuming the `EpochGuard` by value, 
    /// Rust's ownership semantics guarantee that the RAII `Drop` handler is called immediately 
    /// after the deep copy is completed, forcefully unpinning the global RCU tracker.
    #[cold] // Marked cold because this is an anomaly path, instructing LLVM to optimize the hot path.
    pub fn isolate(shared_ref: &T, guard: EpochGuard) -> Self {
        // 1. Perform the localized deep copy. (The trade-off penalty)
        let owned_data = shared_ref.clone();

        // 2. Consume and drop the RAII guard. 
        // This instantly decrements `active_old_epoch_readers`, unblocking the background GC.
        drop(guard);

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            owned_data,
            isolation_timestamp_ms: now_ms,
        }
    }

    /// Checks if this isolated transaction has violated the absolute database session limits.
    pub fn is_violating_absolute_timeout(&self) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        now_ms.saturating_sub(self.isolation_timestamp_ms) > ABSOLUTE_SESSION_TIMEOUT_MS
    }
}

// -----------------------------------------------------------------------------
// Hot Path Integration Example (Conceptual integration logic)
// -----------------------------------------------------------------------------

/// Helper method demonstrating how a read task interacts with the LRT Breaker.
///
/// Foreground tasks periodically invoke this during long range-scans to ensure 
/// they are not inadvertently halting the storage engine's garbage collector.
#[inline]
pub fn ensure_isolation_if_stagnant<T: Clone>(
    shared_data: &T,
    epoch_idx: usize,
    guard: EpochGuard,
) -> Result<(), IsolatedSnapshot<T>> {
    if GLOBAL_WATERMARK.is_epoch_stagnant(epoch_idx) {
        // Trip the breaker: Return the isolated (owned) snapshot and consume the guard.
        Err(IsolatedSnapshot::isolate(shared_data, guard))
    } else {
        // Safe to continue using the shared RCU reference.
        Ok(())
    }
}

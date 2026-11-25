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

//! CASTLE Runtime: Collaborative Cancellation Token
//!
//! This module implements the cooperative task abort mechanism defined in **Section 6.1 
//! (Collaborative Cancellation in Dedicated Pools)** and **ADR-002** of the VLDB 2027 paper.
//!
//! # Academic Context & The Termination Hazard
//! In C++ and Rust systems, forcefully terminating a thread (e.g., `pthread_cancel`) while 
//! it is allocating memory or holding locks results in permanent memory leaks, orphaned 
//! pointers, and systemic deadlocks. Since learned index rebuilds are heavy matrix 
//! decompositions (CPU-bound), they cannot be safely killed by the OS or the Tokio executor 
//! from the outside.
//!
//! # The CASTLE Mechanism: Checkpointed Cooperative Yielding
//! We utilize a lightweight, atomic `CastleCancelToken`. 
//! 1. **The Watchdog (Trigger):** If a rebuild task exceeds its P99 latency budget, the 
//!    Watchdog calls `.cancel()` on the token.
//! 2. **The Worker (Poller):** The dense mathematical loops inside the `DedicatedRebuildPool` 
//!    do not execute blindly. They are instrumented to poll `.is_cancelled()` at safe, 
//!    coarse-grained boundaries (e.g., after processing a Mini-Block of 1,000 records).
//! 3. **The Unwind (Safety):** Upon detecting a cancellation, the worker early-returns 
//!    `Err(CastleCancelledError)`. Rust's ownership model naturally triggers the `Drop` 
//!    handlers of the partially constructed `LeafSnap` arena, freeing the memory cleanly 
//!    and gracefully returning the OS thread back to the pool.

#![deny(missing_docs)]

use std::fmt;
use tokio_util::sync::CancellationToken;

/// Error indicating that a CPU-bound rebuild task was cooperatively aborted.
///
/// Returning this error via the `?` operator ensures that the execution stack unwinds 
/// safely, running all necessary destructors without a panic.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct CastleCancelledError;

impl fmt::Display for CastleCancelledError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CASTLE Rebuild Task was cooperatively cancelled by the Watchdog.")
    }
}

impl std::error::Error for CastleCancelledError {}

/// A highly efficient, thread-safe token for signaling cooperative task cancellation.
/// 
/// Under the hood, this wraps Tokio's `CancellationToken`, providing a unified API 
/// tailored for CASTLE's tight mathematical loops.
#[derive(Clone, Default)]
pub struct CastleCancelToken {
    inner: CancellationToken,
}

impl CastleCancelToken {
    /// Creates a new, un-cancelled token.
    pub fn new() -> Self {
        Self {
            inner: CancellationToken::new(),
        }
    }

    /// Triggers the cancellation signal.
    /// 
    /// # Invocation Context
    /// Called exclusively by the `WatchdogMonitor` when an isolated task within the 
    /// `DedicatedRebuildPool` has exceeded its permissible latency budget.
    #[inline]
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Checks if the cancellation signal has been triggered.
    /// 
    /// # Microarchitectural Trade-off (Section 6.1)
    /// Polling an atomic boolean requires a load instruction. Doing this inside the inner-most 
    /// vector loop (e.g., a PLM regression fit over 1 million keys) would introduce a control-flow 
    /// branch, destroying LLVM's loop unrolling heuristics and preventing SIMD auto-vectorization.
    /// 
    /// **Therefore, this MUST only be polled at coarse-grained check-points** (e.g., every 
    /// outer loop iteration or after scanning a 4KB block), ensuring the cancellation latency 
    /// remains sub-millisecond without sacrificing raw ALU throughput.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// An ergonomic helper for mathematical loops to abort cleanly.
    ///
    /// # Usage
    /// ```rust,ignore
    /// for chunk in data.chunks(1000) {
    ///     cancel_token.ensure_active()?; // Returns Err if cancelled, unwinding the stack safely
    ///     compute_dense_matrix(chunk);
    /// }
    /// ```
    #[inline(always)]
    pub fn ensure_active(&self) -> Result<(), CastleCancelledError> {
        if self.is_cancelled() {
            Err(CastleCancelledError)
        } else {
            Ok(())
        }
    }
}

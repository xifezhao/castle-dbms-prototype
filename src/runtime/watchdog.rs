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

//! CASTLE Runtime: The Collaborative Watchdog Monitor
//!
//! This module implements the precise latency enforcement and memory abandonment 
//! protocols defined in **Section 6.1 (Collaborative Cancellation in Dedicated Pools)** 
//! and **ADR-002** of the VLDB 2027 paper.
//!
//! # Academic Context & The `JoinHandle` Fallacy
//! A widespread anti-pattern in asynchronous DBMS prototypes is relying on 
//! `tokio::time::timeout` to abort CPU-bound tasks in `spawn_blocking`. 
//! Dropping a blocking `JoinHandle` merely detaches the caller; the underlying OS 
//! thread continues executing its dense mathematical loop (e.g., matrix inversion) 
//! indefinitely. During a data skew burst, this detached execution guarantees 
//! `DedicatedRebuildPool` exhaustion and silent system paralysis.
//!
//! # The CASTLE Mechanism
//! The `WatchdogMonitor` addresses this by strictly orchestrating the `CastleCancelToken`.
//! 1. **Budget Enforcement:** It awaits the rebuild task with a strict wall-clock deadline.
//! 2. **Signal & Detach:** Upon timeout, it triggers the token and IMMEDIATELY returns 
//!    control to the background causal scheduler (Section 6.3) to initiate `DegradedScan`.
//! 3. **Asynchronous Memory Abandonment:** The blocking thread, continuing its execution 
//!    in the background, reaches its next `token.ensure_active()` checkpoint (as required 
//!    by ADR-002). It observes the cancellation, unwinds its stack, automatically drops 
//!    any partially constructed `LeafSnap` arenas via RAII, and seamlessly returns 
//!    its OS thread to the pool. Zero leaks, zero forced aborts.

#![deny(missing_docs)]
#![deny(unsafe_code)] // Watchdog logic strictly relies on safe concurrency bounds.

use std::fmt;
use std::time::Duration;
use tokio::task::JoinHandle;

use crate::runtime::cancel_token::CastleCancelToken;

/// Error emitted when a rebuild task exceeds its maximum permissible execution window.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum RebuildTimeoutError {
    /// The task failed to complete within the SLA budget and was cooperatively cancelled.
    BudgetExceeded,
    /// The underlying OS thread panicked (e.g., due to an unexpected hardware exception 
    /// or strict assertion failure). The watchdog propagates this upward to prevent 
    /// silent data corruption.
    TaskPanicked,
}

impl fmt::Display for RebuildTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExceeded => write!(f, "Rebuild task exceeded latency budget and was cancelled"),
            Self::TaskPanicked => write!(f, "Rebuild task panicked internally"),
        }
    }
}

impl std::error::Error for RebuildTimeoutError {}

/// The centralized monitor ensuring bounded execution for CPU-heavy tasks.
pub struct WatchdogMonitor;

impl WatchdogMonitor {
    /// Enforces a strict latency budget on an active rebuild task.
    ///
    /// # Architectural Guarantee
    /// This method is designed to be called by the `off-path causal scheduling loop`.
    /// By returning `Err(BudgetExceeded)` immediately upon timeout, the scheduler is 
    /// unblocked instantly, allowing the DBMS to mark the node for `DegradedScan` 
    /// and maintain overall P99 latency stability.
    ///
    /// # Arguments
    /// * `handle` - The Tokio `JoinHandle` representing the execution of the CPU-bound task.
    /// * `token` - The exact collaborative cancellation token passed into the task.
    /// * `budget` - The absolute maximum duration the system tolerates for this rebuild.
    pub async fn enforce_latency_budget<T>(
        handle: JoinHandle<T>,
        token: &CastleCancelToken,
        budget: Duration,
    ) -> Result<T, RebuildTimeoutError> {
        // Evaluate the future against the wall-clock budget.
        match tokio::time::timeout(budget, handle).await {
            
            // Case 1: The math task successfully finished within the time budget.
            Ok(Ok(result)) => Ok(result),
            
            // Case 2: The task panicked (e.g., failed a mathematical assertion or OOM).
            // [Artifact Note]: Since `.cargo/config.toml` dictates `panic = unwind`, 
            // the thread safely dropped its local memory. However, we must notify 
            // the DB control loop that the model cannot be used.
            Ok(Err(join_err)) => {
                if join_err.is_panic() {
                    // Propagate panic awareness, but do not crash the Watchdog itself.
                    Err(RebuildTimeoutError::TaskPanicked)
                } else {
                    // Reaching here indicates the Tokio runtime itself is shutting down.
                    Err(RebuildTimeoutError::TaskPanicked)
                }
            }
            
            // Case 3: The task exceeded the budget (Executor Starvation Threat).
            Err(_) => {
                // Step A: Signal the collaborative token.
                // The underlying OS thread running the tight mathematical loop will 
                // intersect this flag at its next mini-block boundary.
                token.cancel();
                
                // Step B: Relinquish control immediately.
                // We purposefully do NOT `handle.await` after cancellation. 
                // The thread is detached. Its RAII Drop handlers will implicitly 
                // free the `LeafSnap` arena memory without requiring Watchdog supervision.
                Err(RebuildTimeoutError::BudgetExceeded)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// Verifies that the Watchdog successfully triggers the token and detaches immediately,
    /// and that the background task eventually respects it.
    #[tokio::test]
    async fn test_watchdog_collaborative_cancellation() {
        let token = CastleCancelToken::new();
        let token_clone = token.clone();
        
        let cleanup_executed = Arc::new(AtomicBool::new(false));
        let cleanup_clone = cleanup_executed.clone();

        // Simulate a dense, CPU-bound mathematical loop
        let handle = tokio::task::spawn_blocking(move || {
            // Simulate processing 1,000,000 chunks
            for _ in 0..1_000_000 {
                // Polling the token at coarse-grained boundaries
                if token_clone.is_cancelled() {
                    // Simulate RAII drop execution upon cancellation
                    cleanup_clone.store(true, Ordering::SeqCst);
                    return Err("Cancelled");
                }
                // Simulate intensive work
                std::thread::sleep(Duration::from_micros(10));
            }
            Ok("Completed")
        });

        // The watchdog enforces an impossibly tight budget (1 millisecond)
        let budget = Duration::from_millis(1);
        let result = WatchdogMonitor::enforce_latency_budget(handle, &token, budget).await;

        // Assertion 1: Watchdog must return instantly with BudgetExceeded
        assert_eq!(result, Err(RebuildTimeoutError::BudgetExceeded));

        // Yield to let the blocking thread reach its next checkpoint and exit
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Assertion 2: The background thread MUST have executed its cleanup block
        assert!(
            cleanup_executed.load(Ordering::SeqCst),
            "FATAL: The detached blocking thread failed to honor the cancellation token!"
        );
    }
}

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

//! CASTLE Recovery: Parallel LSN-Indexed Strict Binary Reduction
//!
//! This module implements the CPU-bound, embarrassingly parallel reconstruction 
//! phase defined in **Section 5.2 (Deterministic Floating-Point Reduction)** 
//! and **Section 5.3 (Crash Recovery Lifecycle)** of the VLDB 2027 paper.
//!
//! # Academic Context & The Work-Stealing Trap
//! Modern parallel frameworks (like Rust's `rayon` or C++'s `TBB`) employ dynamic 
//! work-stealing schedulers. While excellent for load balancing, invoking default 
//! parallel reducers (e.g., `data.par_iter().reduce(...)`) constructs a reduction 
//! tree whose topology is dynamically determined by real-time thread availability 
//! and CPU load. 
//! 
//! Because IEEE 754 floating-point addition is non-associative, a dynamically 
//! shaped reduction tree yields non-deterministic rounding truncations. If a 
//! model is rebuilt under load utilizing 64 cores, and later crash-recovered 
//! using 16 cores, the resulting floating-point `LeafSnap` parameters will diverge 
//! at the bit level, destroying the database's physical routing offsets.
//!
//! # The CASTLE Resolution: Rigid Mathematical Topology
//! To neutralize this, `ReconstructionEngine` eschews dynamic iterators. Instead, 
//! it utilizes `rayon::join` to recursively construct a **Rigid Binary Reduction Tree** 
//! parameterized exclusively by the LSN-ordered array length. 
//! The mathematical topology of the operations is absolutely immutable, achieving 
//! bit-exact idempotency regardless of the hardware thread count.

#![deny(missing_docs)]
#![deny(unsafe_code)] // Pure mathematical scheduling. Zero memory unsafety permitted.

use rayon::join;

use crate::index::plm::PiecewiseLinearModel;
use crate::recovery::deterministic_math::{
    accumulate_chunk, fit_linear_regression, OlsAggregate,
};

/// The physical size limit (number of records) for the base-case execution chunk.
///
/// **Architectural Invariant:** This value must be globally deterministic and 
/// identical between normal execution and crash recovery. It balances the overhead 
/// of recursive `rayon::join` task spawning against optimal CPU cache utilization.
/// Setting this to 256 (B=256 in Section 5.2) ensures optimal L1 cache line fits and fixed reduction topology.
const DETERMINISTIC_CHUNK_SIZE: usize = 256;

/// The central engine responsible for transforming LSN-ordered logical logs 
/// into physical machine learning routing models.
pub struct ReconstructionEngine;

impl ReconstructionEngine {
    /// Rebuilds a `PiecewiseLinearModel` from an array of LSN-ordered logical records.
    ///
    /// # Invocation Context
    /// Called independently for each `LeafCtrl` block during the Phase 3 (CPU-Bound) 
    /// of the ARIES Crash Recovery Lifecycle, or during background asynchronous 
    /// model retraining.
    pub fn rebuild_model(
        data: &[(u64, u64)], 
        hard_max_error: usize
    ) -> PiecewiseLinearModel {
        if data.is_empty() {
            // Safe fallback for entirely empty regions
            return PiecewiseLinearModel::new(0.0, 0.0, hard_max_error);
        }

        // 1. Execute the rigid, parallel binary reduction
        let aggregate_state = Self::strict_binary_reduce(data);

        // 2. Derive the final OLS floating-point parameters
        let (slope, intercept) = fit_linear_regression(&aggregate_state);

        // 3. Construct and return the verifiable immutable model
        PiecewiseLinearModel::new(slope, intercept, hard_max_error)
    }

    /// Recursively splits the data array to form a mathematically rigid binary tree.
    ///
    /// # Concurrency & Determinism Proof
    /// `rayon::join(A, B)` guarantees that `A` and `B` can be executed concurrently 
    /// if idle threads are available. However, regardless of *when* or *where* they 
    /// execute, `OlsAggregate::merge` is always invoked strictly to combine `Left` 
    /// and `Right` in their original LSN-sorted spatial order. 
    fn strict_binary_reduce(data: &[(u64, u64)]) -> OlsAggregate {
        let len = data.len();

        // BASE CASE: The chunk is small enough. Fall back to the sequential, 
        // compiler-firewalled accumulator (Section 5.2).
        if len <= DETERMINISTIC_CHUNK_SIZE {
            return accumulate_chunk(data);
        }

        // RECURSIVE CASE: Split exactly at the deterministic midpoint.
        // Because `len` is identical during replay, `mid` is mathematically constant.
        let mid = len / 2;

        let (left_agg, right_agg) = join(
            || Self::strict_binary_reduce(&data[..mid]),
            || Self::strict_binary_reduce(&data[mid..]),
        );

        // Combine the results, strictly preserving the Left-to-Right topological order.
        OlsAggregate::merge(&left_agg, &right_agg)
    }
}

// -----------------------------------------------------------------------------
// Artifact Verification Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::sync::Arc;

    /// Generates a SHA-256 hash representing the absolute bit-pattern of a PLM.
    fn hash_model(model: &PiecewiseLinearModel) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model.slope.to_le_bytes());
        hasher.update(model.intercept.to_le_bytes());
        hex::encode(hasher.finalize())
    }

    /// Validates that altering the underlying Thread Pool size does absolutely 
    /// NOTHING to the final floating-point bit pattern.
    ///
    /// This test directly proves the "Bit-Exact Idempotency" claims in Section 7.4.
    #[test]
    fn test_thread_count_invariance() {
        // Generate an arbitrary dataset large enough to trigger deep tree recursion
        let data: Vec<(u64, u64)> = (1..=50_000).map(|i| (i * 13, i * 97)).collect();

        // Execution 1: Forced Single-Threaded Execution (Simulating low-resource environment)
        let pool_single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let model_single = pool_single.install(|| {
            ReconstructionEngine::rebuild_model(&data, 256)
        });
        let hash_single = hash_model(&model_single);

        // Execution 2: Highly Concurrent Execution (Simulating enterprise server recovery)
        let pool_multi = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        let model_multi = pool_multi.install(|| {
            ReconstructionEngine::rebuild_model(&data, 256)
        });
        let hash_multi = hash_model(&model_multi);

        // THE INVARIANT:
        // If we had used `.par_iter().reduce()`, this assertion would probabilistically 
        // fail because the 16-thread pool would steal work dynamically, altering the 
        // floating-point summation order compared to the 1-thread pool.
        // Our strict `join` topology mathematically forces them to be identical.
        assert_eq!(
            hash_single, hash_multi,
            "FATAL: Thread count variance altered the IEEE 754 bit-pattern!"
        );
    }
}

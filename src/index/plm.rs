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

//! CASTLE Core: Piecewise Linear Model (PLM) Inference Engine
//!
//! This module implements the mathematical routing logic described in **Section 3.3 
//! (The Dual-Track Execution Model: Hot Track)**. It maps logical keys to physical 
//! array offsets via lightweight linear regression models.
//!
//! # Academic Integrity & Compiler Contracts
//! 1. **Determinism over Peak Flops (Section 5.2):** Standard ML inference heavily relies 
//!    on Fused Multiply-Add (FMA) instructions for performance. However, because FMA alters 
//!    IEEE 754 rounding behavior, its use is strictly forbidden in CASTLE to ensure 
//!    bit-exact idempotency during ARIES crash recovery. The arithmetic in this module 
//!    is explicitly compiled with `-C target-feature=-fma`.
//! 2. **SIMD Vectorization:** To reclaim performance without FMA, the `predict_batch` 
//!    API is designed strictly without control-flow divergence (branchless clamping), 
//!    allowing the LLVM backend to safely auto-vectorize the loop into AVX2/AVX-512 
//!    instructions where supported.
//! 3. **Memory Layout:** `PiecewiseLinearModel` is defined with `#[repr(C)]` to guarantee 
//!    predictable memory alignment, facilitating seamless zero-copy evaluation across 
//!    the FFI boundary if needed.

#![deny(missing_docs)]

/// A single segment of the Piecewise Linear Model (PLM).
/// 
/// Translates a logical `key` (converted to f64) into a predicted physical offset 
/// via the linear equation: `offset = (key * slope) + intercept`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PiecewiseLinearModel {
    /// The mathematical slope of the regression line.
    pub slope: f64,
    /// The mathematical intercept of the regression line.
    pub intercept: f64,
    /// The maximum observed error during training, used to cap the `DegradedScan`.
    /// Corresponding to `HARD_MAX_ERROR` limits.
    pub max_error: usize,
}

impl PiecewiseLinearModel {
    /// Constructs a new PLM segment.
    ///
    /// # Safety & Validation
    /// Validates that floating-point parameters are finite. NaNs or Infinities 
    /// injected via corrupted models will immediately poison the system and must 
    /// be caught at instantiation.
    pub fn new(slope: f64, intercept: f64, max_error: usize) -> Self {
        assert!(slope.is_finite() && intercept.is_finite(), 
            "FATAL: Attempted to initialize PLM with NaN or Infinity. Model is poisoned.");
            
        Self { slope, intercept, max_error }
    }

    /// Evaluates the model to predict the physical offset for a single key.
    ///
    /// # Instruction Level Details
    /// Executes separate `FMUL` and `FADD` instructions (guaranteed by `.cargo/config.toml`).
    /// The result is cast to an integer, with negative offsets safely clamped to 0.
    #[inline(always)]
    pub fn evaluate(&self, key: f64) -> usize {
        // [VLDB Determinism Note]: No FMA used here.
        // The operation strictly adheres to (key * slope) + intercept with two distinct rounding steps.
        let raw_pos = (key * self.slope) + self.intercept;
        
        // Branchless clamping for negative predictions.
        // float-to-int casting natively saturates, but handling negative explicitely 
        // prevents undefined behavior or massive wraparound values in unsigned `usize`.
        if raw_pos <= 0.0 {
            0
        } else {
            raw_pos as usize
        }
    }

    /// Evaluates a batch of keys against the model, writing to a pre-allocated offset array.
    ///
    /// # The "Hot Track" SIMD Optimization
    /// Modern analytical queries (e.g., TPC-H scans) process batches of keys. 
    /// This function is written specifically to trigger LLVM's loop vectorizer.
    /// By ensuring there are no hidden panic branches or early-returns inside the loop, 
    /// the compiler translates this directly into packed vector instructions (e.g., `vfmadd231pd` 
    /// is forbidden, but packed `vmulpd` and `vaddpd` will be emitted).
    #[inline]
    pub fn predict_batch(&self, keys: &[f64], out_offsets: &mut [usize]) {
        assert_eq!(keys.len(), out_offsets.len(), "Batch arrays must be equal in length");

        let s = self.slope;
        let i = self.intercept;

        // The compiler recognizes this pattern and applies Loop Unrolling and Auto-Vectorization.
        for idx in 0..keys.len() {
            // We use `unsafe` get_unchecked here strictly for the hot path to eliminate 
            // array bounds checking within the loop, fulfilling the "Compute-bound" promise.
            unsafe {
                let k = *keys.get_unchecked(idx);
                let raw_pos = (k * s) + i;
                
                // Branchless clamping using floating point max
                // f64::max is compiled to a single SIMD `vmaxpd` instruction on x86.
                let clamped = raw_pos.max(0.0);
                
                *out_offsets.get_unchecked_mut(idx) = clamped as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plm_clamping() {
        let model = PiecewiseLinearModel::new(2.0, -100.0, 10);
        
        // Input that yields negative result: (10 * 2) - 100 = -80
        // Must safely clamp to 0 instead of causing integer underflow.
        assert_eq!(model.evaluate(10.0), 0);
        
        // Input yielding positive: (60 * 2) - 100 = 20
        assert_eq!(model.evaluate(60.0), 20);
    }

    #[test]
    fn test_predict_batch_consistency() {
        let model = PiecewiseLinearModel::new(1.5, 10.0, 5);
        let keys = vec![0.0, 10.0, -20.0, 100.0];
        let mut offsets = vec![0; 4];

        model.predict_batch(&keys, &mut offsets);

        // 0 * 1.5 + 10 = 10
        assert_eq!(offsets[0], 10);
        // 10 * 1.5 + 10 = 25
        assert_eq!(offsets[1], 25);
        // -20 * 1.5 + 10 = -20 -> clamped to 0
        assert_eq!(offsets[2], 0);
        // 100 * 1.5 + 10 = 160
        assert_eq!(offsets[3], 160);
    }

    #[test]
    #[should_panic(expected = "FATAL: Attempted to initialize PLM with NaN")]
    fn test_nan_poisoning() {
        // Simulates a catastrophic math failure during background retraining
        let _ = PiecewiseLinearModel::new(f64::NAN, 1.0, 0);
    }
}

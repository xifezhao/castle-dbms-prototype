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

//! CASTLE Recovery: Stateless IEEE 754 Deterministic Math Core
//!
//! This module implements the mathematical building blocks required by **Section 5.2 
//! (Deterministic Floating-Point Reduction)** of the VLDB 2027 paper.
//!
//! # Academic Context & The Compiler Firewall
//! As established in the paper, floating-point addition is non-associative: 
//! `(A + B) + C != A + (B + C)`. Modern compilers (like LLVM) aggressively optimize loops 
//! using vectorization and instruction reordering. If a background rebuilder processes 
//! data in chunks of 1024, but the crash recovery loop processes them in chunks of 2048, 
//! the intermediate rounding sequence will differ, breaking bit-exact idempotency.
//!
//! To completely neutralize this hazard, this module operates as a "Compiler Firewall".
//! 1. **Stateless Aggregation:** Data is reduced into an intermediate `OlsAggregate` state.
//! 2. **Fixed-Order Execution:** The `accumulate_chunk` and `merge` functions strictly 
//!    enforce Left-to-Right evaluation.
//! 3. **FMA Isolation:** This code relies on the `-C target-feature=-fma` compiler 
//!    contract established in `.cargo/config.toml` to guarantee two discrete rounding 
//!    steps (MUL then ADD) rather than hardware-dependent fused rounding.

#![deny(missing_docs)]
#![deny(unsafe_code)] // Pure mathematical operations; zero memory unsafety allowed.

use crate::index::leaf_snap::LeafSnap;

// -----------------------------------------------------------------------------
// 1. The Intermediate Reduction State
// -----------------------------------------------------------------------------

/// The strictly defined intermediate state for Ordinary Least Squares (OLS) regression.
///
/// Unlike naive implementations that compute the slope iteratively (which exacerbates 
/// floating-point drift), CASTLE isolates the summation of components. These components 
/// can be merged via binary reduction trees with zero algebraic ambiguity.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct OlsAggregate {
    /// Count of elements ($n$)
    pub count: f64,
    /// Sum of X ($\sum x$)
    pub sum_x: f64,
    /// Sum of Y ($\sum y$)
    pub sum_y: f64,
    /// Sum of X squared ($\sum x^2$)
    pub sum_xx: f64,
    /// Sum of X * Y ($\sum xy$)
    pub sum_xy: f64,
}

impl OlsAggregate {
    /// Deterministically merges two aggregates (Left and Right).
    ///
    /// # The Asymmetry of IEEE 754
    /// Because `Left + Right != Right + Left` at the bit-level under certain catastrophic 
    /// cancellation scenarios, this function strictly enforces that the topologically 
    /// "earlier" (Left/Lower LSN) chunk is ALWAYS the left operand in the addition.
    #[inline(always)]
    pub fn merge(left: &Self, right: &Self) -> Self {
        Self {
            // [Artifact Note]: The left-to-right operand order here is NOT arbitrary.
            // It is an architectural invariant required to ensure the hash outputs match 
            // the `bit_exact_hash.rs` validation.
            count: left.count + right.count,
            sum_x: left.sum_x + right.sum_x,
            sum_y: left.sum_y + right.sum_y,
            sum_xx: left.sum_xx + right.sum_xx,
            sum_xy: left.sum_xy + right.sum_xy,
        }
    }
}

// -----------------------------------------------------------------------------
// 2. The Core Mathematical Operations
// -----------------------------------------------------------------------------

/// Linearly accumulates a discrete, LSN-ordered chunk of data into an aggregate state.
///
/// # Compiler Optimization Defeat
/// We use `#[inline(never)]` deliberately. If this function is inlined into the parallel 
/// binary tree reduction loop, LLVM might attempt cross-loop auto-vectorization, effectively 
/// rewriting the accumulation order. By preventing inlining, we force the compiler to 
/// execute the loop sequentially within the exact chunk boundaries provided.
#[inline(never)]
pub fn accumulate_chunk(data: &[(u64, u64)]) -> OlsAggregate {
    let mut agg = OlsAggregate::default();

    for &(k, v) in data {
        let x = k as f64;
        let y = v as f64;

        agg.count += 1.0;
        agg.sum_x += x;
        agg.sum_y += y;
        agg.sum_xx += x * x;
        agg.sum_xy += x * y;
    }

    agg
}

/// Derives the final Piecewise Linear Model (PLM) parameters from the global aggregate.
///
/// Uses the standard OLS closed-form solution:
/// - $Slope = \frac{n(\sum xy) - (\sum x)(\sum y)}{n(\sum x^2) - (\sum x)^2}$
/// - $Intercept = \frac{\sum y - Slope \times \sum x}{n}$
///
/// # Edge Case Handling
/// Evaluates colinearity (e.g., all X values are identical) to prevent `NaN` or `Infinity` 
/// poisoning (as guarded against in `plm.rs`).
#[inline]
pub fn fit_linear_regression(agg: &OlsAggregate) -> (f64, f64) {
    // If there is no data or a single point, return a flat line.
    if agg.count <= 1.0 {
        return if agg.count == 1.0 { (0.0, agg.sum_y) } else { (0.0, 0.0) };
    }

    let numerator = (agg.count * agg.sum_xy) - (agg.sum_x * agg.sum_y);
    let denominator = (agg.count * agg.sum_xx) - (agg.sum_x * agg.sum_x);

    // Defense against division by zero and catastrophic cancellation (Relative Epsilon Scaling)
    let scale = (agg.count * agg.sum_xx).abs();
    if scale <= 0.0 || denominator.abs() <= scale * f64::EPSILON * 4.0 || denominator.abs() < f64::EPSILON {
        // Fallback: Model is a flat line passing through the average Y
        let average_y = agg.sum_y / agg.count;
        return (0.0, average_y);
    }

    let slope = numerator / denominator;
    let intercept = (agg.sum_y - (slope * agg.sum_x)) / agg.count;

    debug_assert!(slope.is_finite() && intercept.is_finite(), "OLS result not finite");
    (slope, intercept)
}

// -----------------------------------------------------------------------------
// 3. Artifact Verification Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_math_correctness() {
        let data = vec![(1, 2), (2, 4), (3, 6), (4, 8)];
        let agg = accumulate_chunk(&data);
        
        let (slope, intercept) = fit_linear_regression(&agg);
        
        // Exact mathematical expectation for y = 2x
        assert!((slope - 2.0).abs() < f64::EPSILON);
        assert!((intercept - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_singularity_defense() {
        // All X values are identical (Vertical line). Denominator will be 0.
        let data = vec![(5, 10), (5, 20), (5, 30)];
        let agg = accumulate_chunk(&data);
        
        let (slope, intercept) = fit_linear_regression(&agg);
        
        // System must NOT panic with NaN or Infinity. Must degrade safely.
        assert_eq!(slope, 0.0);
        assert_eq!(intercept, 20.0); // Average of Y: (10+20+30)/3
    }
}

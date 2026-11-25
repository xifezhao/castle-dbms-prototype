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

//! CASTLE Crash Recovery & WAL Subsystem
//!
//! This module implements the durability and deterministic reconstruction protocols 
//! described in **Section 5 (Deterministic Recovery over WAL)** of the VLDB 2027 paper.
//!
//! # Academic Context & The Durability Dichotomy
//! Integrating machine learning models into DBMS internals presents a fundamental conflict 
//! between I/O efficiency and deterministic replay:
//! 1. **The I/O Amplification Hazard:** Physical logging of 64-bit floating-point weights 
//!    and cascading array shifts inflates Write-Ahead Log (WAL) volume by orders of magnitude 
//!    (e.g., 47.5 GB vs 1.2 GB, Section 7.4), severely bottlenecking the transaction commit path.
//! 2. **The IEEE 754 Associativity Hazard:** Relying purely on logical logging introduces 
//!    non-determinism. Because floating-point addition is non-associative, standard 
//!    concurrent aggregations yield bit-level divergences during recovery, destroying 
//!    the physical offset predictions of the index.
//!
//! # The CASTLE Resolution: ARIES-Compliant Determinism
//! This module resolves these hazards through a structured two-phase recovery architecture 
//! that seamlessly piggybacks on the traditional ARIES protocol:
//!
//! 1. **LSN-Driven Logical Logging (`logical_wal::*`)**  
//!    *Maps to Section 5.1.*  
//!    We strictly log logical operations (e.g., `INSERT`) bounded by globally monotonic 
//!    Log Sequence Numbers (LSNs). During the ARIES Redo pass, the engine simply appends 
//!    records to memory. **Zero floating-point calculations occur during I/O replay**, 
//!    guaranteeing that CASTLE's Redo phase operates at the exact same I/O limit as a B+Tree.
//!
//! 2. **Deterministic Binary Reduction (`binary_reduction::*` & `deterministic_math::*`)**  
//!    *Maps to Section 5.2.*  
//!    After logical consistency is restored, model reconstruction is triggered as an 
//!    embarrassingly parallel, CPU-bound phase. By constraining the math to a rigid, 
//!    LSN-indexed binary reduction tree, we mathematically neutralize the non-associativity 
//!    of IEEE 754 arithmetic. This guarantees bit-exact idempotency (validated by SHA-256 
//!    hashing in `tests/recovery_sim/bit_exact_hash.rs`).

#![deny(missing_docs)]
#![deny(unsafe_code)] // Recovery logic is strictly mathematical and data-oriented; no unsafe required.

pub mod binary_reduction;
pub mod deterministic_math;
pub mod logical_wal;

// -----------------------------------------------------------------------------
// Public API Re-exports for DBMS Kernel Integration
// -----------------------------------------------------------------------------

pub use binary_reduction::ReconstructionEngine;
pub use logical_wal::{LogicalWalRecord, LsnTracker};

// [Artifact Note]:
// The rest of the DBMS kernel submits logical operations to `LogicalWalRecord`.
// Upon a crash and restart, the transaction manager hands off the replayed LSN-ordered 
// chunks to the `ReconstructionEngine`, which then dispatches parallel workers via Rayon 
// to rebuild the `LeafSnap` structures deterministically.

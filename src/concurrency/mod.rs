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

//! CASTLE Concurrency & Lifecycle Management
//!
//! This module forms the heart of CASTLE's non-blocking, async-native execution model. 
//! It provides the synchronization primitives and memory reclamation infrastructure 
//! required to safely execute the Pure Copy-on-Write (CoW) paradigm in a modern 
//! cooperative multitasking runtime (e.g., Tokio, io_uring).
//!
//! # Academic Context & VLDB Paper Mapping
//! Standard relational databases rely on Thread-Local Storage (TLS) for RCU implementations 
//! and fine-grained latches for tree structural modifications. As detailed in the paper, 
//! applying these legacy primitives to learned indexes within asynchronous environments 
//! triggers catastrophic concurrency collapse, StoreLoad anomalies, and coroutine memory leaks.
//!
//! This module resolves these engineering impedances via three core sub-components:
//!
//! 1. **Task-Local RCU Epochs (`rcu_epoch::*`)**  
//!    *Maps to Section 6.2 (Dealing with Long-Running Transactions).*  
//!    Replaces unsafe TLS with coroutine-context-bound Epoch pinning. It guarantees 
//!    that a reader's immutable view of the `LeafSnap` remains mathematically consistent 
//!    even if the task yields (`.await`) and is migrated across different OS threads 
//!    by the work-stealing scheduler.
//!
//! 2. **Global Watermark & LRT Circuit Breaker (`watermark::*`)**  
//!    *Maps to Section 6.2.*  
//!    A centralized tracker that advances the global safe epoch. To prevent Long-Running 
//!    Transactions (LRTs) from halting garbage collection and causing Out-Of-Memory (OOM) 
//!    cascades, it features a time-bounded circuit breaker that clones isolating snapshots 
//!    for offending tasks and forcefully advances the system watermark.
//!
//! 3. **The Quiescence Protocol (`write_intent::*`)**  
//!    *Maps to Section 4.3 (Memory Ordering Guarantees).*  
//!    Provides the `WriteIntentGuard`, an RAII structure that utilizes explicit `SeqCst` 
//!    memory barriers. This is the physical implementation of the defense against Time-of-Check 
//!    to Time-of-Use (TOCTOU) Store Buffer races between foreground lock-free appenders 
//!    and background rebuilders.
//!
//! # Formal Verification Guarantee
//! All state-transitions governing pointer reclamation and write intents within this module 
//! have been exhaustively permuted against the C11 weak memory model using `loom`. 
//! Refer to `tests/loom/` for the definitive mathematical proofs of these safety contracts.

#![deny(missing_docs)]
#![allow(unsafe_code)] // Unsafe pointer dereferencing for RCU is tightly isolated within `rcu_epoch.rs`.

pub mod rcu_epoch;
pub mod watermark;
pub mod write_intent;

// -----------------------------------------------------------------------------
// Public API Re-exports for DBMS Kernel Integration
// -----------------------------------------------------------------------------

pub use rcu_epoch::{EpochGuard, retire_snapshot};
pub use watermark::{GlobalWatermark, LRTBreaker};
pub use write_intent::WriteIntentGuard;

// [Artifact Note]: 
// By re-exporting these primitives at the module root, we enforce a strict, 
// opaque API boundary. The rest of the DBMS kernel (e.g., `src/index/`) interacts 
// with concurrency controls purely via these high-level RAII guards, entirely 
// abstracting away the complex atomic manipulations and memory fences.

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

//! CASTLE Runtime Orchestration: Async-Native Rebuilding & Causal Scheduling
//!
//! This module implements the compute-isolation and task lifecycle management protocols 
//! described in **Section 6.1 (Collaborative Cancellation in Dedicated Pools)** of the 
//! DASFAA 2027 paper.
//!
//! # Academic Context: The Asynchronous Runtime Trap
//! As argued in **Section 2.3**, treating learned index retraining as a generic async task 
//! or offloading it blindly to Tokio's `spawn_blocking` pool creates catastrophic 
//! systemic vulnerabilities. Heavy matrix decompositions are opaque to the OS scheduler; 
//! they cannot be preempted. During a massive data skew event (e.g., the Skew Burst Microbenchmark), 
//! an influx of these tasks will permanently exhaust the blocking thread pool, starving 
//! legitimate query execution and pushing P99 tail latencies into the hundreds of milliseconds.
//!
//! # The CASTLE Resolution
//! This module replaces standard task offloading with a strictly bounded, defense-in-depth 
//! orchestration subsystem comprising three pillars:
//!
//! 1. **Dedicated Rebuild Pool (`rebuild_pool::*`)**  
//!    Physically partitions index maintenance threads away from the main DBMS blocking pool 
//!    (which must remain free for WAL flushing and page fetching). Configured with a hard 
//!    concurrency cap to strictly limit the "blast radius" of index retraining storms.
//!
//! 2. **Collaborative Cancellation (`cancel_token::*`)**  
//!    Because forced thread termination in C++/Rust causes undefined behavior and memory leaks, 
//!    CASTLE utilizes cooperative tokens. Compute-heavy mathematical loops (e.g., PLM fitting) 
//!    are instrumented to poll this token at coarse-grained mini-block boundaries.
//!
//! 3. **The Watchdog Monitor (`watchdog::*`)**  
//!    An independent timing mechanism wrapping every scheduled rebuild task. If a matrix 
//!    reduction stalls beyond its latency budget, the watchdog revokes the cancellation token, 
//!    forcing the target thread to gracefully drop its intermediate allocations (`LeafSnap` arenas) 
//!    and yield the OS thread back to the pool.

#![deny(missing_docs)]
#![deny(unsafe_code)] // Runtime orchestration relies purely on safe concurrency abstractions.

pub mod cancel_token;
pub mod rebuild_pool;
pub mod watchdog;

// -----------------------------------------------------------------------------
// Public API Re-exports for DBMS Kernel Integration
// -----------------------------------------------------------------------------

pub use cancel_token::CastleCancelToken;
pub use rebuild_pool::DedicatedRebuildPool;
pub use watchdog::{WatchdogMonitor, RebuildTimeoutError};

// [Artifact Note]: 
// By encapsulating the runtime abstractions here, the core index logic in `src/index/` 
// remains completely agnostic to the specific async executor being utilized (e.g., Tokio, 
// async-std, or a custom Seastar-like runtime). It only interacts with the standardized 
// `DedicatedRebuildPool` interface.

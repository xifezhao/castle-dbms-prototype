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

//! CASTLE Benchmark Suite: Index-Stressed TPC-C
//!
//! This module implements the rigorous benchmarking harness for **Section 7.1 and 7.2** 
//! of the DASFAA 2027 paper.
//! 
//! # Methodology Note: Bypassing the Concurrency Funnel
//! Standard TPC-C implementations are fundamentally unsuited for evaluating raw index 
//! scalability. Heavyweight Two-Phase Locking (2PL) or complex MVCC conflict resolution 
//! mechanisms act as a "concurrency funnel", throttling operations long before the 
//! underlying data structure reaches its multi-core scaling limits. 
//! 
//! To expose the raw structural vulnerabilities of in-place learned indexes (e.g., ALEX, LIPP) 
//! and to stress-test CASTLE's Pure CoW architecture, we deploy an **Index-Stressed** variant.
//! This harness strips away the lock manager, isolating the physical index layer. 
//! We focus exclusively on the two most index-intensive transactions:
//! - **New-Order (Write-Heavy):** Simulates massive concurrent appends to `ORDER` and `ORDER_LINE`.
//! - **Stock-Level (Read-Heavy):** Simulates localized, high-frequency scans/lookups.

#![deny(missing_docs)]

use std::sync::Arc;
use std::time::Instant;
use tokio::task::JoinHandle;
use hdrhistogram::Histogram;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;

use crate::ffi_wrapper::AsyncIndex; // The unified async facade from Section 7.1

/// Configuration for the Index-Stressed TPC-C Workload.
#[derive(Debug, Clone)]
pub struct TpccStressConfig {
    /// Number of simulated Warehouses (W_ID). Dictates the logical partitioning.
    pub num_warehouses: u64,
    /// Number of transactions to execute per task.
    pub txns_per_task: u64,
    /// Number of concurrent Tokio asynchronous tasks.
    pub num_tasks: usize,
    /// Ratio of New-Order vs Stock-Level transactions (e.g., 0.45 means 45% New-Order).
    pub new_order_ratio: f64,
}

/// The aggregate results for the VLDB evaluation tables.
#[derive(Debug)]
pub struct TpccBenchmarkResult {
    /// Total throughput in Transactions Per Second (TPS).
    pub throughput_tps: f64,
    /// P99 latency in microseconds.
    pub p99_latency_us: u64,
}

// -----------------------------------------------------------------------------
// TPC-C Key Encoding Scheme
// -----------------------------------------------------------------------------
// To support unified 64-bit integer keys across all C++ baselines and CASTLE, 
// we encode the composite keys using bit-shifting. This maintains the physical 
// locality and ordering semantics inherent to TPC-C tables.

#[inline(always)]
fn encode_order_line_key(w_id: u64, d_id: u64, o_id: u64, ol_number: u64) -> u64 {
    // 64-bit Key Layout:
    // [ W_ID: 16 bits ] | [ D_ID: 8 bits ] | [ O_ID: 32 bits ] | [ OL_NUMBER: 8 bits ]
    (w_id << 48) | ((d_id & 0xFF) << 40) | ((o_id & 0xFFFFFFFF) << 8) | (ol_number & 0xFF)
}

#[inline(always)]
fn encode_stock_key(w_id: u64, i_id: u64) -> u64 {
    // 64-bit Key Layout:
    // [ W_ID: 16 bits ] | [ I_ID: 48 bits ]
    (w_id << 48) | (i_id & 0xFFFFFFFFFFFF)
}

/// Executes the Index-Stressed TPC-C benchmark.
pub async fn run_tpcc_stress<I>(index: Arc<I>, config: TpccStressConfig) -> TpccBenchmarkResult 
where 
    I: AsyncIndex + Send + Sync + 'static 
{
    let mut handles: Vec<JoinHandle<Histogram<u64>>> = Vec::with_capacity(config.num_tasks);
    let start_time = Instant::now();

    for task_id in 0..config.num_tasks {
        let index_clone = index.clone();
        let cfg = config.clone();

        handles.push(tokio::spawn(async move {
            // [Integrity Guarantee]: Task-local RNG eliminates test-harness bottleneck.
            let mut rng = SmallRng::seed_from_u64(task_id as u64 ^ 0xDEADBEEF);
            
            let mut local_hist = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3)
                .expect("Failed to initialize HDR Histogram");

            // Local state to simulate auto-incrementing Order IDs per district
            // Array of 10 districts per warehouse.
            let mut local_o_ids = vec![3000_u64; 10]; 

            for _ in 0..cfg.txns_per_task {
                let w_id = rng.gen_range(1..=cfg.num_warehouses);
                let d_id = rng.gen_range(1..=10);
                
                let is_new_order = rng.gen_bool(cfg.new_order_ratio);
                let txn_start = Instant::now();

                if is_new_order {
                    // ---------------------------------------------------------
                    // Simulated New-Order Transaction (Write-Heavy)
                    // ---------------------------------------------------------
                    // Simulates inserting an order and 5 to 15 order lines.
                    // This generates localized write bursts that trigger intense 
                    // structural modifications (SMOs) in learned indexes.
                    
                    let o_id = local_o_ids[(d_id - 1) as usize];
                    local_o_ids[(d_id - 1) as usize] += 1;

                    let ol_cnt = rng.gen_range(5..=15);
                    for ol_number in 1..=ol_cnt {
                        let i_id = rng.gen_range(1..=100_000);
                        
                        // Point query to read stock (Read)
                        let stock_key = encode_stock_key(w_id, i_id);
                        let _ = index_clone.read(stock_key).await;

                        // Insert Order-Line (Write)
                        let ol_key = encode_order_line_key(w_id, d_id, o_id, ol_number);
                        // Mock payload: simulated pointer or embedded data
                        let payload = ol_key ^ 0xABCD; 
                        index_clone.insert(ol_key, payload).await;
                    }
                } else {
                    // ---------------------------------------------------------
                    // Simulated Stock-Level Transaction (Read-Heavy / Scan)
                    // ---------------------------------------------------------
                    // Examines the items from the last 20 orders.
                    // Note: To maintain the strict FFI interface compatibility with `lib.rs` 
                    // (which only exposes point reads to accommodate diverse C++ APIs uniformly),
                    // we implement the range scan logic via localized, contiguous point lookups.
                    // This subjects the PLM inference path to heavy sequential load.
                    
                    let current_o_id = local_o_ids[(d_id - 1) as usize];
                    let start_o_id = if current_o_id > 20 { current_o_id - 20 } else { 1 };

                    for o_id in start_o_id..current_o_id {
                        // Scan up to a max of 15 order lines per order
                        for ol_number in 1..=15 {
                            let ol_key = encode_order_line_key(w_id, d_id, o_id, ol_number);
                            let _ = index_clone.read(ol_key).await;
                            // In standard TPC-C, we would use the retrieved I_ID to 
                            // check the stock table. Mocked out for raw index stress.
                        }
                    }
                }

                let elapsed_us = txn_start.elapsed().as_micros() as u64;
                let _ = local_hist.record(elapsed_us);
            }
            local_hist
        }));
    }

    // -------------------------------------------------------------------------
    // Phase 2: Aggregation
    // -------------------------------------------------------------------------
    let mut global_hist = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3).unwrap();
    
    for handle in handles {
        global_hist.add(handle.await.unwrap()).unwrap();
    }

    let total_duration = start_time.elapsed();
    let throughput_tps = ((config.num_tasks as u64) * config.txns_per_task) as f64 / total_duration.as_secs_f64();

    TpccBenchmarkResult {
        throughput_tps,
        p99_latency_us: global_hist.value_at_quantile(0.99),
    }
}

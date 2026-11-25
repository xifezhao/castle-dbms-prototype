# CASTLE: Pure CoW and Deterministic Recovery for Learned Indexes

Rust + C++17 prototype accompanying the paper *"CASTLE: Bridging the Engineering Impedance of Learned Indexes via Pure CoW and Deterministic Recovery"*.

## Components
- `src/index/`: State bifurcation (`LeafCtrl` / `LeafSnap`), lazy `OnceLock` 256-slot double buffers (`OverflowBuffer`), and deterministic Piecewise Linear Model (`PiecewiseLinearModel`).
- `src/concurrency/`: `!Send` thread-local Epoch-Based Reclamation (`rcu_epoch.rs`) and phase-indexed `SeqCst` write-intent guards (`write_intent.rs`).
- `src/recovery/`: 24-byte logical WAL with 1-bit LSN record type tag (`logical_wal.rs`) and fixed-topology binary-tree OLS reduction (`binary_reduction.rs`, `deterministic_math.rs`).
- `src/runtime/`: Physically isolated `DedicatedRebuildPool` and cooperative `CastleCancelToken`.
- `baselines/ffi_wrapper/`: CXX FFI bridge for `microsoft/ALEX` (global & 2,048-partitioned `ALEX-Part`), `Jiacheng-WU/lipp`, cache-aligned `B+-Tree OLC`, and `Delta-Learned`.
- `tests/loom/`: 7 bounded model-checking tests (`loom 0.7`, preemption bound 3).
- `src/bin/castle_bench.rs`: Unified multi-threaded, multi-seed benchmark runner (YCSB-A/C/E, TPC-C, Skew Burst, Recovery, Determinism Ablation).

## Build & Verify
```bash
# 1. Bounded model checking (7 tests, preemption bound 3)
RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=3 cargo test \
  --test test_toctou_race --test test_atomic_swap --test test_rcu_lifetimes

# 2. Production benchmark build (FMA disabled)
RUSTFLAGS="-C target-feature=-fma" cargo build --release --bin castle_bench
```

# CASTLE: Bridging the Engineering Impedance of Learned Indexes via Pure CoW and Deterministic Recovery

[![Artifact CI](https://github.com/xifezhao/castle-dbms-prototype/actions/workflows/ci.yml/badge.svg)](https://github.com/xifezhao/castle-dbms-prototype/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Official prototype implementation, bounded model-checking suite (`loom`), C++17 baseline FFI bridge, and verified 16-physical-core (32-vCPU) 5-seed experimental artifact accompanying:

> **CASTLE: Bridging the Engineering Impedance of Learned Indexes via Pure CoW and Deterministic Recovery**  
> *Status: Under Double-Blind Review at DASFAA 2027 (Research Track).*  
> *Release Tag:* `v1.0.0-dasfaa2027` — see [AUTHORS.md](AUTHORS.md) for double-blind stewardship details.

---

## 1. Repository Structure

| Path | Description |
|---|---|
| `src/index/` | Physical state bifurcation (`LeafCtrl` / `LeafSnap`), lazy `OnceLock` 256-slot double buffers (`OverflowBuffer`), deterministic PLM (`PiecewiseLinearModel`), and `CastleIndex`. |
| `src/concurrency/` | `!Send` thread-local Epoch-Based Reclamation (`rcu_epoch.rs`) and phase-indexed `SeqCst` write-intent guards (`write_intent.rs`). |
| `src/recovery/` | 24-byte logical WAL with 1-bit MSB LSN record type tag (`logical_wal.rs`) and fixed-topology binary-tree OLS reduction (`binary_reduction.rs`, `deterministic_math.rs`). |
| `src/runtime/` | Physically isolated `DedicatedRebuildPool` and cooperative `CastleCancelToken`. |
| `baselines/ffi_wrapper/` | CXX FFI bridge for official `microsoft/ALEX` (global & 2,048-partitioned `ALEX-Part`), `Jiacheng-WU/lipp`, cache-aligned `B+-Tree OLC`, and `Delta-Learned`. |
| `tests/loom/` | 7 bounded model-checking tests (`loom 0.7`, preemption bound 3), including negative controls, `CapacityExhausted` tail rollback, and EBR buffer-reuse grace period. |
| `tests/recovery_sim/` | WAL serialization/deserialization and bit-exact crash-recovery simulation tests. |
| `scripts/fetch_third_party.sh` | Fetches official `microsoft/ALEX` and `Jiacheng-WU/lipp` C++ headers into `baselines/third_party/`. |
| `scripts/analyze_and_plot.py` | Aggregates 5-seed CSVs and regenerates paper Figures 2–4. |
| `scripts/plot_architecture.py` | Generates the vector architecture diagram (Figure 1). |
| `results/raw_ecs32/` | Complete 16-physical-core / 32-vCPU 5-seed raw CSVs, Loom logs, recovery logs, and hardware specs (`lscpu`, `free`, compiler versions). |
| `results/figures/` | Published vector PDF figures (`fig_architecture.pdf`, `fig1_throughput_scalability.pdf`, `fig2_skew_burst_p99_latency.pdf`, `fig3_tpcc_rangescan.pdf`). |
| `results/SHA256SUMS.txt` | SHA-256 integrity manifest covering all published raw data and figure files. |

---

## 2. Quickstart & Reproduction

### Prerequisites
- Linux x86_64 (tested on Ubuntu 24.04 LTS, Linux 6.8.0)
- Rust stable toolchain (`rustc >= 1.80`, tested on `rustc 1.100.0`)
- C++17 toolchain (`g++ 13.3.0`, `cmake >= 3.15`, `git`)
- Python 3 + `matplotlib` (optional, for regenerating PDF figures)

### Step 1: Fetch Third-Party C++ Baselines (`ALEX` & `LIPP`)
```bash
bash scripts/fetch_third_party.sh
```
This clones official `microsoft/ALEX` and `Jiacheng-WU/lipp` into `baselines/third_party/{ALEX,LIPP}`. You can also override the path via `-DCASTLE_THIRD_PARTY_DIR=/path/to/third_party`.

### Step 2: Run Bounded Model Checking (`loom`, 7/7 Tests)
```bash
RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=3 cargo test \
  --test test_toctou_race \
  --test test_atomic_swap \
  --test test_rcu_lifetimes -- --nocapture
```

### Step 3: Run Crash Recovery & Determinism Unit Tests
```bash
RUSTFLAGS="-C target-feature=-fma" cargo test \
  --test bit_exact_hash \
  --test crash_simulator -- --nocapture
```

### Step 4: Build & Run Production Benchmark (`-C target-feature=-fma`)
```bash
RUSTFLAGS="-C target-feature=-fma" cargo build --release --bin castle_bench

# Example: 16-core YCSB-A on 500K keys (2,048 leaves)
./target/release/castle_bench \
  --workload ycsb_a --index CASTLE --threads 16 \
  --records 500000 --ops 50000 --leaves 2048 --seed 42 \
  --out /tmp/ycsb_a_castle_t16.csv

# Example: 1M-record disk-backed redo recovery (100 trials across 1, 2, 4, 8, 16 threads)
./target/release/castle_bench \
  --workload recovery_bench --index CASTLE --records 1000000 \
  --out /tmp/recovery.csv
```

---

## 3. Verifying Published Experimental Results

All tables and figures in the DASFAA 2027 manuscript are derived directly from the 5-seed (`42, 1337, 2027, 7777, 9999`) raw CSVs in `results/raw_ecs32/` (collected on an Aliyun `ecs.u1-c1m1.8xlarge` Intel Xeon Platinum server with 16 physical cores / 32 hardware threads, 33 MiB L3, 32 GiB RAM).

Verify file integrity and inspect aggregated statistics:
```bash
(cd results && sha256sum -c SHA256SUMS.txt)
python3 scripts/analyze_and_plot.py
```

---

## 4. Citation

See [CITATION.cff](CITATION.cff) or use the BibTeX entry below (author list will be unblinded upon completion of DASFAA 2027 double-blind review):

```bibtex
@inproceedings{castle2027dasfaa,
  title     = {{CASTLE}: Bridging the Engineering Impedance of Learned Indexes via Pure {CoW} and Deterministic Recovery},
  author    = {{CASTLE Authors (Anonymized for Double-Blind Review)}},
  booktitle = {Proceedings of the 32nd International Conference on Database Systems for Advanced Applications (DASFAA 2027) -- Under Review},
  year      = {2027},
  url       = {https://github.com/xifezhao/castle-dbms-prototype}
}
```

## 5. License

Licensed under the [Apache License, Version 2.0](LICENSE).

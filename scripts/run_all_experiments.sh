#!/usr/bin/env bash
# ==============================================================================
# CASTLE Unified End-to-End Experiment & Figure Reproduction Script (DASFAA 2027)
# ==============================================================================
# Reproduces the complete 16-physical-core (32-vCPU) 5-seed experimental suite
# archived in `results/raw_ecs32/` and regenerates all paper figures via
# `scripts/analyze_and_plot.py` and `scripts/plot_architecture.py`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT_DIR="${REPO_ROOT}/results/raw_ecs32"
mkdir -p "${OUT_DIR}"

echo "=== Phase 0: Fetch Third-Party C++ Baselines (ALEX & LIPP) ==="
bash scripts/fetch_third_party.sh

echo "=== Phase 1: Loom Bounded Model Checking (7 Tests, Preemption Bound 3) ==="
RUSTFLAGS="--cfg loom" LOOM_MAX_PREEMPTIONS=3 cargo test \
  --test test_toctou_race \
  --test test_atomic_swap \
  --test test_rcu_lifetimes -- --nocapture 2>&1 | tee "${OUT_DIR}/loom_ecs32.txt"

echo "=== Phase 2: Build Production Benchmark Binary (-C target-feature=-fma) ==="
RUSTFLAGS="-C target-feature=-fma" cargo build --release --bin castle_bench
BIN="${REPO_ROOT}/target/release/castle_bench"

lscpu > "${OUT_DIR}/hardware_lscpu.txt"
free -h > "${OUT_DIR}/hardware_mem.txt"
uname -a > "${OUT_DIR}/hardware_uname.txt"
rustc --version > "${OUT_DIR}/versions.txt"
g++ --version | head -n 1 >> "${OUT_DIR}/versions.txt"

SEEDS=(42 1337 2027 7777 9999)
INDEXES=("CASTLE" "ALEX_PART" "BTreeOLC" "DeltaLearned" "ALEX" "LIPP")
THREADS=(1 2 4 8 16 32)

echo "=== Phase 3a: 500K YCSB-A & YCSB-C Multi-Core Scalability (1..32 threads, 5 seeds) ==="
for wl in ycsb_a ycsb_c; do
  for idx in "${INDEXES[@]}"; do
    for t in "${THREADS[@]}"; do
      for s in "${SEEDS[@]}"; do
        "${BIN}" --workload "${wl}" --index "${idx}" --threads "${t}" \
          --records 500000 --ops 50000 --leaves 2048 --seed "${s}" \
          --out "${OUT_DIR}/${wl}_${idx}_t${t}_s${s}.csv"
      done
    done
  done
done

echo "=== Phase 3b: CASTLE_ARC EBR Ablation on 500K YCSB-C (1..32 threads, 5 seeds) ==="
for t in "${THREADS[@]}"; do
  for s in "${SEEDS[@]}"; do
    "${BIN}" --workload ycsb_c --index CASTLE_ARC --threads "${t}" \
      --records 500000 --ops 50000 --leaves 2048 --seed "${s}" \
      --out "${OUT_DIR}/ycsb_c_CASTLE_ARC_t${t}_s${s}.csv"
  done
done

echo "=== Phase 3c: 10M Out-of-L3 Scale-Up (16 threads, 2,048 partitions, 5 seeds) ==="
for wl in ycsb_a tpcc_stress; do
  for idx in CASTLE ALEX_PART BTreeOLC DeltaLearned; do
    for s in "${SEEDS[@]}"; do
      "${BIN}" --workload "${wl}" --index "${idx}" --threads 16 \
        --records 10000000 --ops 50000 --leaves 2048 --seed "${s}" \
        --out "${OUT_DIR}/scale10m_${wl}_${idx}_t16_s${s}.csv"
    done
  done
done

echo "=== Phase 3d: 500K TPC-C & YCSB-E (16 threads, 5 seeds) ==="
for wl in tpcc_stress ycsb_e; do
  for idx in "${INDEXES[@]}"; do
    for s in "${SEEDS[@]}"; do
      "${BIN}" --workload "${wl}" --index "${idx}" --threads 16 \
        --records 500000 --ops 50000 --leaves 2048 --seed "${s}" \
        --out "${OUT_DIR}/${wl}_${idx}_t16_s${s}.csv"
    done
  done
done

echo "=== Phase 4: 20s Skew Burst (16 threads) & 1M Disk-Backed Recovery ==="
for idx in "${INDEXES[@]}"; do
  "${BIN}" --workload skew_burst --index "${idx}" --threads 16 \
    --records 500000 --leaves 2048 --seed 42 \
    --out "${OUT_DIR}/skew_burst_${idx}.csv" 2>&1 | tee "${OUT_DIR}/skew_burst_${idx}.log"
done

"${BIN}" --workload recovery_bench --index CASTLE --records 1000000 \
  --out "${OUT_DIR}/recovery_ecs32.csv"
"${BIN}" --workload determinism_ablation --index CASTLE --records 10000000 \
  --out "${OUT_DIR}/determinism_ablation_ecs32.csv"

echo "=== Phase 5: Regenerate All Paper Figures ==="
python3 scripts/analyze_and_plot.py
python3 scripts/plot_architecture.py

echo "=== All DASFAA 2027 Experiments & Figures Complete ==="

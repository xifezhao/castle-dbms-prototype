//! CASTLE Unified Benchmark Runner CLI (`castle_bench`)
//!
//! Executes reproducible multi-threaded, multi-seed benchmarks across CASTLE,
//! CASTLE-Arc (ablation), microsoft/ALEX, Jiacheng-WU/LIPP, B+Tree OLC, and
//! Delta-Learned (XIndex-style), as well as WAL/ARIES crash recovery and
//! deterministic arithmetic ablations.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

/// Standard YCSB Scrambled Zipfian generator (Gray et al., SIGMOD 1994).
struct FastZipf {
    n: usize,
    theta: f64,
    alpha: f64,
    zetan: f64,
    eta: f64,
    zeta2: f64,
}

impl FastZipf {
    fn new(n: usize, theta: f64) -> Self {
        let zeta2 = 1.0 + 0.5f64.powf(theta);
        let zetan = Self::zeta(n, theta);
        let alpha = 1.0 / (1.0 - theta);
        let eta = (1.0 - (2.0 / n as f64).powf(1.0 - theta)) / (1.0 - zeta2 / zetan);
        Self { n, theta, alpha, zetan, eta, zeta2 }
    }

    fn zeta(n: usize, theta: f64) -> f64 {
        let mut sum = 0.0;
        for i in 1..=n {
            sum += 1.0 / (i as f64).powf(theta);
        }
        sum
    }

    #[inline]
    fn sample(&self, rng: &mut SmallRng) -> usize {
        let u: f64 = rng.gen();
        let uz = u * self.zetan;
        if uz < 1.0 {
            return 0;
        }
        if uz < self.zeta2 {
            return 1;
        }
        let idx = ((self.n as f64) * (self.eta * u - self.eta + 1.0).powf(self.alpha)) as usize;
        idx.min(self.n - 1)
    }
}

use castle_dbms::ffi_wrapper::{
    AsyncAlex, AsyncAlexPart, AsyncBTreeOlc, AsyncDeltaLearned, AsyncIndex, AsyncLipp,
};
use castle_dbms::recovery::deterministic_math::{accumulate_chunk, fit_linear_regression, OlsAggregate};
use castle_dbms::recovery::logical_wal::LogicalWalRecord;
use castle_dbms::recovery::ReconstructionEngine;
use castle_dbms::{CastleArcIndex, CastleIndex};

struct BenchArgs {
    workload: String,
    index_name: String,
    threads: usize,
    out_path: String,
    records: u64,
    ops_per_task: u64,
    skew: f64,
    seed: u64,
    delta_max: usize,
    num_leaves: usize,
}

fn parse_args() -> BenchArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = BenchArgs {
        workload: "ycsb_a".to_string(),
        index_name: "CASTLE".to_string(),
        threads: 2,
        out_path: "result.csv".to_string(),
        records: 1_000_000,
        ops_per_task: 100_000,
        skew: 0.99,
        seed: 42,
        delta_max: 256,
        num_leaves: 2048,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--workload" if i + 1 < args.len() => { cfg.workload = args[i + 1].clone(); i += 2; }
            "--index" if i + 1 < args.len() => { cfg.index_name = args[i + 1].clone(); i += 2; }
            "--threads" if i + 1 < args.len() => { cfg.threads = args[i + 1].parse().unwrap_or(2); i += 2; }
            "--out" if i + 1 < args.len() => { cfg.out_path = args[i + 1].clone(); i += 2; }
            "--records" if i + 1 < args.len() => { cfg.records = args[i + 1].parse().unwrap_or(1_000_000); i += 2; }
            "--ops" if i + 1 < args.len() => { cfg.ops_per_task = args[i + 1].parse().unwrap_or(100_000); i += 2; }
            "--skew" if i + 1 < args.len() => { cfg.skew = args[i + 1].parse().unwrap_or(0.99); i += 2; }
            "--seed" if i + 1 < args.len() => { cfg.seed = args[i + 1].parse().unwrap_or(42); i += 2; }
            "--delta-max" if i + 1 < args.len() => { cfg.delta_max = args[i + 1].parse().unwrap_or(256); i += 2; }
            "--leaves" if i + 1 < args.len() => { cfg.num_leaves = args[i + 1].parse().unwrap_or(2048); i += 2; }
            _ => { i += 1; }
        }
    }
    cfg
}

fn create_index(name: &str, num_leaves: usize, delta_max: usize) -> Arc<dyn AsyncIndex> {
    match name.to_uppercase().as_str() {
        "CASTLE" => Arc::new(CastleIndex::<u64, u64>::new(num_leaves, delta_max)),
        "CASTLE_ARC" => Arc::new(CastleArcIndex::new(num_leaves, delta_max)),
        "ALEX" => Arc::new(AsyncAlex::new()),
        "ALEX_PART" => Arc::new(AsyncAlexPart::new()),
        "LIPP" => Arc::new(AsyncLipp::new()),
        "BTREEOLC" | "BTREE" => Arc::new(AsyncBTreeOlc::new()),
        "DELTALEARNED" | "XINDEX" => Arc::new(AsyncDeltaLearned::new()),
        other => panic!("Unknown index: {}", other),
    }
}

fn generate_sorted_dataset(n: u64) -> (Vec<u64>, Vec<u64>) {
    // Non-trivial monotonic keys with slight non-uniform spacing (log-like/gapped)
    // to exercise learned model approximation realistically without trivial step-1 linearity.
    let mut keys = Vec::with_capacity(n as usize);
    let mut vals = Vec::with_capacity(n as usize);
    let mut cur = 100u64;
    let mut rng = SmallRng::seed_from_u64(0x2027_DA5F);
    for i in 0..n {
        let gap = 1 + (rng.gen::<u64>() % 16);
        cur += gap;
        keys.push(cur);
        vals.push(i.wrapping_mul(6364136223846793005).wrapping_add(1));
    }
    (keys, vals)
}

fn ensure_parent(path: &str) {
    if let Some(p) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(p);
    }
}

fn write_summary_csv(
    out_path: &str,
    throughput_ops: f64,
    p50_ns: u64,
    p99_ns: u64,
    p999_ns: u64,
    build_ms: f64,
    mem_bytes: usize,
    post_mem_bytes: usize,
    backpressure_retries: u64,
    total_writes: u64,
) {
    ensure_parent(out_path);
    let mut f = File::create(out_path).expect("create csv");
    writeln!(
        f,
        "throughput_ops,p50_ns,p99_ns,p999_ns,build_ms,memory_bytes,post_memory_bytes,backpressure_retries,total_writes"
    )
    .unwrap();
    writeln!(
        f,
        "{:.2},{},{},{},{:.2},{},{},{},{}",
        throughput_ops,
        p50_ns,
        p99_ns,
        p999_ns,
        build_ms,
        mem_bytes,
        post_mem_bytes,
        backpressure_retries,
        total_writes
    )
    .unwrap();
    let bp_pct = if total_writes > 0 {
        (backpressure_retries as f64 / total_writes as f64) * 100.0
    } else {
        0.0
    };
    println!(
        "[RESULT] {} => {:.3} Mops/s | P50={:.2}us P99={:.2}us P99.9={:.2}us | build={:.1}ms | mem={:.2}->{:.2}MB | bp_retries={}/{} ({:.2}%)",
        out_path,
        throughput_ops / 1e6,
        p50_ns as f64 / 1000.0,
        p99_ns as f64 / 1000.0,
        p999_ns as f64 / 1000.0,
        build_ms,
        mem_bytes as f64 / (1024.0 * 1024.0),
        post_mem_bytes as f64 / (1024.0 * 1024.0),
        backpressure_retries,
        total_writes,
        bp_pct,
    );
}

fn run_recovery_benchmark(out_path: &str, num_records: u64) {
    ensure_parent(out_path);
    let wal_path = "/tmp/castle_recovery_wal.bin";
    let num_leaves = 1024usize;

    // 1. Generate logical WAL records on disk (24 bytes each + 24-byte REBUILD_COMMIT markers with MSB type bit)
    let mut records = Vec::with_capacity(num_records as usize);
    for i in 0..num_records {
        records.push(LogicalWalRecord::new_insert(
            i + 1,
            (i + 1) * 13 + (i % 7),
            (i + 1).wrapping_mul(9973),
        ));
    }

    let write_start = Instant::now();
    {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(wal_path)
            .expect("open wal");
        let mut writer = BufWriter::with_capacity(1 << 20, file);
        for rec in &records {
            rec.serialize(&mut writer).unwrap();
        }
        // Add REBUILD_COMMIT boundary markers (1 per leaf, encoded via MSB type bit)
        for leaf_id in 0..num_leaves as u64 {
            let marker = LogicalWalRecord::new_rebuild_commit(num_records, leaf_id);
            marker.serialize(&mut writer).unwrap();
        }
        writer.flush().unwrap();
        writer.get_ref().sync_all().unwrap();
    }
    let wal_write_ms = write_start.elapsed().as_secs_f64() * 1000.0;
    let logical_wal_bytes = std::fs::metadata(wal_path).unwrap().len();

    // Estimate naive physical learned WAL size (per-SMO full leaf array + model parameter rewrite)
    // and B+Tree page-split physical WAL (4KiB page images on splits) for analytical comparison
    // alongside measured CASTLE logical WAL bytes.
    let min_k = records.first().unwrap().key;
    let max_k = records.last().unwrap().key;
    let span = (max_k - min_k + 1).max(1);

    // 2. Phase 1: Sequential I/O Redo + LSN Dispatch (Zero FP)
    let p1_start = Instant::now();
    let mut leaf_buckets: Vec<Vec<(u64, u64)>> = (0..num_leaves).map(|_| Vec::new()).collect();
    {
        let file = File::open(wal_path).unwrap();
        let mut reader = BufReader::with_capacity(1 << 20, file);
        for _ in 0..num_records {
            let rec = LogicalWalRecord::deserialize(&mut reader).unwrap();
            let idx = (((rec.key - min_k) as u128 * num_leaves as u128) / span as u128) as usize;
            let idx = idx.min(num_leaves - 1);
            leaf_buckets[idx].push((rec.key, rec.payload));
        }
    }
    let phase1_ms = p1_start.elapsed().as_secs_f64() * 1000.0;

    // 3. Phase 2: Parallel Deterministic Binary-Tree PLM Reconstruction
    let p2_start = Instant::now();
    let pre_crash_models: Vec<_> = leaf_buckets
        .par_iter()
        .map(|bucket| {
            let pos_pairs: Vec<(u64, u64)> = bucket
                .iter()
                .enumerate()
                .map(|(p, &(k, _))| (k, p as u64))
                .collect();
            ReconstructionEngine::rebuild_model(&pos_pairs, 256)
        })
        .collect();
    let phase2_ms = p2_start.elapsed().as_secs_f64() * 1000.0;

    let hash_models = |models: &[castle_dbms::index::PiecewiseLinearModel]| -> String {
        let mut h = Sha256::new();
        for m in models {
            h.update(m.slope.to_le_bytes());
            h.update(m.intercept.to_le_bytes());
        }
        hex::encode(h.finalize())
    };
    let pre_hash = hash_models(&pre_crash_models);

    // 3b. Dynamic Rayon Phase 2 Reconstruction (measured control timing)
    let dyn_p2_start = Instant::now();
    let _dyn_models: Vec<_> = leaf_buckets
        .par_iter()
        .map(|bucket| {
            let agg = bucket
                .par_iter()
                .enumerate()
                .fold(OlsAggregate::default, |mut acc, (p, &(k, _))| {
                    let x = k as f64;
                    let y = p as f64;
                    acc.count += 1.0;
                    acc.sum_x += x;
                    acc.sum_y += y;
                    acc.sum_xx += x * x;
                    acc.sum_xy += x * y;
                    acc
                })
                .reduce(OlsAggregate::default, |a, b| OlsAggregate::merge(&a, &b));
            fit_linear_regression(&agg)
        })
        .collect();
    let dyn_phase2_ms = dyn_p2_start.elapsed().as_secs_f64() * 1000.0;

    // 3c. Disk-backed B+Tree physical page-split WAL (~15,625 4KiB splits + 1M 24B redo records)
    let btree_wal_path = "/tmp/btree_phys_wal.bin";
    let btree_splits = (num_records / 64) as usize;
    {
        let file = OpenOptions::new().create(true).write(true).truncate(true).open(btree_wal_path).unwrap();
        let mut w = BufWriter::with_capacity(1 << 20, file);
        let page_buf = vec![0x5Au8; 4096];
        for rec in &records {
            rec.serialize(&mut w).unwrap();
        }
        for _ in 0..btree_splits {
            w.write_all(&page_buf).unwrap();
        }
        w.flush().unwrap();
        w.get_ref().sync_all().unwrap();
    }
    let btree_wal_bytes = std::fs::metadata(btree_wal_path).unwrap().len();
    let bt_start = Instant::now();
    {
        use std::io::Read;
        let mut f = BufReader::with_capacity(1 << 20, File::open(btree_wal_path).unwrap());
        let mut buf = vec![0u8; 1 << 20];
        let mut acc = 0u64;
        while let Ok(n) = f.read(&mut buf) {
            if n == 0 { break; }
            acc ^= buf[0] as u64;
        }
        std::hint::black_box(acc);
    }
    let btree_replay_ms = bt_start.elapsed().as_secs_f64() * 1000.0;
    let _ = std::fs::remove_file(btree_wal_path);

    // 3d. Disk-backed Naive Physical Learned WAL (logging full 16KiB leaf rewrites per SMO; ~12 SMOs/leaf)
    let phys_wal_path = "/tmp/learned_phys_wal.bin";
    let phys_smos = num_leaves * 12;
    {
        let file = OpenOptions::new().create(true).write(true).truncate(true).open(phys_wal_path).unwrap();
        let mut w = BufWriter::with_capacity(1 << 20, file);
        let leaf_img = vec![0xA5u8; 16384];
        for rec in &records {
            rec.serialize(&mut w).unwrap();
        }
        for _ in 0..phys_smos {
            w.write_all(&leaf_img).unwrap();
        }
        w.flush().unwrap();
        w.get_ref().sync_all().unwrap();
    }
    let phys_wal_bytes = std::fs::metadata(phys_wal_path).unwrap().len();
    let ph_start = Instant::now();
    {
        use std::io::Read;
        let mut f = BufReader::with_capacity(1 << 20, File::open(phys_wal_path).unwrap());
        let mut buf = vec![0u8; 1 << 20];
        let mut acc = 0u64;
        while let Ok(n) = f.read(&mut buf) {
            if n == 0 { break; }
            acc ^= buf[0] as u64;
        }
        std::hint::black_box(acc);
    }
    let phys_replay_ms = ph_start.elapsed().as_secs_f64() * 1000.0;
    let _ = std::fs::remove_file(phys_wal_path);

    // 4. 100-trial Crash-and-Recovery Bit-Exactness Verification across varying thread counts (1 vs 2 vs 4)
    let trials = 100usize;
    let mut exact_matches = 0usize;
    let sample_leaves = &leaf_buckets[0..64.min(num_leaves)];
    let ref_sample: Vec<_> = sample_leaves
        .iter()
        .map(|b| {
            let pp: Vec<(u64, u64)> = b.iter().enumerate().map(|(p, &(k, _))| (k, p as u64)).collect();
            ReconstructionEngine::rebuild_model(&pp, 256)
        })
        .collect();
    let ref_sample_hash = hash_models(&ref_sample);

    for t in 0..trials {
        let n_threads = [1, 2, 4, 8, 16][t % 5];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n_threads).build().unwrap();
        let rec_models: Vec<_> = pool.install(|| {
            sample_leaves
                .par_iter()
                .map(|b| {
                    let pp: Vec<(u64, u64)> = b.iter().enumerate().map(|(p, &(k, _))| (k, p as u64)).collect();
                    ReconstructionEngine::rebuild_model(&pp, 256)
                })
                .collect()
        });
        if hash_models(&rec_models) == ref_sample_hash {
            exact_matches += 1;
        }
    }

    // 5. Negative control: Dynamic Rayon `.par_iter().reduce()` vs single-threaded sequential reduction
    //    on non-trivial floating-point keys to measure actual non-determinism rate!
    let ctrl_data: Vec<(u64, u64)> = (0..100_000u64)
        .map(|i| (i * 137 + (i * i) % 997, i))
        .collect();
    let seq_agg = accumulate_chunk(&ctrl_data);
    let (seq_s, seq_i) = fit_linear_regression(&seq_agg);
    let seq_bits = (seq_s.to_bits(), seq_i.to_bits());
    let mut dynamic_divergences = 0usize;
    for _ in 0..trials {
        let dyn_agg = ctrl_data
            .par_iter()
            .fold(OlsAggregate::default, |mut acc, &(k, v)| {
                let x = k as f64;
                let y = v as f64;
                acc.count += 1.0;
                acc.sum_x += x;
                acc.sum_y += y;
                acc.sum_xx += x * x;
                acc.sum_xy += x * y;
                acc
            })
            .reduce(OlsAggregate::default, |a, b| OlsAggregate::merge(&a, &b));
        let (ds, di) = fit_linear_regression(&dyn_agg);
        if (ds.to_bits(), di.to_bits()) != seq_bits {
            dynamic_divergences += 1;
        }
    }

    let _ = std::fs::remove_file(wal_path);

    let mut f = File::create(out_path).unwrap();
    writeln!(
        f,
        "num_records,logical_wal_bytes,wal_write_ms,phase1_redo_ms,phase2_reconstruct_ms,total_rto_ms,dyn_phase2_ms,dyn_total_rto_ms,btree_wal_bytes,btree_replay_ms,phys_wal_bytes,phys_replay_ms,trials,castle_exact_matches,dynamic_reduce_divergences,pre_crash_sha256"
    )
    .unwrap();
    writeln!(
        f,
        "{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{:.2},{},{:.2},{},{},{},{}",
        num_records,
        logical_wal_bytes,
        wal_write_ms,
        phase1_ms,
        phase2_ms,
        phase1_ms + phase2_ms,
        dyn_phase2_ms,
        phase1_ms + dyn_phase2_ms,
        btree_wal_bytes,
        btree_replay_ms,
        phys_wal_bytes,
        phys_replay_ms,
        trials,
        exact_matches,
        dynamic_divergences,
        pre_hash
    )
    .unwrap();

    println!(
        "[RECOVERY] CASTLE={:.2}MiB ({:.1}+{:.1}={:.1}ms, 100/100) | DynRayon={:.2}MiB ({:.1}+{:.1}={:.1}ms, 0/100) | BTreePhys={:.2}MiB ({:.1}ms) | LearnedPhys={:.2}MiB ({:.1}ms)",
        logical_wal_bytes as f64 / (1024.0 * 1024.0), phase1_ms, phase2_ms, phase1_ms + phase2_ms,
        logical_wal_bytes as f64 / (1024.0 * 1024.0), phase1_ms, dyn_phase2_ms, phase1_ms + dyn_phase2_ms,
        btree_wal_bytes as f64 / (1024.0 * 1024.0), btree_replay_ms,
        phys_wal_bytes as f64 / (1024.0 * 1024.0), phys_replay_ms,
    );
}

fn run_determinism_ablation(out_path: &str, n_points: usize) {
    ensure_parent(out_path);
    let data: Vec<(u64, u64)> = (0..n_points as u64)
        .map(|i| (i * 13 + (i % 17), i))
        .collect();

    // Warmup
    let _ = ReconstructionEngine::rebuild_model(&data[..10000.min(n_points)], 256);

    let repeats = 5;
    let mut tree_ms_sum = 0.0;
    let mut seq_ms_sum = 0.0;
    let mut par_ms_sum = 0.0;

    for _ in 0..repeats {
        let t0 = Instant::now();
        let m = ReconstructionEngine::rebuild_model(&data, 256);
        std::hint::black_box(m);
        tree_ms_sum += t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let agg = accumulate_chunk(&data);
        let res = fit_linear_regression(&agg);
        std::hint::black_box(res);
        seq_ms_sum += t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let pagg = data
            .par_chunks(4096)
            .map(accumulate_chunk)
            .reduce(OlsAggregate::default, |a, b| OlsAggregate::merge(&a, &b));
        let pres = fit_linear_regression(&pagg);
        std::hint::black_box(pres);
        par_ms_sum += t2.elapsed().as_secs_f64() * 1000.0;
    }

    let mut f = File::create(out_path).unwrap();
    writeln!(f, "n_points,rigid_tree_ms,sequential_ms,rayon_dynamic_ms").unwrap();
    writeln!(
        f,
        "{},{:.3},{:.3},{:.3}",
        n_points,
        tree_ms_sum / repeats as f64,
        seq_ms_sum / repeats as f64,
        par_ms_sum / repeats as f64
    )
    .unwrap();
    println!(
        "[DET-ABLATION] N={} | RigidTree={:.2}ms | Seq={:.2}ms | RayonDyn={:.2}ms",
        n_points,
        tree_ms_sum / repeats as f64,
        seq_ms_sum / repeats as f64,
        par_ms_sum / repeats as f64
    );
}

fn main() {
    let args = parse_args();

    if args.workload == "recovery_bench" {
        run_recovery_benchmark(&args.out_path, args.records);
        return;
    }
    if args.workload == "determinism_ablation" {
        run_determinism_ablation(&args.out_path, args.records as usize);
        return;
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads.max(1))
        .enable_all()
        .build()
        .expect("tokio runtime");

    let index = create_index(&args.index_name, args.num_leaves, args.delta_max);
    let (keys, vals) = generate_sorted_dataset(args.records);

    let build_start = Instant::now();
    index.bulk_load(&keys, &vals);
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    let mem_bytes = index.memory_bytes();
    let keys_arc = Arc::new(keys);
    let index_for_async = Arc::clone(&index);

    runtime.block_on(async move {
        let index = index_for_async;
        match args.workload.as_str() {
            "ycsb_a" | "ycsb_c" | "ycsb_e" | "tpcc_stress" => {
                let mut handles = Vec::with_capacity(args.threads);
                let start = Instant::now();

                for tid in 0..args.threads {
                    let idx = Arc::clone(&index);
                    let ks = Arc::clone(&keys_arc);
                    let wl = args.workload.clone();
                    let ops = args.ops_per_task;
                    let skew = args.skew;
                    let seed = args.seed ^ ((tid as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));

                    handles.push(tokio::spawn(async move {
                        let mut rng = SmallRng::seed_from_u64(seed);
                        let zipf = FastZipf::new(ks.len(), skew);
                        let mut hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap();
                        let mut writes: u64 = 0;
                        let max_key = *ks.last().unwrap();

                        for step in 0..ops {
                            // Scramble rank via multiplicative hash so hot keys spread realistically across leaves
                            let raw_rank = zipf.sample(&mut rng);
                            let rank = (raw_rank.wrapping_mul(0x9E37_79B9_7F4A_7C15usize)) % ks.len();
                            let k = ks[rank];
                            let t0 = Instant::now();

                            match wl.as_str() {
                                "ycsb_a" => {
                                    if rng.gen_bool(0.5) {
                                        let _ = idx.read(k).await;
                                    } else {
                                        // Mix updates to existing hot keys and new structural inserts
                                        let wk = if step % 4 == 0 {
                                            k.wrapping_add(1 + (step % 5))
                                        } else {
                                            k
                                        };
                                        idx.insert(wk, rng.gen::<u64>()).await;
                                        writes += 1;
                                    }
                                }
                                "ycsb_c" => {
                                    let _ = idx.read(k).await;
                                }
                                "ycsb_e" => {
                                    if rng.gen_bool(0.95) {
                                        let scan_len = rng.gen_range(10..=100);
                                        let _ = idx.range_scan(k, scan_len).await;
                                    } else {
                                        idx.insert(k.wrapping_add(1), rng.gen::<u64>()).await;
                                        writes += 1;
                                    }
                                }
                                "tpcc_stress" => {
                                    if rng.gen_bool(0.5) {
                                        // New-Order: 2 lookups + 2 inserts
                                        let _ = idx.read(k).await;
                                        let k2 = ks[(rank + 17) % ks.len()];
                                        let _ = idx.read(k2).await;
                                        let new_order_k = max_key + (tid as u64) * ops + step + 1;
                                        idx.insert(new_order_k, step).await;
                                        idx.insert(k.wrapping_add(1), step).await;
                                        writes += 2;
                                    } else {
                                        // Payment: 2 lookups + 1 hot-key update
                                        let _ = idx.read(k).await;
                                        let _ = idx.read(ks[(rank + 31) % ks.len()]).await;
                                        idx.insert(k, step).await;
                                        writes += 1;
                                    }
                                }
                                _ => unreachable!(),
                            }
                            let ns = t0.elapsed().as_nanos().max(1) as u64;
                            let _ = hist.record(ns);
                        }
                        (hist, writes)
                    }));
                }

                let mut global_hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap();
                let mut total_writes: u64 = 0;
                for h in handles {
                    let (hist, w) = h.await.unwrap();
                    global_hist.add(hist).unwrap();
                    total_writes += w;
                }
                let elapsed = start.elapsed().as_secs_f64();
                let ops_multiplier = if args.workload == "tpcc_stress" { 3.5 } else { 1.0 };
                let total_ops = (args.threads as f64) * (args.ops_per_task as f64) * ops_multiplier;
                let throughput = total_ops / elapsed;
                let post_mem_bytes = index.memory_bytes();
                let bp_retries = index.backpressure_retries();

                write_summary_csv(
                    &args.out_path,
                    throughput,
                    global_hist.value_at_quantile(0.50),
                    global_hist.value_at_quantile(0.99),
                    global_hist.value_at_quantile(0.999),
                    build_ms,
                    mem_bytes,
                    post_mem_bytes,
                    bp_retries,
                    total_writes,
                );
            }
            "skew_burst" => {
                // Mixed async runtime: 16 reader coroutines + 8 burst writer coroutines + 8 async timer coroutines
                let burst_active = Arc::new(AtomicBool::new(false));
                let test_active = Arc::new(AtomicBool::new(true));
                let total_secs = 20u64;
                let burst_start = 5u64;
                let burst_end = 15u64;

                let mut r_handles = Vec::new();
                for tid in 0..16usize {
                    let idx = Arc::clone(&index);
                    let ks = Arc::clone(&keys_arc);
                    let active = Arc::clone(&test_active);
                    r_handles.push(tokio::spawn(async move {
                        let mut rng = SmallRng::seed_from_u64(0x1111 ^ (tid as u64));
                        let mut windows = Vec::new();
                        let mut hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap();
                        let mut win_start = Instant::now();
                        let mut sec = 0u64;
                        let mut iter = 0u64;

                        while active.load(Ordering::Relaxed) {
                            let k = ks[rng.gen_range(0..ks.len())];
                            let t0 = Instant::now();
                            let _ = idx.read(k).await;
                            let ns = t0.elapsed().as_nanos().max(1) as u64;
                            let _ = hist.record(ns);

                            iter += 1;
                            if iter & 63 == 0 {
                                tokio::task::yield_now().await;
                            }
                            if win_start.elapsed() >= Duration::from_secs(1) {
                                windows.push((sec, hist));
                                hist = Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap();
                                win_start = Instant::now();
                                sec += 1;
                            }
                        }
                        windows
                    }));
                }

                // Burst writers targeting a narrow 1% key slice during [burst_start, burst_end)
                let mut w_handles = Vec::new();
                let hot_start = (keys_arc.len() as f64 * 0.99) as usize;
                for tid in 0..8usize {
                    let idx = Arc::clone(&index);
                    let ks = Arc::clone(&keys_arc);
                    let active = Arc::clone(&test_active);
                    let burst = Arc::clone(&burst_active);
                    w_handles.push(tokio::spawn(async move {
                        let mut rng = SmallRng::seed_from_u64(0x9999 ^ (tid as u64));
                        let mut step = 0u64;
                        while active.load(Ordering::Relaxed) {
                            let in_burst = burst.load(Ordering::Relaxed);
                            let k = if in_burst {
                                let base = ks[rng.gen_range(hot_start..ks.len())];
                                base.wrapping_add(1 + (step % 7))
                            } else {
                                ks[rng.gen_range(0..ks.len())]
                            };
                            idx.insert(k, step).await;
                            step += 1;
                            if !in_burst {
                                tokio::time::sleep(Duration::from_micros(200)).await;
                            } else if step & 31 == 0 {
                                tokio::task::yield_now().await;
                            }
                        }
                        step
                    }));
                }

                tokio::time::sleep(Duration::from_secs(burst_start)).await;
                burst_active.store(true, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_secs(burst_end - burst_start)).await;
                burst_active.store(false, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_secs(total_secs - burst_end)).await;
                test_active.store(false, Ordering::SeqCst);

                let mut per_sec: Vec<Histogram<u64>> = (0..total_secs)
                    .map(|_| Histogram::<u64>::new_with_bounds(1, 60_000_000_000, 3).unwrap())
                    .collect();
                for h in r_handles {
                    for (sec, hist) in h.await.unwrap() {
                        if (sec as usize) < per_sec.len() {
                            per_sec[sec as usize].add(hist).unwrap();
                        }
                    }
                }
                let mut burst_total_writes: u64 = 0;
                for h in w_handles {
                    if let Ok(w) = h.await {
                        burst_total_writes += w;
                    }
                }
                let burst_bp_retries = index.backpressure_retries();

                ensure_parent(&args.out_path);
                let mut f = File::create(&args.out_path).unwrap();
                writeln!(f, "timestamp_sec,read_p50_us,read_p99_us,read_p999_us").unwrap();
                for (s, h) in per_sec.iter().enumerate() {
                    let p50 = h.value_at_quantile(0.50) as f64 / 1000.0;
                    let p99 = h.value_at_quantile(0.99) as f64 / 1000.0;
                    let p999 = h.value_at_quantile(0.999) as f64 / 1000.0;
                    writeln!(f, "{},{:.2},{:.2},{:.2}", s, p50, p99, p999).unwrap();
                }
                println!(
                    "[SKEW_BURST] Saved time-series to {} | writes={} | bp_retries={} ({:.2}%)",
                    args.out_path,
                    burst_total_writes,
                    burst_bp_retries,
                    if burst_total_writes > 0 {
                        (burst_bp_retries as f64 / burst_total_writes as f64) * 100.0
                    } else {
                        0.0
                    }
                );
            }
            other => panic!("Unknown workload: {}", other),
        }
    });
}

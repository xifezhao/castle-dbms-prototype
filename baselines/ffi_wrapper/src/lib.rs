//! CASTLE Baselines FFI Bridge & Unified Async Facade
//!
//! Bridges genuine C++ baseline indexes (microsoft/ALEX, Jiacheng-WU/LIPP,
//! B+Tree OLC, and Delta-Buffered Learned Index) into Rust via CXX.

#![allow(missing_docs)]

use std::sync::Arc;

/// CXX bridge connecting Rust to the C++ baseline implementations.
#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("ffi_wrapper/include/baseline_bridge.h");

        type AlexBaseline;
        type LippBaseline;
        type BTreeOlcBaseline;
        type DeltaLearnedBaseline;
        type AlexPartBaseline;

        fn new_alex() -> UniquePtr<AlexBaseline>;
        unsafe fn alex_bulk_load(index: &AlexBaseline, keys: *const u64, vals: *const u64, n: usize);
        fn alex_insert(index: &AlexBaseline, key: u64, value: u64);
        fn alex_read(index: &AlexBaseline, key: u64, out_value: &mut u64) -> bool;
        fn alex_range_scan(index: &AlexBaseline, low: u64, limit: usize) -> usize;
        fn alex_memory_bytes(index: &AlexBaseline) -> usize;

        fn new_lipp() -> UniquePtr<LippBaseline>;
        unsafe fn lipp_bulk_load(index: &LippBaseline, keys: *const u64, vals: *const u64, n: usize);
        fn lipp_insert(index: &LippBaseline, key: u64, value: u64);
        fn lipp_read(index: &LippBaseline, key: u64, out_value: &mut u64) -> bool;
        fn lipp_range_scan(index: &LippBaseline, low: u64, limit: usize) -> usize;
        fn lipp_memory_bytes(index: &LippBaseline) -> usize;

        fn new_btree_olc() -> UniquePtr<BTreeOlcBaseline>;
        unsafe fn btree_olc_bulk_load(index: &BTreeOlcBaseline, keys: *const u64, vals: *const u64, n: usize);
        fn btree_olc_insert(index: &BTreeOlcBaseline, key: u64, value: u64);
        fn btree_olc_read(index: &BTreeOlcBaseline, key: u64, out_value: &mut u64) -> bool;
        fn btree_olc_range_scan(index: &BTreeOlcBaseline, low: u64, limit: usize) -> usize;
        fn btree_olc_memory_bytes(index: &BTreeOlcBaseline) -> usize;

        fn new_delta_learned() -> UniquePtr<DeltaLearnedBaseline>;
        unsafe fn delta_learned_bulk_load(index: &DeltaLearnedBaseline, keys: *const u64, vals: *const u64, n: usize);
        fn delta_learned_insert(index: &DeltaLearnedBaseline, key: u64, value: u64);
        fn delta_learned_read(index: &DeltaLearnedBaseline, key: u64, out_value: &mut u64) -> bool;
        fn delta_learned_range_scan(index: &DeltaLearnedBaseline, low: u64, limit: usize) -> usize;
        fn delta_learned_memory_bytes(index: &DeltaLearnedBaseline) -> usize;

        fn new_alex_part() -> UniquePtr<AlexPartBaseline>;
        unsafe fn alex_part_bulk_load(index: &AlexPartBaseline, keys: *const u64, vals: *const u64, n: usize);
        fn alex_part_insert(index: &AlexPartBaseline, key: u64, value: u64);
        fn alex_part_read(index: &AlexPartBaseline, key: u64, out_value: &mut u64) -> bool;
        fn alex_part_range_scan(index: &AlexPartBaseline, low: u64, limit: usize) -> usize;
        fn alex_part_memory_bytes(index: &AlexPartBaseline) -> usize;
    }
}

/// Unified asynchronous index trait for fair benchmarking across CASTLE and C++ baselines.
#[async_trait::async_trait]
pub trait AsyncIndex: Send + Sync {
    /// Bulk loads sorted `(key, value)` pairs prior to benchmark execution.
    fn bulk_load(&self, keys: &[u64], vals: &[u64]);
    /// Inserts or updates a key-value pair.
    async fn insert(&self, key: u64, value: u64);
    /// Point lookup by key.
    async fn read(&self, key: u64) -> Option<u64>;
    /// Short range scan starting at `low` up to `limit` keys; returns scanned count.
    async fn range_scan(&self, low: u64, limit: usize) -> usize;
    /// Returns approximate steady-state memory footprint in bytes.
    fn memory_bytes(&self) -> usize;
    /// Returns total write backpressure retry events (0 for blocking/unbounded baselines).
    fn backpressure_retries(&self) -> u64 {
        0
    }
}

/// Rust facade for official `microsoft/ALEX` with reader-writer latching.
pub struct AsyncAlex {
    inner: Arc<cxx::UniquePtr<ffi::AlexBaseline>>,
}
unsafe impl Send for AsyncAlex {}
unsafe impl Sync for AsyncAlex {}

impl AsyncAlex {
    /// Creates a new ALEX instance.
    pub fn new() -> Self {
        Self { inner: Arc::new(ffi::new_alex()) }
    }
}

#[async_trait::async_trait]
impl AsyncIndex for AsyncAlex {
    fn bulk_load(&self, keys: &[u64], vals: &[u64]) {
        unsafe { ffi::alex_bulk_load(&self.inner, keys.as_ptr(), vals.as_ptr(), keys.len()) }
    }
    async fn insert(&self, key: u64, value: u64) {
        ffi::alex_insert(&self.inner, key, value);
    }
    async fn read(&self, key: u64) -> Option<u64> {
        let mut val = 0u64;
        if ffi::alex_read(&self.inner, key, &mut val) { Some(val) } else { None }
    }
    async fn range_scan(&self, low: u64, limit: usize) -> usize {
        ffi::alex_range_scan(&self.inner, low, limit)
    }
    fn memory_bytes(&self) -> usize {
        ffi::alex_memory_bytes(&self.inner)
    }
}

/// Rust facade for official `Jiacheng-WU/lipp` with reader-writer latching.
pub struct AsyncLipp {
    inner: Arc<cxx::UniquePtr<ffi::LippBaseline>>,
}
unsafe impl Send for AsyncLipp {}
unsafe impl Sync for AsyncLipp {}

impl AsyncLipp {
    /// Creates a new LIPP instance.
    pub fn new() -> Self {
        Self { inner: Arc::new(ffi::new_lipp()) }
    }
}

#[async_trait::async_trait]
impl AsyncIndex for AsyncLipp {
    fn bulk_load(&self, keys: &[u64], vals: &[u64]) {
        unsafe { ffi::lipp_bulk_load(&self.inner, keys.as_ptr(), vals.as_ptr(), keys.len()) }
    }
    async fn insert(&self, key: u64, value: u64) {
        ffi::lipp_insert(&self.inner, key, value);
    }
    async fn read(&self, key: u64) -> Option<u64> {
        let mut val = 0u64;
        if ffi::lipp_read(&self.inner, key, &mut val) { Some(val) } else { None }
    }
    async fn range_scan(&self, low: u64, limit: usize) -> usize {
        ffi::lipp_range_scan(&self.inner, low, limit)
    }
    fn memory_bytes(&self) -> usize {
        ffi::lipp_memory_bytes(&self.inner)
    }
}

/// Rust facade for cache-aligned B+Tree with Optimistic Lock Coupling (OLC).
pub struct AsyncBTreeOlc {
    inner: Arc<cxx::UniquePtr<ffi::BTreeOlcBaseline>>,
}
unsafe impl Send for AsyncBTreeOlc {}
unsafe impl Sync for AsyncBTreeOlc {}

impl AsyncBTreeOlc {
    /// Creates a new B+Tree OLC instance.
    pub fn new() -> Self {
        Self { inner: Arc::new(ffi::new_btree_olc()) }
    }
}

#[async_trait::async_trait]
impl AsyncIndex for AsyncBTreeOlc {
    fn bulk_load(&self, keys: &[u64], vals: &[u64]) {
        unsafe { ffi::btree_olc_bulk_load(&self.inner, keys.as_ptr(), vals.as_ptr(), keys.len()) }
    }
    async fn insert(&self, key: u64, value: u64) {
        ffi::btree_olc_insert(&self.inner, key, value);
    }
    async fn read(&self, key: u64) -> Option<u64> {
        let mut val = 0u64;
        if ffi::btree_olc_read(&self.inner, key, &mut val) { Some(val) } else { None }
    }
    async fn range_scan(&self, low: u64, limit: usize) -> usize {
        ffi::btree_olc_range_scan(&self.inner, low, limit)
    }
    fn memory_bytes(&self) -> usize {
        ffi::btree_olc_memory_bytes(&self.inner)
    }
}

/// Rust facade for Delta-Buffered Learned Index (XIndex-style foreground-merge baseline).
pub struct AsyncDeltaLearned {
    inner: Arc<cxx::UniquePtr<ffi::DeltaLearnedBaseline>>,
}
unsafe impl Send for AsyncDeltaLearned {}
unsafe impl Sync for AsyncDeltaLearned {}

impl AsyncDeltaLearned {
    /// Creates a new Delta-Learned baseline instance.
    pub fn new() -> Self {
        Self { inner: Arc::new(ffi::new_delta_learned()) }
    }
}

#[async_trait::async_trait]
impl AsyncIndex for AsyncDeltaLearned {
    fn bulk_load(&self, keys: &[u64], vals: &[u64]) {
        unsafe { ffi::delta_learned_bulk_load(&self.inner, keys.as_ptr(), vals.as_ptr(), keys.len()) }
    }
    async fn insert(&self, key: u64, value: u64) {
        ffi::delta_learned_insert(&self.inner, key, value);
    }
    async fn read(&self, key: u64) -> Option<u64> {
        let mut val = 0u64;
        if ffi::delta_learned_read(&self.inner, key, &mut val) { Some(val) } else { None }
    }
    async fn range_scan(&self, low: u64, limit: usize) -> usize {
        ffi::delta_learned_range_scan(&self.inner, low, limit)
    }
    fn memory_bytes(&self) -> usize {
        ffi::delta_learned_memory_bytes(&self.inner)
    }
}

/// Rust facade for Range-Partitioned Concurrent ALEX (2,048 partitions, per-partition RWLock).
pub struct AsyncAlexPart {
    inner: Arc<cxx::UniquePtr<ffi::AlexPartBaseline>>,
}
unsafe impl Send for AsyncAlexPart {}
unsafe impl Sync for AsyncAlexPart {}

impl AsyncAlexPart {
    pub fn new() -> Self {
        Self { inner: Arc::new(ffi::new_alex_part()) }
    }
}

#[async_trait::async_trait]
impl AsyncIndex for AsyncAlexPart {
    fn bulk_load(&self, keys: &[u64], vals: &[u64]) {
        unsafe { ffi::alex_part_bulk_load(&self.inner, keys.as_ptr(), vals.as_ptr(), keys.len()) }
    }
    async fn insert(&self, key: u64, value: u64) {
        ffi::alex_part_insert(&self.inner, key, value);
    }
    async fn read(&self, key: u64) -> Option<u64> {
        let mut val = 0u64;
        if ffi::alex_part_read(&self.inner, key, &mut val) { Some(val) } else { None }
    }
    async fn range_scan(&self, low: u64, limit: usize) -> usize {
        ffi::alex_part_range_scan(&self.inner, low, limit)
    }
    fn memory_bytes(&self) -> usize {
        ffi::alex_part_memory_bytes(&self.inner)
    }
}

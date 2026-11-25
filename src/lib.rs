//! CASTLE: Bridging the Engineering Impedance of Learned Indexes via Pure CoW and Deterministic Recovery
//!
//! Root crate facade exporting the core components.

#![deny(missing_docs)]

pub mod concurrency;
pub mod index;
pub mod recovery;
pub mod runtime;
pub mod storage;

#[path = "../baselines/ffi_wrapper/src/lib.rs"]
pub mod ffi_wrapper;

pub use ffi_wrapper::AsyncIndex;
pub use index::{CastleArcIndex, CastleIndex, IndexKey, IndexValue};

#[async_trait::async_trait]
impl AsyncIndex for CastleIndex<u64, u64> {
    fn bulk_load(&self, keys: &[u64], vals: &[u64]) {
        let pairs: Vec<(u64, u64)> = keys.iter().copied().zip(vals.iter().copied()).collect();
        CastleIndex::bulk_load(self, &pairs);
    }

    async fn insert(&self, key: u64, value: u64) {
        loop {
            let res = {
                let guard = concurrency::rcu_epoch::EpochGuard::enter();
                self.insert_with_guard(key, value, &guard)
            };
            match res {
                Ok(()) => return,
                Err(index::AtomicAppendError::CapacityExhausted) => {
                    // Cooperative backpressure retry until background consolidation frees the buffer
                    tokio::task::yield_now().await;
                }
            }
        }
    }

    async fn read(&self, key: u64) -> Option<u64> {
        let guard = concurrency::rcu_epoch::EpochGuard::enter();
        self.get(&key, &guard)
    }

    async fn range_scan(&self, low: u64, limit: usize) -> usize {
        CastleIndex::range_scan(self, low, limit).len()
    }

    fn memory_bytes(&self) -> usize {
        CastleIndex::memory_bytes(self)
    }

    fn backpressure_retries(&self) -> u64 {
        CastleIndex::backpressure_retries(self)
    }
}

#[async_trait::async_trait]
impl AsyncIndex for CastleArcIndex {
    fn bulk_load(&self, keys: &[u64], vals: &[u64]) {
        let pairs: Vec<(u64, u64)> = keys.iter().copied().zip(vals.iter().copied()).collect();
        CastleArcIndex::bulk_load(self, &pairs);
    }

    async fn insert(&self, _key: u64, _value: u64) {}

    async fn read(&self, key: u64) -> Option<u64> {
        CastleArcIndex::get(self, key)
    }

    async fn range_scan(&self, _low: u64, _limit: usize) -> usize {
        0
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}
